"""Multi-provider LLM adapters with deterministic local mock for testing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, AsyncIterator
import urllib.request
import httpx

from domain.ports import LLMProviderPort


def deterministic_text_embedding(text: str, dimensions: int = 128) -> list[float]:
    """Generate a deterministic normalized float vector for any string."""
    clean = text.strip().lower()
    vec = [0.0] * dimensions
    for i, char in enumerate(clean):
        idx = (ord(char) * (i + 1)) % dimensions
        vec[idx] += 1.0

    # Include hash components
    sha = hashlib.sha256(clean.encode("utf-8")).digest()
    for i in range(min(dimensions, len(sha))):
        vec[i] += float(sha[i]) / 255.0

    # L2 normalize
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        return [x / norm for x in vec]
    return vec


class MockLLMAdapter(LLMProviderPort):
    """Deterministic local mock LLM provider for unit tests, replay, and benchmark runs."""

    def __init__(self) -> None:
        self.scripted_responses: dict[str, dict[str, Any]] = {}
        self.custom_handler: Any = None
        self.call_history: list[dict[str, Any]] = []

    def set_scripted_response(self, user_query_substring: str, response: dict[str, Any]) -> None:
        """Register a canned response for a specific prompt pattern."""
        self.scripted_responses[user_query_substring.lower()] = response

    def set_handler(self, handler: Any) -> None:
        """Set dynamic callback handler for generating model outputs."""
        self.custom_handler = handler

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Generate response matching scripted patterns or dynamic callback."""
        self.call_history.append({
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
        })

        if self.custom_handler:
            return self.custom_handler(messages, tools)

        # Look for matching scripted response
        last_message = messages[-1]["content"] if messages else ""
        if isinstance(last_message, str):
            for pattern, resp in self.scripted_responses.items():
                if pattern in last_message.lower():
                    return resp

        # Default fallback response
        return {
            "content": f"Processed message: '{last_message}'",
            "tool_calls": [],
            "tokens_in": len(str(messages)) // 4,
            "tokens_out": 25,
            "cost_usd": 0.0001,
        }

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Asynchronously return response matching scripted patterns or dynamic callback."""
        return self.generate(messages, tools, temperature)

    async def stream_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chunks asynchronously with incremental token deltas or tool calls."""
        resp = self.generate(messages, tools, temperature)
        raw_tool_calls = resp.get("tool_calls", [])
        if raw_tool_calls:
            yield {
                "delta": "",
                "content": "",
                "tool_calls": raw_tool_calls,
                "tokens_in": resp.get("tokens_in", 0),
                "tokens_out": resp.get("tokens_out", 0),
                "cost_usd": resp.get("cost_usd", 0.0),
            }
            return

        content = resp.get("content", "")
        if not content:
            yield {
                "delta": "",
                "content": "",
                "tool_calls": [],
                "tokens_in": resp.get("tokens_in", 0),
                "tokens_out": 0,
                "cost_usd": resp.get("cost_usd", 0.0),
            }
            return

        words = content.split(" ")
        for idx, word in enumerate(words):
            chunk = word if idx == 0 else " " + word
            yield {
                "delta": chunk,
                "content": chunk,
                "tool_calls": [],
                "tokens_in": resp.get("tokens_in", 0) if idx == 0 else 0,
                "tokens_out": 1,
                "cost_usd": resp.get("cost_usd", 0.0) / max(len(words), 1),
            }

    def get_embedding(self, text: str) -> list[float]:
        """Return deterministic normalized embedding."""
        return deterministic_text_embedding(text)


class OpenRouterLLMAdapter(LLMProviderPort):
    """Adapter for OpenRouter multi-model gateway."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        fallback_models: list[str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.fallback_models = fallback_models or ["anthropic/claude-3-haiku", "meta-llama/llama-3-8b-instruct"]
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Call OpenRouter with fallback cascade."""
        models_to_try = [self.model] + self.fallback_models
        last_exception: Exception | None = None

        for target_model in models_to_try:
            try:
                payload: dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if tools:
                    payload["tools"] = tools

                req = urllib.request.Request(
                    self.api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://berlinailabs.de",
                        "X-Title": "Agent Kernel",
                    },
                )

                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choice = data["choices"][0]["message"]
                    usage = data.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)
                    cost_usd = (tokens_in * 0.00000015) + (tokens_out * 0.0000006)

                    return {
                        "content": choice.get("content") or "",
                        "tool_calls": choice.get("tool_calls") or [],
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "cost_usd": cost_usd,
                    }
            except Exception as exc:
                last_exception = exc
                continue

        raise RuntimeError(f"All LLM models in cascade failed. Last error: {last_exception}")

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run generate in asyncio worker thread."""
        return await asyncio.to_thread(self.generate, messages, tools, temperature)

    async def stream_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chunks asynchronously."""
        resp = await self.generate_async(messages, tools, temperature)
        raw_tool_calls = resp.get("tool_calls", [])
        if raw_tool_calls:
            yield {
                "delta": "",
                "content": "",
                "tool_calls": raw_tool_calls,
                "tokens_in": resp.get("tokens_in", 0),
                "tokens_out": resp.get("tokens_out", 0),
                "cost_usd": resp.get("cost_usd", 0.0),
            }
            return

        content = resp.get("content", "")
        words = content.split(" ")
        for idx, word in enumerate(words):
            chunk = word if idx == 0 else " " + word
            yield {
                "delta": chunk,
                "content": chunk,
                "tool_calls": [],
                "tokens_in": resp.get("tokens_in", 0) if idx == 0 else 0,
                "tokens_out": 1,
                "cost_usd": resp.get("cost_usd", 0.0) / max(len(words), 1),
            }

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding vector."""
        return deterministic_text_embedding(text)


class AsyncOpenRouterAdapter(LLMProviderPort):
    """Asynchronous adapter for OpenRouter and OpenAI-compatible endpoints with true SSE streaming."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        fallback_models: list[str] | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.fallback_models = fallback_models if fallback_models is not None else ["anthropic/claude-3-haiku", "meta-llama/llama-3-8b-instruct"]
        self.max_retries = max_retries
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://berlinailabs.de",
                    "X-Title": "Agent Kernel",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Synchronous generation using an isolated event loop run."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.generate_async(messages, tools, temperature)).result()
        return asyncio.run(self.generate_async(messages, tools, temperature))

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        client = self._get_client()
        models_to_try = [self.model] + self.fallback_models
        last_exception: Exception | None = None

        for target_model in models_to_try:
            payload: dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                payload["tools"] = tools

            for attempt in range(self.max_retries):
                try:
                    import asyncio
                    await asyncio.sleep(4.5)
                except ImportError:
                    pass
                try:
                    import asyncio
                    await asyncio.sleep(4.5)
                    print(f"Calling LLM attempt {attempt} for {target_model}...", flush=True)
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    choice = data["choices"][0]["message"]
                    usage = data.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)
                    cost_usd = (tokens_in * 0.00000015) + (tokens_out * 0.0000006)
                    print(f"LLM returned successfully.", flush=True)

                    return {
                        "content": choice.get("content") or "",
                        "tool_calls": choice.get("tool_calls") or [],
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "cost_usd": cost_usd,
                    }
                except httpx.HTTPStatusError as exc:
                    last_exception = exc
                    print(f"HTTP Error {exc.response.status_code}: {exc}", flush=True)
                    if exc.response.status_code == 429:
                        import asyncio
                        print("Rate limited by Google API. Sleeping for 35 seconds...", flush=True)
                        await asyncio.sleep(35.0)
                    elif attempt < self.max_retries - 1:
                        import asyncio, random
                        backoff = (2 ** attempt) * 1.0 + random.uniform(0.0, 0.5)
                        await asyncio.sleep(backoff)
                    continue
                except Exception as exc:
                    last_exception = exc
                    print(f"LLM Error: {exc}", flush=True)
                    if attempt < self.max_retries - 1:
                        import asyncio, random
                        backoff = (2 ** attempt) * 1.0 + random.uniform(0.0, 0.5)
                        await asyncio.sleep(backoff)
                    continue

        raise RuntimeError(f"All LLM models in cascade failed. Last error: {last_exception}")

    async def stream_async(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[dict[str, Any]]:
        client = self._get_client()
        models_to_try = [self.model] + self.fallback_models
        last_exception: Exception | None = None

        for target_model in models_to_try:
            payload: dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if tools:
                payload["tools"] = tools

            for attempt in range(self.max_retries):
                try:
                    import asyncio
                    await asyncio.sleep(4.5)
                except ImportError:
                    pass
                try:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        json=payload,
                    ) as response:
                        if response.status_code == 429:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(12.0)
                                continue
                        response.raise_for_status()
                        tool_calls_acc: dict[int, dict[str, Any]] = {}

                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})

                            # Text delta
                            text_delta = delta.get("content") or ""

                            # Tool call delta assembly
                            raw_tc_deltas = delta.get("tool_calls") or []
                            for tc_delta in raw_tc_deltas:
                                tc_id = tc_delta.get("id")
                                idx = tc_delta.get("index")
                                key = idx if idx is not None else (tc_id if tc_id else len(tool_calls_acc))
                                if key not in tool_calls_acc:
                                    tool_calls_acc[key] = {
                                        "id": tc_delta.get("id", ""),
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc_id:
                                    tool_calls_acc[key]["id"] = tc_id
                                if "extra_content" in tc_delta:
                                    tool_calls_acc[key]["extra_content"] = tc_delta["extra_content"]
                                fn = tc_delta.get("function", {})
                                if "name" in fn and fn["name"]:
                                    if not tool_calls_acc[key]["function"]["name"]:
                                        tool_calls_acc[key]["function"]["name"] = fn["name"]
                                    elif fn["name"] not in tool_calls_acc[key]["function"]["name"]:
                                        tool_calls_acc[key]["function"]["name"] += fn["name"]
                                if "arguments" in fn and fn["arguments"]:
                                    tool_calls_acc[key]["function"]["arguments"] += fn["arguments"]

                            if text_delta:
                                yield {
                                    "delta": text_delta,
                                    "content": text_delta,
                                    "tool_calls": [],
                                    "tokens_in": 0,
                                    "tokens_out": 1,
                                    "cost_usd": 0.0000006,
                                }

                        # If tool calls were accumulated across chunks
                        if tool_calls_acc:
                            ordered_calls = [tool_calls_acc[k] for k in sorted(tool_calls_acc.keys(), key=lambda x: str(x))]
                            yield {
                                "delta": "",
                                "content": "",
                                "tool_calls": ordered_calls,
                                "tokens_in": 0,
                                "tokens_out": 10,
                                "cost_usd": 0.000006,
                            }
                        return
                except Exception as exc:
                    last_exception = exc
                    if attempt < self.max_retries - 1:
                        backoff = (2 ** attempt) * 1.0 + random.uniform(0.0, 0.5)
                        await asyncio.sleep(backoff)
                    continue

        raise RuntimeError(f"All streaming models failed. Last error: {last_exception}")

    def get_embedding(self, text: str) -> list[float]:
        return deterministic_text_embedding(text)


class AsyncXAIAdapter(AsyncOpenRouterAdapter):
    """Asynchronous adapter for xAI Grok API."""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-2-latest",
        fallback_models: list[str] | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.x.ai/v1",
            fallback_models=fallback_models or ["grok-beta"],
            max_retries=max_retries,
            timeout=timeout,
        )


class AsyncGeminiAdapter(AsyncOpenRouterAdapter):
    """Asynchronous adapter for Google Gemini via Google AI Studio OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        fallback_models: list[str] | None = None,
        max_retries: int = 10,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            fallback_models=fallback_models or [],
            max_retries=max_retries,
            timeout=timeout,
        )


class AsyncOllamaAdapter(AsyncOpenRouterAdapter):
    """Asynchronous adapter for local Ollama OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            api_key="ollama",
            model=model,
            base_url=base_url,
            fallback_models=[],
            max_retries=max_retries,
            timeout=timeout,
        )


def _load_env_if_present() -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v


def create_default_llm_adapter() -> LLMProviderPort:
    """Factory creating the appropriate LLM adapter based on available environment variables."""
    _load_env_if_present()
    # 1. xAI / Grok
    xai_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if xai_key:
        return AsyncXAIAdapter(api_key=xai_key)

    # 2. Gemini / Google AI Studio
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return AsyncGeminiAdapter(api_key=gemini_key)

    # 3. OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return AsyncOpenRouterAdapter(api_key=openrouter_key)

    # 4. Standard OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return AsyncOpenRouterAdapter(
            api_key=openai_key,
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            fallback_models=[],
        )

    # 5. Local Ollama
    if os.getenv("USE_OLLAMA") == "1":
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return AsyncOllamaAdapter(model=model)

    # Default to deterministic mock adapter
    return MockLLMAdapter()

