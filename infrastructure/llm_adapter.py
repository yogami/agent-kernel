"""Multi-provider LLM adapters with deterministic local mock for testing."""

from __future__ import annotations

import hashlib
import json
from typing import Any
import urllib.request

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

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding vector."""
        return deterministic_text_embedding(text)
