import os
import pytest
from infrastructure.llm_adapter import (
    AsyncXAIAdapter,
    AsyncGeminiAdapter,
    AsyncOllamaAdapter,
    AsyncOpenRouterAdapter,
    MockLLMAdapter,
    create_default_llm_adapter,
)

def test_xai_adapter_initialization():
    adapter = AsyncXAIAdapter(api_key="xai-test-key")
    assert adapter.base_url == "https://api.x.ai/v1"
    assert adapter.model == "grok-2-latest"
    assert "grok-beta" in adapter.fallback_models

def test_gemini_adapter_initialization():
    adapter = AsyncGeminiAdapter(api_key="gemini-test-key")
    assert adapter.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert adapter.model == "gemini-3.5-flash-lite"
    assert adapter.fallback_models == []

    custom_adapter = AsyncGeminiAdapter(
        api_key="gemini-test-key",
        model="gemini-2.5-flash",
        fallback_models=["gemini-2.5-pro"],
    )
    assert custom_adapter.model == "gemini-2.5-flash"
    assert "gemini-2.5-pro" in custom_adapter.fallback_models

def test_ollama_adapter_initialization():
    adapter = AsyncOllamaAdapter(model="llama3.1:8b")
    assert adapter.base_url == "http://localhost:11434/v1"
    assert adapter.model == "llama3.1:8b"

def test_factory_xai_priority(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("USE_OLLAMA", raising=False)
    adapter = create_default_llm_adapter()
    assert isinstance(adapter, AsyncXAIAdapter)

def test_factory_gemini(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("USE_OLLAMA", raising=False)
    adapter = create_default_llm_adapter()
    assert isinstance(adapter, AsyncGeminiAdapter)

def test_factory_ollama(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("USE_OLLAMA", "1")
    adapter = create_default_llm_adapter()
    assert isinstance(adapter, AsyncOllamaAdapter)

def test_factory_fallback_mock(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("USE_OLLAMA", raising=False)
    adapter = create_default_llm_adapter()
    assert isinstance(adapter, MockLLMAdapter)
