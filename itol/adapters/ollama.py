"""Ollama adapter — §8.3. OpenAI-compatible local endpoint, no auth."""
from __future__ import annotations
from typing import Any
from itol.adapters.openai_compatible_base import OpenAICompatibleAdapter


class OllamaAdapter(OpenAICompatibleAdapter):
    _name = "ollama"
    base_url = "http://localhost:11434/v1"

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "none",
            "cache_read_discount": 0.0,
            "max_context": 32_768,
        }
