"""Groq adapter — §8.3. OpenAI-compatible, no prompt caching."""
from __future__ import annotations
from typing import Any
from itol.adapters.openai_compatible_base import OpenAICompatibleAdapter


class GroqAdapter(OpenAICompatibleAdapter):
    _name = "groq"
    base_url = "https://api.groq.com/openai/v1"

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "none",
            "cache_read_discount": 0.0,
            "max_context": 131_072,
        }
