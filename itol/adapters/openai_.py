"""OpenAI adapter — §8.3."""
from __future__ import annotations
from typing import Any
from itol.adapters.openai_compatible_base import OpenAICompatibleAdapter


class OpenAIAdapter(OpenAICompatibleAdapter):
    _name = "openai"
    base_url = "https://api.openai.com/v1"

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "prefix",
            "cache_read_discount": 0.10,
            "max_context": 128_000,
        }
