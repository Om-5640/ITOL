"""Mistral adapter — §8.3. OpenAI-compatible, no prompt caching."""
from __future__ import annotations
from typing import Any
from itol.adapters.openai_compatible_base import OpenAICompatibleAdapter


class MistralAdapter(OpenAICompatibleAdapter):
    _name = "mistral"
    base_url = "https://api.mistral.ai/v1"

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "none",
            "cache_read_discount": 0.0,
            "max_context": 32_768,
        }
