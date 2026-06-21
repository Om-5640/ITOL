"""Google Gemini adapter — OpenAI-compatible endpoint (v1beta), no prompt caching.

Base URL:  https://generativelanguage.googleapis.com/v1beta/openai/
Auth:      Authorization: Bearer {GEMINI_API_KEY}
Models:    gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, etc.
Caching:   None (cached_content via extra_body is outside ITOL's passthrough)
Context:   1 048 576 tokens (Gemini 2.5 Pro/Flash; smaller for 2.0-flash)

Quirks: unknown params are silently ignored by the Gemini compat layer —
safe to forward standard OpenAI params without stripping.
"""
from __future__ import annotations
from typing import Any
from itol.adapters.openai_compatible_base import OpenAICompatibleAdapter


class GeminiAdapter(OpenAICompatibleAdapter):
    _name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "none",
            "cache_read_discount": 0.0,
            "max_context": 1_048_576,
        }
