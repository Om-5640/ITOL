"""DeepSeek adapter — OpenAI-compatible endpoint, no prompt caching.

Base URL:  https://api.deepseek.com/v1
Auth:      Authorization: Bearer {DEEPSEEK_API_KEY}
Models:    deepseek-chat (128K context), deepseek-reasoner
Caching:   None (response.usage includes cache_read_input_tokens but there
           is no documented request-side cache parameter in the compat API)
Context:   128 000 tokens (deepseek-chat / deepseek-reasoner)

Quirks: thinking mode is enabled by default for all models; callers who
need deterministic outputs should pass thinking={"type":"disabled"} in
params — the base passthrough forwards any params in icr.params.
DeepSeek response usage also carries cache_creation_input_tokens and
cache_read_input_tokens; parse_response (base) extracts prompt_tokens and
completion_tokens, so those extra fields are preserved in ICRResponse.raw.
"""
from __future__ import annotations
from typing import Any
from itol.adapters.openai_compatible_base import OpenAICompatibleAdapter


class DeepSeekAdapter(OpenAICompatibleAdapter):
    _name = "deepseek"
    base_url = "https://api.deepseek.com/v1"

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "none",
            "cache_read_discount": 0.0,
            "max_context": 128_000,
        }
