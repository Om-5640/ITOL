"""OpenRouter adapter — OpenAI-compatible endpoint, no prompt caching.

Base URL:  https://openrouter.ai/api/v1
Auth:      Authorization: Bearer {OPENROUTER_API_KEY}
Models:    provider-prefixed, e.g. "openai/gpt-4o", "anthropic/claude-sonnet-4-5"
           Browse: https://openrouter.ai/models
Caching:   None (caching depends on underlying provider; not standardised)
Context:   200 000 tokens default (varies per model, up to 1M for some)

Optional headers OpenRouter recommends (for attribution / leaderboard):
  HTTP-Referer: <your-site-url>
  X-Title: <your-app-name>
These are NOT part of the request body; callers inject them at the HTTP
layer. The adapter itself stays body-only to remain transport-agnostic.

Quirks: model names MUST include the provider prefix ("openai/gpt-4o",
NOT "gpt-4o"). This is the caller's responsibility — the adapter forwards
icr.model verbatim. OpenRouter-specific params (openrouter_provider,
session_id) can be passed via icr.params and flow through body.update().
"""
from __future__ import annotations
from typing import Any
from itol.adapters.openai_compatible_base import OpenAICompatibleAdapter


class OpenRouterAdapter(OpenAICompatibleAdapter):
    _name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "none",
            "cache_read_discount": 0.0,
            "max_context": 200_000,
        }
