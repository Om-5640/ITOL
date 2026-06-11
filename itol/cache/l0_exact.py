"""
L0 exact-match cache — §6.1 + §6.2.

Key derivation
--------------
sha256( provider + model + normalize_params(params) + canonical_messages(messages) )

normalize_params
----------------
- Round temperature to 1 decimal place.
- Skip (return None → uncacheable) if temperature > 0.3 and NOT cache_high_temp.
- Exclude `stream` parameter.

TTL defaults (§6.2)
-------------------
CLASSIFICATION_SHORT  7d   (604 800 s)
EXTRACTION/FACTUAL    72h  (259 200 s)
SUMMARIZATION         72h
date-reference query   1h  (  3 600 s)
fallback              24h  ( 86 400 s)

CR-20
-----
set() silently skips caching if the response has error or finish_reason not in
{"stop", "end_turn", "tool_use"} — prevents partial/refused responses
from polluting the cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from itol.cache.store import Store
from itol.icr import ContentBlock, ContentType, ICR, ICRResponse, UsageStats


# ---------------------------------------------------------------------------
# Default TTLs by request class  (§6.2)
# ---------------------------------------------------------------------------

_CLASS_TTL: dict[str, int] = {
    "CLASSIFICATION_SHORT": 7 * 24 * 3600,    # 7 days
    "EXTRACTION":           72 * 3600,         # 72 hours
    "GENERATION_FACTUAL":   72 * 3600,
    "SUMMARIZATION":        72 * 3600,
    "REASONING":            72 * 3600,
    "GENERATION_CREATIVE":  24 * 3600,
    "AGENT_TOOL_LOOP":      3600,              # 1 hour
    "CHAT_OPEN":            24 * 3600,
    "AMBIGUOUS":            24 * 3600,
}

_DEFAULT_TTL = 24 * 3600    # fallback
_DATE_REF_TTL = 3600        # 1 hour for date-sensitive queries

# Finish reasons that indicate a complete, cacheable response
_CACHEABLE_FINISH_REASONS = frozenset({"stop", "end_turn", "tool_use", None})


# ---------------------------------------------------------------------------
# Param normalisation
# ---------------------------------------------------------------------------

def _normalize_params(params: dict[str, Any], cache_high_temp: bool = False) -> dict[str, Any] | None:
    """
    Normalise params for cache key inclusion.
    Returns None when the request must not be cached (high-temp, no cache_high_temp).
    """
    temp = params.get("temperature")
    if temp is not None:
        temp_f = float(temp)
        if temp_f > 0.3 and not cache_high_temp:
            return None             # §6.1: high-temperature → skip L0 caching
        params = {**params, "temperature": round(temp_f, 1)}

    return {k: v for k, v in params.items() if k != "stream"}


def _canonical_messages(icr: ICR) -> list[dict[str, Any]]:
    """Minimal canonical serialisation of messages for the cache key."""
    result = []
    # Include system blocks
    for b in icr.system:
        if b.text:
            result.append({"role": "system", "content": b.text})
    # Include conversation messages
    for msg in icr.messages:
        result.append({"role": msg.role, "content": msg.text_content()})
    # Include tool definitions (schema)
    for t in icr.tools:
        result.append({"tool": t.name, "params_hash": _sha256_short(json.dumps(t.parameters, sort_keys=True))})
    return result


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Response serialisation
# ---------------------------------------------------------------------------

def _serialize_response(r: ICRResponse) -> str:
    return json.dumps({
        "request_id": r.request_id,
        "provider":   r.provider,
        "model":      r.model,
        "content":    [{"type": b.type.value, "text": b.text} for b in r.content],
        "usage": {
            "input_tokens":       r.usage.input_tokens,
            "output_tokens":      r.usage.output_tokens,
            "cache_read_tokens":  r.usage.cache_read_tokens,
            "cache_write_tokens": r.usage.cache_write_tokens,
        },
        "finish_reason": r.finish_reason,
        "latency_ms":    r.latency_ms,
    })


def _deserialize_response(raw: str) -> ICRResponse:
    d = json.loads(raw)
    content = [
        ContentBlock(type=ContentType(c["type"]), text=c.get("text"))
        for c in d["content"]
    ]
    u = d["usage"]
    usage = UsageStats(
        input_tokens=u["input_tokens"],
        output_tokens=u["output_tokens"],
        cache_read_tokens=u.get("cache_read_tokens", 0),
        cache_write_tokens=u.get("cache_write_tokens", 0),
    )
    return ICRResponse(
        request_id=d["request_id"],
        provider=d["provider"],
        model=d["model"],
        content=content,
        usage=usage,
        finish_reason=d.get("finish_reason"),
        latency_ms=d.get("latency_ms", 0.0),
    )


# ---------------------------------------------------------------------------
# L0Cache
# ---------------------------------------------------------------------------

class L0Cache:
    """
    Exact-match L0 cache backed by SQLite (via Store).

    Key = sha256(provider + model + normalised_params + canonical_messages).
    Responses with error or non-terminal finish_reason are silently skipped (CR-20).
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def make_key(
        self,
        icr: ICR,
        cache_high_temp: bool = False,
    ) -> str | None:
        """
        Derive the cache key for this ICR.
        Returns None if the request is not cacheable (e.g. high temperature).
        """
        norm = _normalize_params(icr.params, cache_high_temp)
        if norm is None:
            return None

        key_obj = {
            "provider": icr.provider,
            "model":    icr.model,
            "params":   norm,
            "messages": _canonical_messages(icr),
        }
        serial = json.dumps(key_obj, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serial.encode()).hexdigest()

    def get(self, key: str, tenant_id: str) -> ICRResponse | None:
        """Return cached ICRResponse or None on miss/expiry."""
        raw = self._store.get_l0(key, tenant_id)
        if raw is None:
            return None
        try:
            return _deserialize_response(raw)
        except Exception:
            return None

    def set(
        self,
        key: str,
        tenant_id: str,
        response: ICRResponse,
        ttl_seconds: int = _DEFAULT_TTL,
        tokens_saved: int = 0,
    ) -> None:
        """
        Cache a response.

        CR-20: silently drops caching if:
          - response.error is not None
          - response.finish_reason is not a terminal reason
        """
        if response.error is not None:
            return
        if response.finish_reason not in _CACHEABLE_FINISH_REASONS:
            return
        self._store.set_l0(
            key, tenant_id, _serialize_response(response), ttl_seconds, tokens_saved
        )

    @staticmethod
    def ttl_for_class(request_class: str | None) -> int:
        """Return the default TTL in seconds for a given request class (§6.2)."""
        if request_class is None:
            return _DEFAULT_TTL
        return _CLASS_TTL.get(request_class.upper(), _DEFAULT_TTL)
