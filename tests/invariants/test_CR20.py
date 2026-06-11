"""
§14.3 Invariant tests for CR-20 — error/partial responses must not be cached.

CR-20: set() must silently skip caching if:
  - response.error is not None            (error / refusal)
  - response.finish_reason = "length"     (truncated / partial)
  - response.finish_reason = "content_filter"  (safety refusal)

Rules:
- NEVER weaken conditions to make tests pass.
"""

import tempfile

import pytest

from itol.cache.l0_exact import L0Cache, _CACHEABLE_FINISH_REASONS
from itol.cache.store import Store
from itol.icr import ContentBlock, ICRResponse, UsageStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_and_cache():
    tmpdir = tempfile.mkdtemp()
    store = Store(tmpdir)
    l0 = L0Cache(store)
    return l0, tmpdir


def _ok_response(request_id: str = "r1") -> ICRResponse:
    return ICRResponse(
        request_id=request_id,
        provider="openai",
        model="gpt-4o",
        content=[ContentBlock.text("OK response text.")],
        usage=UsageStats(input_tokens=10, output_tokens=5),
        finish_reason="stop",
    )


def _error_response(request_id: str = "r2") -> ICRResponse:
    return ICRResponse(
        request_id=request_id,
        provider="openai",
        model="gpt-4o",
        content=[],
        usage=UsageStats(),
        finish_reason="stop",
        error="Internal server error",
    )


def _partial_response(request_id: str = "r3", finish_reason: str = "length") -> ICRResponse:
    return ICRResponse(
        request_id=request_id,
        provider="openai",
        model="gpt-4o",
        content=[ContentBlock.text("Truncated...")],
        usage=UsageStats(input_tokens=10, output_tokens=3),
        finish_reason=finish_reason,
    )


# ===========================================================================
# CR-20-a: Error response must not be cached
# ===========================================================================

class TestCR20_ErrorNotCached:

    def test_error_response_not_cached(self):
        """
        CR-20: calling set() with response.error != None must not persist
        the entry; get() must return None.
        """
        l0, _ = _store_and_cache()
        resp = _error_response()
        key = "key_error"

        l0.set(key, "tenant1", resp, ttl_seconds=3600)
        result = l0.get(key, "tenant1")

        assert result is None, (
            "CR-20: error responses must not be stored in the L0 cache"
        )

    def test_normal_response_is_cached(self):
        """Control: a normal response (no error, finish_reason=stop) must be cached."""
        l0, _ = _store_and_cache()
        resp = _ok_response()
        key = "key_ok"

        l0.set(key, "tenant1", resp, ttl_seconds=3600)
        result = l0.get(key, "tenant1")

        assert result is not None, "Normal response must be cached (CR-20 control)"
        assert result.request_id == resp.request_id

    def test_none_error_field_is_cached(self):
        """response.error=None (the default) must be treated as cacheable."""
        l0, _ = _store_and_cache()
        resp = ICRResponse(
            request_id="r_none_err",
            provider="openai",
            model="gpt-4o",
            content=[ContentBlock.text("Hello")],
            usage=UsageStats(),
            finish_reason="stop",
            error=None,
        )
        key = "key_none_err"
        l0.set(key, "tenant1", resp, ttl_seconds=3600)
        assert l0.get(key, "tenant1") is not None


# ===========================================================================
# CR-20-b: Partial / stream-truncated responses must not be cached
# ===========================================================================

class TestCR20_PartialNotCached:

    def test_partial_stream_not_cached_length(self):
        """
        CR-20: finish_reason='length' (max_tokens truncation) must NOT be cached.
        """
        l0, _ = _store_and_cache()
        resp = _partial_response(finish_reason="length")
        key = "key_length"

        l0.set(key, "tenant1", resp, ttl_seconds=3600)
        assert l0.get(key, "tenant1") is None, (
            "CR-20: finish_reason='length' (partial stream) must not be cached"
        )

    def test_content_filter_not_cached(self):
        """CR-20: finish_reason='content_filter' (safety refusal) must NOT be cached."""
        l0, _ = _store_and_cache()
        resp = _partial_response(finish_reason="content_filter")
        key = "key_filter"

        l0.set(key, "tenant1", resp, ttl_seconds=3600)
        assert l0.get(key, "tenant1") is None, (
            "CR-20: finish_reason='content_filter' must not be cached"
        )

    def test_end_turn_is_cacheable(self):
        """finish_reason='end_turn' (Anthropic terminal reason) must be cached."""
        l0, _ = _store_and_cache()
        resp = ICRResponse(
            request_id="r_end_turn",
            provider="anthropic",
            model="claude-3-5-sonnet",
            content=[ContentBlock.text("Done.")],
            usage=UsageStats(),
            finish_reason="end_turn",
        )
        key = "key_end_turn"
        l0.set(key, "tenant1", resp, ttl_seconds=3600)
        assert l0.get(key, "tenant1") is not None, (
            "finish_reason='end_turn' is a terminal reason and must be cached"
        )

    def test_cacheable_finish_reasons_constant(self):
        """
        _CACHEABLE_FINISH_REASONS must include 'stop', 'end_turn', 'tool_use'
        and must NOT include 'length' or 'content_filter'.
        """
        assert "stop"        in _CACHEABLE_FINISH_REASONS
        assert "end_turn"    in _CACHEABLE_FINISH_REASONS
        assert "tool_use"    in _CACHEABLE_FINISH_REASONS
        assert "length"      not in _CACHEABLE_FINISH_REASONS
        assert "content_filter" not in _CACHEABLE_FINISH_REASONS


# ===========================================================================
# CR-20-c: High-temperature requests produce no key (uncacheable)
# ===========================================================================

class TestCR20_HighTemp:

    def test_high_temp_no_key(self):
        """
        §6.1: temperature > 0.3 (default cache_high_temp=False) → make_key returns None.
        """
        from itol.icr import ICR, Message
        icr = ICR.create(
            provider="openai",
            model="gpt-4o",
            messages=[Message.user("hi")],
            params={"temperature": 0.9},
            raw={},
        )
        l0, _ = _store_and_cache()
        assert l0.make_key(icr) is None, (
            "§6.1: temperature=0.9 without cache_high_temp → make_key must return None"
        )

    def test_low_temp_has_key(self):
        """temperature ≤ 0.3 must produce a valid key."""
        from itol.icr import ICR, Message
        icr = ICR.create(
            provider="openai",
            model="gpt-4o",
            messages=[Message.user("hi")],
            params={"temperature": 0.2},
            raw={},
        )
        l0, _ = _store_and_cache()
        key = l0.make_key(icr)
        assert key is not None and len(key) == 64, (
            "temperature=0.2 must produce a 64-char sha256 key"
        )

    def test_no_temp_has_key(self):
        """Requests with no temperature set must always produce a key."""
        from itol.icr import ICR, Message
        icr = ICR.create(
            provider="openai",
            model="gpt-4o",
            messages=[Message.user("hi")],
            raw={},
        )
        l0, _ = _store_and_cache()
        key = l0.make_key(icr)
        assert key is not None and len(key) == 64


# ===========================================================================
# CR-20-d: Cache round-trip correctness
# ===========================================================================

class TestCR20_RoundTrip:

    def test_round_trip_preserves_fields(self):
        """set() then get() must reconstruct the response faithfully."""
        l0, _ = _store_and_cache()
        resp = _ok_response("rt1")
        key = "key_rt"

        l0.set(key, "tenant_rt", resp, ttl_seconds=3600)
        retrieved = l0.get(key, "tenant_rt")

        assert retrieved is not None
        assert retrieved.request_id == resp.request_id
        assert retrieved.provider   == resp.provider
        assert retrieved.model      == resp.model
        assert retrieved.finish_reason == resp.finish_reason
        assert len(retrieved.content) == len(resp.content)
        assert retrieved.content[0].text == resp.content[0].text

    def test_expired_entry_not_returned(self):
        """A cached entry with ttl_seconds=0 must not be returned."""
        l0, _ = _store_and_cache()
        resp = _ok_response("exp1")
        key = "key_exp"

        l0.set(key, "tenant_exp", resp, ttl_seconds=0)
        # Entry was set with expires_at = now + 0 → already expired
        result = l0.get(key, "tenant_exp")
        assert result is None, "Expired L0 entry must not be returned"
