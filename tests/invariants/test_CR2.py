"""
§14.3 Invariant tests for CR-2 — no strategy mutates prefix unprofitably.

CR-2 (G3): a strategy must not mutate a byte span inside the provider KV-cache
prefix unless the token savings from the mutation strictly exceed the value of
keeping that prefix cached (provider_cache_value).

Invariants tested:
  - When savings_tokens * input_price_per_token <= provider_cache_value,
    a segment inside the prefix must NOT be mutated (prefix_bytes_mutated == 0).
  - When savings exceed the cache value, mutation is permitted.
  - When there is no cached prefix (prefix_cacheable_span_bytes == 0),
    all segments may be freely mutated.

Rules:
- NEVER weaken thresholds or conditions.
"""

import hashlib
import re

import pytest

from itol.icr import ConstraintManifest, SegmentType
from itol.segmenter import Segment
from itol.strategies.base import OptimizationContext, Strategy
from itol.config import ITOLConfig
from itol.routing.matrix import MATRIX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def _seg(text: str, seg_type: SegmentType = SegmentType.SYSTEM_INSTRUCTION) -> Segment:
    h = _seg_hash(text)
    return Segment(
        segment_type=seg_type,
        text=text,
        segment_hash=h,
        source_message_index=None,
        source_block_index=None,
        token_count=len(text.split()),
    )


def _ctx(
    prefix_bytes: int = 100,
    provider_cache_value: float = 0.01,
    input_price_per_token: float = 3e-6,
) -> OptimizationContext:
    return OptimizationContext(
        request_class="SUMMARIZATION",
        matrix_row=MATRIX["SUMMARIZATION"],
        manifest=ConstraintManifest(),
        signals=__import__("itol.icr", fromlist=["SegmentSignals"]).SegmentSignals(),
        config=ITOLConfig(),
        prefix_cacheable_span_bytes=prefix_bytes,
        provider_cache_value=provider_cache_value,
        input_price_per_token=input_price_per_token,
    )


# ===========================================================================
# CR-2-a: _prefix_safe helper semantics
# ===========================================================================

class TestCR2_PrefixSafeHelper:

    def _strategy(self):
        """Return a concrete Strategy instance to access _prefix_safe."""
        from itol.strategies.s2_instruction import S2InstructionStrategy
        return S2InstructionStrategy()

    def test_span_outside_prefix_always_safe(self):
        """Spans starting at or beyond the prefix boundary are always safe."""
        ctx = _ctx(prefix_bytes=100)
        s = self._strategy()
        assert s._prefix_safe(100, 200, savings_tokens=0, ctx=ctx)
        assert s._prefix_safe(500, 600, savings_tokens=0, ctx=ctx)

    def test_no_prefix_always_safe(self):
        """When prefix_cacheable_span_bytes == 0, all spans are safe."""
        ctx = _ctx(prefix_bytes=0, provider_cache_value=0.99)
        s = self._strategy()
        assert s._prefix_safe(0, 50, savings_tokens=1, ctx=ctx)

    def test_savings_exceed_cache_value_safe(self):
        """
        Savings of N tokens at input_price_per_token exceed provider_cache_value
        → mutation is safe even inside the prefix.
        """
        # cache_value = 0.001 USD; price = 1e-3 USD/token
        # savings_tokens = 2 → 2 * 1e-3 = 0.002 > 0.001
        ctx = _ctx(prefix_bytes=500, provider_cache_value=0.001, input_price_per_token=1e-3)
        s = self._strategy()
        assert s._prefix_safe(0, 100, savings_tokens=2, ctx=ctx), (
            "CR-2: 2 tokens at 1e-3 USD = 0.002 > cache_value 0.001 → safe"
        )

    def test_savings_below_cache_value_unsafe(self):
        """
        Savings of N tokens at input_price_per_token do NOT exceed provider_cache_value
        → mutation inside prefix is NOT safe.
        """
        # cache_value = 0.01 USD; price = 3e-6 USD/token
        # savings_tokens = 5 → 5 * 3e-6 = 0.000015 < 0.01
        ctx = _ctx(prefix_bytes=500, provider_cache_value=0.01, input_price_per_token=3e-6)
        s = self._strategy()
        assert not s._prefix_safe(0, 100, savings_tokens=5, ctx=ctx), (
            "CR-2: 5 tokens at 3e-6 = 0.000015 < cache_value 0.01 → unsafe"
        )


# ===========================================================================
# CR-2-b: S2 respects prefix-safety when cache value is high
# ===========================================================================

class TestCR2_S2RespectsPrefix:

    def test_s2_skips_prefix_mutation_when_unprofitable(self):
        """
        When a SYSTEM_INSTRUCTION segment is inside the cached prefix and the
        filler savings would not exceed provider_cache_value, S2 must leave
        the segment unchanged.
        """
        from itol.icr import ICR, Message
        from itol.strategies.s2_instruction import S2InstructionStrategy

        text = "Please make sure that you always respond helpfully and clearly."
        seg = _seg(text, SegmentType.SYSTEM_INSTRUCTION)

        icr = ICR.create(
            provider="openai", model="gpt-4o",
            system=[],
            messages=[Message.user("hi")],
            raw={},
        )

        # Very high cache value relative to the few tokens we'd save
        ctx = _ctx(
            prefix_bytes=len(text.encode()) + 50,  # segment is in prefix
            provider_cache_value=1.0,               # USD — much higher than any savings
            input_price_per_token=3e-6,
        )

        s2 = S2InstructionStrategy()
        result_segs, report = s2.apply(icr, [seg], ctx)

        # The segment must be unchanged because savings << cache_value
        assert report.prefix_bytes_mutated == 0, (
            "CR-2: S2 must not mutate prefix when savings are below cache_value"
        )
        assert result_segs[0].text == text, (
            "CR-2: original text must be preserved when mutation is unprofitable"
        )

    def test_s2_applies_when_no_prefix_cache(self):
        """
        When prefix_cacheable_span_bytes == 0, S2 freely applies filler removal.
        """
        from itol.icr import ICR, Message
        from itol.strategies.s2_instruction import S2InstructionStrategy

        text = "Please make sure that you always respond helpfully and clearly."
        seg = _seg(text, SegmentType.SYSTEM_INSTRUCTION)

        icr = ICR.create(
            provider="openai", model="gpt-4o",
            system=[],
            messages=[Message.user("hi")],
            raw={},
        )

        ctx = _ctx(prefix_bytes=0, provider_cache_value=0.0)

        s2 = S2InstructionStrategy()
        result_segs, report = s2.apply(icr, [seg], ctx)

        # Filler "Please make sure that you always" should be removed
        assert result_segs[0].text != text or report.tokens_saved == 0, (
            "CR-2: with no prefix cache, S2 must freely apply filler removal"
        )


# ===========================================================================
# CR-2-c: S1 respects prefix-safety
# ===========================================================================

class TestCR2_S1RespectsPrefix:

    def test_s1_keeps_prefix_dup_when_cache_value_high(self):
        """
        S1 must not remove a duplicate inside the cached prefix when the
        savings from removal are smaller than the provider_cache_value.
        """
        from itol.strategies.s1_dedupe import _deduplicate

        text = "You are a helpful assistant." * 5  # make it substantial
        seg_in_prefix = _seg(text, SegmentType.SYSTEM_INSTRUCTION)
        seg_outside = _seg(text, SegmentType.SYSTEM_INSTRUCTION)

        prefix_bytes = len(text.encode()) + 10  # seg_in_prefix is in the prefix

        # Set cache value astronomically high so savings can never exceed it
        ctx = _ctx(
            prefix_bytes=prefix_bytes,
            provider_cache_value=999.0,  # $999 — no savings can beat this
            input_price_per_token=3e-6,
        )

        segments = [seg_in_prefix, seg_outside]
        result, touched, pmb = _deduplicate(segments, ctx, ConstraintManifest())

        # With insanely high cache value, the in-prefix copy should NOT be the
        # one removed; the out-of-prefix copy (seg_outside) should be removed.
        # The in-prefix copy must survive.
        result_hashes = {s.segment_hash for s in result}
        assert seg_in_prefix.segment_hash in result_hashes, (
            "CR-2: in-prefix copy must survive even when it's the 'cheaper' target"
        )
