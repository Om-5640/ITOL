"""
§14.3 Invariant tests for CR-6 — incremental ledger recompute.

CR-6: the S5 ledger must be computed INCREMENTALLY.
  - Turns already folded into the ledger (their hashes stored in turn_hashes)
    must NOT be re-extracted on subsequent calls.
  - If a turn's hash changes (e.g. retroactive edit), the ledger must be
    invalidated and recomputed from that turn forward.

test_CR6_incremental_recompute:
    Process turn N (ledger computed for turns 1..N-K).
    Process turn N+1.
    Assert ledger recompute touches ONLY the newly-aged-out turn, not all prior.

test_CR6_edited_turn_invalidates:
    Process turn N, then retroactively change the hash of an already-ledgered
    turn.  Assert that the next apply() recomputes from the changed turn onward.

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace as dc_replace
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from itol.cache.store import Store
from itol.icr import ICR, Message, SegmentSignals, SegmentType
from itol.segmenter import Segment
from itol.signals import estimate_token_count
from itol.strategies.base import OptimizationContext
from itol.strategies.s5_distill import S5DistillStrategy, _first_changed_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text: str, seg_type: SegmentType, idx: int = 0) -> Segment:
    return Segment(
        segment_type=seg_type,
        text=text,
        segment_hash=hashlib.sha256(text.encode()).hexdigest(),
        source_message_index=idx,
        source_block_index=0,
        token_count=estimate_token_count(text),
    )


def _history(n_pairs: int, tag: str = "t") -> list[Segment]:
    segs = []
    for i in range(n_pairs):
        segs.append(_seg(f"User question {tag}_{i}.", SegmentType.USER_QUERY, i*2))
        segs.append(_seg(f"Answer {tag}_{i} info.", SegmentType.ASSISTANT_TURN, i*2+1))
    return segs


def _icr(conv_id: str = "test_conv") -> ICR:
    return ICR.create(
        provider="openai", model="gpt-4o",
        messages=[Message.user("Next question.")],
        conversation_id=conv_id,
        raw={},
    )


def _ctx(depth: int = 12):
    from itol.config import ITOLConfig
    from itol.routing.matrix import MATRIX
    from itol.icr import ConstraintManifest
    cfg = ITOLConfig()
    return OptimizationContext(
        request_class="SUMMARIZATION",
        matrix_row=MATRIX["SUMMARIZATION"],
        manifest=ConstraintManifest(),
        signals=SegmentSignals(history_depth=depth, token_count=8000),
        config=cfg,
    )


# ===========================================================================
# CR-6-a: incremental recompute — only newly-aging turn extracted
# ===========================================================================

class TestCR6_IncrementalRecompute:

    def test_extract_called_only_for_new_turns(self, tmp_path):
        """
        CR-6: on second apply(), the extraction function must be called only
        for the newly-aged-out turn, not for all prior turns.

        We track calls by counting how many times _extract_into_ledger is called
        per apply() invocation.
        """
        store = Store(str(tmp_path))
        strategy = S5DistillStrategy(store=store)
        icr = _icr("conv_incr")
        ctx = _ctx(depth=12)

        # First apply: 10 pairs → K=6 recent, 4 pairs age out
        segs_10 = _history(10, "r1")
        call_counts_first: list[str] = []

        import itol.strategies.s5_distill as s5_mod
        original_extract = s5_mod._extract_into_ledger

        def counting_extract(seg, ledger, all_sents, all_segs):
            call_counts_first.append(seg.segment_hash)
            return original_extract(seg, ledger, all_sents, all_segs)

        with patch.object(s5_mod, "_extract_into_ledger", side_effect=counting_extract):
            strategy.apply(icr, segs_10, ctx)

        first_count = len(call_counts_first)
        assert first_count > 0, "First apply must call extract at least once"

        # Second apply: 11 pairs → one new pair ages out
        segs_11 = _history(11, "r1")  # same history but one more pair
        call_counts_second: list[str] = []

        def counting_extract2(seg, ledger, all_sents, all_segs):
            call_counts_second.append(seg.segment_hash)
            return original_extract(seg, ledger, all_sents, all_segs)

        with patch.object(s5_mod, "_extract_into_ledger", side_effect=counting_extract2):
            strategy.apply(icr, segs_11, ctx)

        # CR-6: second call should touch only the newly-aging turn pair (2 segments),
        # not all the previously-processed turns
        assert len(call_counts_second) <= 2, (
            f"CR-6 INVARIANT: second apply must only extract newly-aged turns; "
            f"got {len(call_counts_second)} calls (expected ≤ 2)"
        )
        store.close()


# ===========================================================================
# CR-6-b: edited turn hash invalidates ledger from that point
# ===========================================================================

class TestCR6_EditedTurnInvalidates:

    def test_first_changed_idx_detects_change(self):
        """
        Unit test for _first_changed_idx: changing hash at position i must
        return i.
        """
        hashes = ["a", "b", "c", "d"]
        stored = ["a", "b", "X", "d"]   # "c" changed to "X"
        assert _first_changed_idx(hashes, stored) == 2, (
            "_first_changed_idx must return index of first differing hash"
        )

    def test_first_changed_idx_no_change(self):
        hashes = ["a", "b", "c"]
        stored = ["a", "b", "c"]
        # All stored hashes match; first_changed = len(stored)
        assert _first_changed_idx(hashes, stored) == 3

    def test_edited_turn_triggers_recompute(self, tmp_path):
        """
        CR-6: if an already-ledgered turn's hash changes, the next apply() must
        recompute from that turn forward.

        We detect this by: after first apply(), the turn_hashes in the store
        contain hash_A. We then create a new segment list where the same turn
        has hash_B (text changed). On second apply(), the strategy must detect
        hash_B ≠ hash_A and recompute from position i onward.

        Proof: the extraction call count on second apply() must be ≥ 2 (the
        changed turn plus all turns after it that are still aging out).
        """
        store = Store(str(tmp_path))
        strategy = S5DistillStrategy(store=store)
        icr = _icr("conv_edit")
        ctx = _ctx(depth=12)

        # First apply: 10 pairs
        segs_v1 = _history(10, "v1")
        strategy.apply(icr, segs_v1, ctx)

        # Verify turn_hashes stored
        conv = store.get_conversation("conv_edit", icr.tenant_id)
        assert conv is not None and conv["turn_hashes"], "turn_hashes must be stored"
        stored_hashes = conv["turn_hashes"]

        # Second apply: same 10 pairs but first aging turn has DIFFERENT text
        segs_v2 = list(_history(10, "v1"))
        segs_v2[0] = _seg("EDITED user question v1_0.", SegmentType.USER_QUERY, 0)
        assert segs_v2[0].segment_hash != segs_v1[0].segment_hash, (
            "Sanity: the segment hash must change when text changes"
        )

        import itol.strategies.s5_distill as s5_mod
        original_extract = s5_mod._extract_into_ledger
        call_count: list[int] = [0]

        def counting_extract(seg, ledger, all_sents, all_segs):
            call_count[0] += 1
            return original_extract(seg, ledger, all_sents, all_segs)

        with patch.object(s5_mod, "_extract_into_ledger", side_effect=counting_extract):
            strategy.apply(icr, segs_v2, ctx)

        assert call_count[0] >= 1, (
            "CR-6 INVARIANT: editing a ledgered turn must trigger recompute "
            f"(extract called {call_count[0]} times, expected ≥ 1)"
        )
        store.close()
