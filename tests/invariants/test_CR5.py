"""
§14.3 Invariant tests for CR-5 — S5 resurrection archive.

CR-5: BEFORE distilling/dropping a turn, S5 must write the turn's full text to
the docs table keyed by s5_turn:<turn_index>.  The ledger ONLY replaces the turn
in the segment list — the original remains retrievable.

test_CR5_distill_writes_before_replace:
    Verify that after apply(), the docs table contains the turn's full text
    AND the returned segment list contains the ledger block.  This confirms
    the write happened as part of the same apply() call that replaces segments
    (order: write → replace).

test_CR5_resurrection_roundtrip:
    Distill a turn, then call maybe_resurrect() with a query embedding computed
    from a phrase very similar to the distilled turn.  Assert the original text
    is returned verbatim when cos(query, turn) > cos(query, ledger) + 0.45.

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import hashlib
import tempfile

import numpy as np
import pytest

from itol.cache.store import Store
from itol.icr import ICR, Message, SegmentSignals, SegmentType
from itol.segmenter import Segment
from itol.signals import estimate_token_count
from itol.strategies.base import OptimizationContext
from itol.strategies.s5_distill import S5DistillStrategy, maybe_resurrect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seg(text: str, seg_type: SegmentType, msg_idx: int = 0) -> Segment:
    return Segment(
        segment_type=seg_type,
        text=text,
        segment_hash=hashlib.sha256(text.encode()).hexdigest(),
        source_message_index=msg_idx,
        source_block_index=0,
        token_count=estimate_token_count(text),
    )


def _icr(conversation_id: str = "conv_test") -> ICR:
    return ICR.create(
        provider="openai", model="gpt-4o",
        messages=[Message.user("Continue.")],
        conversation_id=conversation_id,
        raw={},
    )


def _ctx(history_depth: int = 10, token_count: int = 5000):
    from itol.config import ITOLConfig
    from itol.routing.matrix import MATRIX
    cfg = ITOLConfig()
    signals = SegmentSignals(history_depth=history_depth, token_count=token_count)
    return OptimizationContext(
        request_class="SUMMARIZATION",
        matrix_row=MATRIX["SUMMARIZATION"],
        manifest=__import__("itol.icr", fromlist=["ConstraintManifest"]).ConstraintManifest(),
        signals=signals,
        config=cfg,
    )


def _make_history(n_pairs: int, base: str = "topic") -> list[Segment]:
    """Build n_pairs of USER_QUERY + ASSISTANT_TURN segments."""
    segs: list[Segment] = []
    for i in range(n_pairs):
        segs.append(_make_seg(f"User question {i} about {base}.", SegmentType.USER_QUERY, msg_idx=i*2))
        segs.append(_make_seg(f"Assistant answer {i} about {base}.", SegmentType.ASSISTANT_TURN, msg_idx=i*2+1))
    return segs


# ===========================================================================
# CR-5-a: docs table written before segment replacement
# ===========================================================================

class TestCR5_WriteBeforeReplace:

    def test_distill_writes_to_docs_table(self, tmp_path):
        """
        CR-5: after apply(), the docs table must contain the full text of
        the aged-out turn AND the returned segments must contain the ledger block.
        """
        store = Store(str(tmp_path))
        strategy = S5DistillStrategy(store=store)
        icr = _icr(conversation_id="test_conv")
        ctx = _ctx(history_depth=10, token_count=5000)

        # Build 10 turn-pairs; with K=6, 4 pairs should age out
        segs = _make_history(10)
        aging_turn_text = segs[0].text  # first user query, will age out

        new_segs, report = strategy.apply(icr, segs, ctx)

        # Docs table must contain the aged turn
        doc = store.get_doc("s5_turn:0", icr.tenant_id, "test_conv")
        assert doc is not None, (
            "CR-5: docs table must contain the aged turn's text after apply()"
        )
        assert doc == aging_turn_text, (
            f"CR-5: docs table must contain exact original text; "
            f"got {doc!r}, expected {aging_turn_text!r}"
        )

        # Segment list must contain a ledger block
        all_text = " ".join(s.text for s in new_segs)
        assert "«Conversation ledger:" in all_text, (
            "CR-5: returned segments must contain the ledger block"
        )

        store.close()

    def test_aging_turns_not_in_output(self, tmp_path):
        """
        After apply(), the raw text of aged turns must not appear in the
        segment list (they have been replaced by the ledger block).
        """
        store = Store(str(tmp_path))
        strategy = S5DistillStrategy(store=store)
        icr = _icr("conv2")
        ctx = _ctx(history_depth=10, token_count=5000)

        segs = _make_history(10)
        # First 4 pairs will age out (10 - K=6 = 4)
        aging_texts = {segs[i].text for i in range(8)}  # 4 pairs = 8 segments

        new_segs, _ = strategy.apply(icr, segs, ctx)
        output_texts = {s.text for s in new_segs}

        # None of the exact aging turn texts should appear as their own segment
        for text in aging_texts:
            assert text not in output_texts, (
                f"CR-5: aged turn text must not appear verbatim in output segments: {text!r}"
            )
        store.close()


# ===========================================================================
# CR-5-b: resurrection round-trip
# ===========================================================================

class TestCR5_ResurrectionRoundtrip:

    def test_resurrection_returns_original_text(self, tmp_path):
        """
        CR-5: maybe_resurrect() must return the original turn text when
        cos(query_emb, turn_emb) > cos(query_emb, ledger_emb) + 0.45.

        We bypass the strategy and write docs directly to the store, then
        call maybe_resurrect with controlled cosine values (monkeypatching the
        embed module inside maybe_resurrect's actual call site).
        """
        store = Store(str(tmp_path))

        conversation_id = "conv_resurrect"
        tenant_id = "default"
        target_text = "The budget must not exceed $5 million for Q3."

        # Write the "aged" turn directly to docs
        store.save_doc("s5_turn:0", tenant_id, conversation_id, target_text, "hash_v1")

        # The ledger text (low relevance to the query)
        ledger_text = "«Conversation ledger: decisions=[none], facts=[none], open=[none]»"

        # Build query: its embedding is [1, 0, ...]; the turn embedding is also [1, 0, ...];
        # the ledger embedding is [0, 1, ...].
        # cos(query, turn) = 1.0; cos(query, ledger) = 0.0
        # delta = 1.0 - 0.0 = 1.0 > threshold=0.45 → resurrect
        query_emb = np.array([1.0] + [0.0]*15, dtype=np.float32)

        def fake_embed(texts, model="minilm"):
            result = []
            for t in texts:
                v = np.zeros(16, dtype=np.float32)
                if "«Conversation ledger" in t:
                    v[1] = 1.0    # orthogonal to query
                else:
                    v[0] = 1.0    # same direction as query
                result.append(v)
            return np.array(result, dtype=np.float32)

        import itol.embed.onnx_embedder as onnx_mod
        original_embed = onnx_mod.embed
        original_cosine = onnx_mod.cosine
        onnx_mod.embed = fake_embed
        onnx_mod.cosine = lambda a, b: float(np.dot(a / (np.linalg.norm(a)+1e-9),
                                                      b / (np.linalg.norm(b)+1e-9)))
        try:
            result = maybe_resurrect(
                query_emb,
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                ledger_text=ledger_text,
                store=store,
            )
        finally:
            onnx_mod.embed = original_embed
            onnx_mod.cosine = original_cosine

        assert result is not None, (
            "CR-5: maybe_resurrect must return original text when cos(query,turn) >> cos(query,ledger)+0.45"
        )
        assert result == target_text, (
            f"CR-5: returned text must be the verbatim original; got {result!r}"
        )
        store.close()

    def test_resurrection_returns_none_below_threshold(self, tmp_path):
        """
        Control: maybe_resurrect() returns None when no turn beats
        cos(query, ledger) + 0.45.
        """
        store = Store(str(tmp_path))
        conversation_id = "conv_no_resurrect"
        tenant_id = "default"

        # Store one turn, but its embedding = ledger embedding (delta = 0)
        store.save_doc("s5_turn:0", tenant_id, conversation_id, "Some turn text.", "h1")

        ledger_text = "«Conversation ledger: decisions=[none], facts=[none], open=[none]»"

        def fake_embed(texts, model="minilm"):
            # All embeddings identical → delta = 0, never above 0.45
            result = []
            for _ in texts:
                v = np.array([1.0] + [0.0]*15, dtype=np.float32)
                result.append(v)
            return np.array(result, dtype=np.float32)

        import itol.embed.onnx_embedder as onnx_mod
        original_embed = onnx_mod.embed
        original_cosine = onnx_mod.cosine
        onnx_mod.embed = fake_embed
        onnx_mod.cosine = lambda a, b: float(np.dot(a / (np.linalg.norm(a)+1e-9),
                                                      b / (np.linalg.norm(b)+1e-9)))
        query_emb = np.array([1.0] + [0.0]*15, dtype=np.float32)
        try:
            result = maybe_resurrect(
                query_emb,
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                ledger_text=ledger_text,
                store=store,
            )
        finally:
            onnx_mod.embed = original_embed
            onnx_mod.cosine = original_cosine

        assert result is None, (
            "Control: maybe_resurrect must return None when no turn exceeds threshold"
        )
        store.close()
