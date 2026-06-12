"""
§14.3 Invariant tests for S4 RACR strategy.

Key invariants:
  - test_s4_full_doc_intent_disables_replacement
    Query containing "entire report" / "whole document" → S4 does not activate
    even when a ≥2000-token doc has been seen ≥2× in the conversation.
    S3 (not S4) should handle such segments.

  - test_s4_matrix_disabled_classes
    S4 must not activate for REASONING, GENERATION_CREATIVE, CLASSIFICATION_SHORT,
    AGENT_TOOL_LOOP, EXTRACTION (verified via matrix).
    Note: AGENT_TOOL_LOOP has S4 ALLOWED in the matrix; but EXTRACTION is DISABLED.

  - test_s4_second_occurrence_replaces
    After a doc is stored in the first request, the second request replaces the
    inline doc with a header + excerpt.

  - test_s4_confidence_floor_falls_back_to_s3
    When all chunk scores are below conf_floor, the fallback produces the
    [mass-floor mass of the original doc], not the header format.

  - test_s4_cr9_new_doc_version_invalidates_l2
    When a near-duplicate (different content) appears, CR-9 must invalidate
    L2 plans that depended on the old doc.

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import hashlib
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from itol.cache.store import Store
from itol.config import ITOLConfig
from itol.icr import ICR, ConstraintManifest, Message, SegmentSignals, SegmentType
from itol.routing.matrix import MATRIX, StrategyStatus
from itol.segmenter import Segment
from itol.signals import estimate_token_count
from itol.strategies.base import OptimizationContext
from itol.strategies.s4_racr import (
    S4RACRStrategy,
    _full_doc_intent,
    _title_guess,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BIG_DOC = (
    "This is a very detailed technical report about quarterly performance. "
    "The report covers financial metrics, operational efficiency, and strategic "
    "initiatives. " * 200   # ~2200 tokens
)

_SHORT_DOC = "Short text. " * 50   # ~200 tokens — below 2000-token floor


def _make_seg(text: str, seg_type: SegmentType = SegmentType.RETRIEVED_DOC) -> Segment:
    h = hashlib.sha256(text.encode()).hexdigest()
    return Segment(
        segment_type=seg_type,
        text=text,
        segment_hash=h,
        source_message_index=0,
        source_block_index=0,
        token_count=estimate_token_count(text),
    )


def _make_icr(query: str = "Tell me about the report.", conv_id: str = "conv_s4") -> ICR:
    return ICR.create(
        provider="openai", model="gpt-4o",
        messages=[Message.user(query)],
        raw={},
        conversation_id=conv_id,
    )


def _ctx(request_class: str = "SUMMARIZATION") -> OptimizationContext:
    return OptimizationContext(
        request_class=request_class,
        matrix_row=MATRIX[request_class],
        manifest=ConstraintManifest(),
        signals=SegmentSignals(history_depth=2, token_count=5000),
        config=ITOLConfig(),
    )


# ===========================================================================
# Full-document intent gate
# ===========================================================================

class TestS4FullDocIntent:

    def test_full_doc_intent_detected_entire_report(self):
        """'entire report' must be detected as full-doc intent."""
        assert _full_doc_intent("Please summarise the entire report.")

    def test_full_doc_intent_detected_whole_document(self):
        assert _full_doc_intent("Extract entities from the whole document.")

    def test_full_doc_intent_detected_all_pages(self):
        assert _full_doc_intent("I need all pages analysed.")

    def test_full_doc_intent_detected_every_section(self):
        assert _full_doc_intent("Review every section of the brief.")

    def test_normal_query_not_full_doc_intent(self):
        """A normal retrieval query must NOT be flagged."""
        assert not _full_doc_intent("What was the revenue in Q3?")
        assert not _full_doc_intent("Summarize the key findings.")

    def test_s4_applies_returns_false_for_full_doc_intent(self, tmp_path):
        """
        S4 MUST NOT activate when the user query contains a full-doc intent
        phrase, even when a qualifying doc has been seen ≥2 times.

        This ensures S3 (not S4) handles such requests.
        """
        store = Store(str(tmp_path))
        # Save the doc as if it was seen before (simulates 1st occurrence)
        doc_hash = hashlib.sha256(_BIG_DOC.encode()).hexdigest()
        store.save_doc(doc_hash, "default", "conv_s4", _BIG_DOC, doc_hash)

        strategy = S4RACRStrategy(store=store)
        # Query with full-doc intent
        icr = _make_icr("Please analyse the entire report.", conv_id="conv_s4")
        segments = [_make_seg(_BIG_DOC)]
        ctx = _ctx("SUMMARIZATION")

        applies = strategy.applies(icr, segments, ctx)
        assert applies is False, (
            "S4 MUST NOT activate when the query requests the entire/whole document"
        )
        store.close()

    def test_s4_applies_true_without_full_doc_intent(self, tmp_path):
        """
        Control: S4 applies when the doc is qualifying and NO full-doc intent.
        """
        store = Store(str(tmp_path))
        doc_hash = hashlib.sha256(_BIG_DOC.encode()).hexdigest()
        store.save_doc(doc_hash, "default", "conv_s4_ctrl", _BIG_DOC, doc_hash)

        strategy = S4RACRStrategy(store=store)
        icr = _make_icr("What were the Q3 revenue figures?", conv_id="conv_s4_ctrl")
        segments = [_make_seg(_BIG_DOC)]
        ctx = _ctx("SUMMARIZATION")

        applies = strategy.applies(icr, segments, ctx)
        assert applies is True, "Control: S4 should apply when no full-doc intent"
        store.close()


# ===========================================================================
# Matrix disabled classes
# ===========================================================================

class TestS4MatrixGating:

    @pytest.mark.parametrize("req_class", [
        "EXTRACTION", "REASONING", "GENERATION_CREATIVE",
        "CLASSIFICATION_SHORT", "CHAT_OPEN",
    ])
    def test_matrix_disabled_classes_do_not_activate_s4(self, tmp_path, req_class):
        """S4 must not activate for classes where the matrix marks it DISABLED."""
        store = Store(str(tmp_path))
        doc_hash = hashlib.sha256(_BIG_DOC.encode()).hexdigest()
        store.save_doc(doc_hash, "default", "conv_mat", _BIG_DOC, doc_hash)

        strategy = S4RACRStrategy(store=store)
        icr = _make_icr("What is described here?", conv_id="conv_mat")
        segments = [_make_seg(_BIG_DOC)]
        ctx = _ctx(req_class)

        # Verify matrix agrees
        assert MATRIX[req_class].strategies.get("S4") == StrategyStatus.DISABLED, (
            f"Test precondition: {req_class} must have S4=DISABLED in matrix"
        )
        applies = strategy.applies(icr, segments, ctx)
        assert applies is False, (
            f"S4 must not activate for {req_class} (matrix DISABLED)"
        )
        store.close()

    def test_s4_allowed_for_summarization(self, tmp_path):
        """SUMMARIZATION must have S4 ALLOWED."""
        store = Store(str(tmp_path))
        doc_hash = hashlib.sha256(_BIG_DOC.encode()).hexdigest()
        store.save_doc(doc_hash, "default", "conv_sum", _BIG_DOC, doc_hash)

        strategy = S4RACRStrategy(store=store)
        icr = _make_icr("What were the key findings?", conv_id="conv_sum")
        segments = [_make_seg(_BIG_DOC)]
        ctx = _ctx("SUMMARIZATION")

        assert MATRIX["SUMMARIZATION"].strategies.get("S4") == StrategyStatus.ALLOWED
        assert strategy.applies(icr, segments, ctx) is True
        store.close()


# ===========================================================================
# Second occurrence replaces inline doc
# ===========================================================================

class TestS4SecondOccurrenceReplacement:

    def test_first_occurrence_not_replaced(self, tmp_path):
        """
        First time S4 sees a doc, it stores it but does NOT replace the segment
        (returns it unchanged).
        """
        store = Store(str(tmp_path))
        strategy = S4RACRStrategy(store=store)
        icr = _make_icr("Summarise this.", conv_id="conv_first")
        seg = _make_seg(_BIG_DOC)
        ctx = _ctx("SUMMARIZATION")

        new_segs, report = strategy.apply(icr, [seg], ctx)
        # Doc should now be stored
        doc_hash = hashlib.sha256(_BIG_DOC.encode()).hexdigest()
        stored = store.get_doc(doc_hash, "default", "conv_first")
        assert stored is not None, "S4 must save the doc on first occurrence"
        # But segment text must be UNCHANGED (no replacement yet)
        assert new_segs[0].text == _BIG_DOC, (
            "S4 must NOT replace on first occurrence"
        )
        store.close()

    def test_second_occurrence_produces_replacement_header(self, tmp_path):
        """
        Second time the same doc appears, S4 replaces it with a header + excerpts.
        """
        store = Store(str(tmp_path))
        strategy = S4RACRStrategy(store=store)
        conv_id = "conv_second"

        # Simulate first occurrence (store the doc)
        doc_hash = hashlib.sha256(_BIG_DOC.encode()).hexdigest()
        store.save_doc(doc_hash, "default", conv_id, _BIG_DOC, doc_hash)

        icr = _make_icr("What are the key metrics?", conv_id=conv_id)
        seg = _make_seg(_BIG_DOC)
        ctx = _ctx("SUMMARIZATION")

        new_segs, report = strategy.apply(icr, [seg], ctx)

        # Header format must be present
        assert f"Document {doc_hash[:8]}" in new_segs[0].text, (
            "S4 replacement must include the doc header with hash prefix"
        )
        assert "available — relevant excerpts below" in new_segs[0].text, (
            "S4 replacement must include the availability line"
        )
        assert new_segs[0].text != _BIG_DOC, "S4 must change the segment text"
        assert new_segs[0].token_count < seg.token_count, (
            "Replacement must be smaller than original"
        )
        assert report.activated is True
        store.close()

    def test_short_doc_below_floor_not_replaced(self, tmp_path):
        """Docs below s4_min_doc_tokens (2000) must not be touched by S4."""
        store = Store(str(tmp_path))
        strategy = S4RACRStrategy(store=store)
        conv_id = "conv_short"

        short_doc = _SHORT_DOC
        doc_hash = hashlib.sha256(short_doc.encode()).hexdigest()
        store.save_doc(doc_hash, "default", conv_id, short_doc, doc_hash)

        icr = _make_icr("What does it say?", conv_id=conv_id)
        seg = _make_seg(short_doc)
        ctx = _ctx("SUMMARIZATION")

        new_segs, report = strategy.apply(icr, [seg], ctx)
        assert new_segs[0].text == short_doc, (
            "S4 must not touch docs below the minimum token threshold"
        )
        store.close()


# ===========================================================================
# Retrieval confidence floor — fallback to S3 mass
# ===========================================================================

class TestS4ConfidenceFloorFallback:

    def test_low_confidence_falls_back_to_mass_format(self, tmp_path):
        """
        When top-1 chunk score < conf_floor (0.35), the replacement must NOT
        use the header format; it should use S3-style mass inclusion instead.
        """
        store = Store(str(tmp_path))
        conv_id = "conv_conf"
        doc_hash = hashlib.sha256(_BIG_DOC.encode()).hexdigest()
        store.save_doc(doc_hash, "default", conv_id, _BIG_DOC, doc_hash)

        strategy = S4RACRStrategy(store=store)

        import itol.embed.onnx_embedder as onnx_mod
        original_embed = onnx_mod.embed
        original_cosine = onnx_mod.cosine

        # Force all scores to be below conf_floor (cosine=0.0 for all chunks)
        def zero_embed(texts, model="minilm"):
            return np.zeros((len(texts), 16), dtype=np.float32)

        def zero_cosine(a, b):
            return 0.0

        onnx_mod.embed = zero_embed
        onnx_mod.cosine = zero_cosine
        try:
            import itol.strategies.s4_racr as s4_mod
            s4_mod.embed = zero_embed
            s4_mod.cosine = zero_cosine

            icr = _make_icr("What does the report say?", conv_id=conv_id)
            seg = _make_seg(_BIG_DOC)
            ctx = _ctx("SUMMARIZATION")

            new_segs, report = strategy.apply(icr, [seg], ctx)
            replacement = new_segs[0].text

            # Must NOT have the header format (confidence too low)
            assert "available — relevant excerpts below" not in replacement, (
                "When confidence < floor, must NOT use header+excerpt format"
            )
            # The fallback path must have returned SOMETHING (not empty)
            assert len(replacement) > 0, "Fallback must return non-empty text"
        finally:
            onnx_mod.embed = original_embed
            onnx_mod.cosine = original_cosine
            s4_mod.embed = original_embed
            s4_mod.cosine = original_cosine

        store.close()


# ===========================================================================
# CR-9 — L2 invalidation on new doc version
# ===========================================================================

class TestS4CR9Invalidation:

    def test_new_doc_version_invalidates_l2_plans(self, tmp_path):
        """
        CR-9: when a near-duplicate doc (same shape, different content) appears
        for the second time, any L2 plan depending on the OLD doc hash must be
        tombstoned.
        """
        store = Store(str(tmp_path))
        conv_id = "conv_cr9"
        tenant_id = "default"

        # v1: original doc
        doc_v1 = _BIG_DOC
        hash_v1 = hashlib.sha256(doc_v1.encode()).hexdigest()

        # v2: slightly edited (different hash, but cosine ≥ 0.97)
        doc_v2 = doc_v1.replace(
            "quarterly performance",
            "quarterly performance (updated)",
        )
        hash_v2 = hashlib.sha256(doc_v2.encode()).hexdigest()
        assert hash_v1 != hash_v2

        # Store v1 and an L2 plan that depends on it
        store.save_doc(hash_v1, tenant_id, conv_id, doc_v1, hash_v1)
        import json
        store.set_l2_plan(
            template_sig="tmpl_cr9",
            tenant_id=tenant_id,
            plan_json='{"plan": "test"}',
            depends_on_doc_versions=json.dumps([hash_v1]),
        )

        strategy = S4RACRStrategy(store=store)

        import itol.embed.onnx_embedder as onnx_mod
        original_embed = onnx_mod.embed
        original_cosine = onnx_mod.cosine

        # Force cosine to return ≥ 0.97 for the near-dup check
        call_count = {"n": 0}
        def high_cosine_embed(texts, model="minilm"):
            vecs = []
            for _ in texts:
                vecs.append(np.array([1.0] + [0.0] * 15, dtype=np.float32))
            return np.array(vecs, dtype=np.float32)

        def high_cosine(a, b):
            return 0.98   # near-dup detected

        onnx_mod.embed = high_cosine_embed
        onnx_mod.cosine = high_cosine
        try:
            import itol.strategies.s4_racr as s4_mod
            s4_mod.embed = high_cosine_embed
            s4_mod.cosine = high_cosine

            icr = _make_icr("Summarise the key updates.", conv_id=conv_id)
            seg = _make_seg(doc_v2)
            ctx = _ctx("SUMMARIZATION")

            _, report = strategy.apply(icr, [seg], ctx)

            # The L2 plan for hash_v1 must now be gone
            remaining = store.get_l2_plan("tmpl_cr9", tenant_id)
            assert remaining is None, (
                "CR-9: L2 plan depending on old doc version must be tombstoned "
                "when a new doc version appears"
            )
        finally:
            onnx_mod.embed = original_embed
            onnx_mod.cosine = original_cosine
            s4_mod.embed = original_embed
            s4_mod.cosine = original_cosine

        store.close()


# ===========================================================================
# Title guess helper
# ===========================================================================

class TestTitleGuess:

    def test_short_first_line_used_as_title(self):
        text = "Quarterly Performance Report\nThis report covers..."
        assert _title_guess(text) == "Quarterly Performance Report"

    def test_sentence_ending_first_line_falls_back_to_five_words(self):
        text = "This is a sentence.\nMore content follows."
        title = _title_guess(text)
        assert "..." in title
        words = title.replace("...", "").split()
        assert len(words) == 5

    def test_long_first_line_falls_back_to_five_words(self):
        text = "A very very very very very long title that exceeds the limit\nMore."
        title = _title_guess(text)
        assert "..." in title
