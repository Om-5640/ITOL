"""
§14.3 Invariant test for CR-21 — governing-span neighbor force-include in S3.

CR-21 (§15.1): after the greedy loop, for every manifest item whose:
  - value is found in a kept chunk, AND
  - governing_span is non-empty, AND
  - governing_span text overlaps a DROPPED neighboring chunk (±1 chunk)

…that neighbor must be force-included in the output.

This ensures that the qualifier/polarity context for a kept entity value is
never silently dropped at a chunk boundary.

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from itol.icr import (
    ConstraintManifest,
    ICR,
    ManifestItem,
    Message,
    SegmentSignals,
    SegmentType,
)
from itol.routing.matrix import ClassMatrix, L1Status, StrategyStatus
from itol.segmenter import Segment
from itol.signals import estimate_token_count
from itol.strategies.base import OptimizationContext
from itol.strategies.s3_window import S3WindowStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text: str, seg_type: SegmentType = SegmentType.RETRIEVED_DOC) -> Segment:
    return Segment(
        segment_type=seg_type,
        text=text,
        segment_hash=hashlib.sha256(text.encode()).hexdigest(),
        source_message_index=1,
        source_block_index=0,
        token_count=estimate_token_count(text),
    )


def _icr(query: str = "Find the value.") -> ICR:
    return ICR.create(
        provider="openai", model="gpt-4o",
        system=[],
        messages=[Message.user(query)],
        raw={},
    )


def _matrix_allow_s3() -> ClassMatrix:
    return ClassMatrix(
        strategies={
            "S1": StrategyStatus.ALLOWED,
            "S2": StrategyStatus.ALLOWED,
            "S3": StrategyStatus.ALLOWED,
            "S4": StrategyStatus.ALLOWED,
            "S5": StrategyStatus.ALLOWED,
            "S6": StrategyStatus.ALLOWED,
            "S7": StrategyStatus.ALLOWED,
        },
        l1=L1Status.ALLOWED,
        l1_tau=0.95,
        restrictions={"S3": 0.20},  # very low floor so greedy stops early
    )


def _ctx(manifest: ConstraintManifest, token_count: int = 20_000) -> OptimizationContext:
    from itol.config import ITOLConfig, ClassConfig

    cfg = ITOLConfig()
    cls_cfg = ClassConfig(s3_enabled=True, s3_class_budget=100, s3_mass_floor=0.20)
    cfg.class_configs["SUMMARIZATION"] = cls_cfg
    return OptimizationContext(
        request_class="SUMMARIZATION",
        matrix_row=_matrix_allow_s3(),
        manifest=manifest,
        signals=SegmentSignals(token_count=token_count),
        config=cfg,
    )


# ===========================================================================
# CR-21-S3: governing-span neighbor force-include
# ===========================================================================

class TestCR21_S3_GoverningSpanForceInclude:

    def test_neighbor_forced_when_governing_span_overlaps(self, monkeypatch):
        """
        CR-21-S3: when a kept chunk contains a manifest entity and the entity's
        governing_span text appears in the DROPPED neighboring chunk, that
        neighbor must be force-included.

        Design:
          - Two consecutive chunks in one RETRIEVED_DOC segment.
          - chunk 0: contains entity value "CRITICAL_VALUE_777"
            (high cosine → picked first; mass floor 0.20 met immediately
             since chunk 0 = ~50% of total tokens in this segment)
          - chunk 1: contains governing_span text "must never exceed"
            (low cosine → would not be picked by greedy alone)
          - governing_span = "CRITICAL_VALUE_777 must never exceed"

        Without CR-21, chunk 1 is dropped.
        With CR-21, chunk 1 must be force-included.
        """
        # Embed: chunk0 → high similarity, chunk1 → low similarity
        def fake_embed(texts, model="minilm"):
            result = []
            for t in texts:
                if "CRITICAL_VALUE_777" in t:
                    result.append(np.array([1.0, 0.0] + [0.0]*14, dtype=np.float32))
                elif "must never exceed" in t:
                    # Very low cosine → greedy ignores it
                    result.append(np.array([0.0, 0.1] + [0.0]*14, dtype=np.float32))
                else:
                    # Query or other chunks
                    result.append(np.array([1.0, 0.0] + [0.0]*14, dtype=np.float32))
            return np.array(result, dtype=np.float32)

        monkeypatch.setattr("itol.strategies.s3_window.embed", fake_embed)
        monkeypatch.setattr(
            "itol.strategies.s3_window.cosine",
            lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)),
        )

        # Build text where the two chunks will be clearly separated.
        # Use roughly 400 chars per chunk so they split into at least 2 windows.
        chunk0_text = "CRITICAL_VALUE_777 is specified in section 3. " + "context " * 40
        chunk1_text = "must never exceed the limits provided herein. " + "details " * 40

        # Combine so the chunker naturally splits them
        doc_text = chunk0_text.strip() + "\n\n" + chunk1_text.strip()

        # Manifest item: value in chunk0, governing_span spans into chunk1
        manifest = ConstraintManifest(items=[
            ManifestItem(
                item_type=ManifestItem.ItemType.ENTITY,
                value="CRITICAL_VALUE_777",
                governing_span="CRITICAL_VALUE_777 must never exceed",
            ),
        ])

        ctx = _ctx(manifest, token_count=20_000)
        strategy = S3WindowStrategy()
        icr = _icr("Find the critical value.")
        doc_seg = _seg(doc_text)

        new_segs, report = strategy.apply(icr, [doc_seg], ctx)
        kept_text = " ".join(s.text for s in new_segs)

        assert "CRITICAL_VALUE_777" in kept_text, (
            "Entity value must be in output (sanity check)"
        )
        assert "must never exceed" in kept_text, (
            "CR-21-S3 INVARIANT: governing_span neighbor chunk must be force-included "
            "when it contains qualifying context ('must never exceed') for the kept entity"
        )
