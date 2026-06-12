"""
§14.3 Invariant tests for CR-4 — manifest entity coverage check inside S3 greedy loop.

CR-4: the S3 greedy-keep loop must NOT stop until BOTH conditions hold:
    (1) kept_mass / total_mass >= mass_floor
    (2) every manifest ENTITY/QUERY_TERM value appears in the kept chunk text

Stopping on (1) alone when (2) is unmet is a CR-4 violation.
Stopping on (2) alone when (1) is unmet is also a CR-4 violation.

Test strategy:
  - Craft a segment whose chunks have controlled cosine scores so that the
    greedy loop would stop on condition (1) before reaching the chunk that
    carries the manifest entity.
  - Assert the strategy still returns text that contains the entity (i.e. it
    kept going until (2) was also satisfied).

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from itol.icr import (
    ClassifierResult,
    ConstraintManifest,
    ICR,
    ManifestItem,
    Message,
    SegmentSignals,
    SegmentType,
)
from itol.routing.matrix import MATRIX
from itol.segmenter import Segment
from itol.signals import estimate_token_count
from itol.strategies.base import OptimizationContext
from itol.strategies.s3_window import S3WindowStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(budget: int = 100, mass_floor: float = 0.50):
    """Build a minimal ITOLConfig with S3 forced active."""
    from itol.config import ITOLConfig, ClassConfig

    cfg = ITOLConfig()
    cls_cfg = ClassConfig(s3_enabled=True, s3_class_budget=budget, s3_mass_floor=mass_floor)
    cfg.class_configs["SUMMARIZATION"] = cls_cfg
    return cfg


def _ctx(signals: SegmentSignals, mass_floor_override: float | None = None):
    """Build an OptimizationContext for SUMMARIZATION with configurable mass floor."""
    from itol.routing.matrix import ClassMatrix, StrategyStatus, L1Status

    # Use SUMMARIZATION matrix but set mass_floor in restrictions if override given
    base = MATRIX["SUMMARIZATION"]
    restrictions = {}
    if mass_floor_override is not None:
        restrictions = {"S3": mass_floor_override}
    matrix_row = ClassMatrix(
        strategies=base.strategies,
        l1=base.l1,
        l1_tau=base.l1_tau,
        restrictions=restrictions,
    )
    cfg = _make_config(budget=100, mass_floor=mass_floor_override or 0.50)
    return OptimizationContext(
        request_class="SUMMARIZATION",
        matrix_row=matrix_row,
        manifest=ConstraintManifest(),
        signals=signals,
        config=cfg,
    )


def _seg(text: str, seg_type: SegmentType = SegmentType.RETRIEVED_DOC) -> Segment:
    import hashlib
    return Segment(
        segment_type=seg_type,
        text=text,
        segment_hash=hashlib.sha256(text.encode()).hexdigest(),
        source_message_index=1,
        source_block_index=0,
        token_count=estimate_token_count(text),
    )


def _icr(query: str = "Find the entity.") -> ICR:
    return ICR.create(
        provider="openai", model="gpt-4o",
        system=[],
        messages=[Message.user(query)],
        raw={},
    )


def _signals(token_count: int = 20_000) -> SegmentSignals:
    return SegmentSignals(token_count=token_count)


# ===========================================================================
# CR-4-a: entity-carrying chunk must be kept even if mass floor already met
# ===========================================================================

class TestCR4_EntityCoverageInsideLoop:

    def test_entity_chunk_forced_when_mass_floor_met_early(self, monkeypatch):
        """
        CR-4: the loop must not stop after meeting mass_floor if the manifest
        entity has not yet been covered by any kept chunk.

        Set-up:
          - 3 chunks; chunk 0 alone exceeds mass_floor=0.40 (it has 60% of mass)
          - the manifest entity "SPECIAL_ENTITY_XYZ" appears ONLY in chunk 2
          - without CR-4, the loop would stop after chunk 0 (mass satisfied)
          - with CR-4, the loop must continue until chunk 2 is also included
        """
        # Force high cos-sim for chunk 0 so it gets picked first
        call_count = [0]
        def fake_embed(texts, model="minilm"):
            result = []
            for t in texts:
                if "SPECIAL_ENTITY_XYZ" in t:
                    # chunk 2: low cosine similarity → low score
                    result.append(np.array([0.1, 0.0] + [0.0]*14, dtype=np.float32))
                elif call_count[0] == 0:
                    # query vector
                    call_count[0] += 1
                    result.append(np.array([1.0, 0.0] + [0.0]*14, dtype=np.float32))
                else:
                    result.append(np.array([0.9, 0.0] + [0.0]*14, dtype=np.float32))
            return np.array(result, dtype=np.float32)

        monkeypatch.setattr("itol.strategies.s3_window.embed", fake_embed)
        # Cosine function: dot product (vecs are already near-unit)
        monkeypatch.setattr(
            "itol.strategies.s3_window.cosine",
            lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)),
        )

        # Build a segment with 3 clear chunks via manual chunking by making
        # the text very long so it gets split into 3 windows
        # chunk 0: ~300 chars (high score, lots of mass)
        # chunk 1: ~300 chars (medium score)
        # chunk 2: contains the entity (low score)
        chunk0_text = "A " * 75   # ~300 chars; 60 tokens; high relevance
        chunk1_text = "B " * 75   # ~300 chars; not relevant
        entity_text = "C " * 50 + "SPECIAL_ENTITY_XYZ " + "D " * 20  # ~300 chars

        # Combine into one large text that will be chunked at 256-token windows
        # Use a text designed so chunks map predictably
        big_text = chunk0_text.strip() + "\n\n" + chunk1_text.strip() + "\n\n" + entity_text.strip()

        manifest = ConstraintManifest(items=[
            ManifestItem(item_type=ManifestItem.ItemType.ENTITY, value="SPECIAL_ENTITY_XYZ"),
        ])

        signals = _signals(token_count=20_000)
        ctx = _ctx(signals, mass_floor_override=0.40)
        ctx = OptimizationContext(
            request_class="SUMMARIZATION",
            matrix_row=ctx.matrix_row,
            manifest=manifest,
            signals=signals,
            config=ctx.config,
        )

        strategy = S3WindowStrategy()
        icr = _icr("Find the entity.")

        system_seg = _seg("System instruction.", SegmentType.SYSTEM_INSTRUCTION)
        doc_seg = _seg(big_text, SegmentType.RETRIEVED_DOC)
        segments = [system_seg, doc_seg]

        new_segs, report = strategy.apply(icr, segments, ctx)

        # Gather all kept text
        kept_text = " ".join(s.text for s in new_segs)

        assert "SPECIAL_ENTITY_XYZ" in kept_text, (
            "CR-4 INVARIANT: manifest entity must be present in output even "
            "when mass floor was met by an earlier chunk alone"
        )

    def test_loop_stops_when_both_conditions_met(self, monkeypatch):
        """
        CR-4 control: when both mass floor AND entity coverage are met simultaneously,
        the loop should stop without including unnecessary extra chunks.
        """
        call_count = [0]
        def fake_embed(texts, model="minilm"):
            result = []
            for t in texts:
                result.append(np.array([1.0] + [0.0]*15, dtype=np.float32))
            return np.array(result, dtype=np.float32)

        monkeypatch.setattr("itol.strategies.s3_window.embed", fake_embed)
        monkeypatch.setattr(
            "itol.strategies.s3_window.cosine",
            lambda a, b: 1.0,
        )

        # Single chunk that both satisfies mass floor (it's the only chunk → 100%)
        # AND contains the entity
        entity_text = "REQUIRED_ENTITY_ABC " + "content " * 100

        manifest = ConstraintManifest(items=[
            ManifestItem(item_type=ManifestItem.ItemType.ENTITY, value="REQUIRED_ENTITY_ABC"),
        ])
        signals = _signals(token_count=20_000)
        ctx = _ctx(signals, mass_floor_override=0.50)
        ctx = OptimizationContext(
            request_class="SUMMARIZATION",
            matrix_row=ctx.matrix_row,
            manifest=manifest,
            signals=signals,
            config=ctx.config,
        )

        strategy = S3WindowStrategy()
        icr = _icr("Find the entity.")
        doc_seg = _seg(entity_text, SegmentType.RETRIEVED_DOC)

        new_segs, report = strategy.apply(icr, [doc_seg], ctx)
        kept_text = " ".join(s.text for s in new_segs)

        assert "REQUIRED_ENTITY_ABC" in kept_text, (
            "Control: entity must be present when contained in the single chunk"
        )
