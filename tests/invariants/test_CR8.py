"""
§14.3 Invariant tests for CR-8 — manifest number mismatch blocks L1 hit.

CR-8: a cache hit additionally requires:
    cached_entry.manifest.numbers_entities ⊇ new_query.manifest.numbers_entities

This check is AND-ed with cosine ≥ τ AND cross_score ≥ 0.85.
It can NEVER be OR-ed or bypassed.

Specifically: if the new query contains a number entity not present in the
cached entry's manifest, L1 must return None regardless of cosine similarity.

To test this in CI (where only the BOW fallback is available), we monkeypatch
the embed function to return identical vectors (cosine = 1.0) and the rerank
function to return high scores (0.99).  The ONLY thing that can block the hit
is the CR-8 manifest numbers check.

Rules:
- NEVER weaken thresholds or conditions.
"""

import json
import tempfile

import numpy as np
import pytest

from itol.cache.l1_semantic import L1Cache, _numbers_entities
from itol.cache.store import Store
from itol.icr import (
    ClassifierResult, ConstraintManifest, ContentBlock, ICR,
    ICRResponse, ManifestItem, Message, UsageStats,
)
from itol.routing.matrix import MATRIX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _icr(user_text: str = "Summarize the report.") -> ICR:
    return ICR.create(
        provider="openai", model="gpt-4o",
        system=[],
        messages=[Message.user(user_text)],
        raw={},
    )


def _manifest(number_values: list[str]) -> ConstraintManifest:
    items = [ManifestItem(item_type="NUMBER", value=v) for v in number_values]
    return ConstraintManifest(items=items)


def _ok_response() -> ICRResponse:
    return ICRResponse(
        request_id="resp1",
        provider="openai", model="gpt-4o",
        content=[ContentBlock.text("The budget is as specified.")],
        usage=UsageStats(input_tokens=50, output_tokens=10),
        finish_reason="stop",
    )


class _FakeCtx:
    def __init__(self, primary: str = "SUMMARIZATION"):
        self.matrix_row = MATRIX[primary]


def _high_vec() -> np.ndarray:
    v = np.ones(16, dtype=np.float32)
    return v / np.linalg.norm(v)


# ===========================================================================
# CR-8-a: different number in new query → None
# ===========================================================================

class TestCR8_DifferentNumberBlocked:

    def test_different_number_never_hits(self, monkeypatch, tmp_path):
        """
        CR-8 core: identical cosine (1.0) + high rerank score (0.99), but the
        new query's manifest number is absent from the cached entry → must return None.

        We force cosine=1.0 and cross_score=0.99 so the ONLY possible blocker
        is the CR-8 manifest check.
        """
        high_vec = _high_vec()

        monkeypatch.setattr(
            "itol.cache.l1_semantic.embed",
            lambda texts, model="bge": np.array([high_vec.copy() for _ in texts]),
        )
        monkeypatch.setattr(
            "itol.cache.l1_semantic.rerank",
            lambda query, candidates: [0.99] * len(candidates),
        )

        store = Store(str(tmp_path))
        l1 = L1Cache(store)
        ctx = _FakeCtx("SUMMARIZATION")

        # Store a response for query with number $1234
        icr_stored = _icr("Summarize the quarterly report. Budget: $1234 million.")
        manifest_stored = _manifest(["$1234"])
        l1.store(icr_stored, manifest_stored, ClassifierResult(primary="SUMMARIZATION", confidence=0.8), _ok_response(), ctx)

        # Lookup with DIFFERENT number ($9999)
        icr_lookup = _icr("Summarize the quarterly report. Budget: $9999 million.")
        manifest_lookup = _manifest(["$9999"])
        result = l1.lookup(icr_lookup, manifest_lookup, ClassifierResult(primary="SUMMARIZATION", confidence=0.8), ctx)

        assert result is None, (
            "CR-8: different manifest number must block L1 cache hit, "
            "even when cosine=1.0 and cross_score=0.99"
        )
        store.close()

    def test_same_number_allows_hit(self, monkeypatch, tmp_path):
        """
        Control: same number in new query → hit is allowed (given high cosine + rerank).
        """
        high_vec = _high_vec()

        monkeypatch.setattr(
            "itol.cache.l1_semantic.embed",
            lambda texts, model="bge": np.array([high_vec.copy() for _ in texts]),
        )
        monkeypatch.setattr(
            "itol.cache.l1_semantic.rerank",
            lambda query, candidates: [0.99] * len(candidates),
        )

        store = Store(str(tmp_path))
        l1 = L1Cache(store)
        ctx = _FakeCtx("SUMMARIZATION")
        cls = ClassifierResult(primary="SUMMARIZATION", confidence=0.8)
        response = _ok_response()

        icr_stored = _icr("Summarize the quarterly report. Budget: $1234 million.")
        manifest_stored = _manifest(["$1234"])
        l1.store(icr_stored, manifest_stored, cls, response, ctx)

        # Lookup with SAME number → should hit
        icr_lookup = _icr("Summarize the quarterly report. Budget: $1234 million.")
        manifest_lookup = _manifest(["$1234"])
        result = l1.lookup(icr_lookup, manifest_lookup, cls, ctx)

        assert result is not None, (
            "CR-8 control: same number should allow a cache hit"
        )
        store.close()

    def test_cached_superset_allows_hit(self, monkeypatch, tmp_path):
        """
        CR-8: cached = {$1234, $9999} ⊇ new = {$1234} → hit is allowed.
        """
        high_vec = _high_vec()

        monkeypatch.setattr(
            "itol.cache.l1_semantic.embed",
            lambda texts, model="bge": np.array([high_vec.copy() for _ in texts]),
        )
        monkeypatch.setattr(
            "itol.cache.l1_semantic.rerank",
            lambda query, candidates: [0.99] * len(candidates),
        )

        store = Store(str(tmp_path))
        l1 = L1Cache(store)
        ctx = _FakeCtx("SUMMARIZATION")
        cls = ClassifierResult(primary="SUMMARIZATION", confidence=0.8)

        # Cache entry has BOTH numbers
        icr_stored = _icr("Report. Budget $1234. Forecast $9999.")
        manifest_stored = _manifest(["$1234", "$9999"])
        l1.store(icr_stored, manifest_stored, cls, _ok_response(), ctx)

        # Lookup with subset {$1234} — superset check cached ⊇ new passes
        icr_lookup = _icr("Report. Budget $1234.")
        manifest_lookup = _manifest(["$1234"])
        result = l1.lookup(icr_lookup, manifest_lookup, cls, ctx)

        assert result is not None, (
            "CR-8: cached {$1234,$9999} ⊇ new {$1234} → hit must be allowed"
        )
        store.close()

    def test_no_numbers_in_query_always_passes_cr8(self, monkeypatch, tmp_path):
        """
        CR-8: if the new query has NO number entities, the superset check vacuously
        passes (any set ⊇ empty set).
        """
        high_vec = _high_vec()

        monkeypatch.setattr(
            "itol.cache.l1_semantic.embed",
            lambda texts, model="bge": np.array([high_vec.copy() for _ in texts]),
        )
        monkeypatch.setattr(
            "itol.cache.l1_semantic.rerank",
            lambda query, candidates: [0.99] * len(candidates),
        )

        store = Store(str(tmp_path))
        l1 = L1Cache(store)
        ctx = _FakeCtx("SUMMARIZATION")
        cls = ClassifierResult(primary="SUMMARIZATION", confidence=0.8)

        icr_stored = _icr("Summarize the report.")
        manifest_stored = _manifest([])  # no numbers
        l1.store(icr_stored, manifest_stored, cls, _ok_response(), ctx)

        icr_lookup = _icr("Summarize the report.")
        manifest_lookup = _manifest([])  # still no numbers
        result = l1.lookup(icr_lookup, manifest_lookup, cls, ctx)

        assert result is not None, "With no number entities, CR-8 vacuously passes"
        store.close()


# ===========================================================================
# CR-8-b: _numbers_entities extracts NUMBER items from manifest
# ===========================================================================

class TestCR8_NumbersEntities:

    def test_numbers_extracted(self):
        """_numbers_entities must return all NUMBER-type manifest items' values."""
        m = _manifest(["$1234", "$5M", "42%"])
        assert _numbers_entities(m) == {"$1234", "$5M", "42%"}

    def test_non_number_items_excluded(self):
        """Non-NUMBER manifest items must NOT be in the numbers set."""
        items = [
            ManifestItem(item_type="NUMBER", value="$100"),
            ManifestItem(item_type="DATE", value="Q3 2024"),
            ManifestItem(item_type="NORMATIVE", value="must"),
        ]
        m = ConstraintManifest(items=items)
        assert _numbers_entities(m) == {"$100"}, (
            "Only NUMBER-type items must be in numbers_entities"
        )

    def test_empty_manifest_empty_set(self):
        """Empty manifest → empty numbers set."""
        assert _numbers_entities(ConstraintManifest()) == set()


# ===========================================================================
# CR-8-c: L1 disabled classes return None immediately (no embedding)
# ===========================================================================

class TestCR8_DisabledClassNoWork:

    def test_generation_creative_returns_none_immediately(self, monkeypatch, tmp_path):
        """
        L1 is DISABLED for GENERATION_CREATIVE; lookup must return None
        without calling embed (no wasted work).
        """
        embed_called = []

        def fail_if_called(texts, model="bge"):
            embed_called.append(True)
            raise AssertionError("embed must NOT be called when L1 is disabled")

        monkeypatch.setattr("itol.cache.l1_semantic.embed", fail_if_called)

        store = Store(str(tmp_path))
        l1 = L1Cache(store)
        ctx = _FakeCtx("GENERATION_CREATIVE")  # L1 DISABLED

        result = l1.lookup(
            _icr(), _manifest([]),
            ClassifierResult(primary="GENERATION_CREATIVE", confidence=0.8),
            ctx,
        )
        assert result is None, "GENERATION_CREATIVE must return None (L1 disabled)"
        assert not embed_called, "embed must not be called when L1 is disabled"
        store.close()
