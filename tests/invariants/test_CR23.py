"""
§14.3 Invariant test for CR-23 — sliding_window_fidelity catches dropped sentences.

CR-23: A segment where one sentence is removed must produce
  sliding_window_fidelity < 0.75 for the affected window
  while segment-level cosine remains > 0.90.

This proves that the window probe catches quality losses that segment-average
cosine misses — justifying §15.2's replacement of structure_integrity.

Rules:
- NEVER weaken thresholds (0.75, 0.90) to make the test pass.
"""

import pytest
import numpy as np

from itol.embed.onnx_embedder import embed, cosine, sliding_window_fidelity


# ---------------------------------------------------------------------------
# Test corpus — designed for BOW cosine (CI fallback)
# ---------------------------------------------------------------------------

# 4-sentence original. Sentence 1 ("She leads the engineering team.") is
# semantically distinct from its neighbours — dropping it creates a window
# mismatch that the sliding probe catches while the full-text cosine stays high.

_ORIGINAL = (
    "Alice works at the company. "
    "She leads the engineering team. "
    "The project is on schedule. "
    "Delivery is expected next quarter."
)

# Sentence 1 dropped (index counting from 0)
_OPTIMIZED = (
    "Alice works at the company. "
    "The project is on schedule. "
    "Delivery is expected next quarter."
)


class TestCR23:

    def test_window_fidelity_catches_dropped_sentence(self):
        """
        CR-23: sliding_window_fidelity must return < 0.75 when a semantically
        distinct sentence is removed, even though segment-level cosine stays > 0.90.
        """
        # --- segment-level cosine must stay HIGH (> 0.90) ---
        vecs = embed([_ORIGINAL, _OPTIMIZED])
        seg_cosine = cosine(vecs[0], vecs[1])

        assert seg_cosine > 0.90, (
            f"CR-23: segment-level cosine must stay > 0.90 for this corpus "
            f"(got {seg_cosine:.4f}) — test design error"
        )

        # --- window fidelity must detect the drop (< 0.75) ---
        wf = sliding_window_fidelity(_ORIGINAL, _OPTIMIZED, window_sentences=2)

        assert wf < 0.75, (
            f"CR-23: sliding_window_fidelity must be < 0.75 when a sentence is "
            f"dropped (got {wf:.4f}). The window probe is not catching the gap."
        )

    def test_window_fidelity_identical_segments_is_1(self):
        """Identical text → sliding_window_fidelity must be 1.0."""
        text = "Revenue is $4.2M. Growth rate is 12%. Delivery is Q3."
        wf = sliding_window_fidelity(text, text, window_sentences=2)
        assert wf == pytest.approx(1.0, abs=1e-5), (
            "Identical segments must produce window fidelity = 1.0"
        )

    def test_window_fidelity_completely_different_is_low(self):
        """Completely unrelated segments must produce very low window fidelity."""
        orig = "The cat sat on the mat. Dogs are good pets."
        opt  = "Quantum mechanics governs subatomic particles. Nuclear fusion powers stars."
        wf = sliding_window_fidelity(orig, opt, window_sentences=2)
        assert wf < 0.5, (
            f"Completely unrelated segments must have very low window fidelity (got {wf:.4f})"
        )

    def test_window_fidelity_short_fallback_to_full_cosine(self):
        """
        When a segment has fewer sentences than window_sentences, the function
        must fall back to a full-segment cosine (not raise).
        """
        orig = "Short text."
        opt  = "Short text too."
        wf = sliding_window_fidelity(orig, opt, window_sentences=2)
        assert 0.0 <= wf <= 1.0, "Must return a valid float in [0, 1]"

    def test_segment_cosine_above_threshold_even_with_drop(self):
        """
        Explicit check: segment cosine > 0.90 for the CR-23 corpus,
        confirming this is a case where segment average would NOT trigger rollback.
        """
        vecs = embed([_ORIGINAL, _OPTIMIZED])
        seg_cos = cosine(vecs[0], vecs[1])
        assert seg_cos > 0.90, (
            f"Segment cosine should stay above 0.90 even with the dropped sentence "
            f"(got {seg_cos:.4f}). If this fails the corpus needs redesign."
        )
        # And window fidelity must catch what segment cosine misses
        wf = sliding_window_fidelity(_ORIGINAL, _OPTIMIZED)
        assert wf < seg_cos, (
            "Window fidelity must be lower than segment cosine when a sentence is dropped"
        )
