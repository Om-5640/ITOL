"""
§14.3 Invariant test — Stage B ML classifier graceful fallback.

When itol/data/calibration/classifier_ml.json is absent:
  - ClassifierML.load() must return None
  - classify_with_ml() must return None
  - ClassifierML.predict() (when loaded) must return None for wrong-shaped features

The system must fall back to Stage A's result without raising exceptions.

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from itol.analysis.classifier_ml import ClassifierML, classify_with_ml


# ===========================================================================
# Graceful fallback when calibration file is absent
# ===========================================================================

class TestGracefulFallback:

    def test_load_returns_none_when_file_absent(self, tmp_path):
        """ClassifierML.load() must return None when the calibration file does not exist."""
        absent_path = tmp_path / "nonexistent_classifier.json"
        result = ClassifierML.load(absent_path)
        assert result is None, (
            "ClassifierML.load() must return None when calibration file is absent"
        )

    def test_classify_with_ml_returns_none_when_file_absent(self, tmp_path):
        """
        classify_with_ml() must return None when calibration is absent,
        never raising an exception.
        """
        from itol.icr import ICR, Message

        icr = ICR.create(
            provider="openai", model="gpt-4o",
            system=[],
            messages=[Message.user("Summarize the report.")],
            raw={},
        )
        absent_path = tmp_path / "nonexistent.json"

        # Must not raise; must return None
        result = classify_with_ml(icr, path=absent_path)
        assert result is None, (
            "classify_with_ml() must return None when calibration file is absent"
        )

    def test_predict_returns_none_for_wrong_shape(self, tmp_path):
        """
        ClassifierML.predict() must return None when the feature vector
        is the wrong shape (not exactly (73,)).
        """
        # Create a minimal valid calibration file
        n_classes = 4
        calib = {
            "classes": ["EXTRACTION", "GENERATION_CREATIVE", "REASONING", "SUMMARIZATION"],
            "W": np.zeros((n_classes, 73)).tolist(),
            "b": np.zeros(n_classes).tolist(),
            "feature_names": [f"f{i}" for i in range(73)],
        }
        calib_path = tmp_path / "classifier_ml.json"
        with open(calib_path, "w") as fh:
            json.dump(calib, fh)

        clf = ClassifierML.load(calib_path)
        assert clf is not None, "Should load successfully from valid calibration"

        # Wrong shape
        for bad_features in [
            np.zeros(72, dtype=np.float32),   # one too few
            np.zeros(74, dtype=np.float32),   # one too many
            np.zeros((73, 1), dtype=np.float32),  # 2-D
            None,
        ]:
            result = clf.predict(bad_features)
            assert result is None, (
                f"predict() must return None for features with shape "
                f"{getattr(bad_features, 'shape', None)}"
            )

    def test_predict_returns_result_for_correct_shape(self, tmp_path):
        """
        Control: ClassifierML.predict() must return a ClassifierResult
        for a correctly-shaped (73,) feature vector.
        """
        from itol.icr import ClassifierResult

        n_classes = 4
        classes = ["EXTRACTION", "GENERATION_CREATIVE", "REASONING", "SUMMARIZATION"]
        # Simple weights: identity-like to ensure EXTRACTION wins for first feature = 1
        W = np.zeros((n_classes, 73), dtype=np.float32)
        W[0, 0] = 10.0   # EXTRACTION gets large logit when features[0] is high
        b = np.zeros(n_classes, dtype=np.float32)

        calib = {
            "classes": classes,
            "W": W.tolist(),
            "b": b.tolist(),
            "feature_names": [f"f{i}" for i in range(73)],
        }
        calib_path = tmp_path / "classifier_ml.json"
        with open(calib_path, "w") as fh:
            json.dump(calib, fh)

        clf = ClassifierML.load(calib_path)
        assert clf is not None

        features = np.zeros(73, dtype=np.float32)
        features[0] = 1.0  # push EXTRACTION logit high

        result = clf.predict(features)
        assert isinstance(result, ClassifierResult), (
            "predict() must return ClassifierResult for valid 73-dim features"
        )
        assert result.primary == "EXTRACTION", (
            "With W[0,0]=10.0 and features[0]=1.0, EXTRACTION must win"
        )
        assert 0.0 < result.confidence <= 1.0


# ===========================================================================
# Calibration file format round-trip
# ===========================================================================

class TestCalibrationFormat:

    def test_load_roundtrip(self, tmp_path):
        """Weights loaded from JSON must produce numerically identical inference."""
        n_classes = 3
        classes = ["EXTRACTION", "REASONING", "SUMMARIZATION"]
        W = np.random.default_rng(42).standard_normal((n_classes, 73)).astype(np.float32)
        b = np.random.default_rng(99).standard_normal(n_classes).astype(np.float32)

        calib = {
            "classes": classes,
            "W": W.tolist(),
            "b": b.tolist(),
            "feature_names": [f"f{i}" for i in range(73)],
        }
        calib_path = tmp_path / "classifier_ml.json"
        with open(calib_path, "w") as fh:
            json.dump(calib, fh)

        clf = ClassifierML.load(calib_path)
        assert clf is not None
        assert clf.classes == classes
        np.testing.assert_allclose(clf.W, W, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(clf.b, b, rtol=1e-5, atol=1e-6)
