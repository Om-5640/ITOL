"""
§14.3 Invariant tests for CR-25 — optimize mode requires calibration.

CR-25: if data/calibration/*.json files are absent, setting mode="optimize"
must raise CalibrationRequiredError and the effective mode must remain
"observe_only".

Once all four calibration files are present, mode="optimize" must succeed.

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from itol.engine import Engine, CalibrationRequiredError

_REQUIRED = ["qps.json", "tau.json", "bandit_priors.json", "manifest_recall.json"]


# ===========================================================================
# CR-25-a: optimize blocked without calibration
# ===========================================================================

class TestCR25_OptimizeRequiresCalibration:

    def test_optimize_raises_when_no_calibration_files(self, tmp_path):
        """
        CR-25: with no calibration JSONs present, Engine.mode = 'optimize'
        must raise CalibrationRequiredError.
        """
        engine = Engine(data_dir=tmp_path)
        with pytest.raises(CalibrationRequiredError):
            engine.mode = "optimize"

    def test_effective_mode_is_observe_only_after_blocked(self, tmp_path):
        """
        CR-25: after a failed optimize attempt, the effective mode must be
        'observe_only', not 'optimize'.
        """
        engine = Engine(data_dir=tmp_path)
        try:
            engine.mode = "optimize"
        except CalibrationRequiredError:
            pass
        assert engine.mode == "observe_only", (
            "CR-25: effective mode must be 'observe_only' when calibration is absent"
        )

    def test_partial_calibration_files_still_blocked(self, tmp_path):
        """
        CR-25: having only some (but not all) calibration files must still
        block optimize mode.
        """
        calib_dir = tmp_path / "calibration"
        calib_dir.mkdir()
        # Write only 2 of 4 required files
        (calib_dir / "qps.json").write_text("{}", encoding="utf-8")
        (calib_dir / "tau.json").write_text("{}", encoding="utf-8")

        engine = Engine(data_dir=tmp_path)
        with pytest.raises(CalibrationRequiredError):
            engine.mode = "optimize"


# ===========================================================================
# CR-25-b: optimize allowed after calibration
# ===========================================================================

class TestCR25_OptimizeAllowedAfterCalibration:

    def _write_dummy_calibration(self, calib_dir: Path) -> None:
        calib_dir.mkdir(parents=True, exist_ok=True)
        dummy = {"overall": 0.95, "per_class": {}}
        for fname in _REQUIRED:
            (calib_dir / fname).write_text(json.dumps(dummy), encoding="utf-8")

    def test_optimize_succeeds_with_all_files_present(self, tmp_path):
        """
        CR-25: once all four calibration files exist, mode='optimize' must
        not raise and the mode must be set correctly.
        """
        self._write_dummy_calibration(tmp_path / "calibration")
        engine = Engine(data_dir=tmp_path)
        engine.mode = "optimize"   # must not raise
        assert engine.mode == "optimize", (
            "CR-25: optimize mode must be reachable after calibration"
        )

    def test_observe_only_never_requires_calibration(self, tmp_path):
        """observe_only and passthrough modes must always be settable."""
        engine = Engine(data_dir=tmp_path)
        engine.mode = "observe_only"  # no raise
        assert engine.mode == "observe_only"

    def test_engine_default_mode_is_not_optimize(self, tmp_path):
        """Default mode must not be 'optimize' (safe default before calibration)."""
        engine = Engine(data_dir=tmp_path)
        assert engine.mode != "optimize", (
            "Default mode must be safe (non-optimize) before calibration"
        )
