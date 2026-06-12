"""
§14.3 Invariant tests for CR-26 — manifest recall honest reporting and
conservative bandit prior bump.

CR-26: after calibration, manifest_recall.json must report per-class recall
against 200 gold manifests.  If any class has recall < 0.92, the bandit
controller's conservative arm must use an elevated prior (α=3, β=1) for
that class instead of the normal starting prior (α=2, β=1).

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from itol.engine import Engine, CalibrationRequiredError
from itol.quality.bandit import BanditController


# ===========================================================================
# CR-26-a: manifest recall reported honestly
# ===========================================================================

class TestCR26_RecallReporting:

    def _write_recall(self, calib_dir: Path, data: dict) -> None:
        calib_dir.mkdir(parents=True, exist_ok=True)
        (calib_dir / "manifest_recall.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    def test_recall_report_reflects_json_content(self, tmp_path):
        """
        CR-26: manifest_recall_report() must return the exact content of
        manifest_recall.json — no adjustment, no suppression.
        """
        data = {
            "overall": 0.94,
            "per_class": {
                "EXTRACTION": 0.96,
                "SUMMARIZATION": 0.90,  # below 0.92
            },
        }
        self._write_recall(tmp_path / "calibration", data)
        engine = Engine(data_dir=tmp_path)
        report = engine.manifest_recall_report()

        assert report["overall"] == pytest.approx(0.94, abs=1e-9), (
            "CR-26: overall recall must be reported verbatim"
        )
        assert report["per_class"]["SUMMARIZATION"] == pytest.approx(0.90, abs=1e-9), (
            "CR-26: per-class recall below 0.92 must be reported, not hidden"
        )

    def test_recall_report_raises_when_absent(self, tmp_path):
        """
        CR-26: manifest_recall_report() raises CalibrationRequiredError when
        manifest_recall.json is absent.
        """
        engine = Engine(data_dir=tmp_path)
        with pytest.raises(CalibrationRequiredError):
            engine.manifest_recall_report()

    def test_recall_report_includes_all_written_classes(self, tmp_path):
        """
        CR-26: all per-class entries written to manifest_recall.json must be
        returned by manifest_recall_report().
        """
        classes = [
            "EXTRACTION", "REASONING", "SUMMARIZATION",
            "GENERATION_FACTUAL", "GENERATION_CREATIVE",
            "CLASSIFICATION_SHORT", "AGENT_TOOL_LOOP", "CHAT_OPEN",
        ]
        per_class = {cls: 0.93 for cls in classes}
        data = {"overall": 0.93, "per_class": per_class}
        self._write_recall(tmp_path / "calibration", data)

        engine = Engine(data_dir=tmp_path)
        report = engine.manifest_recall_report()
        for cls in classes:
            assert cls in report["per_class"], (
                f"CR-26: class {cls!r} must appear in recall report"
            )


# ===========================================================================
# CR-26-b: low recall triggers conservative bandit prior bump
# ===========================================================================

class TestCR26_ConservativePriorBump:

    def _write_bandit_priors(
        self, calib_dir: Path, per_class_recall: dict[str, float]
    ) -> None:
        """Write manifest_recall.json and bandit_priors.json."""
        calib_dir.mkdir(parents=True, exist_ok=True)
        recall_data = {
            "overall": sum(per_class_recall.values()) / len(per_class_recall),
            "per_class": per_class_recall,
        }
        (calib_dir / "manifest_recall.json").write_text(
            json.dumps(recall_data), encoding="utf-8"
        )
        # bandit_priors can start empty — the controller fills defaults
        (calib_dir / "bandit_priors.json").write_text("{}", encoding="utf-8")

    def test_low_recall_class_gets_conservative_prior(self, tmp_path):
        """
        CR-26: if recall for a class < 0.92, the bandit controller must use
        α=3, β=1 (elevated conservative prior) for that class, rather than
        the normal α=2, β=1 conservative prior.

        We verify by loading the bandit controller with the recall JSON and
        checking that the conservative arm's (α, β) are (3, 1) for the low-
        recall class.
        """
        calib_dir = tmp_path / "calibration"
        self._write_bandit_priors(
            calib_dir,
            {"SUMMARIZATION": 0.88, "EXTRACTION": 0.95},
        )

        controller = BanditController.load_with_recall(
            str(calib_dir / "manifest_recall.json"),
            store=None,
        )

        # For SUMMARIZATION (recall=0.88 < 0.92), conservative arm must have α=3, β=1
        alpha, beta = controller.get_conservative_prior("SUMMARIZATION")
        assert alpha == 3, (
            f"CR-26: low recall class must have conservative α=3; got α={alpha}"
        )
        assert beta == 1, (
            f"CR-26: low recall class must have conservative β=1; got β={beta}"
        )

    def test_normal_recall_class_keeps_default_prior(self, tmp_path):
        """
        CR-26: a class with recall >= 0.92 must keep the default conservative
        prior (α=2, β=1), not the elevated one.
        """
        calib_dir = tmp_path / "calibration"
        self._write_bandit_priors(
            calib_dir,
            {"EXTRACTION": 0.95, "SUMMARIZATION": 0.88},
        )

        controller = BanditController.load_with_recall(
            str(calib_dir / "manifest_recall.json"),
            store=None,
        )

        # For EXTRACTION (recall=0.95 >= 0.92), conservative prior stays α=2, β=1
        alpha, beta = controller.get_conservative_prior("EXTRACTION")
        assert alpha == 2, (
            f"CR-26: normal recall class must have conservative α=2; got α={alpha}"
        )
        assert beta == 1, (
            f"CR-26: normal recall class must have conservative β=1; got β={beta}"
        )

    def test_threshold_is_exactly_0_92(self, tmp_path):
        """
        CR-26: the recall threshold is exactly 0.92 (not 0.9199 or 0.9201).
        recall=0.92 must NOT trigger the bump (only recall < 0.92 does).
        """
        calib_dir = tmp_path / "calibration"
        self._write_bandit_priors(calib_dir, {"REASONING": 0.92})

        controller = BanditController.load_with_recall(
            str(calib_dir / "manifest_recall.json"),
            store=None,
        )

        alpha, _ = controller.get_conservative_prior("REASONING")
        assert alpha == 2, (
            "CR-26: recall==0.92 must NOT trigger the bump (threshold is strictly <0.92)"
        )
