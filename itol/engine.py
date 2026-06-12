"""
ITOL Engine — §15.4 CR-25/CR-26.

The Engine is the top-level entrypoint that gates optimize mode behind
calibration.  If data/calibration/*.json files are absent, setting
mode="optimize" raises CalibrationRequiredError and the effective mode
silently becomes "observe_only".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from itol.config import ITOLConfig


# Files that must exist before optimize mode is reachable (CR-25)
_REQUIRED_CALIBRATION_FILES = [
    "qps.json",
    "tau.json",
    "bandit_priors.json",
    "manifest_recall.json",
]


class CalibrationRequiredError(Exception):
    """
    Raised when optimize mode is requested but calibration data is absent.

    Run `python -m itol.cli calibrate` (or `calibrate --offline`) to produce
    the required files in data/calibration/.
    """


class Engine:
    """
    Manages ITOL operating mode and exposes calibration metadata.

    Parameters
    ----------
    config : ITOLConfig, optional
        Root configuration; defaults to ITOLConfig().
    data_dir : path, optional
        Root data directory.  Defaults to <package_root>/../data.
    """

    def __init__(
        self,
        config: ITOLConfig | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self._config = config or ITOLConfig()
        if data_dir is not None:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = Path(__file__).parent.parent / "data"
        self._calib_dir = self._data_dir / "calibration"
        # CR-25: if the config starts in optimize mode but calibration is absent,
        # silently downgrade to observe_only.  The mode setter enforces this on
        # explicit writes; here we enforce it at construction time too.
        if self._config.mode == "optimize" and not self._calibration_present():
            self._config.mode = "observe_only"

    # ------------------------------------------------------------------
    # Mode property  (CR-25 gate)
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._config.mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value == "optimize" and not self._calibration_present():
            self._config.mode = "observe_only"
            raise CalibrationRequiredError(
                "Cannot enter optimize mode: calibration data absent. "
                "Run 'python -m itol.cli calibrate --offline' to generate "
                f"{self._calib_dir}/{{qps,tau,bandit_priors,manifest_recall}}.json"
            )
        self._config.mode = value

    def _calibration_present(self) -> bool:
        """Return True iff all required calibration JSON files exist."""
        return all(
            (self._calib_dir / fname).exists()
            for fname in _REQUIRED_CALIBRATION_FILES
        )

    # ------------------------------------------------------------------
    # CR-26 manifest recall report
    # ------------------------------------------------------------------

    def manifest_recall_report(self) -> dict[str, Any]:
        """
        CR-26: return the manifest recall report produced by calibration.

        Returns a dict with keys:
            overall       — float, mean recall across all classes
            per_class     — dict[str, float]

        Raises CalibrationRequiredError if manifest_recall.json is absent.
        """
        recall_path = self._calib_dir / "manifest_recall.json"
        if not recall_path.exists():
            raise CalibrationRequiredError(
                "manifest_recall.json not found. Run calibration first."
            )
        with open(recall_path, encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Config access helpers
    # ------------------------------------------------------------------

    @property
    def config(self) -> ITOLConfig:
        return self._config
