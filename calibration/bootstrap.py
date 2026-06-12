"""
calibration/bootstrap.py — §15.4 `itol calibrate` pipeline entry-point.

Orchestrates the full calibration run:
  1. Load corpus (offline: synthetic only; online: synthetic + optional HF)
  2. Generate synthetic pairs via synth_agent
  3. Run fit.run_fit() → writes 4 JSON artefacts to data/calibration/
  4. Print a summary report

CR-25: After this command succeeds, Engine.mode = 'optimize' becomes reachable.
CR-26: Classes with manifest recall < 0.92 are flagged and bandit_priors.json
       receives elevated conservative priors for those classes.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_CALIB_DIR = _REPO_ROOT / "data" / "calibration"


def run_calibration(
    offline: bool = True,
    calib_dir: Path | None = None,
    n_synth_per_class: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Run the full calibration pipeline.

    Parameters
    ----------
    offline : bool
        If True, use only bundled synthetic data (no HuggingFace downloads).
    calib_dir : Path, optional
        Output directory.  Defaults to data/calibration/.
    n_synth_per_class : int
        Number of synthetic pairs per class to generate.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict with keys: qps, tau, bandit_priors, manifest_recall, low_recall_classes
    """
    out_dir = calib_dir or _CALIB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[calibrate] Output directory: {out_dir}")
        print(f"[calibrate] Mode: {'offline (synthetic only)' if offline else 'online (synthetic + HF)'}")

    # 1. Synthetic pairs
    from calibration.synth_agent import generate_pairs
    synth_pairs = generate_pairs(n_per_class=n_synth_per_class)
    if verbose:
        print(f"[calibrate] Generated {len(synth_pairs)} synthetic pairs.")

    # 2. Corpus (synthetic + optional HF)
    from calibration.corpora import build_corpus_pairs
    corpus_pairs = build_corpus_pairs(offline=offline)
    all_pairs = synth_pairs + corpus_pairs
    if verbose:
        print(f"[calibrate] Total corpus pairs: {len(all_pairs)}")

    # 3. Fit
    from calibration.fit import run_fit
    gold_path = out_dir / "manifest_gold.jsonl"

    result = run_fit(
        corpus_pairs=all_pairs,
        calib_dir=out_dir,
        gold_path=gold_path,
    )

    if verbose:
        _print_report(result)

    return result


def _print_report(result: dict) -> None:
    """Print a human-readable calibration summary."""
    mr = result.get("manifest_recall", {})
    overall = mr.get("overall", 0.0)
    per_class = mr.get("per_class", {})
    low = result.get("low_recall_classes", [])

    print("\n=== Calibration Report ===")
    print(f"  Manifest recall (overall): {overall:.3f}")
    print("  Per-class recall:")
    for cls, score in sorted(per_class.items()):
        flag = " *** LOW (CR-26 bump)" if cls in low else ""
        print(f"    {cls:<28} {score:.3f}{flag}")

    tau = result.get("tau", {})
    print("  tau thresholds:")
    for cls, t in sorted(tau.items()):
        print(f"    {cls:<28} {t}")

    qps = result.get("qps", {})
    print(f"  QPS weights: {qps}")

    if low:
        print(f"\n  [CR-26] Classes with recall < 0.92: {', '.join(low)}")
        print("          Conservative bandit prior bumped to alpha=3, beta=1 for these classes.")

    print("\n  Artefacts written:")
    for fname in ["qps.json", "tau.json", "bandit_priors.json", "manifest_recall.json"]:
        print(f"    data/calibration/{fname}")
    print()


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point: calibration/bootstrap.py --offline / --online."""
    import argparse
    parser = argparse.ArgumentParser(description="Run ITOL calibration pipeline")
    parser.add_argument(
        "--offline", action="store_true", default=True,
        help="Use only synthetic data (default; no internet required)"
    )
    parser.add_argument(
        "--online", action="store_true", default=False,
        help="Also download HuggingFace datasets (requires internet + datasets package)"
    )
    parser.add_argument(
        "--n-synth", type=int, default=10,
        help="Synthetic pairs per class (default: 10)"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    offline = not args.online
    try:
        run_calibration(
            offline=offline,
            n_synth_per_class=args.n_synth,
            verbose=not args.quiet,
        )
        return 0
    except Exception as exc:
        _log.error("Calibration failed: %s", exc, exc_info=True)
        print(f"[calibrate] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
