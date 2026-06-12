"""
itol/cli.py — §15.4 `itol calibrate` CLI entry-point.

Usage:
    python -m itol.cli calibrate [--offline] [--online] [--n-synth N] [--quiet]
    python -m itol.cli status

Subcommands
-----------
calibrate   Run full calibration pipeline → data/calibration/*.json
status      Show current engine mode and calibration status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from calibration.bootstrap import run_calibration

    offline = not args.online
    try:
        run_calibration(
            offline=offline,
            n_synth_per_class=args.n_synth,
            verbose=not args.quiet,
        )
        return 0
    except Exception as exc:
        print(f"[itol calibrate] ERROR: {exc}", file=sys.stderr)
        return 1


def _cmd_status(args: argparse.Namespace) -> int:
    from itol.engine import Engine, CalibrationRequiredError

    engine = Engine()
    print(f"Mode: {engine.mode}")

    try:
        report = engine.manifest_recall_report()
        overall = report.get("overall", "N/A")
        print(f"Manifest recall (overall): {overall:.3f}")
        per_class = report.get("per_class", {})
        for cls, score in sorted(per_class.items()):
            flag = " [LOW]" if isinstance(score, float) and score < 0.92 else ""
            print(f"  {cls:<28} {score:.3f}{flag}")
    except CalibrationRequiredError:
        print("Calibration not present. Run: python -m itol.cli calibrate --offline")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="itol",
        description="ITOL — Intelligent Token Optimization Layer CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # calibrate subcommand
    p_cal = sub.add_parser("calibrate", help="Run calibration pipeline")
    p_cal.add_argument(
        "--offline", action="store_true", default=True,
        help="Use only bundled synthetic data (default)"
    )
    p_cal.add_argument(
        "--online", action="store_true", default=False,
        help="Also fetch HuggingFace datasets (requires internet + datasets package)"
    )
    p_cal.add_argument(
        "--n-synth", type=int, default=10,
        metavar="N",
        help="Synthetic pairs per class (default: 10)"
    )
    p_cal.add_argument("--quiet", action="store_true", help="Suppress output")

    # status subcommand
    sub.add_parser("status", help="Show engine mode and calibration status")

    args = parser.parse_args(argv)

    if args.command == "calibrate":
        return _cmd_calibrate(args)
    elif args.command == "status":
        return _cmd_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
