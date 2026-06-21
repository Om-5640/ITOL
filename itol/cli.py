"""
itol/cli.py — §12 / §15.4 ITOL CLI.

Usage
-----
    python -m itol.cli calibrate [--offline] [--online] [--n-synth N] [--quiet]
    python -m itol.cli status
    python -m itol.cli serve [--port PORT] [--upstream URL] [--reload]
    python -m itol.cli new-adapter --name NAME [--output DIR]

Subcommands
-----------
calibrate    Run full calibration pipeline → data/calibration/*.json
status       Show current engine mode and calibration status
serve        Launch ITOL proxy with uvicorn (§12 Phase 3)
new-adapter  Scaffold a new OpenAI-compatible provider adapter (§8.3)
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


def _cmd_serve(args: argparse.Namespace) -> int:
    """Launch ITOL proxy with uvicorn (§12 Phase 3)."""
    try:
        import uvicorn
    except ImportError:
        print("[itol serve] ERROR: uvicorn not installed. Run: pip install uvicorn", file=sys.stderr)
        return 1

    upstream = args.upstream or ""
    if upstream:
        import os
        os.environ.setdefault("ITOL_DEFAULT_UPSTREAM", upstream)

    uvicorn.run(
        "itol.proxy.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


_NEW_ADAPTER_TEMPLATE = '''\
"""
{name} adapter — extends OpenAICompatibleAdapter (§8.3).

Proof that a new OpenAI-dialect provider can be added in <30 min:
  1. Set _name and base_url.
  2. Override capabilities() with provider limits.
  3. Done — to_icr / from_icr are inherited from OpenAICompatibleAdapter.
"""
from __future__ import annotations
from typing import Any
from itol.adapters.openai_compatible_base import OpenAICompatibleAdapter


class {class_name}Adapter(OpenAICompatibleAdapter):
    _name = "{slug}"
    base_url = "https://api.example.com/v1"  # TODO: replace with real URL

    def capabilities(self) -> dict[str, Any]:
        return {{
            "native_prompt_cache": "none",   # set to "prefix" if provider supports it
            "cache_read_discount": 0.0,       # set to 0.10 for 10% discount
            "max_context": 32_768,            # provider context window
        }}
'''


def _cmd_new_adapter(args: argparse.Namespace) -> int:
    """Scaffold a new provider adapter from the OpenAICompatibleAdapter template."""
    name = args.name.strip()
    if not name:
        print("[itol new-adapter] ERROR: --name is required", file=sys.stderr)
        return 1

    slug = name.lower().replace(" ", "_").replace("-", "_")
    class_name = "".join(part.capitalize() for part in slug.split("_"))

    out_dir = Path(args.output) if args.output else Path("itol/adapters")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{slug}.py"

    if out_file.exists() and not args.force:
        print(f"[itol new-adapter] ERROR: {out_file} already exists. Use --force to overwrite.",
              file=sys.stderr)
        return 1

    content = _NEW_ADAPTER_TEMPLATE.format(
        name=name, class_name=class_name, slug=slug
    )
    out_file.write_text(content, encoding="utf-8")
    print(f"[itol new-adapter] Created: {out_file}")
    print(f"[itol new-adapter] Class:   {class_name}Adapter")
    print(f"[itol new-adapter] Next:    edit base_url and capabilities() in {out_file}")
    return 0


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

    # serve subcommand
    p_serve = sub.add_parser("serve", help="Launch ITOL proxy server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument("--upstream", default="", metavar="URL",
                         help="Default upstream LLM API (e.g. https://api.openai.com/v1/chat/completions)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")

    # new-adapter subcommand
    p_na = sub.add_parser("new-adapter", help="Scaffold a new provider adapter")
    p_na.add_argument("--name", required=True, metavar="NAME",
                      help="Provider name, e.g. 'MyProvider'")
    p_na.add_argument("--output", default="", metavar="DIR",
                      help="Output directory (default: itol/adapters/)")
    p_na.add_argument("--force", action="store_true", help="Overwrite existing file")

    args = parser.parse_args(argv)

    if args.command == "calibrate":
        return _cmd_calibrate(args)
    elif args.command == "status":
        return _cmd_status(args)
    elif args.command == "serve":
        return _cmd_serve(args)
    elif args.command == "new-adapter":
        return _cmd_new_adapter(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
