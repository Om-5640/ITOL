"""
ITOL Benchmark CLI

Subcommands:
  smoke   — Quick smoke test (MockProvider, no API keys, 5 samples/workload)
  check   — Verify API keys and rate-limit headroom
  prepare — Pre-download / generate workload samples to data/bench_corpora/
  run     — Run baseline + ITOL benchmarks
  report  — Generate HTML (+ optional PDF) report from existing JSONL data

Usage examples:
  python -m bench smoke
  python -m bench check
  python -m bench run --providers groq,mistral --workloads rag,faq --n 50
  python -m bench run --providers all --workloads all --n 150
  python -m bench report
  python -m bench report --pdf
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from bench.config import PROVIDERS, WORKLOADS, BenchConfig

logger = logging.getLogger("bench")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommand: check
# ---------------------------------------------------------------------------

def cmd_check(args) -> int:
    _setup_logging(args.verbose)
    print("Checking provider availability...\n")

    available = []
    unavailable = []
    for name, cfg in PROVIDERS.items():
        if name == "mock":
            continue
        if cfg.available:
            available.append(name)
            print(f"  [OK] {name:10s}  {cfg.model}  (key: {cfg.api_key_env})")
        else:
            unavailable.append(name)
            print(f"  [--] {name:10s}  missing ${cfg.api_key_env}")

    print()
    if available:
        print(f"Available: {', '.join(available)}")
    else:
        print("No providers available. Set API key env vars or use --providers mock.")

    if unavailable:
        print(f"Missing keys: {', '.join(unavailable)}")
    print()

    # Quick import check
    try:
        from jinja2 import Environment
        print("[OK] jinja2 installed")
    except ImportError:
        print("[--] jinja2 not installed  (pip install jinja2)")

    try:
        import weasyprint
        print("[OK] weasyprint installed (PDF export available)")
    except ImportError:
        print("[--] weasyprint not installed (PDF export disabled)")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args) -> int:
    _setup_logging(args.verbose)
    from bench.workloads import load_workload

    corpus_dir = Path("data/bench_corpora")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    workloads = _resolve_workloads(args.workloads)
    n = args.n

    for wl_name in workloads:
        print(f"Preparing {wl_name} ({n} samples)...", end=" ", flush=True)
        try:
            samples = load_workload(wl_name, n=n)
            print(f"ok ({len(samples)} samples)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            return 1

    print(f"\nCorpora ready in {corpus_dir}/")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    _setup_logging(args.verbose)

    providers = _resolve_providers(args.providers)
    workloads = _resolve_workloads(args.workloads)
    n = args.n

    if not providers:
        print("No providers available. Run `python -m bench check` for details.", file=sys.stderr)
        return 1

    config = BenchConfig(
        providers=providers,
        workloads=workloads,
        n_samples=n,
        resume=not args.no_resume,
        temperature=0.0,
        seed=42,
    )

    print(f"Benchmark run:")
    print(f"  providers : {providers}")
    print(f"  workloads : {workloads}")
    print(f"  n_samples : {n}")
    print(f"  resume    : {config.resume}")
    print()

    return asyncio.run(_run_all(config, args))


async def _run_all(config: BenchConfig, args) -> int:
    from bench.workloads import load_workload
    from bench.runners.baseline import run_baseline
    from bench.runners.itol_run import run_itol

    def progress(workload, provider, run_type, i, total):
        pct = int(100 * i / total)
        filled = pct // 5
        bar = "#" * filled + "-" * (20 - filled)
        print(f"\r  [{bar}] {pct:3d}%  {run_type}/{workload}/{provider} ({i}/{total})", end="", flush=True)
        if i == total:
            print()

    for prov_name in config.providers:
        prov_cfg = PROVIDERS[prov_name]
        if not prov_cfg.available and prov_name != "mock":
            print(f"  Skipping {prov_name} — API key not set")
            continue

        for wl_name in config.workloads:
            print(f"\n=== {wl_name} x {prov_name} ===")
            samples = load_workload(wl_name, n=config.n_for_workload(), seed=config.seed)
            print(f"  Loaded {len(samples)} samples")

            # 1. Baseline
            print(f"  Running baseline...")
            baseline = await run_baseline(
                workload=wl_name, provider=prov_cfg, config=config,
                samples=samples, progress_cb=progress,
            )
            print(f"  Baseline done: {len(baseline)} results")

            # 2. ITOL
            print(f"  Running ITOL pipeline...")
            itol = await run_itol(
                workload=wl_name, provider=prov_cfg, config=config,
                samples=samples, baseline_results=baseline, progress_cb=progress,
            )
            print(f"  ITOL done: {len(itol)} results")

            # Quick summary
            if baseline and itol:
                _print_quick_summary(baseline, itol, wl_name, prov_name)

    print("\nAll runs complete.")
    print(f"Results in: {config.output_dir}/")
    print("Generate report with: python -m bench report")
    return 0


def _print_quick_summary(baseline, itol, workload, provider):
    from statistics import mean
    bl_tok_in  = mean(r.tokens_in for r in baseline) if baseline else 0
    it_tok_in  = mean(r.tokens_in for r in itol)     if itol else 0
    reduction  = (1 - it_tok_in / bl_tok_in) * 100   if bl_tok_in > 0 else 0
    quality_scores = [r.quality_score for r in itol if r.quality_score is not None]
    avg_qp = mean(quality_scores) if quality_scores else 0.0
    errors = sum(1 for r in baseline if r.error)
    err_note = f"  [{errors} baseline errors]" if errors else ""
    print(f"  Summary: token reduction={reduction:.1f}%  quality_parity={avg_qp:.3f}{err_note}")


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------

def cmd_report(args) -> int:
    _setup_logging(args.verbose)
    from bench.report.generate_html import generate_html

    config = BenchConfig(
        providers=["mock"], workloads=list(WORKLOADS),
        n_samples=150,
    )

    print("Generating HTML report...", flush=True)
    html_path = generate_html(config)
    print(f"Report: {html_path}")

    if args.pdf:
        from bench.report.generate_pdf import generate_pdf
        pdf_path = html_path.parent / "report.pdf"
        print("Generating PDF...", flush=True)
        ok = generate_pdf(html_path, pdf_path)
        if ok:
            print(f"PDF:    {pdf_path}")
        else:
            print("PDF skipped (weasyprint not installed)")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: smoke
# ---------------------------------------------------------------------------

def cmd_smoke(args) -> int:
    """Quick smoke test: MockProvider, 5 samples, all workloads, generates report."""
    _setup_logging(args.verbose)

    print("ITOL Benchmark Smoke Test")
    print("=========================")
    print("Provider: mock (no API keys needed)")
    print("Samples:  5 per workload")
    print()

    config = BenchConfig(
        providers=["mock"],
        workloads=list(WORKLOADS),
        n_samples=5,
        smoke=True,
        resume=False,
    )

    result = asyncio.run(_run_all(config, args))
    if result != 0:
        return result

    # Generate report
    print("\nGenerating smoke report...")
    from bench.report.generate_html import generate_html
    html_path = generate_html(config)
    print(f"\nSmoke test passed!")
    print(f"Report: {html_path}")
    print(f"\nOpen in browser: file://{html_path.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_providers(spec: str) -> list[str]:
    """Parse --providers 'groq,mistral,all,mock' → list of valid names."""
    names = [s.strip() for s in spec.split(",")]
    if "all" in names:
        return [n for n, c in PROVIDERS.items() if n != "mock" and c.available] or ["mock"]
    result = []
    for n in names:
        if n in PROVIDERS:
            result.append(n)
        else:
            print(f"  Warning: unknown provider '{n}' (choices: {list(PROVIDERS)})", file=sys.stderr)
    return result or ["mock"]


def _resolve_workloads(spec: str) -> list[str]:
    """Parse --workloads 'rag,faq,all' → list of workload names."""
    names = [s.strip() for s in spec.split(",")]
    if "all" in names:
        return list(WORKLOADS)
    result = []
    for n in names:
        if n in WORKLOADS:
            result.append(n)
        else:
            print(f"  Warning: unknown workload '{n}' (choices: {list(WORKLOADS)})", file=sys.stderr)
    return result or list(WORKLOADS)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bench",
        description="ITOL Benchmark Harness",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    # check
    sub.add_parser("check", help="Verify API keys and dependencies")

    # prepare
    pr = sub.add_parser("prepare", help="Pre-download workload corpora")
    pr.add_argument("--workloads", default="all", help="Comma-separated workloads (default: all)")
    pr.add_argument("--n", type=int, default=150, help="Samples per workload (default: 150)")

    # run
    ru = sub.add_parser("run", help="Run baseline + ITOL benchmarks")
    ru.add_argument("--providers", default="mock",
                    help="Comma-separated: groq,mistral,cohere,mock,all (default: mock)")
    ru.add_argument("--workloads", default="all",
                    help="Comma-separated: rag,agent,chat,faq,all (default: all)")
    ru.add_argument("--n", type=int, default=150, help="Samples per workload (default: 150)")
    ru.add_argument("--no-resume", action="store_true", help="Don't resume from existing JSONL")

    # report
    rep = sub.add_parser("report", help="Generate HTML report from existing JSONL data")
    rep.add_argument("--pdf", action="store_true", help="Also export PDF (requires weasyprint)")

    # smoke
    sub.add_parser("smoke", help="Quick smoke test (MockProvider, no API keys)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Always add verbose attr
    if not hasattr(args, "verbose"):
        args.verbose = False

    dispatch = {
        "check":   cmd_check,
        "prepare": cmd_prepare,
        "run":     cmd_run,
        "report":  cmd_report,
        "smoke":   cmd_smoke,
    }

    handler = dispatch.get(args.cmd)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args) or 0)
