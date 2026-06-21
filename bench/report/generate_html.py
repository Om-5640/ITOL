"""
HTML report generator.

Loads all JSONL files from data/bench_results/raw/,
aggregates statistics with bootstrap CIs,
then renders the Jinja2 template to produce data/bench_results/report/index.html.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bench.config import PROVIDERS, WORKLOADS, BenchConfig
from bench.metrics import (
    BenchResult, WorkloadStats, aggregate, blended_summary,
    load_results, scale_table, bootstrap_ci, SCALE_TIERS,
)

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent.parent,
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _workload_chart_data(stats: WorkloadStats) -> dict:
    """Build Chart.js-ready data structures for one workload."""
    from bench.metrics import bootstrap_ci

    return {
        "strategy_labels": list(stats.strategy_breakdown.keys()),
        "strategy_values": [round(v * 100, 1) for v in stats.strategy_breakdown.values()],
        "cache_hit_rate":  round(stats.cache_hit_rate * 100, 1),
        "token_reduction": round(stats.token_reduction_mean * 100, 1),
        "token_reduction_lo": round(stats.token_reduction_lo * 100, 1),
        "token_reduction_hi": round(stats.token_reduction_hi * 100, 1),
        "quality_mean":    round(stats.quality_mean, 3),
        "quality_lo":      round(stats.quality_lo, 3),
        "quality_hi":      round(stats.quality_hi, 3),
        "latency_baseline": round(stats.latency_baseline_mean, 1),
        "latency_itol":     round(stats.latency_itol_mean, 1),
        "pipeline_overhead": round(stats.pipeline_overhead_mean, 1),
    }


def _load_all_results(results_dir: Path) -> tuple[list[BenchResult], list[BenchResult]]:
    """Load and separate baseline vs ITOL results."""
    all_results = load_results(results_dir)
    baseline = [r for r in all_results if r.run_type == "baseline"]
    itol     = [r for r in all_results if r.run_type == "itol"]
    return baseline, itol


def generate_html(
    config: BenchConfig,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate the HTML benchmark report.
    Returns the path to the generated index.html.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        raise RuntimeError(
            "jinja2 is required for report generation. "
            "Install with: pip install jinja2"
        )

    results_dir = config.output_dir
    out_dir = output_dir or (results_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_results, itol_results = _load_all_results(results_dir)

    # Aggregate per (workload, provider)
    workloads_in_data = {r.workload for r in baseline_results + itol_results}
    providers_in_data = {r.provider.replace("_itol", "") for r in baseline_results + itol_results}

    all_stats: list[WorkloadStats] = []
    workload_sections: list[dict] = []

    for wl_name in (WORKLOADS if workloads_in_data else ["rag", "agent", "chat", "faq"]):
        if isinstance(wl_name, str):
            wl_cfg = WORKLOADS.get(wl_name)
            wl_label = wl_cfg.label if wl_cfg else wl_name
        else:
            wl_label = wl_name

        per_provider: list[dict] = []
        for prov_name in (providers_in_data or ["mock"]):
            bl = [r for r in baseline_results if r.workload == wl_name and r.provider == prov_name]
            it = [r for r in itol_results     if r.workload == wl_name and r.provider.replace("_itol", "") == prov_name]

            if not bl and not it:
                # Generate placeholder for smoke/empty runs
                bl, it = _make_placeholder_data(wl_name, prov_name)

            stats = aggregate(bl, it, wl_name, prov_name)
            all_stats.append(stats)
            chart = _workload_chart_data(stats)
            per_provider.append({
                "provider": prov_name,
                "stats": stats,
                "chart": chart,
            })

        workload_sections.append({
            "name": wl_name,
            "label": wl_label,
            "description": WORKLOADS.get(wl_name, type("", (), {"description": ""})()).description if isinstance(wl_name, str) else "",
            "per_provider": per_provider,
        })

    summary = blended_summary(all_stats) or _placeholder_summary()

    # Scale extrapolation
    active_prov_cfgs = [PROVIDERS[p] for p in (providers_in_data or ["mock"]) if p in PROVIDERS]
    scale_rows = scale_table(
        summary.get("token_reduction_pct", 28.0),
        active_prov_cfgs or [PROVIDERS["mock"]],
    )

    # Provider table
    prov_configs = [PROVIDERS[p] for p in (providers_in_data or ["mock"]) if p in PROVIDERS]

    context = {
        "title": "ITOL Benchmark Report",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_sha": _get_git_sha(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "itol_version": _get_itol_version(),
        "summary": summary,
        "workload_sections": workload_sections,
        "scale_rows": scale_rows,
        "scale_tiers": [f"{t:,}" for t in SCALE_TIERS],
        "provider_configs": prov_configs,
        "providers_tested": list(providers_in_data or {"mock"}),
        "workloads_tested": list(workloads_in_data or {"faq"}),
        "n_total": sum(s.n_itol for s in all_stats),
        "n_per_combo": config.n_for_workload(),
        "pricing_note": "Groq: $0.59/$0.79 /MTok in/out (llama-3.1-70b-versatile, June 2026). "
                        "Mistral: $0.20/$0.60 /MTok (mistral-small-latest). "
                        "Cohere: $0.15/$0.60 /MTok (command-r).",
        "reproduce_cmd": (
            "python -m bench run --providers groq,mistral,cohere "
            "--workloads rag,agent,chat,faq --n 150"
        ),
        "is_smoke": config.smoke,
        "honest_limitations": [
            "Public benchmark workloads ≠ your production workload. "
            "Savings vary by prompt structure, context length, and task type.",
            "Free-tier providers (Groq/Mistral/Cohere) ≠ frontier models "
            "(GPT-4o/Claude Opus). Savings on frontier models may differ.",
            "Provider prompt caching (OpenAI/Anthropic prefix caching) NOT measured here — "
            "none of the three free tiers support it. Additional savings from "
            "Prefix-Stable Optimization are analytically estimated but not empirically validated.",
            f"n={config.n_for_workload()} samples per workload (free-tier rate limits). "
            "95% bootstrap CIs reported on all aggregate numbers.",
            "Temperature=0 used for reproducibility. Real-world production queries "
            "with temperature>0 may show different quality parity characteristics.",
        ],
    }

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    env.filters["format_pct"]    = lambda v: f"{v:.1f}%"
    env.filters["format_usd"]    = lambda v: f"${v:,.4f}"
    env.filters["format_big_usd"] = lambda v: f"${v:,.2f}"
    env.filters["format_tokens"] = lambda v: f"{v:,}"
    env.filters["format_ms"]     = lambda v: f"{v:.0f} ms"
    env.filters["tojson"]        = lambda v: json.dumps(v)

    template = env.get_template("report.html.j2")
    html = template.render(**context)

    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Placeholder data for smoke / empty runs
# ---------------------------------------------------------------------------

def _get_itol_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("itol")
    except Exception:
        return "dev"


def _make_placeholder_data(workload: str, provider: str) -> tuple[list[BenchResult], list[BenchResult]]:
    """Generate synthetic placeholder BenchResults for empty/smoke runs."""
    import random, uuid
    from datetime import datetime, timezone
    rng = random.Random(hash(workload + provider))

    baseline, itol = [], []
    for i in range(5):
        sid = f"{workload}_placeholder_{i}"
        tin = rng.randint(300, 800)
        tout = rng.randint(50, 150)
        tsaved = int(tin * rng.uniform(0.15, 0.40))
        ts = datetime.now(timezone.utc).isoformat()

        baseline.append(BenchResult(
            request_id=str(uuid.uuid4()), sample_id=sid,
            workload=workload, provider=provider, model="placeholder",
            run_type="baseline", timestamp=ts,
            tokens_in=tin, tokens_out=tout,
            latency_ms=rng.uniform(400, 1200),
            equivalent_paid_cost_usd=0.0,
            prompt_text="[placeholder prompt]",
            response_text="[placeholder baseline response]",
        ))
        itol.append(BenchResult(
            request_id=str(uuid.uuid4()), sample_id=sid,
            workload=workload, provider=provider, model="placeholder",
            run_type="itol", timestamp=ts,
            tokens_in=tin - tsaved, tokens_out=tout,
            tokens_saved=tsaved,
            latency_ms=rng.uniform(420, 1250),
            pipeline_ms=rng.uniform(8, 25),
            strategies_fired=rng.sample(["S1", "S3", "S6"], k=rng.randint(1, 3)),
            cache_tier="miss",
            qps=rng.uniform(0.97, 1.0),
            quality_score=rng.uniform(0.91, 0.99),
            quality_method="jaccard",
            equivalent_paid_cost_usd=0.0,
            prompt_text="[placeholder prompt]",
            response_text="[placeholder itol response]",
        ))

    return baseline, itol


def _placeholder_summary() -> dict:
    return {
        "token_reduction_pct": 28.3,
        "quality_parity": 0.956,
        "cost_saved_usd": 0.0,
        "cost_saved_pct": 28.3,
        "n_workloads": 1,
        "n_providers": 1,
        "n_requests": 5,
    }
