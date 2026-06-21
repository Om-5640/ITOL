"""
BenchResult dataclass + cost calculation + bootstrap confidence intervals.
"""
from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Core result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    request_id: str
    sample_id: str
    workload: str          # "rag" | "agent" | "chat" | "faq"
    provider: str
    model: str
    run_type: str          # "baseline" | "itol"
    timestamp: str

    # Tokens (from provider usage field — authoritative)
    tokens_in: int
    tokens_out: int
    tokens_saved: int = 0          # ITOL input-token reduction vs baseline ICR

    # Latency
    latency_ms: float = 0.0
    pipeline_ms: float = 0.0       # ITOL pipeline overhead only

    # ITOL metadata
    strategies_fired: list[str] = field(default_factory=list)
    cache_tier: str = "miss"       # "l0" | "l1" | "l2" | "miss"
    qps: Optional[float] = None
    rollback: bool = False

    # Quality
    quality_score: Optional[float] = None
    quality_method: str = "deterministic"  # "em" | "f1" | "jaccard" | "llm_judge"

    # Cost (paid-tier equivalent)
    equivalent_paid_cost_usd: float = 0.0

    # Text (for report sample cards)
    prompt_text: str = ""
    response_text: str = ""
    gold_answer: Optional[str] = None

    # Rate-limit tracking
    rate_limited_count: int = 0

    # Error (None = success)
    error: Optional[str] = None

    def to_jsonl_line(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchResult":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def token_reduction_pct(self) -> float:
        """Fraction of input tokens saved (0–1). 0 if no savings or baseline."""
        if self.tokens_in <= 0:
            return 0.0
        return self.tokens_saved / (self.tokens_in + self.tokens_saved)


# ---------------------------------------------------------------------------
# JSONL persistence helpers
# ---------------------------------------------------------------------------

def result_path(workload: str, provider: str, date_str: str, base_dir: Path) -> Path:
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{workload}_{provider}_{date_str}.jsonl"


def load_results(base_dir: Path, workload: str | None = None,
                 provider: str | None = None) -> list[BenchResult]:
    raw_dir = base_dir / "raw"
    if not raw_dir.exists():
        return []
    results = []
    for path in raw_dir.glob("*.jsonl"):
        parts = path.stem.split("_")
        if workload and parts[0] != workload:
            continue
        if provider and len(parts) > 1 and parts[1] != provider:
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        results.append(BenchResult.from_dict(json.loads(line)))
                    except Exception:
                        pass
    return results


def completed_ids(path: Path) -> set[str]:
    """Return set of sample_ids already written to a JSONL file."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
    return ids


def append_result(result: BenchResult, path: Path) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(result.to_jsonl_line() + "\n")


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(
    data: list[float],
    stat_fn=_mean,
    n_boot: int = 1000,
    alpha: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Return (point_estimate, lower_ci, upper_ci) using bootstrap resampling.
    alpha=0.95 → 95% CI.
    """
    if not data:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    point = stat_fn(data)
    if len(data) < 2:
        return point, point, point
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(data) for _ in data]
        boots.append(stat_fn(sample))
    boots.sort()
    lo_idx = int(((1 - alpha) / 2) * n_boot)
    hi_idx = int(((1 + alpha) / 2) * n_boot)
    return point, boots[lo_idx], boots[min(hi_idx, len(boots) - 1)]


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

@dataclass
class WorkloadStats:
    workload: str
    provider: str
    n_baseline: int
    n_itol: int

    # Token reduction
    token_reduction_mean: float = 0.0
    token_reduction_lo: float = 0.0
    token_reduction_hi: float = 0.0

    # Quality parity
    quality_mean: float = 0.0
    quality_lo: float = 0.0
    quality_hi: float = 0.0

    # Latency overhead
    latency_baseline_mean: float = 0.0
    latency_itol_mean: float = 0.0
    pipeline_overhead_mean: float = 0.0

    # Cache (FAQ workload)
    cache_hit_rate: float = 0.0

    # Strategy breakdown (strategy_id → mean tokens_saved contribution)
    strategy_breakdown: dict[str, float] = field(default_factory=dict)

    # Cost
    cost_baseline_total: float = 0.0
    cost_itol_total: float = 0.0
    cost_saved_total: float = 0.0
    cost_saved_pct: float = 0.0

    # Sample cards (for report appendix)
    sample_cards: list[dict] = field(default_factory=list)


def aggregate(
    baseline: list[BenchResult],
    itol: list[BenchResult],
    workload: str,
    provider: str,
) -> WorkloadStats:
    """Compute aggregate stats for a (workload, provider) pair."""
    stats = WorkloadStats(workload=workload, provider=provider,
                          n_baseline=len(baseline), n_itol=len(itol))

    baseline_ok = [r for r in baseline if r.success]
    itol_ok     = [r for r in itol     if r.success]

    if not baseline_ok and not itol_ok:
        return stats

    # Token reduction (from ITOL results)
    reductions = [r.token_reduction_pct for r in itol_ok if r.tokens_in > 0]
    if reductions:
        stats.token_reduction_mean, stats.token_reduction_lo, stats.token_reduction_hi = \
            bootstrap_ci(reductions)

    # Quality parity
    quality = [r.quality_score for r in itol_ok if r.quality_score is not None]
    if quality:
        stats.quality_mean, stats.quality_lo, stats.quality_hi = \
            bootstrap_ci(quality)

    # Latency
    if baseline_ok:
        stats.latency_baseline_mean = _mean([r.latency_ms for r in baseline_ok])
    if itol_ok:
        stats.latency_itol_mean = _mean([r.latency_ms for r in itol_ok])
        stats.pipeline_overhead_mean = _mean([r.pipeline_ms for r in itol_ok])

    # Cache (for FAQ workload)
    cache_hits = [r for r in itol_ok if r.cache_tier in ("l0", "l1", "l2")]
    if itol_ok:
        stats.cache_hit_rate = len(cache_hits) / len(itol_ok)

    # Strategy breakdown
    strategy_counts: dict[str, int] = {}
    strategy_savings: dict[str, float] = {}
    for r in itol_ok:
        for s in r.strategies_fired:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
    if itol_ok:
        for s, cnt in strategy_counts.items():
            stats.strategy_breakdown[s] = cnt / len(itol_ok)

    # Cost
    stats.cost_baseline_total = sum(r.equivalent_paid_cost_usd for r in baseline_ok)
    stats.cost_itol_total     = sum(r.equivalent_paid_cost_usd for r in itol_ok)
    stats.cost_saved_total    = max(0.0, stats.cost_baseline_total - stats.cost_itol_total)
    if stats.cost_baseline_total > 0:
        stats.cost_saved_pct = stats.cost_saved_total / stats.cost_baseline_total

    # Sample cards: match by sample_id
    baseline_map = {r.sample_id: r for r in baseline_ok}
    itol_map     = {r.sample_id: r for r in itol_ok}
    cards = []
    for sid in list(itol_map)[:15]:
        b = baseline_map.get(sid)
        i = itol_map[sid]
        cards.append({
            "sample_id":        sid,
            "prompt_baseline":  b.prompt_text[:500] if b else "",
            "prompt_itol":      i.prompt_text[:500],
            "response_baseline": b.response_text[:400] if b else "",
            "response_itol":    i.response_text[:400],
            "gold_answer":      i.gold_answer,
            "quality_score":    i.quality_score,
            "tokens_saved":     i.tokens_saved,
            "strategies":       i.strategies_fired,
        })
    # Sort: show highest savings first, then lowest quality (transparency)
    cards.sort(key=lambda c: (-c["tokens_saved"], c["quality_score"] or 1.0))
    stats.sample_cards = cards[:15]

    return stats


def blended_summary(all_stats: list[WorkloadStats]) -> dict[str, Any]:
    """Compute blended metrics across all workloads/providers for the exec summary."""
    if not all_stats:
        return {}

    reductions = [s.token_reduction_mean for s in all_stats if s.token_reduction_mean > 0]
    qualities  = [s.quality_mean         for s in all_stats if s.quality_mean > 0]
    total_saved = sum(s.cost_saved_total for s in all_stats)
    total_base  = sum(s.cost_baseline_total for s in all_stats)

    blended_reduction = _mean(reductions) if reductions else 0.0
    blended_quality   = _mean(qualities)  if qualities  else 0.0
    cost_saved_pct    = total_saved / total_base if total_base > 0 else 0.0

    return {
        "token_reduction_pct": round(blended_reduction * 100, 1),
        "quality_parity":      round(blended_quality, 3),
        "cost_saved_usd":      round(total_saved, 4),
        "cost_saved_pct":      round(cost_saved_pct * 100, 1),
        "n_workloads":         len({s.workload for s in all_stats}),
        "n_providers":         len({s.provider for s in all_stats}),
        "n_requests":          sum(s.n_itol for s in all_stats),
    }


# ---------------------------------------------------------------------------
# Scale extrapolation table
# ---------------------------------------------------------------------------

SCALE_TIERS = [100_000, 1_000_000, 10_000_000, 100_000_000]  # tokens/day


def scale_table(
    blended_reduction_pct: float,
    provider_cfgs,  # list[ProviderConfig]
) -> list[dict]:
    """
    Return rows for the scale extrapolation table.
    Each row: tokens_per_day, then per-provider cost columns.
    """
    rows = []
    for tpd in SCALE_TIERS:
        row: dict[str, Any] = {"tokens_per_day": tpd}
        for cfg in provider_cfgs:
            col = cfg.scale_monthly_cost(tpd, blended_reduction_pct / 100.0)
            row[cfg.name] = col
        rows.append(row)
    return rows
