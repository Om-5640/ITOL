"""
Quality Preservation Score gate and rollback orchestrator — §5.2 + §15.2.

Public API
----------
compute_qps(...)     → QPSResult          pure scoring, no side-effects
score_and_rollback(...)  → ScoreResult    score + incremental rollback + CR-16 guard

Rollback constants
------------------
PIPELINE_ORDER   — execution order of each strategy (used to find earliest snapshot)
ROLLBACK_ORDER   — order in which strategies are disabled (S7→S4→S3→S5→S1)
NO_ROLLBACK      — strategies that are NEVER rolled back (S2, S6 — verified-lossless)

Placeholder note
----------------
semantic_fidelity and min_window_fidelity default to 1.0 until the embedder
is built in Step 6.  The interface is final; the real computation slots in
without changing this file.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from itol.config import QualityConfig
from itol.icr import ConstraintManifest, ICR, StrategyReport


# ---------------------------------------------------------------------------
# Strategy ordering constants
# ---------------------------------------------------------------------------

# Position in the execution pipeline (S2 → S1 → S6 → S3 → S5 → S4 → S7)
PIPELINE_ORDER: dict[str, int] = {
    "S2": 0,
    "S1": 1,
    "S6": 2,
    "S3": 3,
    "S5": 4,
    "S4": 5,
    "S7": 6,
}

# Rollback disables strategies in this exact order (§5.2 CR-12)
ROLLBACK_ORDER: tuple[str, ...] = ("S7", "S4", "S3", "S5", "S1")

# S2 and S6 are verified-lossless and are NEVER rolled back (§5.2)
NO_ROLLBACK: frozenset[str] = frozenset({"S2", "S6"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QPSResult:
    """Scoring output — all components retained for telemetry (§9)."""
    qps: float
    manifest_coverage: float
    semantic_fidelity: float       # placeholder 1.0 until Step-6 embedder
    min_window_fidelity: float     # §15.2 placeholder 1.0 until Step-6 embedder
    coverage_margin: float         # raw S3/S4 relevance mass (pre-rescale)
    passed: bool                   # qps >= floor_used
    floor_used: float              # 0.99 if S7 participated, else 0.98
    s7_participated: bool
    polarity_intact: bool = True   # CR-22 result
    rollback_stages_tried: list[str] = field(default_factory=list)
    rollback_stage_passed: str | None = None   # strategy whose rollback let it pass


@dataclass
class ScoreResult:
    """
    Outcome of score_and_rollback.

    use_raw=True  → caller MUST dispatch icr.raw byte-identical (CR-3b guarantee)
    use_raw=False → caller uses `segments` to assemble the provider payload
    """
    use_raw: bool
    segments: list[Any] | None   # None when use_raw=True
    qps_result: QPSResult


# ---------------------------------------------------------------------------
# Coverage-margin rescaling  (§5.2: [0.9, 1] → [0, 1])
# ---------------------------------------------------------------------------

def _rescale_margin(margin: float) -> float:
    """
    §5.2: coverage_margin is S3/S4 relevance mass, rescaled [0.9,1]→[0,1].
    Values ≤ 0.9 clamp to 0; 1.0 → 1.0.
    """
    return max(0.0, min(1.0, (margin - 0.9) / 0.1))


# ---------------------------------------------------------------------------
# Core QPS formula  (§5.2 + §15.2)
# ---------------------------------------------------------------------------

def compute_qps(
    manifest: ConstraintManifest,
    optimised_text: str,
    quality_cfg: QualityConfig,
    *,
    semantic_fidelity: float = 1.0,
    min_window_fidelity: float = 1.0,
    coverage_margin: float = 1.0,
    s7_participated: bool = False,
    optimised_segments: list | None = None,
) -> QPSResult:
    """
    §5.2 QPS formula with §15.2 min_window_fidelity substitution.

    QPS = 1.0 · hard_gate
        × ( 0.45 · manifest_coverage
          + 0.30 · semantic_fidelity
          + 0.15 · min_window_fidelity      ← §15.2 replaces structure_integrity
          + 0.10 · rescaled_coverage_margin )

    Hard gates (multiplicative, short-circuit to 0):
      • manifest_coverage < 1.0
      • polarity_intact == False  (CR-22)

    Floors (read from quality_cfg — never hardcoded):
      • 0.99 when S7 participated
      • 0.98 otherwise

    optimised_segments: when provided, polarity_intact checks each item against
    its source segment text (same scope as manifest construction) rather than
    the full concatenated text, preventing false negatives from cross-segment
    context bleed (e.g. after S1 removes exact-duplicate document blocks).
    """
    from itol.analysis.manifest import polarity_intact as _polarity_intact

    cov = manifest.coverage(optimised_text)
    pol_ok = _polarity_intact(manifest, optimised_text, optimised_segments)

    if cov < 1.0 or not pol_ok:
        qps = 0.0
    else:
        qps = 1.0 * (
            0.45 * cov
            + 0.30 * semantic_fidelity
            + 0.15 * min_window_fidelity
            + 0.10 * _rescale_margin(coverage_margin)
        )

    floor = quality_cfg.qps_floor_with_s7 if s7_participated else quality_cfg.qps_floor

    return QPSResult(
        qps=qps,
        manifest_coverage=cov,
        semantic_fidelity=semantic_fidelity,
        min_window_fidelity=min_window_fidelity,
        coverage_margin=coverage_margin,
        passed=qps >= floor,
        floor_used=floor,
        s7_participated=s7_participated,
        polarity_intact=pol_ok,
    )


# ---------------------------------------------------------------------------
# Segment-text helper (import here to avoid circular imports at module level)
# ---------------------------------------------------------------------------

def _segments_to_text(segments: list[Any]) -> str:
    from itol.segmenter import segments_full_text
    return segments_full_text(segments)


# ---------------------------------------------------------------------------
# Rollback helpers
# ---------------------------------------------------------------------------

def _find_snapshot_for_disabled(
    disabled: list[str],
    reports_by_id: dict[str, StrategyReport],
) -> list[Any] | None:
    """
    Return the segment snapshot corresponding to rolling back all `disabled`
    strategies simultaneously.

    Because strategies run in pipeline order, the correct state is the
    snapshot of the EARLIEST disabled strategy (the one that ran first).
    That snapshot captures the segment list before any of the disabled
    strategies had a chance to mutate it.

    Returns None if no report with a snapshot exists for the earliest strategy.
    """
    if not disabled:
        return None

    earliest = min(disabled, key=lambda s: PIPELINE_ORDER.get(s, 999))
    report = reports_by_id.get(earliest)
    if report is None or report.segment_snapshot is None:
        return None
    return report.segment_snapshot


def _applied_strategies(reports: list[StrategyReport]) -> list[str]:
    """Strategy IDs that were actually applied (have a report)."""
    return [r.strategy_id for r in reports]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_and_rollback(
    icr: ICR,
    strategy_reports: list[StrategyReport],
    manifest: ConstraintManifest,
    quality_cfg: QualityConfig,
    optimised_segments: list[Any],
    *,
    semantic_fidelity: float = 1.0,
    min_window_fidelity: float = 1.0,
    coverage_margin: float = 1.0,
) -> ScoreResult:
    """
    Score the optimised prompt.  If QPS < floor, disable strategies one by one
    in ROLLBACK_ORDER and re-score.  Stop at the first passing state.
    If still failing after all eligible rollbacks → dispatch icr.raw (CR-3b).

    Wrapped in a blanket try/except: any unhandled exception → return icr.raw
    without propagating (CR-16).

    Rollback rules:
    • Strategies are disabled in ROLLBACK_ORDER: S7 → S4 → S3 → S5 → S1
    • S2 and S6 are NEVER rolled back (NO_ROLLBACK)
    • Disabling N strategies simultaneously uses the snapshot of the EARLIEST
      disabled strategy in PIPELINE_ORDER (the one that ran first)
    """
    try:
        return _score_and_rollback_inner(
            icr, strategy_reports, manifest, quality_cfg, optimised_segments,
            semantic_fidelity=semantic_fidelity,
            min_window_fidelity=min_window_fidelity,
            coverage_margin=coverage_margin,
        )
    except Exception:
        # CR-16: exceptions must never propagate — return raw, log quietly
        _log_exception()
        fallback_qps = QPSResult(
            qps=0.0,
            manifest_coverage=0.0,
            semantic_fidelity=1.0,
            min_window_fidelity=1.0,
            coverage_margin=1.0,
            passed=False,
            floor_used=quality_cfg.qps_floor,
            s7_participated=False,
            polarity_intact=False,
        )
        return ScoreResult(use_raw=True, segments=None, qps_result=fallback_qps)


def _score_and_rollback_inner(
    icr: ICR,
    strategy_reports: list[StrategyReport],
    manifest: ConstraintManifest,
    quality_cfg: QualityConfig,
    optimised_segments: list[Any],
    *,
    semantic_fidelity: float,
    min_window_fidelity: float,
    coverage_margin: float,
) -> ScoreResult:
    applied = _applied_strategies(strategy_reports)
    reports_by_id: dict[str, StrategyReport] = {r.strategy_id: r for r in strategy_reports}
    s7_participated = "S7" in applied

    # --- Score the full optimised result first ---
    opt_text = _segments_to_text(optimised_segments)
    result = compute_qps(
        manifest, opt_text, quality_cfg,
        semantic_fidelity=semantic_fidelity,
        min_window_fidelity=min_window_fidelity,
        coverage_margin=coverage_margin,
        s7_participated=s7_participated,
        optimised_segments=optimised_segments,
    )

    if result.passed:
        return ScoreResult(use_raw=False, segments=optimised_segments, qps_result=result)

    # --- Incremental rollback (CR-12) ---
    disabled: list[str] = []
    stages_tried: list[str] = []

    for strategy_id in ROLLBACK_ORDER:
        if strategy_id not in applied:
            continue  # strategy was never run — nothing to roll back
        if strategy_id in NO_ROLLBACK:
            continue  # safety check (ROLLBACK_ORDER already excludes S2/S6)

        disabled.append(strategy_id)
        stages_tried.append(strategy_id)

        snapshot = _find_snapshot_for_disabled(disabled, reports_by_id)
        if snapshot is None:
            # No snapshot available — cannot roll back further, try next
            continue

        # S7 is only considered "participating" if it's still in the disabled-free set
        s7_still_active = "S7" in applied and "S7" not in disabled
        candidate_text = _segments_to_text(snapshot)
        candidate_result = compute_qps(
            manifest, candidate_text, quality_cfg,
            semantic_fidelity=semantic_fidelity,
            min_window_fidelity=min_window_fidelity,
            coverage_margin=coverage_margin,
            s7_participated=s7_still_active,
            optimised_segments=snapshot,
        )
        candidate_result.rollback_stages_tried = stages_tried[:]
        candidate_result.rollback_stage_passed = strategy_id if candidate_result.passed else None

        if candidate_result.passed:
            return ScoreResult(
                use_raw=False,
                segments=snapshot,
                qps_result=candidate_result,
            )

    # --- Full rollback exhausted — dispatch icr.raw (CR-3b) ---
    fallback = QPSResult(
        qps=0.0,
        manifest_coverage=result.manifest_coverage,
        semantic_fidelity=semantic_fidelity,
        min_window_fidelity=min_window_fidelity,
        coverage_margin=coverage_margin,
        passed=False,
        floor_used=result.floor_used,
        s7_participated=s7_participated,
        polarity_intact=result.polarity_intact,
        rollback_stages_tried=stages_tried,
        rollback_stage_passed=None,
    )
    return ScoreResult(use_raw=True, segments=None, qps_result=fallback)


def _log_exception() -> None:
    """Quiet exception log — never raises."""
    try:
        import sys
        print(f"[ITOL QPS] exception suppressed (CR-16):\n{traceback.format_exc()}",
              file=sys.stderr)
    except Exception:
        pass
