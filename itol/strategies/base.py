"""
Strategy base classes and shared utilities — §4.

All strategies implement the Strategy ABC.  OptimizationContext carries all
per-request state needed for gating and cost decisions.

CR-2 Prefix-Stable rule: no strategy may mutate a span inside the provider's
KV-cache prefix unless the token savings exceed the cost of breaking it.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from itol.config import ITOLConfig
from itol.icr import ConstraintManifest, ICR, SegmentSignals, StrategyReport
from itol.routing.matrix import ClassMatrix
from itol.segmenter import Segment


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def seg_hash(text: str) -> str:
    """Compute the canonical segment hash (sha256 of normalised text)."""
    normalised = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalised.encode()).hexdigest()


def update_segment(seg: Segment, new_text: str) -> Segment:
    """Return a copy of `seg` with updated text, hash, and token_count."""
    from dataclasses import replace as dc_replace
    from itol.signals import estimate_token_count
    new_hash = seg_hash(new_text)
    return dc_replace(
        seg,
        text=new_text,
        segment_hash=new_hash,
        token_count=estimate_token_count(new_text),
    )


# ---------------------------------------------------------------------------
# OptimizationContext
# ---------------------------------------------------------------------------

@dataclass
class OptimizationContext:
    """
    Per-request context passed to every strategy.

    Carries request class, routing matrix row, manifest, signals, config,
    and provider-cache economics for CR-2 prefix-safety enforcement.
    """
    request_class: str
    matrix_row: ClassMatrix
    manifest: ConstraintManifest
    signals: SegmentSignals
    config: ITOLConfig
    prefix_cacheable_span_bytes: int = 0
    provider_cache_value: float = 0.0   # USD value of the cached prefix (CR-2)
    input_price_per_token: float = 3e-6  # USD per input token; override per model


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Abstract base for all ITOL optimisation strategies."""

    strategy_id: str
    risk_class: str  # "LOSSLESS" | "NEAR_LOSSLESS" | "LOSSY_BOUNDED" | "LOSSY_AGGRESSIVE"

    @abstractmethod
    def applies(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> bool:
        """Return True if this strategy should run for this request."""
        ...

    @abstractmethod
    def apply(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> tuple[list[Segment], StrategyReport]:
        """
        Apply the strategy.  Returns (new_segments, report).

        If not activated, MUST return (segments_unchanged, report with
        activated=False) — never mutate segments when not activated.

        MUST set report.segment_snapshot to a shallow copy of `segments`
        taken BEFORE any mutation (required for QPS-gate rollback).
        """
        ...

    def _prefix_safe(
        self,
        span_start_byte: int,
        span_end_byte: int,
        savings_tokens: int,
        ctx: OptimizationContext,
    ) -> bool:
        """
        CR-2: return True iff mutating this byte span is prefix-safe.

        True when:
        - span lies entirely outside the cacheable prefix, OR
        - no prefix is cached (prefix_cacheable_span_bytes == 0), OR
        - savings_tokens × input_price_per_token > provider_cache_value
        """
        if span_start_byte >= ctx.prefix_cacheable_span_bytes:
            return True
        if ctx.prefix_cacheable_span_bytes == 0 or ctx.provider_cache_value <= 0:
            return True
        savings_value = savings_tokens * ctx.input_price_per_token
        return savings_value > ctx.provider_cache_value

    def _make_report(
        self,
        segments_before: list[Segment],
        tokens_before: int,
        tokens_after: int,
        touched: list[str],
        removed_spans: list[tuple[int, int]],
        prefix_bytes_mutated: int,
        prefix_savings: int,
        activated: bool,
        notes: str = "",
    ) -> StrategyReport:
        """Build a populated StrategyReport with all required fields."""
        saved = tokens_before - tokens_after
        return StrategyReport(
            strategy_id=self.strategy_id,
            tokens_removed=saved,       # backward-compat
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=saved,
            risk_class=self.risk_class,
            touched_segments=touched,
            removed_spans=removed_spans,
            segment_snapshot=list(segments_before),
            prefix_bytes_mutated=prefix_bytes_mutated,
            prefix_savings=prefix_savings,
            activated=activated,
            notes=notes,
        )
