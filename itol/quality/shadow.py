"""
Shadow evaluation — §5.3 + §15.3.

Adaptive sampling rate:
    n < 100   → 0.20   (cold cell — high uncertainty)
    n < 300   → 0.10
    n < 1000  → 0.03
    else      → 0.015

Modifiers:
    S7 gets ×5 (§4.S7 explicit instruction)
    on_probation → ×3
    (min(1.0, ...))

Hard daily floor: 5 shadow calls per active (strategy, class, tenant) cell,
regardless of traffic.  If floor not yet met today, should_sample() returns True.

ShadowResult carries the parity score and shadow call cost (for CR-13 subtraction).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from itol.icr import ICR, ICRResponse
from itol.quality.parity import compute_parity

# Daily floor — 5 shadow calls per cell
_DAILY_FLOOR = 5


def adaptive_rate(n: int, on_probation: bool, strategy_id: str) -> float:
    """
    §15.3 adaptive sampling rate formula.

    n              — cumulative shadow samples in this cell
    on_probation   — whether this cell is in 7-day probation (10× shadow)
    strategy_id    — e.g. "S3", "S7"
    """
    if n < 100:
        base = 0.20
    elif n < 300:
        base = 0.10
    elif n < 1000:
        base = 0.03
    else:
        base = 0.015

    if strategy_id == "S7":
        base = min(1.0, base * 5)
    if on_probation:
        base = min(1.0, base * 3)

    return base


@dataclass
class ShadowResult:
    parity: float
    shadow_cost_usd: float
    original_response: ICRResponse | None = None


class ShadowEvaluator:
    """
    Evaluates whether to shadow-sample a given cell and, when sampling,
    dispatches the original prompt, computes parity, and records cost.

    `dispatch_fn` is an async callable `(icr: ICR) -> ICRResponse`.  In
    production this is the provider gateway; in tests it can be a stub.
    `store` is required for the daily floor counter.
    """

    def __init__(
        self,
        dispatch_fn: Callable[[ICR], Any] | None = None,
        store=None,
    ) -> None:
        self._dispatch = dispatch_fn
        self._store = store

    # ------------------------------------------------------------------
    # Sampling decision
    # ------------------------------------------------------------------

    def should_sample(
        self,
        cell_key: str,
        n_samples_in_cell: int,
        on_probation: bool,
        strategy_id: str,
    ) -> bool:
        """
        Returns True if this request should be shadow-sampled.

        Combines:
          1. Adaptive rate check (probabilistic)
          2. Daily floor: if today's count < 5, force True regardless of rate
        """
        import random

        # Check daily floor first
        if self._store is not None:
            today = datetime.date.today().isoformat()
            daily_count = self._store.get_shadow_floor_count(cell_key, today)
            if daily_count < _DAILY_FLOOR:
                return True

        rate = adaptive_rate(n_samples_in_cell, on_probation, strategy_id)
        return random.random() < rate

    # ------------------------------------------------------------------
    # Shadow execution
    # ------------------------------------------------------------------

    async def run_shadow_async(
        self,
        original_icr: ICR,
        optimized_response: ICRResponse,
        manifest=None,
    ) -> ShadowResult:
        """
        Dispatch the ORIGINAL prompt and compute parity between
        the original response and the optimized response.

        CR-13: the shadow call's cost (in USD) is returned so the
        recorder can subtract it from reported savings.
        """
        if self._dispatch is None:
            return ShadowResult(parity=1.0, shadow_cost_usd=0.0)

        original_response: ICRResponse = await self._dispatch(original_icr)

        # Compute parity
        from itol.icr import ConstraintManifest
        mfst = manifest or ConstraintManifest()
        parity = compute_parity(optimized_response, original_response, mfst)

        # Estimate shadow cost (input tokens × ~3 USD/1M tokens for approximation)
        shadow_tokens = (
            original_response.usage.input_tokens if original_response.usage else 0
        )
        shadow_cost = shadow_tokens * 3e-6

        return ShadowResult(
            parity=parity,
            shadow_cost_usd=shadow_cost,
            original_response=original_response,
        )

    def run_shadow(
        self,
        original_icr: ICR,
        optimized_response: ICRResponse,
        manifest=None,
    ) -> "Coroutine[Any, Any, ShadowResult]":
        """Return a coroutine for async dispatch (mirrors run_shadow_async)."""
        return self.run_shadow_async(original_icr, optimized_response, manifest)

    # ------------------------------------------------------------------
    # Floor accounting
    # ------------------------------------------------------------------

    def record_sample(self, cell_key: str) -> None:
        """Increment today's shadow count for the cell."""
        if self._store is not None:
            today = datetime.date.today().isoformat()
            self._store.increment_shadow_floor(cell_key, today)
