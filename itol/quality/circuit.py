"""
Circuit breaker — §5.4 CR-14.

Rolling 200-sample parity mean per (strategy, class, tenant).

OPEN conditions (either triggers):
  - rolling mean < 0.95, OR
  - P(parity < 0.85) > 2% over the rolling window

When OPEN: force conservative arm (CR-14, checked in BanditController.select_arm).

Second violation within 24h → strategy disabled for that cell;
QUALITY_DEGRADATION alert logged.

Re-enable: manual flag (disabled=False in DB) OR 7-day probation at 10× sampling.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from itol.cache.store import Store


_WINDOW_SIZE = 200
_MEAN_FLOOR  = 0.95
_TAIL_FLOOR  = 0.85
_TAIL_RATE   = 0.02   # P(parity < 0.85) > 2%
_24H         = 86_400.0

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    PROBATION = "PROBATION"


class CircuitBreaker:
    """
    Per-(strategy_id, request_class, tenant_id) rolling parity monitor.

    `store` is required for persistence; without it CircuitBreaker is stateless
    (always CLOSED) — acceptable for lightweight tests of BanditController logic.
    """

    def __init__(self, store: "Store | None" = None) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self, strategy_id: str, request_class: str, tenant_id: str
    ) -> CircuitState:
        """Return the current circuit state for this cell."""
        if self._store is None:
            return CircuitState.CLOSED
        row = self._store.get_circuit(strategy_id, request_class, tenant_id)
        if row is None:
            return CircuitState.CLOSED
        if row["disabled"]:
            return CircuitState.OPEN
        state = row.get("state", "CLOSED")
        return CircuitState(state)

    def record_shadow_result(
        self,
        strategy_id: str,
        request_class: str,
        tenant_id: str,
        parity: float,
    ) -> None:
        """Append parity to the rolling window and update circuit state."""
        if self._store is None:
            return

        row = self._store.get_circuit(strategy_id, request_class, tenant_id)
        if row is None:
            samples: list[float] = []
            violations_24h = 0
            last_violation_ts = 0.0
            disabled = False
            state_str = CircuitState.CLOSED.value
        else:
            samples = list(row["parity_samples"])
            violations_24h = row["violations_24h"]
            last_violation_ts = row["last_violation_ts"]
            disabled = row["disabled"]
            state_str = row["state"]

        if disabled:
            return

        # Maintain rolling window
        samples.append(parity)
        if len(samples) > _WINDOW_SIZE:
            samples = samples[-_WINDOW_SIZE:]

        # Evaluate open conditions
        is_open = False
        if len(samples) >= 10:  # need at least 10 samples to judge
            mean_parity = sum(samples) / len(samples)
            below_tail = sum(1 for p in samples if p < _TAIL_FLOOR)
            tail_rate = below_tail / len(samples)
            is_open = (mean_parity < _MEAN_FLOOR) or (tail_rate > _TAIL_RATE)

        now = time.time()
        if is_open:
            # Count violations within 24h
            if now - last_violation_ts < _24H:
                violations_24h += 1
            else:
                violations_24h = 1
            last_violation_ts = now

            if violations_24h >= 2:
                disabled = True
                state_str = CircuitState.OPEN.value
                logger.warning(
                    "QUALITY_DEGRADATION: strategy=%s class=%s tenant=%s "
                    "disabled after 2nd violation within 24h",
                    strategy_id, request_class, tenant_id,
                )
            else:
                state_str = CircuitState.OPEN.value
        else:
            state_str = CircuitState.CLOSED.value

        self._store.set_circuit(
            strategy_id, request_class, tenant_id,
            parity_samples=samples,
            state=state_str,
            violations_24h=violations_24h,
            last_violation_ts=last_violation_ts,
            disabled=disabled,
        )

    def enable(self, strategy_id: str, request_class: str, tenant_id: str) -> None:
        """Manual re-enable: clear disabled flag and reset to CLOSED."""
        if self._store is None:
            return
        row = self._store.get_circuit(strategy_id, request_class, tenant_id)
        samples = row["parity_samples"] if row else []
        self._store.set_circuit(
            strategy_id, request_class, tenant_id,
            parity_samples=samples,
            state=CircuitState.CLOSED.value,
            violations_24h=0,
            last_violation_ts=0.0,
            disabled=False,
        )
