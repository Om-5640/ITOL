"""
Thompson-sampling bandit controller — §5.4.

Per (strategy_id, request_class) cell, 4 discretized arms for the relevant
aggressiveness parameter (e.g., S3's mass floor ∈ {0.88, 0.90, 0.93, 0.97}).

Beta posteriors (α, β) per arm, updated ONLY from shadow-evaluated requests:
    reward = parity_normalised - 0.25 × (1 - token_reduction)
    parity_normalised = max(0, (parity - 0.85) / 0.15)   maps [0.85,1.0] → [0,1]

Priors: all arms α=1, β=1 EXCEPT the most conservative arm starts α=2, β=1.
(§5.4: "priors set at the conservative arm")

CR-14: select_arm() queries CircuitBreaker first; OPEN → return conservative arm.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from itol.quality.circuit import CircuitBreaker

# Default arms per strategy (most conservative last = highest mass floor)
_DEFAULT_ARMS: dict[str, list[float]] = {
    "S3": [0.88, 0.90, 0.93, 0.97],
    "S5": [0.70, 0.80, 0.90, 0.95],
    "S7": [0.80, 0.85, 0.90, 0.95],
}
# For unknown strategies, use a generic 4-arm range
_FALLBACK_ARMS = [0.70, 0.80, 0.90, 0.97]


def _conservative_arm(arms: list[float]) -> float:
    """The most conservative arm is the one with the highest value (strictest threshold)."""
    return max(arms)


def _arms_for(strategy_id: str) -> list[float]:
    return list(_DEFAULT_ARMS.get(strategy_id, _FALLBACK_ARMS))


def _parity_normalised(parity: float) -> float:
    return max(0.0, (parity - 0.85) / 0.15)


def _compute_reward(parity: float, token_reduction: float) -> float:
    return _parity_normalised(parity) - 0.25 * (1.0 - token_reduction)


class BanditController:
    """
    Manages Thompson-sampling arm selection and posterior updates.

    `store` holds bandit_state rows; if None, posteriors live only in memory
    (useful for testing without a real database).
    `circuit_breaker` is optional; when provided, select_arm() respects CR-14.
    """

    def __init__(
        self,
        store=None,
        circuit_breaker: "CircuitBreaker | None" = None,
    ) -> None:
        self._store = store
        self._cb = circuit_breaker
        # In-memory fallback (used when store is None)
        self._memory: dict[tuple[str, str, float], tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Alpha/beta accessors
    # ------------------------------------------------------------------

    def _get(self, strategy_id: str, request_class: str, arm: float) -> tuple[float, float]:
        """Return (alpha, beta) for arm; seed conservatism prior for the conservative arm."""
        if self._store is not None:
            ab = self._store.get_bandit_arm(strategy_id, request_class, arm)
            # If the store returns the default (1, 1) and this is the conservative arm,
            # honour the spec prior only on the very first read (no existing row).
            # We detect "no row exists" by checking all arms.
            stored_arms = {a for a, _, _ in self._store.get_all_bandit_arms(strategy_id, request_class)}
            if arm not in stored_arms:
                arms = _arms_for(strategy_id)
                alpha = 2.0 if arm == _conservative_arm(arms) else 1.0
                beta  = 1.0
                self._store.set_bandit_arm(strategy_id, request_class, arm, alpha, beta)
                return alpha, beta
            return ab
        else:
            key = (strategy_id, request_class, arm)
            if key not in self._memory:
                arms = _arms_for(strategy_id)
                alpha = 2.0 if arm == _conservative_arm(arms) else 1.0
                self._memory[key] = (alpha, 1.0)
            return self._memory[key]

    def _set(self, strategy_id: str, request_class: str, arm: float, alpha: float, beta: float) -> None:
        if self._store is not None:
            self._store.set_bandit_arm(strategy_id, request_class, arm, alpha, beta)
        else:
            self._memory[(strategy_id, request_class, arm)] = (alpha, beta)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_arm(
        self,
        strategy_id: str,
        request_class: str,
        tenant_id: str = "default",
    ) -> float:
        """
        CR-14: if CircuitBreaker.check returns OPEN, return the conservative arm.
        Otherwise, Thompson-sample from Beta posteriors.
        """
        if self._cb is not None:
            from itol.quality.circuit import CircuitState
            state = self._cb.check(strategy_id, request_class, tenant_id)
            if state == CircuitState.OPEN:
                return _conservative_arm(_arms_for(strategy_id))

        arms = _arms_for(strategy_id)
        # Thompson sample: draw θ ~ Beta(α, β) per arm, pick argmax
        best_arm = arms[0]
        best_sample = -1.0
        for arm in arms:
            alpha, beta = self._get(strategy_id, request_class, arm)
            sample = random.betavariate(max(alpha, 1e-9), max(beta, 1e-9))
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        return best_arm

    def update(
        self,
        strategy_id: str,
        request_class: str,
        arm_value: float,
        reward: float,
    ) -> None:
        """
        Update Beta posterior for arm_value given reward ∈ [-∞, 1].

        Success increment (+1 to α) if reward > 0; failure increment (+1 to β) otherwise.
        """
        alpha, beta = self._get(strategy_id, request_class, arm_value)
        if reward > 0:
            alpha += reward
        else:
            beta += abs(reward)
        self._set(strategy_id, request_class, arm_value, alpha, beta)

    def compute_reward(self, parity: float, token_reduction: float) -> float:
        return _compute_reward(parity, token_reduction)
