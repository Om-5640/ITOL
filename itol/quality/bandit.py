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


_RECALL_BUMP_THRESHOLD = 0.92  # CR-26: recall below this triggers α=3 conservative prior


class BanditController:
    """
    Manages Thompson-sampling arm selection and posterior updates.

    `store` holds bandit_state rows; if None, posteriors live only in memory
    (useful for testing without a real database).
    `circuit_breaker` is optional; when provided, select_arm() respects CR-14.
    `_low_recall_classes` holds classes whose manifest recall < 0.92 (CR-26).
    """

    def __init__(
        self,
        store=None,
        circuit_breaker: "CircuitBreaker | None" = None,
        low_recall_classes: set[str] | None = None,
    ) -> None:
        self._store = store
        self._cb = circuit_breaker
        # CR-26: classes with recall < 0.92 use α=3 conservative prior
        self._low_recall_classes: set[str] = low_recall_classes or set()
        # In-memory fallback (used when store is None)
        self._memory: dict[tuple[str, str, float], tuple[float, float]] = {}

    @classmethod
    def load_with_recall(
        cls,
        manifest_recall_path: str,
        store=None,
        circuit_breaker: "CircuitBreaker | None" = None,
    ) -> "BanditController":
        """
        CR-26: load manifest_recall.json and return a BanditController that
        applies an elevated conservative prior (α=3, β=1) for any class whose
        recall < 0.92.
        """
        import json
        from pathlib import Path

        low_recall: set[str] = set()
        recall_path = Path(manifest_recall_path)
        if recall_path.exists():
            with open(recall_path, encoding="utf-8") as fh:
                data = json.load(fh)
            for req_class, recall in data.get("per_class", {}).items():
                if recall < _RECALL_BUMP_THRESHOLD:
                    low_recall.add(req_class)
        return cls(store=store, circuit_breaker=circuit_breaker, low_recall_classes=low_recall)

    def get_conservative_prior(self, request_class: str) -> tuple[float, float]:
        """
        CR-26: return (α, β) for the conservative arm prior.
        Low-recall classes → (3, 1); normal classes → (2, 1).
        """
        alpha = 3.0 if request_class in self._low_recall_classes else 2.0
        return alpha, 1.0

    # ------------------------------------------------------------------
    # Alpha/beta accessors
    # ------------------------------------------------------------------

    def _conservative_alpha(self, request_class: str) -> float:
        """Return the conservative arm starting α — elevated for CR-26 low-recall classes."""
        return self.get_conservative_prior(request_class)[0]

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
                if arm == _conservative_arm(arms):
                    alpha = self._conservative_alpha(request_class)
                else:
                    alpha = 1.0
                beta  = 1.0
                self._store.set_bandit_arm(strategy_id, request_class, arm, alpha, beta)
                return alpha, beta
            return ab
        else:
            key = (strategy_id, request_class, arm)
            if key not in self._memory:
                arms = _arms_for(strategy_id)
                if arm == _conservative_arm(arms):
                    alpha = self._conservative_alpha(request_class)
                else:
                    alpha = 1.0
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
