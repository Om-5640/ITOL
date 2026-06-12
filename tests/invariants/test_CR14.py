"""
§14.3 Invariant test for CR-14 — circuit breaker overrides bandit arm selection.

CR-14: when CircuitBreaker.check() returns CircuitState.OPEN for a
(strategy_id, request_class, tenant_id) cell, BanditController.select_arm()
MUST return the most conservative arm, regardless of the Beta posterior.

This is an unconditional override — even if the bandit posterior strongly
favours an aggressive arm, the conservative arm must be returned when
the circuit is OPEN.

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import pytest

from itol.quality.bandit import BanditController, _conservative_arm, _arms_for
from itol.quality.circuit import CircuitBreaker, CircuitState


# ===========================================================================
# CR-14: breaker OPEN → conservative arm unconditionally
# ===========================================================================

class TestCR14_BreakerOverridesBandit:

    def test_open_circuit_returns_conservative_arm(self):
        """
        CR-14: set the bandit posterior to strongly favour the most aggressive
        arm (lowest mass floor = 0.88); open the circuit; assert select_arm
        returns the CONSERVATIVE arm (0.97), not the aggressive one.
        """
        # Build a bandit with the aggressive arm favoured (α=100, β=1)
        bandit = BanditController(store=None)
        strategy_id   = "S3"
        request_class = "EXTRACTION"
        tenant_id     = "t1"

        arms = _arms_for(strategy_id)
        aggressive = min(arms)    # 0.88
        conservative = max(arms)  # 0.97

        # Set aggressive arm posterior to very high α
        bandit._memory[(strategy_id, request_class, aggressive)] = (100.0, 1.0)
        # Set conservative arm to low α (posterior says avoid it)
        bandit._memory[(strategy_id, request_class, conservative)] = (1.0, 100.0)
        bandit._memory[(strategy_id, request_class, 0.90)] = (1.0, 1.0)
        bandit._memory[(strategy_id, request_class, 0.93)] = (1.0, 1.0)

        # Create a mock circuit breaker that always reports OPEN
        class _AlwaysOpenBreaker:
            def check(self, sid, rc, tid):
                return CircuitState.OPEN

        bandit._cb = _AlwaysOpenBreaker()

        # Run many trials — must ALWAYS return conservative arm
        for _ in range(50):
            arm = bandit.select_arm(strategy_id, request_class, tenant_id)
            assert arm == conservative, (
                f"CR-14 INVARIANT: circuit OPEN must return conservative arm {conservative}, "
                f"got {arm}"
            )

    def test_closed_circuit_uses_bandit_posterior(self):
        """
        Control: when circuit is CLOSED, the bandit posterior is free to select
        any arm.  With an extremely high posterior for the aggressive arm, it
        should be selected most of the time.
        """
        bandit = BanditController(store=None)
        strategy_id   = "S3"
        request_class = "SUMMARIZATION"
        tenant_id     = "t1"

        arms = _arms_for(strategy_id)
        aggressive = min(arms)   # 0.88

        # Strongly favour the aggressive arm
        for arm in arms:
            if arm == aggressive:
                bandit._memory[(strategy_id, request_class, arm)] = (1000.0, 1.0)
            else:
                bandit._memory[(strategy_id, request_class, arm)] = (1.0, 1000.0)

        class _AlwaysClosedBreaker:
            def check(self, sid, rc, tid):
                return CircuitState.CLOSED

        bandit._cb = _AlwaysClosedBreaker()

        selections = [bandit.select_arm(strategy_id, request_class, tenant_id) for _ in range(100)]
        aggressive_count = sum(1 for s in selections if s == aggressive)

        assert aggressive_count >= 80, (
            f"Control: with strong posterior, aggressive arm should win most trials; "
            f"got {aggressive_count}/100"
        )

    def test_no_circuit_breaker_uses_bandit_normally(self):
        """Without a circuit breaker, BanditController works normally."""
        bandit = BanditController(store=None, circuit_breaker=None)
        # Should not raise
        arm = bandit.select_arm("S3", "REASONING")
        arms = _arms_for("S3")
        assert arm in arms, f"select_arm must return a valid arm; got {arm}"
