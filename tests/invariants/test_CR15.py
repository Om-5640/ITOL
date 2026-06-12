"""
§14.3 Invariant tests for CR-15 — break-even gate.

CR-15 invariants:
  1. P_in_discounted < P_in_list when cache_read_discount > 0, and
     breakeven_check receives the discounted value (not list price).
  2. Every call to breakeven_check logs both sides (lhs, rhs, decision)
     to the breakeven_log table.
  3. §4.S5 worked example reproduces correctly:
       delta_T=6800, R=6, $3/MTok target
       Case A: C_opt≈0 (local)  → passes
       Case B: C_opt=$0.00375   → passes (5×C_opt=$0.01875 < $0.1224)

Invariant on the safety multiplier: 5× is locked — tests MUST NOT weaken
this to make them pass.
"""

from __future__ import annotations

import pytest

from itol.quality.breakeven import (
    BreakevenResult,
    _SAFETY_MULTIPLIER,
    breakeven_check,
    compute_p_in_discounted,
)
from itol.cache.store import Store


# ===========================================================================
# CR-15-a: P_in_discounted uses discounted price
# ===========================================================================

class TestCR15_DiscountedPrice:

    def test_discounted_price_less_than_list_when_discount_positive(self):
        """
        CR-15: when cache_read_discount > 0, P_in_discounted must be strictly
        less than the base (list) price.
        """
        list_price = 3e-6   # $3 / MTok
        discount   = 0.50   # 50% (OpenAI-style)
        P_disc = compute_p_in_discounted(list_price, discount)
        assert P_disc < list_price, (
            f"CR-15: discounted price {P_disc} must be < list price {list_price}"
        )
        assert abs(P_disc - list_price * 0.50) < 1e-12

    def test_zero_discount_equals_list_price(self):
        """With discount=0.0, compute_p_in_discounted must return list price unchanged."""
        list_price = 5e-6
        P_disc = compute_p_in_discounted(list_price, 0.0)
        assert abs(P_disc - list_price) < 1e-12

    def test_anthropic_discount_90_percent(self):
        """Anthropic cache_read_discount=0.90 → P_in_disc = 10% of list."""
        list_price = 3e-6
        P_disc = compute_p_in_discounted(list_price, 0.90)
        assert abs(P_disc - list_price * 0.10) < 1e-12

    def test_breakeven_check_uses_lower_price_when_discount_applied(self):
        """
        CR-15: breakeven_check with discounted price must produce a LOWER lhs
        than with list price, making the gate harder to pass (conservative).
        """
        R, delta_T, C_opt = 5, 500, 0.0
        list_price = 3e-6
        discount   = 0.50

        result_list = breakeven_check(R, delta_T, list_price, C_opt)
        result_disc = breakeven_check(R, delta_T, compute_p_in_discounted(list_price, discount), C_opt)

        assert result_disc.lhs < result_list.lhs, (
            "CR-15: discounted lhs must be lower than list-price lhs"
        )

    def test_safety_multiplier_is_5(self):
        """The §4.S5 safety multiplier must be exactly 5 — locked invariant."""
        assert _SAFETY_MULTIPLIER == 5.0, (
            f"CR-15: safety multiplier must be 5; got {_SAFETY_MULTIPLIER}"
        )


# ===========================================================================
# CR-15-b: logs both sides to breakeven_log table
# ===========================================================================

class TestCR15_Logging:

    def test_every_call_logs_lhs_rhs_and_decision(self, tmp_path):
        """
        CR-15: after calling breakeven_check with a store, breakeven_log must
        contain a row with the correct lhs, rhs, and passed values.
        """
        store = Store(str(tmp_path))
        R, delta_T = 6, 6800
        P = compute_p_in_discounted(3e-6, 0.0)
        C_opt = 0.00375

        result = breakeven_check(R, delta_T, P, C_opt, store=store)

        rows = store.get_breakeven_log(limit=10)
        assert len(rows) >= 1, "CR-15: at least one row must be written to breakeven_log"

        last = rows[0]   # get_breakeven_log returns newest first
        assert abs(last["lhs"] - result.lhs) < 1e-12, "lhs must match"
        assert abs(last["rhs"] - result.rhs) < 1e-12, "rhs must match"
        assert last["passed"] == result.passed, "passed must match"
        store.close()

    def test_multiple_calls_each_logged(self, tmp_path):
        """Each call to breakeven_check must produce a separate log row."""
        store = Store(str(tmp_path))
        for _ in range(3):
            breakeven_check(4, 100, 3e-6, 0.001, store=store)
        rows = store.get_breakeven_log(limit=10)
        assert len(rows) >= 3, "CR-15: each call must produce a log row"
        store.close()

    def test_call_without_store_does_not_raise(self):
        """breakeven_check with store=None must not raise."""
        result = breakeven_check(3, 500, 3e-6, 0.0, store=None)
        assert isinstance(result, BreakevenResult)

    def test_log_contains_inputs(self, tmp_path):
        """Log row must preserve all four inputs: R, delta_T, P_in_disc, C_opt."""
        store = Store(str(tmp_path))
        P = 1.5e-6
        breakeven_check(10, 500, P, 0.002, store=store)
        rows = store.get_breakeven_log(limit=1)
        row = rows[0]
        assert row["r_reuses"] == 10
        assert row["delta_t"] == 500
        assert abs(row["p_in_disc"] - P) < 1e-12
        assert abs(row["c_opt"] - 0.002) < 1e-12
        store.close()


# ===========================================================================
# CR-15-c: §4.S5 worked example
# ===========================================================================

class TestCR15_WorkedExample:
    """
    Reproduce the §4.S5 worked example verbatim.

    Context: 8000-token document compressed to 1200 tokens.
    delta_T = 8000 - 1200 = 6800 tokens saved per reuse.
    R = 6 expected reuses.
    P_in (target model input price) = $3 / MTok = 3×10⁻⁶ USD per token.
    No cache discount applied to match the spec's worked-example numbers.

    Expected lhs = 6 × 6800 × 3×10⁻⁶ = 0.1224 USD
    """

    _R       = 6
    _DELTA_T = 6800          # tokens: 8000 - 1200
    _P_IN    = 3e-6          # $3/MTok

    def test_lhs_value_exact(self):
        """lhs must equal exactly R × delta_T × P_in."""
        result = breakeven_check(self._R, self._DELTA_T, self._P_IN, 0.0)
        expected_lhs = self._R * self._DELTA_T * self._P_IN  # 0.1224
        assert abs(result.lhs - expected_lhs) < 1e-9, (
            f"Expected lhs={expected_lhs}, got {result.lhs}"
        )

    def test_case_a_local_model_passes(self):
        """
        Case A: local model, C_opt ≈ 0.
        rhs = 5 × 0 = 0 → passed = True (lhs > 0 > rhs is vacuously True).
        """
        result = breakeven_check(self._R, self._DELTA_T, self._P_IN, C_opt=0.0)
        assert result.passed is True, (
            f"Case A: lhs={result.lhs:.4f} must pass against rhs={result.rhs}"
        )
        assert result.ratio == float("inf"), "ratio must be inf when C_opt=0"

    def test_case_b_hosted_small_model_passes(self):
        """
        Case B: hosted small model, C_opt = $0.00375.
        rhs = 5 × $0.00375 = $0.01875
        lhs = $0.1224 > rhs → passed = True.
        """
        C_opt = 0.00375
        result = breakeven_check(self._R, self._DELTA_T, self._P_IN, C_opt=C_opt)
        expected_rhs = 5 * C_opt   # 0.01875
        assert abs(result.rhs - expected_rhs) < 1e-9, (
            f"Expected rhs={expected_rhs}, got {result.rhs}"
        )
        assert result.passed is True, (
            f"Case B: lhs={result.lhs:.4f} must pass against rhs={result.rhs:.4f}"
        )

    def test_case_c_expensive_model_fails(self):
        """
        Control: a very expensive optimisation call MUST fail.
        If C_opt = $0.10 → rhs = $0.50 > lhs = $0.1224.
        """
        C_opt = 0.10
        result = breakeven_check(self._R, self._DELTA_T, self._P_IN, C_opt=C_opt)
        assert result.passed is False, (
            f"Control: lhs={result.lhs:.4f} must NOT pass against rhs={result.rhs:.4f}"
        )

    def test_ratio_reflects_margin(self):
        """ratio = lhs / rhs must indicate how much headroom the gate has."""
        result = breakeven_check(self._R, self._DELTA_T, self._P_IN, C_opt=0.00375)
        # lhs=0.1224, rhs=0.01875 → ratio ≈ 6.528
        assert result.ratio > 6.0, (
            f"Expected ratio > 6.0 for Case B; got {result.ratio:.3f}"
        )
