"""
§14.3 Invariant tests for CR-24 / §15.3 — adaptive shadow sampling.

§15.3 formula (all four cases must be exact):
    n < 100   → base = 0.20
    n < 300   → base = 0.10
    n < 1000  → base = 0.03
    else      → base = 0.015

Modifiers (applied after base is set):
    strategy_id == "S7" → base = min(1.0, base × 5)
    on_probation        → base = min(1.0, base × 3)

Hard daily floor:
    even at n ≥ 1000 with rate = 0.015, should_sample() returns True
    if today's count < 5 for this cell.

Rules:
- NEVER weaken thresholds or conditions.
"""

from __future__ import annotations

import datetime
import tempfile

import pytest

from itol.cache.store import Store
from itol.quality.shadow import ShadowEvaluator, adaptive_rate


# ===========================================================================
# §15.3-a: base rate by traffic bucket
# ===========================================================================

class TestCR24_AdaptiveRate:

    def test_cold_cell_rate_is_0_20(self):
        """n < 100 → rate = 0.20."""
        assert adaptive_rate(0, False, "S3") == 0.20
        assert adaptive_rate(99, False, "S3") == 0.20

    def test_medium_cell_rate_is_0_10(self):
        """100 ≤ n < 300 → rate = 0.10."""
        assert adaptive_rate(100, False, "S3") == 0.10
        assert adaptive_rate(299, False, "S3") == 0.10

    def test_warm_cell_rate_is_0_03(self):
        """300 ≤ n < 1000 → rate = 0.03."""
        assert adaptive_rate(300, False, "S3") == 0.03
        assert adaptive_rate(999, False, "S3") == 0.03

    def test_hot_cell_rate_is_0_015(self):
        """n ≥ 1000 → rate = 0.015."""
        assert adaptive_rate(1000, False, "S3") == 0.015
        assert adaptive_rate(100_000, False, "S3") == 0.015


# ===========================================================================
# §15.3-b: S7 five-times multiplier
# ===========================================================================

class TestCR24_S7Rate:

    def test_s7_cold_cell_5x(self):
        """S7 cold: 0.20 × 5 = 1.0."""
        assert adaptive_rate(0, False, "S7") == 1.0

    def test_s7_medium_cell_5x(self):
        """S7 medium: min(1.0, 0.10 × 5) = 0.50."""
        assert abs(adaptive_rate(100, False, "S7") - 0.50) < 1e-9

    def test_s7_hot_cell_5x(self):
        """S7 hot: min(1.0, 0.015 × 5) = 0.075."""
        assert abs(adaptive_rate(1000, False, "S7") - 0.075) < 1e-9


# ===========================================================================
# §15.3-c: probation triples the rate
# ===========================================================================

class TestCR24_ProbationRate:

    def test_probation_triples_base(self):
        """on_probation=True → base × 3 (capped at 1.0)."""
        # cold: min(1.0, 0.20 × 3) = 0.60
        assert abs(adaptive_rate(0, True, "S3") - 0.60) < 1e-9
        # medium: min(1.0, 0.10 × 3) = 0.30
        assert abs(adaptive_rate(100, True, "S3") - 0.30) < 1e-9
        # warm: min(1.0, 0.03 × 3) = 0.09
        assert abs(adaptive_rate(300, True, "S3") - 0.09) < 1e-9
        # hot: min(1.0, 0.015 × 3) = 0.045
        assert abs(adaptive_rate(1000, True, "S3") - 0.045) < 1e-9

    def test_probation_caps_at_1(self):
        """S7 cold + probation: min(1.0, 1.0 × 3) = 1.0."""
        assert adaptive_rate(0, True, "S7") == 1.0


# ===========================================================================
# §15.3-d: daily floor — force True when count < 5
# ===========================================================================

class TestCR24_DailyFloor:

    def test_daily_floor_forces_sample_at_hot_cell(self, tmp_path):
        """
        Hard floor: even when n ≥ 1000 (rate = 0.015), should_sample must
        return True if today's count < 5 for this cell.
        """
        store = Store(str(tmp_path))
        evaluator = ShadowEvaluator(store=store)
        cell_key = "S3|EXTRACTION|default"

        # n=10000 → rate = 0.015; without floor, chance of True = 1.5%
        # With floor (today count = 0), must always return True
        assert evaluator.should_sample(
            cell_key,
            n_samples_in_cell=10_000,
            on_probation=False,
            strategy_id="S3",
        ) is True, (
            "§15.3 INVARIANT: daily floor must force should_sample=True "
            "when today's count < 5"
        )
        store.close()

    def test_daily_floor_not_enforced_after_5_calls(self, tmp_path, monkeypatch):
        """
        Once today's count ≥ 5, the daily floor no longer forces True.
        At n=10000 (rate 0.015), with floor met, sampling is probabilistic.
        We verify by running 1000 trials with random() patched to 0.99 (above rate):
        should_sample must return False for all of them.
        """
        monkeypatch.setattr("random.random", lambda: 0.99)

        store = Store(str(tmp_path))
        today = datetime.date.today().isoformat()
        # Seed the counter to 5 (floor met)
        for _ in range(5):
            store.increment_shadow_floor("S3|EXTRACTION|t2", today)

        evaluator = ShadowEvaluator(store=store)
        result = evaluator.should_sample(
            "S3|EXTRACTION|t2",
            n_samples_in_cell=10_000,
            on_probation=False,
            strategy_id="S3",
        )
        # random() = 0.99 >> rate=0.015 → should be False
        assert result is False, (
            "Once daily floor met, should_sample must be probabilistic (False when random > rate)"
        )
        store.close()

    def test_record_sample_increments_floor_counter(self, tmp_path):
        """record_sample() must increment the daily floor counter."""
        store = Store(str(tmp_path))
        evaluator = ShadowEvaluator(store=store)
        cell_key = "S5|CHAT_OPEN|tenant1"
        today = datetime.date.today().isoformat()

        assert store.get_shadow_floor_count(cell_key, today) == 0

        evaluator.record_sample(cell_key)
        assert store.get_shadow_floor_count(cell_key, today) == 1

        evaluator.record_sample(cell_key)
        assert store.get_shadow_floor_count(cell_key, today) == 2
        store.close()
