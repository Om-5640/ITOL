"""
§14.3 Invariant tests for CR-13 — savings are net of shadow eval cost.

CR-13: est_cost_saved_usd = gross_saved - shadow_cost_usd
       The field shadow_cost_usd is stored separately in the requests table.
       est_cost_saved_usd must NEVER equal gross_saved when shadow_cost_usd > 0.

Rules:
- NEVER weaken thresholds or conditions.
"""

import tempfile

import pytest

from itol.cache.store import Store
from itol.telemetry.recorder import Recorder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recorder():
    tmpdir = tempfile.mkdtemp()
    store = Store(tmpdir)
    return Recorder(store, data_dir=tmpdir), store, tmpdir


# ===========================================================================
# CR-13-a: est_cost_saved_usd = gross - shadow_cost, never gross alone
# ===========================================================================

class TestCR13_NetSavings:

    def test_savings_net_of_shadow_cost(self):
        """
        CR-13: when shadow_cost_usd > 0, the stored est_cost_saved_usd must
        equal gross_saved_usd - shadow_cost_usd, not gross alone.
        """
        rec, store, _ = _recorder()
        gross   = 0.010
        shadow  = 0.002
        expected_net = gross - shadow   # 0.008

        rec.record(
            request_id="cr13_test_1",
            tenant_id="t1",
            provider="openai",
            model="gpt-4o",
            gross_cost_saved_usd=gross,
            shadow_cost_usd=shadow,
            provider_usage={"completion_tokens": 20, "prompt_tokens": 50},
        )

        row = store.get_request("cr13_test_1")
        assert row is not None
        assert abs(row["est_cost_saved_usd"] - expected_net) < 1e-9, (
            f"CR-13: est_cost_saved_usd must be gross - shadow = {expected_net}, "
            f"got {row['est_cost_saved_usd']}"
        )

    def test_savings_not_gross_alone(self):
        """
        CR-13: the stored value must NOT equal gross when shadow_cost > 0.
        This directly tests that we subtract, not pass-through.
        """
        rec, store, _ = _recorder()
        gross  = 0.010
        shadow = 0.003

        rec.record(
            request_id="cr13_test_2",
            tenant_id="t1",
            gross_cost_saved_usd=gross,
            shadow_cost_usd=shadow,
        )

        row = store.get_request("cr13_test_2")
        stored = row["est_cost_saved_usd"]
        assert abs(stored - gross) > 1e-9, (
            "CR-13: est_cost_saved_usd must not equal gross_saved when shadow_cost > 0"
        )

    def test_shadow_cost_zero_net_equals_gross(self):
        """
        When shadow_cost_usd = 0, net = gross (no subtraction effect).
        """
        rec, store, _ = _recorder()
        gross = 0.005

        rec.record(
            request_id="cr13_test_3",
            tenant_id="t1",
            gross_cost_saved_usd=gross,
            shadow_cost_usd=0.0,
        )

        row = store.get_request("cr13_test_3")
        assert abs(row["est_cost_saved_usd"] - gross) < 1e-9, (
            "When shadow_cost=0, est_cost_saved_usd must equal gross"
        )

    def test_shadow_cost_stored_separately(self):
        """
        CR-13: shadow_cost_usd must be stored in its own column so it can be
        reported and audited independently.
        """
        rec, store, _ = _recorder()
        shadow = 0.0025

        rec.record(
            request_id="cr13_test_4",
            tenant_id="t1",
            gross_cost_saved_usd=0.01,
            shadow_cost_usd=shadow,
        )

        row = store.get_request("cr13_test_4")
        assert "shadow_cost_usd" in row, (
            "CR-13: shadow_cost_usd must be stored as a separate column in requests"
        )
        assert abs(row["shadow_cost_usd"] - shadow) < 1e-9, (
            f"shadow_cost_usd stored incorrectly: expected {shadow}, got {row['shadow_cost_usd']}"
        )

    def test_net_savings_negative_when_shadow_exceeds_gross(self):
        """
        If shadow eval cost > gross savings, net must be negative (no clamping).
        Clamping would hide over-spending on shadow evals.
        """
        rec, store, _ = _recorder()

        rec.record(
            request_id="cr13_test_5",
            tenant_id="t1",
            gross_cost_saved_usd=0.001,
            shadow_cost_usd=0.005,
        )

        row = store.get_request("cr13_test_5")
        assert row["est_cost_saved_usd"] < 0, (
            "CR-13: when shadow_cost > gross_saved, net must be negative (not clamped to 0)"
        )


# ===========================================================================
# CR-13-b: CR-17 — tokens_out from provider_usage, not estimates
# ===========================================================================

class TestCR13_CR17TokenSource:

    def test_tokens_out_from_provider_usage(self):
        """
        CR-17: the recorder must use provider_usage['completion_tokens'] for
        tokens_out, never a local estimate.
        """
        rec, store, _ = _recorder()
        actual_out = 42

        rec.record(
            request_id="cr17_test_1",
            tenant_id="t1",
            tokens_in_original=100,
            tokens_in_optimized=80,
            provider_usage={"completion_tokens": actual_out, "prompt_tokens": 80},
        )

        row = store.get_request("cr17_test_1")
        assert row["tokens_out"] == actual_out, (
            f"CR-17: tokens_out must come from provider_usage (expected {actual_out}, "
            f"got {row['tokens_out']})"
        )

    def test_tokens_out_none_when_no_provider_usage(self):
        """When provider_usage is not provided, tokens_out must be NULL."""
        rec, store, _ = _recorder()

        rec.record(
            request_id="cr17_test_2",
            tenant_id="t1",
            tokens_in_original=100,
            provider_usage=None,
        )

        row = store.get_request("cr17_test_2")
        assert row["tokens_out"] is None, (
            "CR-17: tokens_out must be NULL when no provider_usage is given"
        )

    def test_openai_usage_key_completion_tokens(self):
        """OpenAI uses 'completion_tokens' as the output token key."""
        rec, store, _ = _recorder()

        rec.record(
            request_id="cr17_test_3",
            tenant_id="t1",
            provider_usage={"prompt_tokens": 50, "completion_tokens": 15},
        )

        row = store.get_request("cr17_test_3")
        assert row["tokens_out"] == 15

    def test_alt_usage_key_output_tokens(self):
        """Some providers use 'output_tokens'; recorder must handle both."""
        rec, store, _ = _recorder()

        rec.record(
            request_id="cr17_test_4",
            tenant_id="t1",
            provider_usage={"input_tokens": 50, "output_tokens": 22},
        )

        row = store.get_request("cr17_test_4")
        assert row["tokens_out"] == 22


# ===========================================================================
# CR-13-c: Telemetry jsonl written
# ===========================================================================

class TestCR13_JsonlOutput:

    def test_jsonl_written(self):
        """Recorder must write a JSON-lines record to telemetry.jsonl."""
        import json
        from pathlib import Path

        rec, store, tmpdir = _recorder()
        rec.record(
            request_id="jsonl_1",
            tenant_id="t1",
            provider="openai",
            model="gpt-4o",
            gross_cost_saved_usd=0.003,
            shadow_cost_usd=0.001,
        )

        jsonl_path = Path(tmpdir) / "telemetry.jsonl"
        assert jsonl_path.exists(), "telemetry.jsonl must be created by recorder"

        with jsonl_path.open() as f:
            line = f.readline()
        record = json.loads(line)

        assert record["request_id"] == "jsonl_1"
        assert record["tenant_id"]  == "t1"

    def test_jsonl_stores_net_savings(self):
        """The jsonl record must also store net (not gross) savings."""
        import json
        from pathlib import Path

        rec, store, tmpdir = _recorder()
        gross = 0.010
        shadow = 0.002
        rec.record(
            request_id="jsonl_2",
            tenant_id="t1",
            gross_cost_saved_usd=gross,
            shadow_cost_usd=shadow,
        )

        jsonl_path = Path(tmpdir) / "telemetry.jsonl"
        with jsonl_path.open() as f:
            record = json.loads(f.readline())

        net = gross - shadow
        assert abs(record["est_cost_saved_usd"] - net) < 1e-9, (
            "CR-13: jsonl record must store net savings, not gross"
        )
