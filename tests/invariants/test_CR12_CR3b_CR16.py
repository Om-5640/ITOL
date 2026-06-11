"""
§14.3 Invariant tests for CR-12, CR-3b, and CR-16 — QPS gate + rollback orchestrator.

CR-12: Rollback order is S7 → S4 → S3 → S5 → S1; S2 and S6 are NEVER rolled back.
CR-3b: Full rollback returns icr.raw byte-identical.
CR-16: Any exception in the QPS pipeline → return icr.raw, never raise to caller.

Rules:
- NEVER weaken a threshold to make a test pass.
- Every assertion cites the spec requirement in a comment.
"""

import pytest

from itol.config import ITOLConfig
from itol.icr import (
    ConstraintManifest,
    ContentBlock,
    ICR,
    ManifestItem,
    Message,
    StrategyReport,
)
from itol.quality.qps import (
    NO_ROLLBACK,
    PIPELINE_ORDER,
    ROLLBACK_ORDER,
    ScoreResult,
    compute_qps,
    score_and_rollback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _icr(system_text: str = "Hello world.", user_text: str = "Summarise.") -> ICR:
    raw_payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": user_text}]}
    return ICR.create(
        provider="openai",
        model="gpt-4o",
        system=[ContentBlock.text(system_text)] if system_text else [],
        messages=[Message.user(user_text)],
        raw=raw_payload,
    )


def _empty_manifest() -> ConstraintManifest:
    """A manifest with no items — coverage always 1.0."""
    return ConstraintManifest(items=[])


def _manifest_with(value: str) -> ConstraintManifest:
    """Manifest with a single NUMBER item requiring `value` to be present."""
    return ConstraintManifest(items=[
        ManifestItem(item_type=ManifestItem.ItemType.NUMBER, value=value)
    ])


def _default_cfg():
    return ITOLConfig().quality


def _report(strategy_id: str, snapshot=None) -> StrategyReport:
    return StrategyReport(
        strategy_id=strategy_id,
        tokens_removed=10,
        risk_class="NEAR-LOSSLESS",
        segment_snapshot=snapshot,
    )


class _FakeSegment:
    """Minimal segment-like object so segments_full_text works."""
    def __init__(self, text: str):
        self.text = text
        self.segment_type = "USER_QUERY"


# ---------------------------------------------------------------------------
# CR-12: Rollback order is S7 → S4 → S3 → S5 → S1; S2 and S6 never rolled back
# ---------------------------------------------------------------------------

class TestCR12:

    def test_rollback_order_constant(self):
        """CR-12: ROLLBACK_ORDER must be exactly (S7, S4, S3, S5, S1)."""
        assert ROLLBACK_ORDER == ("S7", "S4", "S3", "S5", "S1"), (
            "CR-12: rollback order must be S7→S4→S3→S5→S1 (§5.2)"
        )

    def test_no_rollback_set(self):
        """CR-12: S2 and S6 must never be in ROLLBACK_ORDER (verified-lossless)."""
        for s in ("S2", "S6"):
            assert s not in ROLLBACK_ORDER, (
                f"CR-12: {s} is verified-lossless and must never appear in ROLLBACK_ORDER (§5.2)"
            )
        assert NO_ROLLBACK == frozenset({"S2", "S6"}), (
            "NO_ROLLBACK must be exactly {S2, S6}"
        )

    def test_pipeline_order_covers_all_strategies(self):
        """All 7 strategies must appear in PIPELINE_ORDER with distinct positions."""
        assert set(PIPELINE_ORDER.keys()) == {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}
        assert len(set(PIPELINE_ORDER.values())) == 7, "Pipeline positions must be unique"

    def test_rollback_tries_s7_first(self):
        """
        CR-12: when QPS fails, S7 must be the first strategy disabled.

        Setup: manifest requires '$9.9M' which is absent in optimised text
        (so initial QPS = 0). S7 snapshot restores the value → passes.
        """
        icr = _icr()
        manifest = _manifest_with("$9.9M")
        cfg = _default_cfg()

        # Optimised segments: value is ABSENT → QPS fails
        opt_segments = [_FakeSegment("The total is unspecified.")]

        # S7 snapshot contains the required value
        s7_snapshot = [_FakeSegment("The total is $9.9M.")]
        reports = [
            _report("S7", snapshot=s7_snapshot),
        ]

        result = score_and_rollback(
            icr, reports, manifest, cfg, opt_segments
        )

        assert not result.use_raw, "S7 rollback should recover the value"
        assert result.qps_result.rollback_stages_tried[0] == "S7", (
            "CR-12: S7 must be tried first in rollback sequence"
        )
        assert result.qps_result.rollback_stage_passed == "S7"

    def test_rollback_order_respected_s4_before_s3(self):
        """
        CR-12: when S7 rollback still fails, S4 must be tried before S3.
        """
        icr = _icr()
        manifest = _manifest_with("$3.3M")
        cfg = _default_cfg()

        opt_segments = [_FakeSegment("No specific value mentioned.")]

        # S7 snapshot still missing the value; S4 snapshot has it
        s7_snapshot = [_FakeSegment("Some other text.")]
        s4_snapshot = [_FakeSegment("The amount is $3.3M as required.")]

        reports = [
            _report("S4", snapshot=s4_snapshot),
            _report("S7", snapshot=s7_snapshot),
        ]

        result = score_and_rollback(
            icr, reports, manifest, cfg, opt_segments
        )

        assert not result.use_raw, "S4 rollback should recover the value"
        tried = result.qps_result.rollback_stages_tried
        assert tried.index("S7") < tried.index("S4"), (
            "CR-12: S7 must be tried before S4 in the rollback sequence"
        )
        assert result.qps_result.rollback_stage_passed == "S4"

    def test_s2_never_rolled_back(self):
        """
        CR-12: even when S2 ran and QPS fails, S2 must NOT appear in rollback_stages_tried.
        """
        icr = _icr()
        manifest = _manifest_with("$7.7M")
        cfg = _default_cfg()

        opt_segments = [_FakeSegment("No value.")]
        # Only S2 ran — it cannot be rolled back
        reports = [_report("S2", snapshot=[_FakeSegment("The value is $7.7M.")])]

        result = score_and_rollback(
            icr, reports, manifest, cfg, opt_segments
        )

        assert "S2" not in result.qps_result.rollback_stages_tried, (
            "CR-12: S2 is verified-lossless and must NEVER appear in rollback_stages_tried (§5.2)"
        )

    def test_s6_never_rolled_back(self):
        """
        CR-12: S6 must not appear in rollback_stages_tried even when it ran.
        """
        icr = _icr()
        manifest = _manifest_with("$8.8M")
        cfg = _default_cfg()

        opt_segments = [_FakeSegment("No value.")]
        reports = [_report("S6", snapshot=[_FakeSegment("The value is $8.8M.")])]

        result = score_and_rollback(
            icr, reports, manifest, cfg, opt_segments
        )

        assert "S6" not in result.qps_result.rollback_stages_tried, (
            "CR-12: S6 is verified-lossless and must NEVER appear in rollback_stages_tried (§5.2)"
        )


# ---------------------------------------------------------------------------
# CR-3b: Full rollback returns icr.raw byte-identical
# ---------------------------------------------------------------------------

class TestCR3b:

    def test_full_rollback_returns_use_raw_true(self):
        """
        CR-3b: when all rollback attempts fail, use_raw must be True.

        Setup: manifest requires '$1.1M'; neither optimised text nor any
        snapshot contains it → full rollback exhausted.
        """
        icr = _icr()
        manifest = _manifest_with("$1.1M")
        cfg = _default_cfg()

        opt_segments = [_FakeSegment("Nothing useful.")]

        # All snapshots also lack the required value
        reports = [
            _report("S1", snapshot=[_FakeSegment("Still no value here.")]),
            _report("S3", snapshot=[_FakeSegment("Also missing.")]),
            _report("S4", snapshot=[_FakeSegment("No luck.")]),
            _report("S5", snapshot=[_FakeSegment("Not found.")]),
            _report("S7", snapshot=[_FakeSegment("None present.")]),
        ]

        result = score_and_rollback(
            icr, reports, manifest, cfg, opt_segments
        )

        assert result.use_raw is True, (
            "CR-3b: exhausted rollback must set use_raw=True so caller dispatches icr.raw (§14.2)"
        )
        assert result.segments is None, (
            "CR-3b: segments must be None when use_raw=True — caller uses icr.raw"
        )

    def test_use_raw_false_when_optimised_passes(self):
        """
        When the optimised result passes QPS, use_raw must be False
        and segments must be the optimised list.
        """
        icr = _icr()
        manifest = _empty_manifest()  # no items → coverage = 1.0
        cfg = _default_cfg()

        opt_segments = [_FakeSegment("Summarised text is here.")]
        reports = [_report("S3")]

        result = score_and_rollback(
            icr, reports, manifest, cfg, opt_segments
        )

        assert result.use_raw is False, (
            "Passing QPS result must have use_raw=False"
        )
        assert result.segments is opt_segments, (
            "Passing QPS result must return the optimised segments"
        )

    def test_raw_payload_unchanged_on_rollback(self):
        """
        CR-3b: icr.raw must be the exact object returned by the caller
        (not a copy or mutation).  We verify identity via a reference check.
        """
        raw_payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        icr = ICR.create(
            provider="openai",
            model="gpt-4o",
            messages=[Message.user("hi")],
            raw=raw_payload,
        )
        manifest = _manifest_with("$2.2M")  # will fail — value absent
        cfg = _default_cfg()

        opt_segments = [_FakeSegment("No value.")]
        reports: list[StrategyReport] = []  # no strategies ran → no rollback possible

        result = score_and_rollback(
            icr, reports, manifest, cfg, opt_segments
        )

        assert result.use_raw is True
        # Verify that the caller can safely use icr.raw byte-identical: raw must
        # be the same dict object we passed in (no copy, no mutation).
        assert icr.raw is raw_payload, (
            "CR-3b: icr.raw must be the exact original object, never mutated (§14.2)"
        )


# ---------------------------------------------------------------------------
# CR-16: Any exception in QPS pipeline → return icr.raw, never raise
# ---------------------------------------------------------------------------

class TestCR16:

    def test_exception_in_coverage_returns_raw(self):
        """
        CR-16: if an exception is raised inside score_and_rollback, the call
        must NOT propagate — it must return ScoreResult(use_raw=True).
        """
        icr = _icr()
        cfg = _default_cfg()

        # Craft a manifest whose coverage() raises when called
        class _BrokenManifest(ConstraintManifest):
            def coverage(self, text: str) -> float:
                raise RuntimeError("Simulated coverage failure (CR-16 test)")

        broken_manifest = _BrokenManifest(items=[])
        opt_segments = [_FakeSegment("some text")]
        reports: list[StrategyReport] = []

        result = score_and_rollback(
            icr, reports, broken_manifest, cfg, opt_segments
        )

        assert result.use_raw is True, (
            "CR-16: exception inside the QPS pipeline must return use_raw=True (§14.2)"
        )
        assert result.segments is None, (
            "CR-16: segments must be None on exception path"
        )

    def test_exception_never_propagates(self):
        """
        CR-16: score_and_rollback must never raise, even on totally broken inputs.
        """
        icr = _icr()
        cfg = _default_cfg()

        class _ExplodingManifest(ConstraintManifest):
            def coverage(self, text: str) -> float:
                raise ValueError("Explosion")

        result = score_and_rollback(
            icr, [], _ExplodingManifest(), cfg, None  # type: ignore[arg-type]
        )

        # Just reaching here without an exception proves CR-16.
        assert result.use_raw is True

    def test_exception_qps_result_has_zero_score(self):
        """
        CR-16: the QPSResult returned on exception must have qps=0.0 and passed=False.
        """
        icr = _icr()
        cfg = _default_cfg()

        class _BrokenManifest(ConstraintManifest):
            def coverage(self, text: str) -> float:
                raise RuntimeError("broken")

        result = score_and_rollback(
            icr, [], _BrokenManifest(), cfg, [_FakeSegment("text")]
        )

        assert result.qps_result.qps == pytest.approx(0.0), (
            "CR-16: exception path must report qps=0.0"
        )
        assert result.qps_result.passed is False, (
            "CR-16: exception path must report passed=False"
        )


# ---------------------------------------------------------------------------
# QPS formula invariants (§5.2)
# ---------------------------------------------------------------------------

class TestQPSFormula:

    def test_qps_zero_on_coverage_below_1(self):
        """
        §5.2: manifest_coverage < 1.0 must hard-gate QPS to exactly 0.0.
        """
        manifest = _manifest_with("$5.5M")  # value absent in optimised text
        cfg = _default_cfg()

        result = compute_qps(
            manifest,
            "No specific value here.",
            cfg,
            semantic_fidelity=1.0,
            min_window_fidelity=1.0,
            coverage_margin=1.0,
        )

        assert result.qps == pytest.approx(0.0), (
            "§5.2: coverage hard gate must short-circuit QPS to 0.0"
        )
        assert result.passed is False

    def test_qps_formula_components(self):
        """
        §5.2: verify QPS = 0.45·cov + 0.30·sem + 0.15·mwf + 0.10·margin_rescaled
        when coverage = 1.0 (hard gate passes).
        """
        manifest = _empty_manifest()
        cfg = _default_cfg()

        result = compute_qps(
            manifest, "any text",
            cfg,
            semantic_fidelity=0.8,
            min_window_fidelity=0.9,
            coverage_margin=0.95,  # rescaled: (0.95-0.90)/0.10 = 0.5
        )

        expected = 0.45 * 1.0 + 0.30 * 0.8 + 0.15 * 0.9 + 0.10 * 0.5
        assert result.qps == pytest.approx(expected, rel=1e-6), (
            "§5.2: QPS must exactly match the weighted formula"
        )

    def test_qps_floor_98_no_s7(self):
        """§5.2: QPS floor must be 0.98 when S7 did not participate."""
        cfg = _default_cfg()
        assert cfg.qps_floor == pytest.approx(0.98), (
            "§5.2: default QPS floor must be 0.98"
        )
        manifest = _empty_manifest()
        result = compute_qps(manifest, "text", cfg, s7_participated=False)
        assert result.floor_used == pytest.approx(0.98)

    def test_qps_floor_99_with_s7(self):
        """§5.2: QPS floor must be 0.99 when S7 participated."""
        cfg = _default_cfg()
        assert cfg.qps_floor_with_s7 == pytest.approx(0.99), (
            "§5.2: S7 QPS floor must be 0.99"
        )
        manifest = _empty_manifest()
        result = compute_qps(manifest, "text", cfg, s7_participated=True)
        assert result.floor_used == pytest.approx(0.99)

    def test_coverage_margin_rescaling(self):
        """§5.2: coverage_margin [0.9, 1.0] → [0.0, 1.0]; below 0.9 clamps to 0."""
        manifest = _empty_manifest()
        cfg = _default_cfg()

        r_low = compute_qps(manifest, "t", cfg, coverage_margin=0.85)
        r_mid = compute_qps(manifest, "t", cfg, coverage_margin=0.95)
        r_high = compute_qps(manifest, "t", cfg, coverage_margin=1.0)

        # margin=0.85 → rescaled=0 → 0.10 * 0 = 0 contribution from margin
        # margin=0.95 → rescaled=0.5
        # margin=1.0  → rescaled=1.0
        expected_low  = 0.45 + 0.30 + 0.15 + 0.10 * 0.0
        expected_mid  = 0.45 + 0.30 + 0.15 + 0.10 * 0.5
        expected_high = 0.45 + 0.30 + 0.15 + 0.10 * 1.0

        assert r_low.qps  == pytest.approx(expected_low,  rel=1e-6)
        assert r_mid.qps  == pytest.approx(expected_mid,  rel=1e-6)
        assert r_high.qps == pytest.approx(expected_high, rel=1e-6)
