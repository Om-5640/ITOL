"""
§14.3 Invariant tests for Step 1: icr.py + config.py

Rules:
- NEVER weaken a threshold to make a test pass.
- Every assertion encodes a spec requirement; the comment cites the section.
"""

import copy
import uuid

import pytest

from itol.icr import (
    ICR,
    ICRResponse,
    AnalysisMeta,
    ClassifierResult,
    ContentBlock,
    ContentType,
    ConstraintManifest,
    ManifestItem,
    Message,
    SegmentSignals,
    SegmentType,
    StrategyReport,
    ToolDef,
    UsageStats,
)
from itol.config import (
    ClassConfig,
    ITOLConfig,
    ModelConfig,
    ProviderCacheConfig,
    QualityConfig,
    StorageConfig,
    StrategyConfig,
    TelemetryConfig,
    default_class_configs,
    default_provider_cache_configs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_icr(**kwargs) -> ICR:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        messages=[Message.user("hello")],
        raw={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
    )
    defaults.update(kwargs)
    return ICR.create(**defaults)


# ===========================================================================
# ContentBlock invariants  (§3.1)
# ===========================================================================

class TestContentBlock:

    def test_text_block_requires_text(self):
        """TEXT ContentBlock must have a non-None text field."""
        with pytest.raises(ValueError, match="text"):
            ContentBlock(type=ContentType.TEXT, text=None)

    def test_tool_use_requires_id_and_name(self):
        """TOOL_USE block must carry tool_use_id and tool_name (§3.1)."""
        with pytest.raises(ValueError):
            ContentBlock(type=ContentType.TOOL_USE, tool_use_id=None, tool_name=None)

    def test_tool_result_requires_for_id(self):
        """TOOL_RESULT block must carry tool_result_for_id (§3.1)."""
        with pytest.raises(ValueError):
            ContentBlock(type=ContentType.TOOL_RESULT, tool_result_for_id=None)

    def test_text_convenience_constructor(self):
        b = ContentBlock.text("hello world")
        assert b.type is ContentType.TEXT
        assert b.text == "hello world"

    def test_tool_use_convenience_constructor(self):
        b = ContentBlock.tool_use("id1", "my_tool", {"arg": 1})
        assert b.type is ContentType.TOOL_USE
        assert b.tool_name == "my_tool"
        assert b.tool_input == {"arg": 1}

    def test_tool_result_convenience_constructor(self):
        b = ContentBlock.tool_result("id1", "output text")
        assert b.type is ContentType.TOOL_RESULT
        assert b.tool_result_for_id == "id1"
        assert b.text == "output text"
        assert b.is_error is False

    def test_tool_result_error_flag(self):
        b = ContentBlock.tool_result("id2", "boom", is_error=True)
        assert b.is_error is True

    def test_segment_type_defaults_to_unknown(self):
        b = ContentBlock.text("x")
        assert b.segment_type is SegmentType.UNKNOWN


# ===========================================================================
# Message invariants  (§3.1)
# ===========================================================================

class TestMessage:

    def test_user_factory(self):
        m = Message.user("what is 2+2?")
        assert m.role == "user"
        assert len(m.content) == 1
        assert m.content[0].text == "what is 2+2?"

    def test_assistant_factory(self):
        m = Message.assistant("4")
        assert m.role == "assistant"

    def test_system_factory(self):
        m = Message.system("You are a helpful assistant.")
        assert m.role == "system"

    def test_text_content_concatenates_text_blocks(self):
        m = Message(
            role="user",
            content=[ContentBlock.text("Hello"), ContentBlock.text("World")],
        )
        result = m.text_content()
        assert "Hello" in result
        assert "World" in result

    def test_text_content_skips_non_text_blocks(self):
        m = Message(
            role="assistant",
            content=[
                ContentBlock.tool_use("t1", "search", {}),
                ContentBlock.text("result"),
            ],
        )
        assert m.text_content() == "result"


# ===========================================================================
# ToolDef invariants
# ===========================================================================

class TestToolDef:

    def test_requires_nonempty_name(self):
        with pytest.raises(ValueError, match="name"):
            ToolDef(name="", description="desc", parameters={})

    def test_valid_tool_def(self):
        t = ToolDef(name="search", description="Search the web", parameters={"q": "string"})
        assert t.name == "search"
        assert t.call_count_last_20 == 0


# ===========================================================================
# ICR invariants  (§3.1 — hard constraints 2, 3)
# ===========================================================================

class TestICR:

    def test_create_factory_generates_uuid(self):
        icr = _minimal_icr()
        # Must be a valid UUID4
        parsed = uuid.UUID(icr.request_id, version=4)
        assert str(parsed) == icr.request_id

    def test_requires_nonempty_provider(self):
        with pytest.raises(ValueError, match="provider"):
            ICR.create(provider="", model="gpt-4o", messages=[Message.user("hi")], raw={})

    def test_requires_nonempty_model(self):
        with pytest.raises(ValueError, match="model"):
            ICR.create(provider="openai", model="", messages=[Message.user("hi")], raw={})

    def test_raw_must_not_be_none(self):
        """raw is required for rollback — hard constraint 3 (§5.2)."""
        with pytest.raises((ValueError, TypeError)):
            ICR(
                request_id=str(uuid.uuid4()),
                tenant_id="t",
                provider="openai",
                model="gpt-4o",
                system=[],
                messages=[Message.user("hi")],
                tools=[],
                params={},
                raw=None,   # type: ignore[arg-type]
            )

    def test_raw_is_never_mutated_by_copy(self):
        """Ensure the original raw dict is not aliased — rollback safety."""
        icr = _minimal_icr()
        raw_id = id(icr.raw)
        icr2 = copy.deepcopy(icr)
        # Mutating the copy's raw does not affect original
        icr2.raw["injected"] = True
        assert "injected" not in icr.raw
        assert id(icr.raw) == raw_id

    def test_all_text_includes_system_and_messages(self):
        icr = ICR.create(
            provider="openai",
            model="gpt-4o",
            system=[ContentBlock.text("System instruction here.")],
            messages=[Message.user("User asks this.")],
            raw={},
        )
        text = icr.all_text()
        assert "System instruction here." in text
        assert "User asks this." in text

    def test_final_user_query_returns_last_user_turn(self):
        icr = ICR.create(
            provider="openai",
            model="gpt-4o",
            messages=[
                Message.user("first question"),
                Message.assistant("first answer"),
                Message.user("second question"),
            ],
            raw={},
        )
        assert icr.final_user_query() == "second question"

    def test_final_user_query_empty_when_no_user_message(self):
        icr = ICR.create(
            provider="openai",
            model="gpt-4o",
            messages=[Message.assistant("only assistant")],
            raw={},
        )
        assert icr.final_user_query() == ""

    def test_meta_is_none_before_analysis(self):
        """Analysis metadata is populated by the pipeline, not at construction."""
        icr = _minimal_icr()
        assert icr.meta is None

    def test_default_tenant_id(self):
        icr = _minimal_icr()
        assert icr.tenant_id == "default"

    def test_explicit_tenant_id(self):
        icr = _minimal_icr(tenant_id="acme")
        assert icr.tenant_id == "acme"

    def test_tools_defaults_to_empty_list(self):
        icr = _minimal_icr()
        assert icr.tools == []

    def test_params_defaults_to_empty_dict(self):
        icr = _minimal_icr()
        assert icr.params == {}


# ===========================================================================
# ConstraintManifest invariants  (§5.1 — G1 gap)
# ===========================================================================

class TestConstraintManifest:

    def test_coverage_empty_manifest_is_one(self):
        """Empty manifest ⟹ nothing to lose ⟹ coverage = 1.0 (§5.1)."""
        m = ConstraintManifest()
        assert m.coverage("anything") == 1.0

    def test_coverage_all_present_is_one(self):
        m = ConstraintManifest(items=[
            ManifestItem(ManifestItem.ItemType.NUMBER, "42"),
            ManifestItem(ManifestItem.ItemType.ENTITY, "Anthropic"),
        ])
        assert m.coverage("The answer is 42 from Anthropic.") == 1.0

    def test_coverage_partial_miss(self):
        m = ConstraintManifest(items=[
            ManifestItem(ManifestItem.ItemType.NUMBER, "42"),
            ManifestItem(ManifestItem.ItemType.ENTITY, "Anthropic"),
        ])
        # Only "42" present, "Anthropic" missing
        assert m.coverage("The answer is 42.") == pytest.approx(0.5)

    def test_coverage_all_missing_is_zero(self):
        m = ConstraintManifest(items=[
            ManifestItem(ManifestItem.ItemType.NORMATIVE, "must not delete"),
        ])
        assert m.coverage("do whatever you like") == 0.0

    def test_coverage_floor_enforcement(self):
        """Any optimised prompt below 1.0 MUST trigger rollback (§5.2).
        This test verifies the computation the Guarantor uses; rollback logic
        is tested in the strategy tests."""
        m = ConstraintManifest(items=[
            ManifestItem(ManifestItem.ItemType.QUERY_TERM, "invoice"),
            ManifestItem(ManifestItem.ItemType.NUMBER, "INV-9042"),
        ])
        assert m.coverage("show me the invoice INV-9042") == 1.0
        assert m.coverage("show me the document") < 1.0   # "invoice" and "INV-9042" dropped


# ===========================================================================
# AnalysisMeta / SegmentSignals invariants
# ===========================================================================

class TestAnalysisMeta:

    def test_defaults_are_safe(self):
        """Conservative defaults — redundancy 0 ⟹ no dedupe fires; density 1 ⟹ not compressible."""
        s = SegmentSignals()
        assert s.redundancy_score == 0.0
        assert s.semantic_density == 1.0
        assert s.history_depth == 0
        assert s.stale_mass == 0
        assert s.prefix_cacheable_span == 0

    def test_meta_initialises_empty_strategy_reports(self):
        meta = AnalysisMeta()
        assert meta.strategy_reports == []

    def test_meta_classifier_none_until_set(self):
        meta = AnalysisMeta()
        assert meta.classifier is None

    def test_meta_qps_none_until_guarantor_runs(self):
        meta = AnalysisMeta()
        assert meta.qps is None


class TestClassifierResult:

    def test_ambiguous_flag_set_when_confidence_below_threshold(self):
        """Classifier confidence < 0.6 ⟹ AMBIGUOUS routing (§3.4)."""
        r = ClassifierResult(primary="EXTRACTION", confidence=0.55)
        assert r.ambiguous is True

    def test_not_ambiguous_at_threshold(self):
        r = ClassifierResult(primary="SUMMARIZATION", confidence=0.60)
        assert r.ambiguous is False

    def test_not_ambiguous_above_threshold(self):
        r = ClassifierResult(primary="AGENT_TOOL_LOOP", confidence=0.90)
        assert r.ambiguous is False


# ===========================================================================
# ICRResponse / UsageStats
# ===========================================================================

class TestICRResponse:

    def test_usage_defaults_zero(self):
        u = UsageStats()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_read_tokens == 0
        assert u.cache_write_tokens == 0

    def test_response_construction(self):
        resp = ICRResponse(
            request_id="req-1",
            provider="openai",
            model="gpt-4o",
            content=[ContentBlock.text("hello")],
            usage=UsageStats(input_tokens=10, output_tokens=5),
        )
        assert resp.request_id == "req-1"
        assert resp.usage.input_tokens == 10


# ===========================================================================
# ITOLConfig invariants  (§10.1, hard constraint 3)
# ===========================================================================

class TestITOLConfig:

    def test_default_config_is_valid(self):
        cfg = ITOLConfig()
        assert cfg.mode == "optimize"

    def test_valid_modes(self):
        for mode in ("optimize", "cache_only", "observe_only", "bypass"):
            cfg = ITOLConfig(mode=mode)
            assert cfg.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            ITOLConfig(mode="turbo")   # type: ignore[arg-type]

    def test_qps_floor_must_be_positive(self):
        with pytest.raises(ValueError, match="qps_floor"):
            ITOLConfig(quality=QualityConfig(qps_floor=0.0))

    def test_qps_floor_must_not_exceed_one(self):
        with pytest.raises(ValueError, match="qps_floor"):
            ITOLConfig(quality=QualityConfig(qps_floor=1.01))

    def test_qps_floor_with_s7_must_be_gte_floor(self):
        """S7 requires a stricter floor (§5.2): 0.99 vs 0.98 default."""
        with pytest.raises(ValueError):
            ITOLConfig(quality=QualityConfig(qps_floor=0.98, qps_floor_with_s7=0.97))

    def test_default_qps_floor(self):
        """Spec (§5.2) mandates floor = 0.98 base, 0.99 with S7."""
        cfg = ITOLConfig()
        assert cfg.quality.qps_floor == pytest.approx(0.98)
        assert cfg.quality.qps_floor_with_s7 == pytest.approx(0.99)

    def test_default_shadow_eval_rate(self):
        """Shadow eval rate = 1.5% (§5.3)."""
        cfg = ITOLConfig()
        assert cfg.quality.shadow_eval_rate == pytest.approx(0.015)

    def test_s7_shadow_eval_rate_is_five_x(self):
        """S7 shadow rate must be 5× the base rate (§5.3)."""
        cfg = ITOLConfig()
        assert cfg.quality.shadow_eval_rate_s7 == pytest.approx(
            cfg.quality.shadow_eval_rate * 5, rel=1e-6
        )

    def test_parity_floor(self):
        """Rolling parity floor = 0.95 (§5.3)."""
        cfg = ITOLConfig()
        assert cfg.quality.parity_floor == pytest.approx(0.95)

    def test_parity_tail_limit(self):
        """P(parity < 0.85) ≤ 2% (§5.3)."""
        cfg = ITOLConfig()
        assert cfg.quality.parity_tail_limit == pytest.approx(0.02)
        assert cfg.quality.parity_tail_threshold == pytest.approx(0.85)

    def test_shadow_eval_rate_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ITOLConfig(quality=QualityConfig(shadow_eval_rate=1.5))

    def test_parity_floor_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ITOLConfig(quality=QualityConfig(parity_floor=0.0))


# ===========================================================================
# Class-config invariants  (§7.2 compatibility matrix)
# ===========================================================================

class TestClassConfigs:

    def test_all_eight_classes_present(self):
        """The spec defines exactly 8 request classes (§3.4)."""
        cfg = default_class_configs()
        required = {
            "EXTRACTION", "GENERATION_CREATIVE", "GENERATION_FACTUAL",
            "REASONING", "SUMMARIZATION", "CLASSIFICATION_SHORT",
            "AGENT_TOOL_LOOP", "CHAT_OPEN",
        }
        assert required.issubset(cfg.keys())

    def test_extraction_s7_disabled(self):
        """EXTRACTION forbids lossy compression (§7.2)."""
        cfg = default_class_configs()
        assert cfg["EXTRACTION"].s7_enabled is False

    def test_reasoning_s7_disabled(self):
        """REASONING forbids lossy compression (§7.2)."""
        cfg = default_class_configs()
        assert cfg["REASONING"].s7_enabled is False

    def test_extraction_s3_mass_floor_at_least_097(self):
        """EXTRACTION: context windowing only at mass ≥ 0.97 (§7.2, §4.S3)."""
        cfg = default_class_configs()
        assert cfg["EXTRACTION"].s3_mass_floor >= 0.97

    def test_reasoning_s3_mass_floor_at_least_097(self):
        cfg = default_class_configs()
        assert cfg["REASONING"].s3_mass_floor >= 0.97

    def test_extraction_l1_threshold_at_least_097(self):
        """EXTRACTION cache τ = 0.97 (§6.1)."""
        cfg = default_class_configs()
        assert cfg["EXTRACTION"].l1_similarity_threshold >= 0.97

    def test_reasoning_l1_threshold_at_least_097(self):
        cfg = default_class_configs()
        assert cfg["REASONING"].l1_similarity_threshold >= 0.97

    def test_generation_creative_l1_serve_disabled(self):
        """GENERATION_CREATIVE: semantic cache serving off (§7.2, §6.1 — users expect novelty)."""
        cfg = default_class_configs()
        assert cfg["GENERATION_CREATIVE"].l1_serve is False

    def test_agent_tool_loop_l1_serve_disabled(self):
        """AGENT_TOOL_LOOP: L0 only (§6.1)."""
        cfg = default_class_configs()
        assert cfg["AGENT_TOOL_LOOP"].l1_serve is False

    def test_agent_tool_loop_s6_tool_hygiene_enabled(self):
        """AGENT_TOOL_LOOP must enable S6(d+e) tool-result expiry (§4.S6)."""
        cfg = default_class_configs()
        assert cfg["AGENT_TOOL_LOOP"].s6_tool_hygiene is True

    def test_classification_short_ttl_is_7_days(self):
        """CLASSIFICATION_SHORT TTL = 7d (§6.2)."""
        cfg = default_class_configs()
        assert cfg["CLASSIFICATION_SHORT"].cache_ttl_seconds == 7 * 24 * 3600

    def test_class_config_mass_floor_in_valid_range(self):
        """s3_mass_floor must be in [0.5, 1.0] for every class (validation §config.py)."""
        for name, cc in default_class_configs().items():
            assert 0.5 <= cc.s3_mass_floor <= 1.0, (
                f"{name}: s3_mass_floor={cc.s3_mass_floor} out of range"
            )

    def test_class_config_mutating_one_does_not_affect_another(self):
        """default_class_configs() returns independent copies, not aliases."""
        a = default_class_configs()
        b = default_class_configs()
        a["EXTRACTION"].s3_mass_floor = 0.50
        assert b["EXTRACTION"].s3_mass_floor >= 0.97


# ===========================================================================
# ProviderCacheConfig invariants  (§4, Prefix-Stable rule — G3 gap)
# ===========================================================================

class TestProviderCacheConfig:

    def test_anthropic_discount_is_ninety_percent(self):
        """Anthropic cache reads at ~10% of list price = 90% discount (§4, G3)."""
        cfg = default_provider_cache_configs()
        assert cfg["anthropic"].cache_read_discount == pytest.approx(0.90)

    def test_openai_discount_is_fifty_percent(self):
        """OpenAI cache reads at ~50% discount (§4, G3)."""
        cfg = default_provider_cache_configs()
        assert cfg["openai"].cache_read_discount == pytest.approx(0.50)

    def test_anthropic_uses_explicit_breakpoints(self):
        cfg = default_provider_cache_configs()
        assert cfg["anthropic"].native_prompt_cache == "explicit_breakpoints"

    def test_openai_uses_auto_prefix(self):
        cfg = default_provider_cache_configs()
        assert cfg["openai"].native_prompt_cache == "auto_prefix"

    def test_discount_validation_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            ITOLConfig(
                provider_cache={"openai": ProviderCacheConfig(cache_read_discount=1.5)}
            )

    def test_provider_cache_copy_is_independent(self):
        a = default_provider_cache_configs()
        b = default_provider_cache_configs()
        a["anthropic"].cache_read_discount = 0.0
        assert b["anthropic"].cache_read_discount == pytest.approx(0.90)


# ===========================================================================
# StrategyConfig invariants  (§4 strategy parameters)
# ===========================================================================

class TestStrategyConfig:

    def test_break_even_safety_multiplier_is_five(self):
        """S5 break-even safety multiplier must be 5× (§4.S5, hard constraint 5)."""
        sc = StrategyConfig()
        assert sc.s5_break_even_safety == pytest.approx(5.0)

    def test_s1_cosine_cluster_threshold(self):
        """S1 deduplication clusters at cosine ≥ 0.92 (§4.S1)."""
        sc = StrategyConfig()
        assert sc.s1_cosine_cluster >= 0.92

    def test_s3_relevance_weights_sum_to_one(self):
        """S3 scorer weights must sum to 1.0 (§4.S3)."""
        sc = StrategyConfig()
        total = (
            sc.s3_relevance_weight_semantic
            + sc.s3_relevance_weight_bm25
            + sc.s3_relevance_weight_position
        )
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_s7_max_ratio_cap(self):
        """S7 lossy compression ratio capped at 2× (§4.S7)."""
        sc = StrategyConfig()
        assert sc.s7_max_ratio == pytest.approx(2.0)

    def test_s4_retrieval_confidence_floor(self):
        """S4: top-1 chunk score ≥ 0.35 else fall back (§4.S4)."""
        sc = StrategyConfig()
        assert sc.s4_retrieval_confidence_floor >= 0.35


# ===========================================================================
# for_tenant override invariant  (hard constraint 3 — quality floors only go up)
# ===========================================================================

class TestTenantOverride:

    def test_tenant_with_no_override_returns_same_config(self):
        cfg = ITOLConfig()
        assert cfg.for_tenant("unknown") is cfg

    def test_tenant_can_raise_shadow_eval_rate(self):
        cfg = ITOLConfig(
            tenant_overrides={"acme": {"shadow_eval_rate": 0.05}}
        )
        acme = cfg.for_tenant("acme")
        assert acme.quality.shadow_eval_rate == pytest.approx(0.05)

    def test_tenant_cannot_lower_shadow_eval_rate(self):
        """Tenant overrides may not weaken quality controls."""
        cfg = ITOLConfig(
            tenant_overrides={"cheapo": {"shadow_eval_rate": 0.001}}
        )
        cheapo = cfg.for_tenant("cheapo")
        # Must clamp to base rate, not the requested lower value
        assert cheapo.quality.shadow_eval_rate >= cfg.quality.shadow_eval_rate

    def test_tenant_mode_can_be_more_restrictive(self):
        cfg = ITOLConfig(
            mode="optimize",
            tenant_overrides={"readonly": {"mode": "observe_only"}}
        )
        assert cfg.for_tenant("readonly").mode == "observe_only"

    def test_tenant_mode_cannot_be_less_restrictive(self):
        cfg = ITOLConfig(
            mode="observe_only",
            tenant_overrides={"sneaky": {"mode": "optimize"}}
        )
        # "optimize" is more powerful than "observe_only" — must be rejected
        assert cfg.for_tenant("sneaky").mode == "observe_only"


# ===========================================================================
# Stage latency budget  (§7.4 — hard constraint 4)
# ===========================================================================

class TestLatencyBudget:

    def test_total_overhead_budget_is_200ms(self):
        """Full path p95 target ≤ 200ms (§7.4)."""
        cfg = ITOLConfig()
        assert cfg.stage_deadline_ms["total_overhead"] == pytest.approx(200.0)

    def test_cached_path_budget_is_50ms(self):
        """
        Cached path p95 ≤ 50ms: L0 + classify_manifest (combined, §7.4 table) + L1.
        The spec groups classify + manifest as one 9ms stage; summing them separately
        would be incorrect and would inflate the budget.
        """
        cfg = ITOLConfig()
        cached_path = (
            cfg.stage_deadline_ms["l0"]
            + cfg.stage_deadline_ms["classify_manifest"]
            + cfg.stage_deadline_ms["l1"]
        )
        assert cached_path <= 50.0, (
            f"Cached path sum {cached_path}ms exceeds 50ms budget (§7.4)"
        )

    def test_all_stage_deadlines_positive(self):
        cfg = ITOLConfig()
        for stage, ms in cfg.stage_deadline_ms.items():
            assert ms > 0, f"Stage {stage!r} deadline must be positive"
