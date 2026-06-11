"""
ITOL system configuration  (§10.1, §7.2, §5, §6).

All tuneable parameters live here.  Runtime components import `ITOLConfig`
and receive it at construction; nothing reads environment variables directly
(callers set those up before building a config if desired).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any


# ---------------------------------------------------------------------------
# Per-class strategy overrides  (§7.2 compatibility matrix)
# ---------------------------------------------------------------------------

@dataclass
class ClassConfig:
    """
    Per-request-class strategy knobs.  The matrix defaults below encode the
    §7.2 table; operators may tighten but not loosen quality floors.
    """
    s1_enabled: bool = True          # semantic deduplication
    s2_enabled: bool = True          # instruction compression (template mining)
    s3_enabled: bool = True          # dynamic context windowing
    s3_mass_floor: float = 0.90      # minimum relevance mass retained by S3
    s4_enabled: bool = True          # retrieval-augmented context replacement
    s5_enabled: bool = True          # conversation history distillation
    s5_k_turns: int = 6              # last-K verbatim turns preserved by S5
    s6_enabled: bool = True          # structural minification & trajectory hygiene
    s6_tool_hygiene: bool = False    # S6(d+e) tool-result expiry / schema pruning
    s7_enabled: bool = False         # lossy token-level compression (opt-in only)
    l1_serve: bool = True            # semantic cache serving
    l1_similarity_threshold: float = 0.95   # per-class default; bandit-tuned at runtime
    cache_ttl_seconds: int = 72 * 3600      # L0/L1 TTL


# Spec §7.2 defaults — encoded as a factory so mutations don't alias
_CLASS_DEFAULTS: dict[str, ClassConfig] = {
    "EXTRACTION": ClassConfig(
        s3_mass_floor=0.97,
        s4_enabled=True,      # ⚠ no-full-doc-intent check in strategy
        s7_enabled=False,
        l1_similarity_threshold=0.97,
        cache_ttl_seconds=72 * 3600,
    ),
    "REASONING": ClassConfig(
        s3_mass_floor=0.97,
        s4_enabled=False,
        s7_enabled=False,
        l1_similarity_threshold=0.97,
        cache_ttl_seconds=72 * 3600,
    ),
    "SUMMARIZATION": ClassConfig(
        s3_enabled=True,
        s4_enabled=True,
        s7_enabled=False,        # opt-in only
        l1_similarity_threshold=0.95,
        cache_ttl_seconds=72 * 3600,
    ),
    "GENERATION_FACTUAL": ClassConfig(
        s4_enabled=True,
        s7_enabled=False,
        l1_similarity_threshold=0.96,
        cache_ttl_seconds=72 * 3600,
    ),
    "GENERATION_CREATIVE": ClassConfig(
        s3_mass_floor=0.93,      # conservative windowing allowed
        s4_enabled=False,
        s5_k_turns=6,
        s7_enabled=False,
        l1_serve=False,          # semantic serving disabled; expect novelty
        cache_ttl_seconds=24 * 3600,
    ),
    "CLASSIFICATION_SHORT": ClassConfig(
        s4_enabled=False,
        s5_enabled=False,        # typically stateless
        s7_enabled=False,
        l1_similarity_threshold=0.93,
        cache_ttl_seconds=7 * 24 * 3600,
    ),
    "AGENT_TOOL_LOOP": ClassConfig(
        s6_tool_hygiene=True,    # S6(d+e) enabled
        s7_enabled=False,
        l1_serve=False,          # L0 only per §6.1
        cache_ttl_seconds=3600,
    ),
    "CHAT_OPEN": ClassConfig(
        s3_mass_floor=0.93,      # conservative arm
        s4_enabled=False,
        s7_enabled=False,
        l1_serve=False,
        cache_ttl_seconds=24 * 3600,
    ),
    "AMBIGUOUS": ClassConfig(
        s3_enabled=False,
        s4_enabled=False,
        s7_enabled=False,
        l1_serve=False,
    ),
}


def default_class_configs() -> dict[str, ClassConfig]:
    """Return a deep copy of the default per-class configs."""
    return {k: ClassConfig(**vars(v)) for k, v in _CLASS_DEFAULTS.items()}


# ---------------------------------------------------------------------------
# Provider-cache settings  (§4, Prefix-Stable rule)
# ---------------------------------------------------------------------------

@dataclass
class ProviderCacheConfig:
    """
    Controls the Prefix-Stable optimisation and cache_control injection.
    One instance per provider (keyed by provider name in ITOLConfig).
    """
    native_prompt_cache: Literal["none", "auto_prefix", "explicit_breakpoints"] = "none"
    cache_read_discount: float = 0.0       # fraction of input price saved on a cache hit
    min_cacheable_tokens: int = 1024
    inject_breakpoints: bool = True        # whether ITOL should inject cache_control markers


_PROVIDER_CACHE_DEFAULTS: dict[str, ProviderCacheConfig] = {
    "anthropic": ProviderCacheConfig(
        native_prompt_cache="explicit_breakpoints",
        cache_read_discount=0.90,
        min_cacheable_tokens=1024,
        inject_breakpoints=True,
    ),
    "openai": ProviderCacheConfig(
        native_prompt_cache="auto_prefix",
        cache_read_discount=0.50,
        min_cacheable_tokens=1024,
        inject_breakpoints=False,   # OpenAI handles prefix caching automatically
    ),
    "mistral": ProviderCacheConfig(),
    "cohere": ProviderCacheConfig(),
}


def default_provider_cache_configs() -> dict[str, ProviderCacheConfig]:
    return {k: ProviderCacheConfig(**vars(v)) for k, v in _PROVIDER_CACHE_DEFAULTS.items()}


# ---------------------------------------------------------------------------
# Quality Preservation settings  (§5.2, §5.3, §5.4)
# ---------------------------------------------------------------------------

@dataclass
class QualityConfig:
    qps_floor: float = 0.98            # minimum QPS to dispatch optimised prompt
    qps_floor_with_s7: float = 0.99    # stricter floor when S7 participated
    shadow_eval_rate: float = 0.015    # fraction of requests sent for shadow evaluation
    shadow_eval_rate_s7: float = 0.075 # 5× rate when S7 participated
    shadow_eval_daily_cap: int = 200   # max shadow calls per day per tenant
    parity_floor: float = 0.95         # rolling-200 parity mean floor (§5.3)
    parity_tail_limit: float = 0.02    # P(parity < 0.85) must stay ≤ 2%
    parity_tail_threshold: float = 0.85
    circuit_breaker_window: int = 200  # rolling window size for parity checks
    circuit_breaker_second_strike_hours: int = 24   # reset window for 2nd violation
    circuit_breaker_probation_days: int = 7
    bandit_lambda: float = 0.25        # reward = parity_norm − λ·(1 − token_reduction)


# ---------------------------------------------------------------------------
# Storage settings  (§6, §9)
# ---------------------------------------------------------------------------

@dataclass
class StorageConfig:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("ITOL_DATA_DIR", "~/.itol")).expanduser())
    sqlite_db: str = "itol.db"        # relative to data_dir
    vector_backend: Literal["sqlite_vec", "qdrant"] = "sqlite_vec"
    sqlite_vec_max_entries: int = 500_000   # switch to Qdrant beyond this
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "itol_cache"
    redis_url: str | None = None       # None = no Redis; L0 falls back to SQLite
    max_cache_size_bytes: int = 1_073_741_824   # 1 GB

    def db_path(self) -> Path:
        return self.data_dir / self.sqlite_db


# ---------------------------------------------------------------------------
# Strategy-level tuning knobs  (§4)
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    # S1 — semantic deduplication
    s1_minhash_perms: int = 128
    s1_jaccard_prefilter: float = 0.55
    s1_cosine_cluster: float = 0.92
    s1_redundancy_score_gate: float = 0.15

    # S2 — instruction compression
    s2_min_reuse_count: int = 10       # minimum template hits before offline job fires
    s2_offline_parity_threshold: float = 0.97  # A/B parity required for acceptance

    # S3 — dynamic context windowing
    s3_chunk_size_tokens: int = 256
    s3_chunk_overlap_tokens: int = 32
    s3_relevance_weight_semantic: float = 0.70
    s3_relevance_weight_bm25: float = 0.20
    s3_relevance_weight_position: float = 0.10
    s3_activation_multiplier: float = 1.5   # context must be > 1.5× class budget

    # S4 — RACR
    s4_min_doc_tokens: int = 2000
    s4_min_appearances: int = 2
    s4_retrieval_confidence_floor: float = 0.35

    # S5 — history distillation
    s5_history_depth_gate: int = 8
    s5_history_tokens_gate: int = 4000
    s5_draft_superseded_jaccard: float = 0.60
    s5_resurrection_delta: float = 0.45   # margin above ledger-match to restore a turn
    s5_break_even_safety: float = 5.0     # multiplier on C_opt for LLM-assisted distil

    # S6 — structural minification
    s6_tool_superseded_jaccard: float = 0.50
    s6_tool_expiry_turns: int = 3
    s6_schema_prune_turns: int = 20

    # S7 — lossy compression
    s7_density_gate: float = 0.45
    s7_max_ratio: float = 2.0
    s7_model_path: str | None = None   # path to int8 ONNX checkpoint; None → strategy disabled


# ---------------------------------------------------------------------------
# Embedding / model settings  (§3.3, §6.1)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Segment similarity — speed-priority (MiniLM)
    segment_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    segment_embed_onnx: str | None = None    # optional: path to int8 ONNX export

    # Cache query embedding — retrieval-grade (BGE)
    cache_embed_model: str = "BAAI/bge-small-en-v1.5"
    cache_embed_onnx: str | None = None

    # Cache verification cross-encoder
    cache_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    cache_rerank_onnx: str | None = None
    cache_rerank_threshold: float = 0.85

    # NER model for manifest entity extraction
    ner_model: str = "dslim/bert-base-NER"
    ner_onnx: str | None = None

    # Classifier (stage B logistic regression)
    classifier_onnx: str | None = None     # None → rules-only (stage A)
    classifier_confidence_floor: float = 0.60   # below this → AMBIGUOUS

    embed_batch_size: int = 32
    embed_max_tokens: int = 512


# ---------------------------------------------------------------------------
# Telemetry & alerting settings  (§9)
# ---------------------------------------------------------------------------

@dataclass
class TelemetryConfig:
    enabled: bool = True
    metrics_endpoint: str = "/metrics"
    dashboard_enabled: bool = True
    dashboard_path: str = "/dashboard"
    alert_webhook_url: str | None = None
    savings_floor_pct: float = 0.08     # SAVINGS_COLLAPSE alert if <8% over 7d
    rollback_spike_threshold: float = 0.10   # ROLLBACK_SPIKE alert if >10% over 1h
    latency_overhead_budget_p95_ms: float = 200.0


# ---------------------------------------------------------------------------
# Top-level config  (§10.1)
# ---------------------------------------------------------------------------

OperatingMode = Literal["optimize", "cache_only", "observe_only", "bypass"]


@dataclass
class ITOLConfig:
    """
    Root configuration object.  Pass to `itol.wrap()` or `ITOL(config=…)`.

    All fields have sensible defaults; a minimal deployment needs only
    `provider_adapters` and optionally `data_dir`.
    """
    mode: OperatingMode = "optimize"

    # Provider → ProviderCacheConfig; missing providers get a default
    provider_cache: dict[str, ProviderCacheConfig] = field(
        default_factory=default_provider_cache_configs
    )

    # Per-request-class strategy settings
    class_configs: dict[str, ClassConfig] = field(
        default_factory=default_class_configs
    )

    quality: QualityConfig = field(default_factory=QualityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    # Per-tenant overrides: tenant_id → partial ITOLConfig-like dict
    tenant_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Hard latency deadlines (ms); exceeded → bypass for that stage
    stage_deadline_ms: dict[str, float] = field(default_factory=lambda: {
        "segment":           10.0,  # normalize + segment + signals (§7.4)
        "l0":                 3.0,
        "classify_manifest":  9.0,  # §7.4 table: "Classify + manifest" is one combined stage
        "l1":                35.0,
        "l2_replay":         45.0,
        "strategy_cold":     95.0,
        "guarantor":         15.0,
        "total_overhead":   200.0,
    })

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.mode not in ("optimize", "cache_only", "observe_only", "bypass"):
            raise ValueError(f"Unknown mode: {self.mode!r}")

        q = self.quality
        if not (0.0 < q.qps_floor <= 1.0):
            raise ValueError(f"qps_floor must be in (0, 1]; got {q.qps_floor}")
        if not (0.0 < q.qps_floor_with_s7 <= 1.0):
            raise ValueError(f"qps_floor_with_s7 must be in (0, 1]; got {q.qps_floor_with_s7}")
        if q.qps_floor_with_s7 < q.qps_floor:
            raise ValueError("qps_floor_with_s7 must be >= qps_floor")
        if not (0.0 <= q.shadow_eval_rate <= 1.0):
            raise ValueError(f"shadow_eval_rate out of range: {q.shadow_eval_rate}")
        if not (0.0 < q.parity_floor <= 1.0):
            raise ValueError(f"parity_floor must be in (0, 1]; got {q.parity_floor}")

        for cls_name, cls_cfg in self.class_configs.items():
            if not (0.5 <= cls_cfg.s3_mass_floor <= 1.0):
                raise ValueError(
                    f"class_configs[{cls_name!r}].s3_mass_floor out of range: {cls_cfg.s3_mass_floor}"
                )
            if not (0.0 < cls_cfg.l1_similarity_threshold <= 1.0):
                raise ValueError(
                    f"class_configs[{cls_name!r}].l1_similarity_threshold out of range"
                )

        for prov, pc in self.provider_cache.items():
            if not (0.0 <= pc.cache_read_discount <= 1.0):
                raise ValueError(
                    f"provider_cache[{prov!r}].cache_read_discount out of range: {pc.cache_read_discount}"
                )

    def for_tenant(self, tenant_id: str) -> "ITOLConfig":
        """
        Return a (possibly mutated) config view for a specific tenant.
        Only quality floors may be raised, not lowered, by tenant overrides
        (enforcing hard constraint 3 — zero quality degradation).
        """
        overrides = self.tenant_overrides.get(tenant_id, {})
        if not overrides:
            return self
        import copy
        cfg = copy.deepcopy(self)
        # Only safe field: mode relaxation is not allowed; quality floors only go up.
        if "shadow_eval_rate" in overrides:
            cfg.quality.shadow_eval_rate = max(
                cfg.quality.shadow_eval_rate, float(overrides["shadow_eval_rate"])
            )
        if "mode" in overrides:
            # Only allow more restrictive mode changes
            order = ["optimize", "cache_only", "observe_only", "bypass"]
            if order.index(overrides["mode"]) >= order.index(cfg.mode):
                cfg.mode = overrides["mode"]
        return cfg
