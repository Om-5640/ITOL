"""
§14.3 Invariant tests for multitenancy (§11).

Invariants:
  1. test_tenant_override_can_only_tighten — tenant quality overrides can only
     RAISE quality floors (max), never lower them.
  2. test_quota_exceeded_bypasses_not_blocks — when quota is exceeded, the
     request still dispatches (as raw/unoptimized), never an error.
  3. test_no_store_tenant_skips_all_caching — no_store=True → L0/L1/L2 writes
     are no-ops; reads return None/empty.
"""

from __future__ import annotations

import pytest

from itol.cache.store import Store
from itol.config import ITOLConfig, QualityConfig
from itol.multitenancy.config import (
    QuotaSpec,
    TenantConfig,
    TenantQualityOverride,
    TenantRegistry,
    apply_tenant_quality_override,
)
from itol.multitenancy.auth import authenticate, resolve_tenant
from itol.multitenancy.quota import QuotaTracker, TenantStoreGuard


# ===========================================================================
# Invariant 1: tenant override can only tighten
# ===========================================================================

class TestTenantOverrideTightenOnly:

    def test_override_cannot_lower_qps_floor(self):
        """
        Attempt to set tenant qps_floor BELOW global default (0.98) → must be
        clamped to the global value.  Operators cannot loosen quality floors.
        """
        global_quality = QualityConfig(qps_floor=0.98)
        override = TenantQualityOverride(qps_floor=0.90)  # trying to LOOSEN

        effective = apply_tenant_quality_override(global_quality, override)
        assert effective.qps_floor == 0.98, (
            "Tenant override must not lower qps_floor below global default"
        )

    def test_override_can_raise_qps_floor(self):
        """Tenant override ABOVE global default is allowed (tightening)."""
        global_quality = QualityConfig(qps_floor=0.98)
        override = TenantQualityOverride(qps_floor=0.99)  # TIGHTENING

        effective = apply_tenant_quality_override(global_quality, override)
        assert effective.qps_floor == 0.99, (
            "Tenant override that raises qps_floor must be applied"
        )

    def test_override_cannot_lower_parity_floor(self):
        """Attempt to lower parity_floor → must be clamped to global value."""
        global_quality = QualityConfig(parity_floor=0.95)
        override = TenantQualityOverride(parity_floor=0.80)  # too low

        effective = apply_tenant_quality_override(global_quality, override)
        assert effective.parity_floor == 0.95, (
            "Tenant override must not lower parity_floor below global default"
        )

    def test_override_cannot_lower_shadow_eval_rate(self):
        """Attempt to reduce shadow_eval_rate → must be clamped to global value."""
        global_quality = QualityConfig(shadow_eval_rate=0.015)
        override = TenantQualityOverride(shadow_eval_rate=0.001)  # too low

        effective = apply_tenant_quality_override(global_quality, override)
        assert effective.shadow_eval_rate == 0.015, (
            "Tenant override must not lower shadow_eval_rate below global default"
        )

    def test_none_override_fields_leave_global_unchanged(self):
        """Override fields that are None must not change the global values."""
        global_quality = QualityConfig(qps_floor=0.98, parity_floor=0.95)
        override = TenantQualityOverride()  # all None

        effective = apply_tenant_quality_override(global_quality, override)
        assert effective.qps_floor == 0.98
        assert effective.parity_floor == 0.95

    def test_override_does_not_mutate_global_config(self):
        """apply_tenant_quality_override must return a COPY, not mutate the original."""
        global_quality = QualityConfig(qps_floor=0.98)
        override = TenantQualityOverride(qps_floor=0.99)

        effective = apply_tenant_quality_override(global_quality, override)
        assert global_quality.qps_floor == 0.98, (
            "Global quality config must not be mutated by apply_tenant_quality_override"
        )
        assert effective is not global_quality

    def test_itol_config_for_tenant_cannot_lower_qps_floor(self):
        """ITOLConfig.for_tenant() must also enforce the tighten-only rule."""
        config = ITOLConfig()
        # Inject a tenant override that tries to lower shadow_eval_rate
        config.tenant_overrides["evil_tenant"] = {"shadow_eval_rate": 0.0}

        tenant_cfg = config.for_tenant("evil_tenant")
        # shadow_eval_rate can only be raised, never lowered
        assert tenant_cfg.quality.shadow_eval_rate >= config.quality.shadow_eval_rate, (
            "ITOLConfig.for_tenant() must not lower shadow_eval_rate"
        )


# ===========================================================================
# Invariant 2: quota exceeded → bypass, not block
# ===========================================================================

class TestQuotaExceededBypassesNotBlocks:

    def test_requests_within_quota_return_true(self, tmp_path):
        """QuotaTracker.check() returns True when request count < daily limit."""
        store = Store(str(tmp_path))
        tracker = QuotaTracker(store)
        tenant = TenantConfig(
            tenant_id="test_tenant",
            quotas=QuotaSpec(requests_per_day=5),
        )

        for _ in range(5):
            assert tracker.check(tenant) is True, (
                "check() must return True while within quota"
            )
        store.close()

    def test_quota_exceeded_returns_false_not_raises(self, tmp_path):
        """
        QuotaTracker.check() must return False (not raise) when quota exceeded.
        This signals 'bypass optimization' not 'deny service'.
        """
        store = Store(str(tmp_path))
        tracker = QuotaTracker(store)
        tenant = TenantConfig(
            tenant_id="test_quota_tenant",
            quotas=QuotaSpec(requests_per_day=3),
        )

        # Exhaust quota
        for _ in range(3):
            tracker.check(tenant)

        # 4th request — quota exceeded → False (bypass), never an exception
        try:
            result = tracker.check(tenant)
        except Exception as exc:
            pytest.fail(
                f"QuotaTracker.check() must not raise on quota exceeded; "
                f"got {type(exc).__name__}: {exc}"
            )

        assert result is False, (
            "QuotaTracker.check() must return False when quota exceeded, "
            "not block or raise (quota exceeded = bypass, not deny)"
        )
        store.close()

    def test_quota_exceeded_does_not_increment_further(self, tmp_path):
        """After quota is exceeded, further calls do not increment the counter."""
        store = Store(str(tmp_path))
        tracker = QuotaTracker(store)
        tenant = TenantConfig(
            tenant_id="tenant_incr",
            quotas=QuotaSpec(requests_per_day=2),
        )

        tracker.check(tenant)
        tracker.check(tenant)  # at limit
        tracker.check(tenant)  # over limit — should NOT increment

        usage = tracker.get_usage("tenant_incr")
        assert usage["requests"] == 2, (
            "Counter must not increment when quota is exceeded"
        )
        store.close()

    def test_unlimited_quota_always_returns_true(self, tmp_path):
        """None quota (unlimited) must always return True."""
        store = Store(str(tmp_path))
        tracker = QuotaTracker(store)
        tenant = TenantConfig(
            tenant_id="unlimited",
            quotas=QuotaSpec(requests_per_day=None, tokens_per_day=None),
        )

        for _ in range(100):
            assert tracker.check(tenant) is True
        store.close()

    def test_token_quota_exceeded_returns_false(self, tmp_path):
        """Token-based quota exceeded → False (bypass), not raise."""
        store = Store(str(tmp_path))
        tracker = QuotaTracker(store)
        tenant = TenantConfig(
            tenant_id="tok_tenant",
            quotas=QuotaSpec(tokens_per_day=1000),
        )

        # Use up 900 tokens
        tracker.check(tenant, tokens=900)
        # Try to use 200 more (total 1100 > 1000) → should fail gracefully
        result = tracker.check(tenant, tokens=200)
        assert result is False, "Token quota exceeded must return False, not raise"
        store.close()


# ===========================================================================
# Invariant 3: no_store tenant skips all caching
# ===========================================================================

class TestNoStoreTenantSkipsCaching:

    def test_no_store_set_l0_is_noop(self, tmp_path):
        """set_l0 must be a no-op for no_store tenants."""
        store = Store(str(tmp_path))
        guard = TenantStoreGuard(store, no_store=True)

        guard.set_l0("key1", "default", '{"result": "test"}', ttl_seconds=3600)
        result = guard.get_l0("key1", "default")
        assert result is None, "get_l0 must return None for no_store tenant"
        store.close()

    def test_no_store_set_l1_is_noop(self, tmp_path):
        """set_l1_entry must be a no-op for no_store tenants."""
        store = Store(str(tmp_path))
        guard = TenantStoreGuard(store, no_store=True)

        guard.set_l1_entry("entry1", "ns1", '{}', "query", "[]", tokens_saved=10)
        result = guard.get_l1_entry("entry1", "ns1")
        assert result is None, "get_l1_entry must return None for no_store tenant"
        store.close()

    def test_no_store_set_l2_plan_is_noop(self, tmp_path):
        """set_l2_plan must be a no-op for no_store tenants."""
        store = Store(str(tmp_path))
        guard = TenantStoreGuard(store, no_store=True)

        guard.set_l2_plan("tmpl1", "default", '{"plan": "test"}', "[]")
        result = guard.get_l2_plan("tmpl1", "default")
        assert result is None, "get_l2_plan must return None for no_store tenant"
        store.close()

    def test_no_store_does_not_affect_real_store(self, tmp_path):
        """
        TenantStoreGuard in no_store mode must NOT write to the underlying
        store — a non-guard direct access to the same store must also return None.
        """
        store = Store(str(tmp_path))
        guard = TenantStoreGuard(store, no_store=True)

        guard.set_l0("direct_key", "default", '{"x": 1}', ttl_seconds=3600)
        # Direct store access must also return None (nothing was written)
        result = store.get_l0("direct_key", "default")
        assert result is None, (
            "TenantStoreGuard (no_store=True) must not write to the underlying store"
        )
        store.close()

    def test_normal_store_guard_passes_through(self, tmp_path):
        """Control: TenantStoreGuard with no_store=False passes writes through."""
        store = Store(str(tmp_path))
        guard = TenantStoreGuard(store, no_store=False)

        guard.set_l0("passkey", "default", '{"result": "ok"}', ttl_seconds=3600)
        result = guard.get_l0("passkey", "default")
        assert result == '{"result": "ok"}', (
            "TenantStoreGuard (no_store=False) must pass writes through to the store"
        )
        store.close()

    def test_no_store_get_docs_for_conversation_returns_empty(self, tmp_path):
        """get_docs_for_conversation must return empty list for no_store tenant."""
        store = Store(str(tmp_path))
        guard = TenantStoreGuard(store, no_store=True)

        result = guard.get_docs_for_conversation("default", "conv_xyz")
        assert result == [], (
            "get_docs_for_conversation must return empty list for no_store tenant"
        )
        store.close()


# ===========================================================================
# Authentication
# ===========================================================================

class TestAuthentication:

    def test_valid_api_key_returns_tenant(self):
        """authenticate() must return the TenantConfig for a known API key."""
        registry = TenantRegistry(tenants={
            "acme": TenantConfig(tenant_id="acme", api_keys=["sk-acme-test"]),
        })
        result = authenticate("sk-acme-test", registry)
        assert result is not None
        assert result.tenant_id == "acme"

    def test_unknown_api_key_returns_none(self):
        """authenticate() must return None for an unrecognised key."""
        registry = TenantRegistry(tenants={
            "acme": TenantConfig(tenant_id="acme", api_keys=["sk-acme-test"]),
        })
        result = authenticate("sk-unknown", registry)
        assert result is None

    def test_empty_key_returns_none(self):
        """authenticate() with empty string must return None."""
        registry = TenantRegistry()
        assert authenticate("", registry) is None

    def test_resolve_tenant_falls_back_to_default(self):
        """resolve_tenant() with unknown key falls back to default tenant."""
        registry = TenantRegistry()
        result = resolve_tenant("sk-unknown", registry)
        assert result.tenant_id == "default"

    def test_default_tenant_always_exists(self):
        """TenantRegistry must always contain a 'default' tenant."""
        registry = TenantRegistry()
        assert registry.get("default") is not None
