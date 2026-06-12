"""
Multitenancy configuration — §11 self-hosted, no external auth provider.

TenantConfig: per-tenant settings (API keys, quotas, quality overrides, data policy).
TenantRegistry: loads from itol.yaml [tenants:] section; falls back to single-tenant
                "default" with unlimited quotas if no config file is present.

Tighten-only invariant
-----------------------
Tenant quality overrides can ONLY raise quality floors, never lower them:
    effective_floor = max(global_floor, tenant_override_floor)

This is enforced by apply_tenant_quality_override() and tested in §14.3.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class QuotaSpec:
    """Daily resource limits for a tenant. None = unlimited."""
    requests_per_day: int | None = None
    tokens_per_day: int | None = None


@dataclass
class TenantQualityOverride:
    """
    Quality knobs a tenant operator can tighten (not loosen) relative to the
    global ITOLConfig.quality defaults.

    All values are applied via max() so they can only raise the floor:
        effective = max(global, tenant_override)
    """
    qps_floor: float | None = None
    qps_floor_with_s7: float | None = None
    shadow_eval_rate: float | None = None
    parity_floor: float | None = None


@dataclass
class TenantConfig:
    """
    Per-tenant runtime configuration.

    Fields
    ------
    tenant_id          : Stable identifier; must match the tenant_id threaded
                         through Store/L0/L1/L2 namespacing.
    api_keys           : List of raw API keys accepted for this tenant.
    quotas             : Daily request / token limits.
    quality_overrides  : Quality floors — can only be TIGHTENED relative to
                         the global ITOLConfig.quality values.
    data_retention_days: How long telemetry rows are kept (default 90 days).
    no_store           : True → all L0/L1/L2 cache writes are no-ops for this
                         tenant; reads also skipped.  Useful for GDPR-sensitive
                         or stateless deployments.
    """
    tenant_id: str
    api_keys: list[str] = field(default_factory=list)
    quotas: QuotaSpec = field(default_factory=QuotaSpec)
    quality_overrides: TenantQualityOverride = field(default_factory=TenantQualityOverride)
    data_retention_days: int = 90
    no_store: bool = False


def apply_tenant_quality_override(
    base_quality: Any,  # itol.config.QualityConfig instance
    overrides: TenantQualityOverride,
) -> Any:
    """
    Return a COPY of base_quality with tenant overrides applied.

    Enforces the tighten-only invariant: each field is set to
        max(current_value, override_value)
    so overrides can ONLY raise (tighten) floors — never lower them.
    """
    q = copy.deepcopy(base_quality)
    if overrides.qps_floor is not None:
        q.qps_floor = max(q.qps_floor, overrides.qps_floor)
    if overrides.qps_floor_with_s7 is not None:
        q.qps_floor_with_s7 = max(q.qps_floor_with_s7, overrides.qps_floor_with_s7)
    if overrides.shadow_eval_rate is not None:
        q.shadow_eval_rate = max(q.shadow_eval_rate, overrides.shadow_eval_rate)
    if overrides.parity_floor is not None:
        q.parity_floor = max(q.parity_floor, overrides.parity_floor)
    return q


class TenantRegistry:
    """
    In-memory registry of tenants.  Loaded once at startup.

    Load order:
      1. itol.yaml [tenants:] section, if the file exists
      2. Single-tenant default (tenant_id="default", unlimited quotas)

    Thread-safe for reads after initialisation (no writes after load).
    """

    def __init__(
        self,
        tenants: dict[str, TenantConfig] | None = None,
    ) -> None:
        # Key: tenant_id → TenantConfig
        self._by_id: dict[str, TenantConfig] = tenants or {}
        # Key: api_key → tenant_id  (built from _by_id)
        self._by_key: dict[str, str] = {
            key: cfg.tenant_id
            for cfg in self._by_id.values()
            for key in cfg.api_keys
        }
        # Ensure "default" always exists
        if "default" not in self._by_id:
            self._by_id["default"] = TenantConfig(tenant_id="default")

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, tenant_id: str) -> TenantConfig | None:
        return self._by_id.get(tenant_id)

    def lookup_by_api_key(self, api_key: str) -> TenantConfig | None:
        tenant_id = self._by_key.get(api_key)
        if tenant_id is None:
            return None
        return self._by_id.get(tenant_id)

    def all_tenants(self) -> list[TenantConfig]:
        return list(self._by_id.values())

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TenantRegistry":
        """
        Load from an itol.yaml file.  Returns a single-tenant default registry
        if the file does not exist or has no [tenants:] section.

        Expected YAML shape:
            tenants:
              acme:
                api_keys: ["sk-acme-123"]
                quotas:
                  requests_per_day: 10000
                  tokens_per_day: 50000000
                quality_overrides:
                  qps_floor: 0.99
                no_store: false
                data_retention_days: 30
        """
        path = Path(config_path)
        if not path.exists():
            _log.debug("TenantRegistry: no config file at %s; using default tenant", path)
            return cls()

        try:
            import yaml  # type: ignore
        except ImportError:
            _log.warning("TenantRegistry: PyYAML not installed; using default tenant")
            return cls()

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        raw_tenants: dict[str, Any] = data.get("tenants", {}) or {}
        tenants: dict[str, TenantConfig] = {}

        for tenant_id, raw in raw_tenants.items():
            raw = raw or {}
            raw_quotas = raw.get("quotas", {}) or {}
            quotas = QuotaSpec(
                requests_per_day=raw_quotas.get("requests_per_day"),
                tokens_per_day=raw_quotas.get("tokens_per_day"),
            )
            raw_overrides = raw.get("quality_overrides", {}) or {}
            overrides = TenantQualityOverride(
                qps_floor=raw_overrides.get("qps_floor"),
                qps_floor_with_s7=raw_overrides.get("qps_floor_with_s7"),
                shadow_eval_rate=raw_overrides.get("shadow_eval_rate"),
                parity_floor=raw_overrides.get("parity_floor"),
            )
            tenants[tenant_id] = TenantConfig(
                tenant_id=tenant_id,
                api_keys=raw.get("api_keys", []),
                quotas=quotas,
                quality_overrides=overrides,
                data_retention_days=raw.get("data_retention_days", 90),
                no_store=bool(raw.get("no_store", False)),
            )

        return cls(tenants=tenants)

    @classmethod
    def single_tenant_default(cls) -> "TenantRegistry":
        """Convenience: a registry with only the default tenant (unlimited quotas)."""
        return cls()
