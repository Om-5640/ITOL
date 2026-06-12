"""
Multitenancy authentication — §11.

authenticate() maps an API key to a TenantConfig.
No external auth provider dependency (self-hosted design).

The per-tenant namespace (tenant_id) is threaded through L0/L1/L2/docs/templates
tables from Steps 6-11; this module is the lookup layer that resolves
an incoming API key to its tenant_id before the request enters the pipeline.
"""

from __future__ import annotations

from itol.multitenancy.config import TenantConfig, TenantRegistry


def authenticate(api_key: str, registry: TenantRegistry) -> TenantConfig | None:
    """
    Resolve an API key to its TenantConfig.

    Returns None if the key is not recognised.  Callers should reject the
    request (HTTP 401) when None is returned.

    For single-tenant deployments with no API key enforcement, callers may
    bypass this function and use registry.get("default") directly.
    """
    if not api_key:
        return None
    return registry.lookup_by_api_key(api_key)


def resolve_tenant(
    api_key: str | None,
    registry: TenantRegistry,
    default_tenant_id: str = "default",
) -> TenantConfig:
    """
    Resolve an optional API key to a TenantConfig.

    If api_key is None or not found, returns the default tenant.  Use this
    for single-tenant or dev deployments where auth is not enforced.
    """
    if api_key:
        cfg = authenticate(api_key, registry)
        if cfg is not None:
            return cfg
    default = registry.get(default_tenant_id)
    if default is None:
        return TenantConfig(tenant_id=default_tenant_id)
    return default
