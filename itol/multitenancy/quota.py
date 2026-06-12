"""
Quota tracking + TenantStoreGuard — §11.

QuotaTracker
------------
Daily per-tenant request and token counters stored in the quota_usage table.

Quota-exceeded semantics (§11 invariant):
    check() returns False (quota exceeded) → caller BYPASSES optimization and
    dispatches the request RAW to the provider.  The request is NEVER blocked
    entirely — quota exceeded means "stop saving money, just pass through."

TenantStoreGuard
----------------
A thin proxy around Store that enforces the no_store flag:
    no_store=True  → all cache writes are no-ops; cache reads return None/empty.
    no_store=False → pass-through to the underlying Store unchanged.

This is the ONLY mechanism that must intercept writes for no_store tenants.
It does NOT affect telemetry writes (record_request) or quota tracking — those
are operational and always proceed.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from itol.cache.store import Store
    from itol.multitenancy.config import TenantConfig

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QuotaTracker
# ---------------------------------------------------------------------------

class QuotaTracker:
    """
    Checks and increments daily per-tenant resource counters.

    The quota_usage table is managed by the Store (added in store.py schema).
    """

    def __init__(self, store: "Store") -> None:
        self._store = store

    def check(
        self,
        tenant_cfg: "TenantConfig",
        tokens: int = 0,
    ) -> bool:
        """
        Returns True if the request should be OPTIMIZED (within quota).
        Returns False if quota exceeded → caller must BYPASS, not block.

        Increments counters when returning True.
        """
        today = date.today().isoformat()
        usage = self._store.get_quota_usage(tenant_cfg.tenant_id, today)

        req_limit = tenant_cfg.quotas.requests_per_day
        tok_limit = tenant_cfg.quotas.tokens_per_day

        if req_limit is not None and usage["requests"] >= req_limit:
            _log.info(
                "QuotaTracker: tenant %s request quota exceeded (%d/%d)",
                tenant_cfg.tenant_id, usage["requests"], req_limit,
            )
            return False

        if tok_limit is not None and usage["tokens"] + tokens > tok_limit:
            _log.info(
                "QuotaTracker: tenant %s token quota exceeded (%d+%d>%d)",
                tenant_cfg.tenant_id, usage["tokens"], tokens, tok_limit,
            )
            return False

        # Within quota — increment
        self._store.increment_quota(
            tenant_id=tenant_cfg.tenant_id,
            date=today,
            requests=1,
            tokens=tokens,
        )
        return True

    def get_usage(self, tenant_id: str) -> dict[str, int]:
        """Return today's usage for a tenant."""
        today = date.today().isoformat()
        return self._store.get_quota_usage(tenant_id, today)


# ---------------------------------------------------------------------------
# TenantStoreGuard
# ---------------------------------------------------------------------------

class TenantStoreGuard:
    """
    Proxy around Store that enforces the no_store flag.

    When no_store=True:
      - Cache write methods (set_l0, set_l1_entry, set_l2_plan, save_doc,
        set_template_compressed, increment_template) are no-ops.
      - Cache read methods (get_l0, get_l1_entry, get_l2_plan, get_doc,
        get_template) return None / empty.
      - Operational writes (record_request, log_breakeven, quota) are
        NOT intercepted — they always proceed.

    When no_store=False: every method delegates to the underlying Store.
    """

    _NO_STORE_READ_NONE = frozenset({
        "get_l0", "get_l1_entry", "get_l2_plan", "get_doc",
        "get_template", "get_docs_for_conversation",
    })
    _NO_STORE_WRITE_NOP = frozenset({
        "set_l0", "set_l1_entry", "set_l2_plan", "save_doc",
        "set_template_compressed", "increment_template",
        "delete_l1_entry", "update_l1_access",
    })

    def __init__(self, store: "Store", no_store: bool) -> None:
        self._store = store
        self._no_store = no_store

    def __getattr__(self, name: str) -> Any:
        if self._no_store:
            if name in self._NO_STORE_READ_NONE:
                if name == "get_docs_for_conversation":
                    return lambda *a, **kw: []
                return lambda *a, **kw: None
            if name in self._NO_STORE_WRITE_NOP:
                return lambda *a, **kw: None
        return getattr(self._store, name)
