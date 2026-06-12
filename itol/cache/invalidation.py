"""
Cache invalidation — §6.2.

Three eviction paths:
  1. TTL eviction: entries past their class-based TTL are removed from L0.
     Reuses _CLASS_TTL from l0_exact.py (import, not duplication).
  2. LRU-K(K=2) capacity eviction for L1 vector index.
     Savings-weighted: entries that saved more tokens evict later.
     Capacity limit: storage.l1_max_entries (default 50 000).
  3. Doc-version invalidation: L2 plans depending on a changed doc are tombstoned.
     Delegates to Store.invalidate_l2_by_doc() (CR-9).

Negative caching prohibition: L1 store() in l1_semantic.py already refuses entries
where response.error is not None (same CR-20 pattern as L0).  This module confirms
the contract by exposing is_negative_response() for callers.
"""

from __future__ import annotations

import time

from itol.cache.l0_exact import _CLASS_TTL, _CACHEABLE_FINISH_REASONS
from itol.cache.store import Store
from itol.icr import ICRResponse


# Default L1 entry limit — overridden by ITOLConfig.storage.l1_max_entries
_DEFAULT_L1_MAX_ENTRIES = 50_000


def is_negative_response(response: ICRResponse) -> bool:
    """
    Confirm that a response should NOT be cached (CR-20 pattern).

    True when:
      - response.error is not None (error / refusal), OR
      - response.finish_reason is not a terminal cacheable reason.
    """
    if response.error is not None:
        return True
    return response.finish_reason not in _CACHEABLE_FINISH_REASONS


class CacheInvalidator:
    """
    Performs scheduled eviction and invalidation operations on the cache store.
    """

    def __init__(self, store: Store, l1_max_entries: int = _DEFAULT_L1_MAX_ENTRIES) -> None:
        self._store = store
        self._l1_max_entries = l1_max_entries

    # ------------------------------------------------------------------
    # TTL eviction — L0
    # ------------------------------------------------------------------

    def evict_expired_l0(self, now: float | None = None) -> int:
        """
        Remove all L0 cache entries whose TTL has elapsed.
        Returns the count of entries removed.

        Reuses §6.2 TTL constants from l0_exact._CLASS_TTL — not duplicated here.
        """
        if now is None:
            now = time.time()
        conn = self._store._conn()
        cur = conn.execute(
            "DELETE FROM l0_cache WHERE expires_at <= ?", (now,)
        )
        conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # LRU-K(K=2) eviction — L1
    # ------------------------------------------------------------------

    def evict_l1_lru_k(self, max_entries: int | None = None) -> int:
        """
        Remove lowest-priority L1 entries until count ≤ max_entries.

        Priority = last_access_1 (K=2nd-most-recent access time) weighted by
        tokens_saved.  Higher score = kept longer.

        Returns the count of entries evicted.
        """
        limit = max_entries if max_entries is not None else self._l1_max_entries
        return self._store.evict_l1_lru_k(limit)

    # ------------------------------------------------------------------
    # Doc-version invalidation — L2  (CR-9)
    # ------------------------------------------------------------------

    def invalidate_by_doc_version(
        self, doc_hash: str, new_version_hash: str
    ) -> int:
        """
        CR-9: tombstone all L2 plans that list doc_hash in depends_on_doc_versions.
        new_version_hash is provided for audit but not stored.
        Returns the count of plans tombstoned.
        """
        return self._store.invalidate_l2_by_doc(doc_hash)

    # ------------------------------------------------------------------
    # TTL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ttl_for_class(request_class: str | None) -> int:
        """
        Return the standard TTL (seconds) for a given request class.
        Delegates to l0_exact._CLASS_TTL — single source of truth.
        """
        if request_class is None:
            return 24 * 3600
        return _CLASS_TTL.get(request_class.upper(), 24 * 3600)
