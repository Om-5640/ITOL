"""
§14.3 Invariant tests for CR-9 — doc-version change tombstones L2 plans.

CR-9: when a document referenced by an L2 template plan is updated, the plan
must be invalidated (tombstoned).  Subsequent get_plan() calls for the affected
template must return None.

The depends_on_doc_versions list on TemplatePlan records all external doc hashes
that the plan assumed.  invalidate_by_doc_version() walks all plans and deletes
those that reference the changed doc.

Rules:
- NEVER weaken thresholds or conditions.
"""

import time
import tempfile

import pytest

from itol.cache.l2_template import L2PlanCache, TemplatePlan
from itol.cache.store import Store
from itol.cache.invalidation import CacheInvalidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_and_caches(tmp_path):
    store = Store(str(tmp_path))
    l2 = L2PlanCache(store)
    invalidator = CacheInvalidator(store)
    return store, l2, invalidator


def _plan(
    template_sig: str = "sig1",
    tenant_id: str = "t1",
    doc_versions: list[str] | None = None,
) -> TemplatePlan:
    return TemplatePlan(
        template_sig=template_sig,
        tenant_id=tenant_id,
        strategies_config={"S2": {"fired": True}},
        segment_token_counts={"abc123": 42},
        breakpoint_positions=[256],
        depends_on_doc_versions=doc_versions or [],
        created_at=time.time(),
    )


# ===========================================================================
# CR-9-a: doc-version change tombstones the plan
# ===========================================================================

class TestCR9_DocVersionInvalidation:

    def test_doc_version_tombstones_plan(self, tmp_path):
        """
        CR-9: storing a plan that depends on doc_v1, then calling
        invalidate_by_doc_version("doc_v1", ...), must make get_plan() return None.
        """
        store, l2, inv = _store_and_caches(tmp_path)

        plan = _plan(doc_versions=["doc_v1"])
        l2.store_plan("sig1", "t1", plan)
        assert l2.get_plan("sig1", "t1") is not None, "Plan must exist before invalidation"

        count = inv.invalidate_by_doc_version("doc_v1", "doc_v2")
        assert count >= 1, f"CR-9: at least 1 plan must be tombstoned; got {count}"

        result = l2.get_plan("sig1", "t1")
        assert result is None, (
            "CR-9: after invalidate_by_doc_version, get_plan must return None"
        )
        store.close()

    def test_unrelated_plan_survives_invalidation(self, tmp_path):
        """
        CR-9: a plan that does NOT depend on the changed doc must NOT be tombstoned.
        """
        store, l2, inv = _store_and_caches(tmp_path)

        plan_affected = _plan("sig1", "t1", doc_versions=["doc_changed"])
        plan_unrelated = _plan("sig2", "t1", doc_versions=["doc_other"])

        l2.store_plan("sig1", "t1", plan_affected)
        l2.store_plan("sig2", "t1", plan_unrelated)

        inv.invalidate_by_doc_version("doc_changed", "doc_changed_v2")

        assert l2.get_plan("sig1", "t1") is None, "Affected plan must be tombstoned"
        assert l2.get_plan("sig2", "t1") is not None, (
            "CR-9: unrelated plan must NOT be tombstoned"
        )
        store.close()

    def test_plan_with_no_dependencies_survives(self, tmp_path):
        """
        A plan with depends_on_doc_versions=[] is never tombstoned by doc invalidation.
        """
        store, l2, inv = _store_and_caches(tmp_path)

        plan = _plan("sig_nodep", "t1", doc_versions=[])
        l2.store_plan("sig_nodep", "t1", plan)

        inv.invalidate_by_doc_version("any_doc", "any_new_version")

        assert l2.get_plan("sig_nodep", "t1") is not None, (
            "A plan with no doc dependencies must survive invalidation"
        )
        store.close()

    def test_multiple_plans_same_doc_all_tombstoned(self, tmp_path):
        """
        All plans referencing the same doc hash must be tombstoned together.
        """
        store, l2, inv = _store_and_caches(tmp_path)

        for i in range(4):
            l2.store_plan(f"sig{i}", "t1", _plan(f"sig{i}", "t1", doc_versions=["shared_doc"]))

        count = inv.invalidate_by_doc_version("shared_doc", "shared_doc_v2")
        assert count == 4, f"CR-9: all 4 plans must be tombstoned; got {count}"

        for i in range(4):
            assert l2.get_plan(f"sig{i}", "t1") is None, (
                f"Plan sig{i} must be tombstoned after shared_doc invalidation"
            )
        store.close()

    def test_invalidation_via_l2_directly(self, tmp_path):
        """L2PlanCache.invalidate_by_doc_version returns the tombstone count."""
        store, l2, _ = _store_and_caches(tmp_path)

        l2.store_plan("s1", "t1", _plan("s1", "t1", doc_versions=["doc_x"]))
        l2.store_plan("s2", "t1", _plan("s2", "t1", doc_versions=["doc_x"]))

        count = l2.invalidate_by_doc_version("doc_x", "doc_x_new")
        assert count == 2
        assert l2.get_plan("s1", "t1") is None
        assert l2.get_plan("s2", "t1") is None
        store.close()


# ===========================================================================
# CR-9-b: L2 plan round-trip correctness
# ===========================================================================

class TestCR9_PlanRoundTrip:

    def test_plan_roundtrip(self, tmp_path):
        """store_plan then get_plan must reconstruct the plan faithfully."""
        store, l2, _ = _store_and_caches(tmp_path)

        original = _plan("rt_sig", "rt_tenant", doc_versions=["doc_abc"])
        l2.store_plan("rt_sig", "rt_tenant", original)

        retrieved = l2.get_plan("rt_sig", "rt_tenant")

        assert retrieved is not None
        assert retrieved.template_sig == original.template_sig
        assert retrieved.tenant_id == original.tenant_id
        assert retrieved.strategies_config == original.strategies_config
        assert retrieved.segment_token_counts == original.segment_token_counts
        assert retrieved.breakpoint_positions == original.breakpoint_positions
        assert retrieved.depends_on_doc_versions == original.depends_on_doc_versions
        store.close()

    def test_get_missing_plan_returns_none(self, tmp_path):
        """get_plan on a non-existent key must return None."""
        store, l2, _ = _store_and_caches(tmp_path)
        assert l2.get_plan("nonexistent", "t1") is None
        store.close()
