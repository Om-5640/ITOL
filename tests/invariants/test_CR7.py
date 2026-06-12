"""
§14.3 Invariant tests for CR-7 — L1 cache namespace isolation.

CR-7: the L1 vector index must NEVER return a cache hit for a lookup
whose namespace differs from the stored entry's namespace, even when
the query vector is identical (cosine = 1.0).

Namespace key: sha256(tenant_id | provider | model | template_sig | history_digest)
Any difference in any of the five fields produces a different namespace.

Rules:
- NEVER weaken thresholds or conditions.
"""

import hashlib
import tempfile

import numpy as np
import pytest

from itol.cache.vector_index import SqliteVecIndex
from itol.cache.l1_semantic import _namespace
from itol.cache.store import Store
from itol.icr import ICR, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(16).astype(np.float32)
    return v / np.linalg.norm(v)


def _icr(tenant_id: str = "t1", provider: str = "openai", model: str = "gpt-4o") -> ICR:
    return ICR.create(
        provider=provider,
        model=model,
        tenant_id=tenant_id,
        system=[],
        messages=[Message.user("What is the capital of France?")],
        raw={},
    )


# ===========================================================================
# CR-7-a: SqliteVecIndex only returns hits within the given namespace
# ===========================================================================

class TestCR7_VectorIndexNamespaceIsolation:

    def test_different_namespace_returns_no_hits(self, tmp_path):
        """
        CR-7: a vector added under namespace N1 must not be returned when
        searching under namespace N2, even with an identical query vector.
        """
        idx = SqliteVecIndex(tmp_path / "test.db")
        vec = _fixed_vec(0)

        idx.add("entry1", vec, namespace="namespace_A")

        # Search in a DIFFERENT namespace with the IDENTICAL vector
        results = idx.search(vec, namespace="namespace_B", top_k=5)

        assert len(results) == 0, (
            "CR-7: SqliteVecIndex must return zero hits when searching "
            "a namespace that has no entries"
        )
        idx.close()

    def test_same_namespace_returns_hit(self, tmp_path):
        """Control: same namespace with identical vector must return a hit."""
        idx = SqliteVecIndex(tmp_path / "test.db")
        vec = _fixed_vec(0)

        idx.add("entry1", vec, namespace="namespace_A")
        results = idx.search(vec, namespace="namespace_A", top_k=5)

        assert len(results) == 1, "Same namespace must return the stored entry"
        assert results[0][0] == "entry1"
        assert results[0][1] > 0.999  # identical vectors → cosine ≈ 1.0
        idx.close()

    def test_multiple_namespaces_isolated(self, tmp_path):
        """CR-7: entries in N1, N2, N3 are each invisible to other namespaces."""
        idx = SqliteVecIndex(tmp_path / "test.db")

        vecs = {f"ns{i}": _fixed_vec(i) for i in range(3)}
        for ns, vec in vecs.items():
            idx.add(f"entry_{ns}", vec, namespace=ns)

        # Each namespace only sees its own entry
        for target_ns in vecs:
            # search with the target namespace's own vector
            results = idx.search(vecs[target_ns], namespace=target_ns, top_k=5)
            assert len(results) == 1, f"ns={target_ns} must only see its own entry"
            assert results[0][0] == f"entry_{target_ns}"

            # search OTHER namespaces with this vector — must return nothing from target
            for other_ns in vecs:
                if other_ns == target_ns:
                    continue
                other_results = idx.search(vecs[target_ns], namespace=other_ns, top_k=5)
                ids_in_other = {r[0] for r in other_results}
                assert f"entry_{target_ns}" not in ids_in_other, (
                    f"CR-7: entry from {target_ns!r} must not appear in {other_ns!r}"
                )
        idx.close()

    def test_cosine_1_still_no_cross_namespace_hit(self, tmp_path):
        """
        CR-7: even cosine = 1.0 (identical vector) must not produce a cross-
        namespace hit.  This is the invariant's most important boundary condition.
        """
        idx = SqliteVecIndex(tmp_path / "test.db")
        vec = _fixed_vec(7)

        idx.add("entry1", vec, namespace="ns_stored")

        # Query with the EXACT SAME vector in a different namespace
        results = idx.search(vec, namespace="ns_different", top_k=5)
        assert len(results) == 0, (
            "CR-7 INVARIANT: cosine=1.0 must not produce a cross-namespace hit"
        )
        idx.close()


# ===========================================================================
# CR-7-b: L1Cache namespace based on all five fields
# ===========================================================================

class TestCR7_NamespaceKey:

    def test_different_tenant_different_namespace(self):
        """Different tenant_id → different namespace hash."""
        icr_a = _icr(tenant_id="tenant_a")
        icr_b = _icr(tenant_id="tenant_b")
        assert _namespace(icr_a) != _namespace(icr_b), (
            "CR-7: different tenant_id must produce different namespace"
        )

    def test_different_provider_different_namespace(self):
        """Different provider → different namespace."""
        icr_a = _icr(provider="openai")
        icr_b = _icr(provider="anthropic")
        assert _namespace(icr_a) != _namespace(icr_b), (
            "CR-7: different provider must produce different namespace"
        )

    def test_different_model_different_namespace(self):
        """Different model → different namespace."""
        icr_a = _icr(model="gpt-4o")
        icr_b = _icr(model="gpt-3.5-turbo")
        assert _namespace(icr_a) != _namespace(icr_b), (
            "CR-7: different model must produce different namespace"
        )

    def test_same_params_same_namespace(self):
        """Identical request params → same namespace (deterministic)."""
        icr_a = _icr()
        icr_b = _icr()
        assert _namespace(icr_a) == _namespace(icr_b), (
            "CR-7: identical request params must produce identical namespace"
        )

    def test_namespace_is_64_char_hex(self):
        """Namespace must be a 64-char hex string (sha256 output)."""
        ns = _namespace(_icr())
        assert len(ns) == 64
        assert all(c in "0123456789abcdef" for c in ns)
