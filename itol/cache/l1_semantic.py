"""
L1 semantic cache — §6.1.

Lookup embeds the query (bge model, last-256-token tail + final user query),
searches a per-namespace ANN index, verifies top candidates with a cross-encoder
reranker, and enforces CR-8 (manifest numbers superset check).

CR-7 namespace key: sha256(tenant_id | provider | model | template_sig | history_digest)
All five fields are required — a missing field degrades to an empty string, not
a removed field, so namespaces remain stable across partial requests.

CR-8: cached_entry.manifest.numbers_entities ⊇ new_query.manifest.numbers_entities
This is an AND condition with cosine ≥ τ and cross_score ≥ 0.85 — never OR-ed.

L1 is entirely disabled (returns None immediately) for request classes where
matrix_row.l1 == L1Status.DISABLED (GENERATION_CREATIVE, AGENT_TOOL_LOOP, CHAT_OPEN).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from itol.cache.l0_exact import _CACHEABLE_FINISH_REASONS, _deserialize_response, _serialize_response
from itol.cache.rerank import rerank
from itol.cache.store import Store
from itol.cache.vector_index import SqliteVecIndex
from itol.embed.onnx_embedder import embed
from itol.icr import ClassifierResult, ConstraintManifest, ICR, ICRResponse, ManifestItem
from itol.routing.matrix import L1Status


_CROSS_SCORE_FLOOR = 0.85
_QUERY_TAIL_CHARS = 256 * 4   # ~256 tokens @ ~4 chars/token


# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------

def _history_digest(icr: ICR) -> str:
    """Hash of all conversation turns except the last user message."""
    prior_msgs = icr.messages[:-1] if len(icr.messages) > 1 else []
    content = "|".join(f"{m.role}:{m.text_content()}" for m in prior_msgs)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _namespace(icr: ICR) -> str:
    """
    CR-7 namespace key — five fields, all required.
    sha256(tenant_id | provider | model | template_signature | history_digest)
    """
    signals = icr.meta.signals if icr.meta else None
    template_sig = (signals.template_signature if signals else None) or ""
    hist = _history_digest(icr)
    raw = "|".join([icr.tenant_id, icr.provider, icr.model, template_sig, hist])
    return hashlib.sha256(raw.encode()).hexdigest()


def _build_query_text(icr: ICR) -> str:
    """Embed query: 256-token tail of prior context + final user query."""
    query = icr.final_user_query() or ""
    prior_msgs = icr.messages[:-1] if icr.messages else []
    full_context = " ".join(m.text_content() for m in prior_msgs)
    tail = full_context[-_QUERY_TAIL_CHARS:] if len(full_context) > _QUERY_TAIL_CHARS else full_context
    parts = [p for p in [tail, query] if p]
    return " ".join(parts).strip() or query


# ---------------------------------------------------------------------------
# Manifest numbers (CR-8)
# ---------------------------------------------------------------------------

def _numbers_entities(manifest: ConstraintManifest) -> set[str]:
    return {item.value for item in manifest.items if item.item_type == "NUMBER" and item.value}


# ---------------------------------------------------------------------------
# L1Cache
# ---------------------------------------------------------------------------

class L1Cache:
    """
    Semantic (bi-encoder + cross-encoder) cache — §6.1 L1.

    Backed by a SQLite vector index (SqliteVecIndex) and the persistent Store.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self._index = SqliteVecIndex(store.db_path)

    def lookup(
        self,
        icr: ICR,
        manifest: ConstraintManifest,
        class_result: ClassifierResult,
        ctx: Any,
    ) -> ICRResponse | None:
        """
        Return a cached ICRResponse or None.

        Returns None immediately (no embedding work) when L1 is disabled for
        this request class (§7.2 expected-value check).
        """
        # §7.2 gate — no embedding work when L1 is disabled
        matrix_row = getattr(ctx, "matrix_row", None)
        if matrix_row is None:
            from itol.routing.matrix import MATRIX
            matrix_row = MATRIX.get(class_result.primary)
        if matrix_row is None or matrix_row.l1 == L1Status.DISABLED:
            return None

        τ = matrix_row.l1_tau or 0.95

        ns = _namespace(icr)
        query_text = _build_query_text(icr)

        try:
            query_vec = embed([query_text], model="bge")[0]
        except Exception:
            return None

        # ANN search — top-5 within namespace
        try:
            candidates = self._index.search(query_vec, ns, top_k=5)
        except Exception:
            return None

        # Bi-encoder filter: cosine ≥ τ
        candidates = [(eid, score) for eid, score in candidates if score >= τ]
        if not candidates:
            return None

        # Fetch metadata for surviving candidates
        entries = []
        for eid, bi_cos in candidates:
            entry = self._store.get_l1_entry(eid, ns)
            if entry is not None:
                entries.append((eid, bi_cos, entry))

        if not entries:
            return None

        # Cross-encoder rerank
        cand_texts = [e["query_text"] for _, _, e in entries]
        try:
            cross_scores = rerank(query_text, cand_texts)
        except Exception:
            cross_scores = [0.0] * len(entries)

        new_numbers = _numbers_entities(manifest)

        for (eid, bi_cos, entry), cross_score in zip(entries, cross_scores):
            if cross_score < _CROSS_SCORE_FLOOR:
                continue

            # CR-8: cached numbers ⊇ new query numbers
            try:
                cached_numbers: set[str] = set(json.loads(entry.get("manifest_numbers", "[]")))
            except (json.JSONDecodeError, TypeError):
                cached_numbers = set()
            if not cached_numbers.issuperset(new_numbers):
                continue

            # All checks passed
            try:
                response = _deserialize_response(entry["response_json"])
            except Exception:
                continue

            # Update LRU-K access times
            try:
                self._store.update_l1_access(eid, ns, time.time())
            except Exception:
                pass

            return response

        return None

    def store(
        self,
        icr: ICR,
        manifest: ConstraintManifest,
        class_result: ClassifierResult,
        response: ICRResponse,
        ctx: Any,
    ) -> None:
        """
        Cache a response in L1.

        Negative caching prohibition (CR-20 pattern): silently drops entries
        where response.error is not None or finish_reason is non-terminal.
        L1 is disabled for classes with matrix_row.l1 == DISABLED.
        """
        # CR-20 pattern — never cache errors or partial responses
        if response.error is not None:
            return
        if response.finish_reason not in _CACHEABLE_FINISH_REASONS:
            return

        matrix_row = getattr(ctx, "matrix_row", None)
        if matrix_row is None:
            from itol.routing.matrix import MATRIX
            matrix_row = MATRIX.get(class_result.primary)
        if matrix_row is None or matrix_row.l1 == L1Status.DISABLED:
            return

        ns = _namespace(icr)
        query_text = _build_query_text(icr)

        try:
            query_vec = embed([query_text], model="bge")[0]
        except Exception:
            return

        entry_id = hashlib.sha256(
            (ns + "|" + query_text).encode()
        ).hexdigest()

        numbers_json = json.dumps(sorted(_numbers_entities(manifest)))
        tokens_saved = (response.usage.input_tokens or 0) if response.usage else 0

        try:
            self._index.add(entry_id, query_vec, ns)
            self._store.set_l1_entry(
                entry_id=entry_id,
                namespace=ns,
                response_json=_serialize_response(response),
                query_text=query_text,
                manifest_numbers_json=numbers_json,
                tokens_saved=tokens_saved,
            )
        except Exception:
            pass
