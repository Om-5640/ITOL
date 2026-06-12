"""
S4 Retrieval-Augmented Context Replacement — LOSSY-BOUNDED (§4).

When a large document (≥ s4_min_doc_tokens) recurs in the SAME conversation
(seen ≥ s4_min_appearances times), replace the repeated inline copy with:
    1-sentence header + top-k retrieved chunks (scored by S3's scorer)

This avoids re-sending the full doc on every turn while preserving the parts
most relevant to the current query.

Detection
---------
A doc is considered "the same" if:
    segment_hash matches exactly (handles normalised whitespace), OR
    cos(embed(doc), embed(stored)) ≥ 0.97 (handles minor edits across turns)

Retrieval confidence floor
--------------------------
If top-1 chunk score < s4_retrieval_confidence_floor (default 0.35) → fall
back to S3-style 90%-mass inclusion of the full doc.

Full-document intent gate
--------------------------
If the user query contains phrases like "whole document", "entire report",
"all pages", "every section" → S4 is globally disabled for this request
(the segment type RETRIEVED_DOC is left for S3 to handle normally).

Matrix constraints
------------------
S4 is DISABLED for: EXTRACTION, REASONING, GENERATION_CREATIVE,
                    CLASSIFICATION_SHORT, CHAT_OPEN
S4 is ALLOWED for:  SUMMARIZATION, GENERATION_FACTUAL, AGENT_TOOL_LOOP

CR-9 link
---------
When a doc's content has changed since it was first stored (different content
→ different doc_hash but cosine ≥ 0.97), call store.invalidate_l2_by_doc()
on the old hash to tombstone any L2 plans that depended on the old version.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

import numpy as np

from itol.embed.onnx_embedder import cosine, embed
from itol.icr import ICR, SegmentType, StrategyReport
from itol.routing.matrix import StrategyStatus
from itol.segmenter import Segment
from itol.signals import estimate_token_count
from itol.strategies.base import OptimizationContext, Strategy, update_segment
# Reuse S3's chunker and scorer (do not duplicate)
from itol.strategies.s3_window import (
    _BM25,
    _chunk_text,
    _position_prior,
    _tokenize,
)

if TYPE_CHECKING:
    from itol.cache.store import Store

_log = logging.getLogger(__name__)

# Segment types eligible for S4 replacement
_ELIGIBLE_TYPES = {SegmentType.RETRIEVED_DOC, SegmentType.STRUCTURED_DATA}

# Cosine threshold for near-duplicate detection
_NEAR_DUP_COSINE = 0.97

# Full-document intent phrases — any match → disable S4 for this request
_FULL_DOC_INTENT_PATTERNS = re.compile(
    r"\b(whole\s+document|entire\s+report|all\s+pages|every\s+section"
    r"|full\s+document|complete\s+report|read\s+everything|entire\s+text)\b",
    re.IGNORECASE,
)

# S3-style fallback mass floor when top-1 < confidence floor
_FALLBACK_MASS_FLOOR = 0.90
# Top-k chunks to include in replacement header
_TOP_K = 5


def _doc_hash(text: str) -> str:
    """Content-addressed hash of a document."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _title_guess(text: str) -> str:
    """
    Return a short title for a document:
    - First line if it looks like a title (≤ 10 words, no sentence-ending punctuation)
    - Otherwise first 5 words + '...'
    """
    first_line = text.split("\n", 1)[0].strip()
    words = first_line.split()
    if words and len(words) <= 10 and not first_line.rstrip().endswith((".", "?", "!")):
        return first_line
    headline_words = text.split()[:5]
    return " ".join(headline_words) + "..."


def _full_doc_intent(query_text: str) -> bool:
    """Return True if the query asks for the whole document."""
    return bool(_FULL_DOC_INTENT_PATTERNS.search(query_text))


def _extract_query_text(icr: ICR) -> str:
    """Pull the last user message text from the ICR."""
    for msg in reversed(icr.messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                return msg.content
            if isinstance(msg.content, list):
                parts = [b.text for b in msg.content if hasattr(b, "text")]
                return " ".join(parts)
    return ""


def _score_chunks(
    chunks: list[str],
    query_text: str,
) -> list[float]:
    """
    Score chunks using S3's formula:
        score = 0.7 × cos_sim + 0.2 × bm25_norm + 0.1 × position_prior
    """
    if not chunks:
        return []

    n = len(chunks)
    tokenised = [_tokenize(c) for c in chunks]
    bm25 = _BM25(tokenised)
    q_tokens = _tokenize(query_text)

    # BM25 scores, normalised to [0,1]
    raw_bm25 = bm25.score_all(q_tokens)
    max_bm25 = max(raw_bm25) if raw_bm25 else 1.0
    bm25_norm = [s / max(max_bm25, 1e-9) for s in raw_bm25]

    # Embed query and all chunks in a single call so the BOW fallback
    # builds a shared vocabulary (consistent vector dimensions).
    all_embs = embed([query_text] + chunks)
    query_emb = all_embs[0]
    chunk_embs = all_embs[1:]

    scores = []
    for i, chunk in enumerate(chunks):
        cos = float(cosine(query_emb, chunk_embs[i]))
        pos = _position_prior(i, n)
        score = 0.70 * cos + 0.20 * bm25_norm[i] + 0.10 * pos
        scores.append(score)
    return scores


def _s3_mass_fallback(doc_text: str, query_text: str, chunk_tokens: int = 256) -> str:
    """
    Fall back to S3-style 90%-mass inclusion when retrieval confidence is low.
    Returns the kept text (may be shorter than original).
    """
    chunks = _chunk_text(doc_text, chunk_tokens=chunk_tokens)
    if not chunks:
        return doc_text

    scores = _score_chunks(chunks, query_text)
    total_tokens = sum(estimate_token_count(c) for c in chunks)
    floor_tokens = int(_FALLBACK_MASS_FLOOR * total_tokens)

    # Sort by score descending, keep until mass floor met
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    kept_indices: set[int] = set()
    kept_tokens = 0
    for i in order:
        kept_indices.add(i)
        kept_tokens += estimate_token_count(chunks[i])
        if kept_tokens >= floor_tokens:
            break

    kept_in_order = sorted(kept_indices)
    parts = []
    prev = None
    for idx in kept_in_order:
        if prev is not None and idx > prev + 1:
            parts.append("[…]")
        parts.append(chunks[idx])
        prev = idx
    return "\n".join(parts)


class S4RACRStrategy(Strategy):
    """
    Retrieval-Augmented Context Replacement — LOSSY-BOUNDED.

    See module docstring for full specification.
    """

    strategy_id = "S4"
    risk_class  = "LOSSY_BOUNDED"

    def __init__(self, store: "Store | None" = None) -> None:
        self._store = store

    def applies(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> bool:
        # Matrix gate
        if ctx.matrix_row.strategies.get("S4") == StrategyStatus.DISABLED:
            return False
        # Class-config gate
        cls_cfg = ctx.config.class_configs.get(ctx.request_class)
        if cls_cfg is not None and not cls_cfg.s4_enabled:
            return False
        # Need conversation tracking
        if icr.conversation_id is None or self._store is None:
            return False
        # Full-document intent → let S3 handle the full doc
        query_text = _extract_query_text(icr)
        if _full_doc_intent(query_text):
            return False
        # At least one eligible segment above size floor
        min_tokens = ctx.config.strategies.s4_min_doc_tokens
        return any(
            s.segment_type in _ELIGIBLE_TYPES and s.token_count >= min_tokens
            for s in segments
        )

    def apply(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> tuple[list[Segment], StrategyReport]:
        segments_before = list(segments)
        tokens_before = sum(s.token_count for s in segments)

        if not self.applies(icr, segments, ctx):
            return segments, self._make_report(
                segments_before, tokens_before, tokens_before,
                [], [], 0, 0, activated=False,
            )

        min_tokens = ctx.config.strategies.s4_min_doc_tokens
        conf_floor = ctx.config.strategies.s4_retrieval_confidence_floor
        conv_id = icr.conversation_id
        tenant_id = icr.tenant_id
        query_text = _extract_query_text(icr)

        new_segments = list(segments)
        touched: list[str] = []

        for seg_idx, seg in enumerate(segments):
            if seg.segment_type not in _ELIGIBLE_TYPES:
                continue
            if seg.token_count < min_tokens:
                continue

            hash_val = _doc_hash(seg.text)
            existing = self._store.get_doc(hash_val, tenant_id, conv_id)

            if existing is not None:
                # 2nd+ occurrence — activate replacement
                replacement_text = self._build_replacement(
                    doc_text=seg.text,
                    doc_hash=hash_val,
                    query_text=query_text,
                    conf_floor=conf_floor,
                )
                new_segments[seg_idx] = update_segment(seg, replacement_text)
                touched.append(seg.segment_hash)
                _log.debug("S4: replaced doc %s... in seg %d", hash_val[:8], seg_idx)
            else:
                # First occurrence — detect near-dups via cosine, then store
                old_hash = self._find_near_duplicate(
                    seg.text, tenant_id, conv_id
                )
                if old_hash is not None and old_hash != hash_val:
                    # Content changed since last time (CR-9)
                    invalidated = self._store.invalidate_l2_by_doc(old_hash)
                    _log.info(
                        "S4 CR-9: doc %s... updated; tombstoned %d L2 plans",
                        old_hash[:8], invalidated,
                    )
                    # Write new version
                    self._store.save_doc(hash_val, tenant_id, conv_id, seg.text, hash_val)
                    # Replace on this turn (near-dup counts as 2nd occurrence)
                    replacement_text = self._build_replacement(
                        doc_text=seg.text,
                        doc_hash=hash_val,
                        query_text=query_text,
                        conf_floor=conf_floor,
                    )
                    new_segments[seg_idx] = update_segment(seg, replacement_text)
                    touched.append(seg.segment_hash)
                else:
                    # Truly first occurrence — store for next turn
                    self._store.save_doc(hash_val, tenant_id, conv_id, seg.text, hash_val)

        tokens_after = sum(s.token_count for s in new_segments)
        activated = len(touched) > 0
        return new_segments, self._make_report(
            segments_before, tokens_before, tokens_after,
            touched, [], 0, 0, activated=activated,
            notes=f"replaced_docs={len(touched)}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_replacement(
        self,
        doc_text: str,
        doc_hash: str,
        query_text: str,
        conf_floor: float,
        chunk_tokens: int = 256,
    ) -> str:
        """
        Build the replacement text: header + top-k chunks.
        Falls back to S3-mass-floor approach when top-1 score < conf_floor.
        """
        chunks = _chunk_text(doc_text, chunk_tokens=chunk_tokens)
        if not chunks:
            return doc_text

        scores = _score_chunks(chunks, query_text)
        top_score = max(scores) if scores else 0.0

        if top_score < conf_floor:
            # Fall back: run S3-style mass inclusion
            return _s3_mass_fallback(doc_text, query_text, chunk_tokens)

        title = _title_guess(doc_text)
        header = (
            f'Document {doc_hash[:8]} "{title}", available — '
            f"relevant excerpts below"
        )

        # Top-k chunks in original order
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        top_k = sorted(order[:_TOP_K])
        parts = [header]
        prev = None
        for idx in top_k:
            if prev is not None and idx > prev + 1:
                parts.append("[…]")
            parts.append(chunks[idx])
            prev = idx
        return "\n".join(parts)

    def _find_near_duplicate(
        self,
        doc_text: str,
        tenant_id: str,
        conv_id: str,
    ) -> str | None:
        """
        Return the doc_hash of a previously stored doc with cosine ≥ 0.97,
        or None if no near-duplicate found.
        """
        stored = self._store.get_docs_for_conversation(tenant_id, conv_id)
        if not stored:
            return None
        try:
            query_emb = embed([doc_text])[0]
            for row in stored:
                stored_emb = embed([row["content"]])[0]
                if float(cosine(query_emb, stored_emb)) >= _NEAR_DUP_COSINE:
                    return row["doc_hash"]
        except Exception as exc:
            _log.debug("S4 near-dup cosine check failed: %s", exc)
        return None
