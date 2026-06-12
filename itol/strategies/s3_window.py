"""
S3 Dynamic Context Windowing — LOSSY-BOUNDED (§4, execution order position 4).

Chunks RETRIEVED_DOC and STRUCTURED_DATA segments; scores each chunk; greedily
keeps the highest-scoring chunks until the relevance-mass floor is met AND
every manifest entity is covered.

Chunk scoring:
    score = 0.7 × cos_sim + 0.2 × bm25_normalised + 0.1 × position_prior

Position prior: 1.0 for first and last chunk, linear decay to 0.3 for middle.

CR-4 — hard constraint INSIDE the greedy loop:
    A stopping condition triggers only when BOTH conditions hold:
      (kept_mass / total_mass) >= mass_floor  AND
      every manifest entity value appears in the kept text.
    Neither condition alone is sufficient.

CR-21 — governing-span neighbor rule (post-greedy):
    For every manifest item whose value is found in a kept chunk AND whose
    governing_span text overlaps a dropped neighboring chunk (±1 chunk), that
    neighbor is force-included.

Reassembly: non-adjacent kept chunks are separated by `[…]` markers.

Activation:
    context_tokens > 1.5 × class_budget   (class_budget from ClassConfig.s3_class_budget)
    AND at least one eligible segment exists
    AND matrix allows S3 (not DISABLED)
    AND ClassConfig.s3_enabled is True
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import replace as dc_replace
from typing import NamedTuple

import numpy as np

from itol.embed.onnx_embedder import cosine, embed
from itol.icr import (
    ConstraintManifest,
    ICR,
    ManifestItem,
    SegmentType,
    StrategyReport,
)
from itol.routing.matrix import StrategyStatus
from itol.segmenter import Segment
from itol.signals import estimate_token_count
from itol.strategies.base import OptimizationContext, Strategy, update_segment

# Eligible segment types for chunking
_ELIGIBLE_TYPES = {SegmentType.RETRIEVED_DOC, SegmentType.STRUCTURED_DATA}

# BM25 hyper-parameters
_BM25_K1 = 1.5
_BM25_B  = 0.75

# Position prior range
_POS_PRIOR_MAX = 1.0
_POS_PRIOR_MIN = 0.3

# Ellipsis marker inserted between non-adjacent kept chunks
_ELLIPSIS = "[…]"


# ---------------------------------------------------------------------------
# Chunk representation
# ---------------------------------------------------------------------------

class _Chunk(NamedTuple):
    seg_idx: int        # index into the segments list
    chunk_idx: int      # position within this segment's chunk list
    n_chunks: int       # total chunks in this segment (for position prior)
    text: str
    token_count: int


# ---------------------------------------------------------------------------
# BM25 — pure Python, no dependencies
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class _BM25:
    def __init__(self, corpus: list[list[str]], k1: float = _BM25_K1, b: float = _BM25_B) -> None:
        self.k1 = k1
        self.b  = b
        self.N  = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.N, 1)
        # Term frequency per document
        self._tf: list[dict[str, int]] = []
        for doc in corpus:
            tf: dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
        # Document frequency
        self._df: dict[str, int] = {}
        for tf in self._tf:
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        dl = sum(self._tf[doc_idx].values())
        tf_map = self._tf[doc_idx]
        s = 0.0
        for t in query_tokens:
            tf = tf_map.get(t, 0)
            if tf == 0:
                continue
            idf = math.log((self.N - self._df.get(t, 0) + 0.5) / (self._df.get(t, 0) + 0.5) + 1)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            s += idf * num / den
        return s

    def score_all(self, query_tokens: list[str]) -> list[float]:
        return [self.score(query_tokens, i) for i in range(self.N)]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_tokens: int = 256, overlap_tokens: int = 32) -> list[str]:
    """
    Split `text` into overlapping windows of ~`chunk_tokens` tokens.

    Uses the 4-char/token heuristic to keep things dependency-free.
    Returns at least one chunk.
    """
    chars_per_chunk  = chunk_tokens  * 4
    chars_per_overlap = overlap_tokens * 4
    step = max(chars_per_chunk - chars_per_overlap, 1)
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chars_per_chunk]
        chunks.append(chunk)
        start += step
        if start >= len(text):
            break
    return chunks or [text]


def _position_prior(chunk_idx: int, n_chunks: int) -> float:
    """1.0 for first/last chunk; linear decay to 0.3 for middle."""
    if n_chunks <= 1:
        return _POS_PRIOR_MAX
    if chunk_idx == 0 or chunk_idx == n_chunks - 1:
        return _POS_PRIOR_MAX
    mid = (n_chunks - 1) / 2
    dist_from_edge = min(chunk_idx, n_chunks - 1 - chunk_idx)
    # dist_from_edge ranges from 1 (just inside edge) to mid
    frac = (dist_from_edge - 1) / max(mid - 1, 1)
    return _POS_PRIOR_MAX - frac * (_POS_PRIOR_MAX - _POS_PRIOR_MIN)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class S3WindowStrategy(Strategy):
    """
    Dynamic context windowing — LOSSY-BOUNDED.

    See module docstring for full spec.
    """

    strategy_id = "S3"
    risk_class  = "LOSSY_BOUNDED"

    def applies(self, icr: ICR, segments: list[Segment], ctx: OptimizationContext) -> bool:
        # Must not be DISABLED in matrix
        if ctx.matrix_row.strategies.get("S3") == StrategyStatus.DISABLED:
            return False
        # Must be enabled in class config
        cls_cfg = ctx.config.class_configs.get(ctx.request_class)
        if cls_cfg is not None and not cls_cfg.s3_enabled:
            return False
        # At least one eligible segment
        if not any(s.segment_type in _ELIGIBLE_TYPES for s in segments):
            return False
        # Context must exceed 1.5 × class_budget
        budget = 4000
        if cls_cfg is not None:
            budget = cls_cfg.s3_class_budget
        activation_mul = ctx.config.strategies.s3_activation_multiplier
        context_tokens = ctx.signals.token_count
        return context_tokens > activation_mul * budget

    def apply(
        self,
        icr: ICR,
        segments: list[Segment],
        ctx: OptimizationContext,
    ) -> tuple[list[Segment], StrategyReport]:
        snapshot = list(segments)
        tokens_before = sum(s.token_count or estimate_token_count(s.text) for s in segments)

        cls_cfg = ctx.config.class_configs.get(ctx.request_class)
        # Mass floor: from matrix restrictions, else class config, else 0.90
        mass_floor = ctx.matrix_row.restrictions.get("S3")
        if mass_floor is None:
            if cls_cfg is not None:
                mass_floor = cls_cfg.s3_mass_floor
            else:
                mass_floor = 0.90

        chunk_tokens   = ctx.config.strategies.s3_chunk_size_tokens
        overlap_tokens = ctx.config.strategies.s3_chunk_overlap_tokens

        # Build query text from last user message
        query_text = _build_query_text(icr)
        query_tokens = _tokenize(query_text)

        # Gather all chunks from eligible segments
        all_chunks: list[_Chunk] = []
        for seg_idx, seg in enumerate(segments):
            if seg.segment_type not in _ELIGIBLE_TYPES:
                continue
            raw_chunks = _chunk_text(seg.text, chunk_tokens, overlap_tokens)
            nc = len(raw_chunks)
            for ci, ctxt in enumerate(raw_chunks):
                tc = estimate_token_count(ctxt)
                all_chunks.append(_Chunk(
                    seg_idx=seg_idx,
                    chunk_idx=ci,
                    n_chunks=nc,
                    text=ctxt,
                    token_count=tc,
                ))

        if not all_chunks:
            report = self._make_report(
                snapshot, tokens_before, tokens_before, [], [], 0, 0, activated=False
            )
            return segments, report

        # Build BM25 corpus and get scores
        corpus_tokens = [_tokenize(c.text) for c in all_chunks]
        bm25 = _BM25(corpus_tokens)
        bm25_raw = bm25.score_all(query_tokens)
        bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_norm = [v / bm25_max for v in bm25_raw]

        # Build embedding similarity
        query_vec = embed([query_text[:512]], model="minilm")[0]
        chunk_vecs = embed([c.text[:512] for c in all_chunks], model="minilm")

        # Compute composite scores
        scored: list[tuple[float, int]] = []  # (score, chunk_list_index)
        for i, chunk in enumerate(all_chunks):
            cos_sim = float(cosine(query_vec, chunk_vecs[i]))
            pos = _position_prior(chunk.chunk_idx, chunk.n_chunks)
            score = (
                ctx.config.strategies.s3_relevance_weight_semantic * cos_sim
                + ctx.config.strategies.s3_relevance_weight_bm25   * bm25_norm[i]
                + ctx.config.strategies.s3_relevance_weight_position * pos
            )
            scored.append((score, i))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Total mass = sum of token counts of all chunks for eligible segments
        total_mass = sum(c.token_count for c in all_chunks)

        # CR-4: greedy loop — stop only when BOTH conditions hold
        manifest = ctx.manifest
        kept_indices: set[int] = set()

        def _kept_text() -> str:
            return "\n".join(all_chunks[i].text for i in sorted(kept_indices))

        def _entities_covered() -> bool:
            ktext = _kept_text()
            for item in manifest.items:
                if item.item_type == ManifestItem.ItemType.ENTITY or item.item_type == ManifestItem.ItemType.QUERY_TERM:
                    if item.value not in ktext:
                        return False
            return True

        def _mass_fraction() -> float:
            return sum(all_chunks[i].token_count for i in kept_indices) / max(total_mass, 1)

        for score_val, idx in scored:
            kept_indices.add(idx)
            # CR-4: check BOTH conditions inside the loop
            if _mass_fraction() >= mass_floor and _entities_covered():
                break
        else:
            # Exhausted all chunks — force-add any chunk covering uncovered entities
            pass

        # CR-21: governing-span neighbor force-include
        # Group chunks by seg_idx for neighbor lookups
        seg_chunk_map: dict[int, list[int]] = defaultdict(list)
        for list_idx, chunk in enumerate(all_chunks):
            seg_chunk_map[chunk.seg_idx].append(list_idx)

        for list_idx in list(kept_indices):
            chunk = all_chunks[list_idx]
            for item in manifest.items:
                if not item.governing_span:
                    continue
                if item.value not in chunk.text:
                    continue
                # Check if governing_span text appears in a dropped neighbor
                neighbors = _get_neighbors(list_idx, seg_chunk_map[chunk.seg_idx])
                for neighbor_idx in neighbors:
                    if neighbor_idx in kept_indices:
                        continue
                    neighbor_text = all_chunks[neighbor_idx].text
                    if _spans_overlap(item.governing_span, neighbor_text):
                        kept_indices.add(neighbor_idx)

        # Reassemble each eligible segment from kept chunks
        new_segments = list(segments)
        touched: list[str] = []

        # Group kept chunk list-indices by seg_idx
        kept_by_seg: dict[int, list[int]] = defaultdict(list)
        for list_idx in kept_indices:
            kept_by_seg[all_chunks[list_idx].seg_idx].append(list_idx)

        for seg_idx in sorted(kept_by_seg.keys()):
            original_seg = segments[seg_idx]
            chunk_list_indices = sorted(kept_by_seg[seg_idx])
            # Map to chunk_idx (position in segment)
            seg_all_indices = seg_chunk_map[seg_idx]  # all list-indices for this seg in order
            reassembled = _reassemble(all_chunks, seg_all_indices, chunk_list_indices)
            if reassembled != original_seg.text:
                new_seg = update_segment(original_seg, reassembled)
                new_segments[seg_idx] = new_seg
                touched.append(original_seg.segment_hash)

        tokens_after = sum(
            s.token_count or estimate_token_count(s.text) for s in new_segments
        )
        report = self._make_report(
            snapshot,
            tokens_before,
            tokens_after,
            touched,
            [],
            prefix_bytes_mutated=0,
            prefix_savings=0,
            activated=len(touched) > 0,
        )
        return new_segments, report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_query_text(icr: ICR) -> str:
    """Build a short query text from the last user message."""
    for msg in reversed(icr.messages):
        if msg.role == "user":
            return msg.text_content()[:512]
    return ""


def _get_neighbors(list_idx: int, seg_all_indices: list[int]) -> list[int]:
    """
    Return the list-indices of the chunks immediately before and after
    `list_idx` within the same segment (±1 neighbors).
    """
    try:
        pos = seg_all_indices.index(list_idx)
    except ValueError:
        return []
    neighbors = []
    if pos > 0:
        neighbors.append(seg_all_indices[pos - 1])
    if pos < len(seg_all_indices) - 1:
        neighbors.append(seg_all_indices[pos + 1])
    return neighbors


def _spans_overlap(governing_span: str, candidate_text: str) -> bool:
    """
    Check whether any token from governing_span appears in candidate_text.
    The governing_span is the ±1-sentence qualifier window from the manifest
    item; we check for substring presence as a proxy for overlap.
    """
    gs_tokens = set(governing_span.lower().split())
    ct_lower = candidate_text.lower()
    # Use short tokens (len ≥ 4) to avoid noise from stop-words
    meaningful = [t for t in gs_tokens if len(t) >= 4]
    if not meaningful:
        return governing_span.lower() in ct_lower
    return any(t in ct_lower for t in meaningful)


def _reassemble(
    all_chunks: list[_Chunk],
    seg_all_indices: list[int],
    kept_list_indices: list[int],
) -> str:
    """
    Reassemble the kept chunks for one segment, inserting `[…]` between
    non-adjacent kept chunks.  `seg_all_indices` is the ordered list of
    list-indices for this segment; `kept_list_indices` is the subset that
    is kept (already sorted ascending by list-index).
    """
    if not kept_list_indices:
        return ""

    # Convert list-indices to positions within seg_all_indices
    pos_map = {li: pos for pos, li in enumerate(seg_all_indices)}
    kept_positions = sorted(pos_map[li] for li in kept_list_indices)

    parts: list[str] = []
    prev_pos: int | None = None
    for pos in kept_positions:
        if prev_pos is not None and pos > prev_pos + 1:
            parts.append(_ELLIPSIS)
        parts.append(all_chunks[seg_all_indices[pos]].text)
        prev_pos = pos

    return "\n".join(parts)
