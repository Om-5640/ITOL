"""
S1 Deduplication — NEAR_LOSSLESS pass (§4, execution order position 2).

Algorithm:
  1. Group segments by SegmentType (only compare within-type).
  2. MinHash pre-filter at Jaccard ≥ 0.55 — cheap O(n²) in group size.
  3. Embed candidate pairs via BOW/ONNX/ST embedder (minilm model).
  4. Union-find clustering at cosine ≥ 0.92.
  5. Per cluster: keep in-prefix copy (CR-1), else keep earliest.
  6. Non-representatives: manifest guard → remove (exact) or pointer stub (near-dup).

CR-1  — keep in-prefix copy when a duplicate cluster spans the prefix boundary.
CR-3b — if full rollback is triggered, icr.raw is returned byte-identical.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import replace as dc_replace

import numpy as np

from itol.icr import ICR, ConstraintManifest, SegmentType, StrategyReport
from itol.segmenter import Segment
from itol.signals import estimate_token_count, jaccard_estimate, minhash_signature
from itol.strategies.base import OptimizationContext, Strategy, update_segment

# MinHash pre-filter threshold
_MINHASH_FLOOR = 0.55

# Embedding cosine threshold for clustering
_COSINE_FLOOR = 0.92

# Segment types excluded from deduplication
_SKIP_TYPES = frozenset({
    SegmentType.TOOL_SCHEMA,
    SegmentType.CODE_BLOCK,
})


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)

    def clusters(self, n: int) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[self.find(i)].append(i)
        return dict(groups)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class S1DedupeStrategy(Strategy):
    """NEAR_LOSSLESS deduplication via MinHash pre-filter + embedding cluster."""

    strategy_id = "S1"
    risk_class = "NEAR_LOSSLESS"

    def applies(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> bool:
        return len(segments) >= 2

    def apply(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> tuple[list[Segment], StrategyReport]:
        snapshot = list(segments)
        tokens_before = sum(
            (s.token_count or estimate_token_count(s.text)) for s in segments
        )

        result, touched, prefix_bytes_mutated = _deduplicate(
            segments, ctx, ctx.manifest
        )

        tokens_after = sum(
            (s.token_count or estimate_token_count(s.text)) for s in result
        )

        before_hashes = {s.segment_hash for s in segments}
        new_touched = [s.segment_hash for s in result if s.segment_hash not in before_hashes]

        report = self._make_report(
            segments_before=snapshot,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            touched=new_touched or touched,
            removed_spans=[],
            prefix_bytes_mutated=prefix_bytes_mutated,
            prefix_savings=tokens_before - tokens_after,
            activated=tokens_after < tokens_before,
        )
        return result, report


# ---------------------------------------------------------------------------
# Core deduplication
# ---------------------------------------------------------------------------

def _cumulative_byte_offsets(segments: list[Segment]) -> list[int]:
    """Return the byte offset of the START of each segment."""
    offsets = []
    cum = 0
    for s in segments:
        offsets.append(cum)
        cum += len(s.text.encode())
    return offsets


def _deduplicate(
    segments: list[Segment],
    ctx: OptimizationContext,
    manifest: ConstraintManifest,
) -> tuple[list[Segment], list[str], int]:
    """
    Returns (new_segments, touched_hashes, prefix_bytes_mutated).
    """
    from itol.embed.onnx_embedder import embed

    offsets = _cumulative_byte_offsets(segments)
    prefix_limit = ctx.prefix_cacheable_span_bytes

    # Group by type for comparison
    by_type: dict[SegmentType, list[int]] = defaultdict(list)
    for idx, seg in enumerate(segments):
        if seg.segment_type not in _SKIP_TYPES:
            by_type[seg.segment_type].append(idx)

    # Track which global indices to replace
    replacements: dict[int, str | None] = {}  # None = exact dup (silent remove)

    for seg_type, indices in by_type.items():
        if len(indices) < 2:
            continue

        type_segs = [segments[i] for i in indices]

        # Step 1: MinHash pre-filter
        sigs = [minhash_signature(s.text) for s in type_segs]
        candidate_pairs: list[tuple[int, int]] = []
        for a in range(len(type_segs)):
            for b in range(a + 1, len(type_segs)):
                j = jaccard_estimate(sigs[a], sigs[b])
                if j >= _MINHASH_FLOOR:
                    candidate_pairs.append((a, b))

        if not candidate_pairs:
            continue

        # Step 2: Embed & cluster
        try:
            vecs = embed([s.text for s in type_segs], model="minilm")
        except Exception:
            continue

        uf = _UnionFind(len(type_segs))
        for a, b in candidate_pairs:
            if _cosine(vecs[a], vecs[b]) >= _COSINE_FLOOR:
                uf.union(a, b)

        # Step 3: Process each cluster
        for root, members in uf.clusters(len(type_segs)).items():
            if len(members) < 2:
                continue

            global_members = [indices[m] for m in members]

            # CR-1: pick representative — in-prefix first, then earliest
            rep_global = _pick_representative(
                global_members, offsets, prefix_limit
            )

            rep_seg = segments[rep_global]

            for gi in global_members:
                if gi == rep_global:
                    continue
                cand_seg = segments[gi]

                # Manifest guard: block removal if manifest item only in this seg
                if _manifest_item_would_be_lost(cand_seg, rep_seg, manifest):
                    continue

                # CR-2: skip if removing this prefix segment is unprofitable
                savings = cand_seg.token_count or estimate_token_count(cand_seg.text)
                if not _seg_prefix_safe(gi, offsets, savings, ctx):
                    continue

                if cand_seg.segment_hash == rep_seg.segment_hash:
                    # Exact duplicate — silent remove
                    replacements[gi] = None
                else:
                    # Near-duplicate — pointer stub
                    replacements[gi] = f"[dup of seg_{rep_seg.segment_hash[:8]}]"

    # Build result
    result: list[Segment] = []
    touched: list[str] = []
    prefix_bytes_mutated = 0

    for i, seg in enumerate(segments):
        if i not in replacements:
            result.append(seg)
            continue

        replacement = replacements[i]
        in_prefix = offsets[i] < prefix_limit

        if replacement is None:
            # Exact dup: remove silently
            if in_prefix:
                prefix_bytes_mutated += len(seg.text.encode())
            touched.append(seg.segment_hash)
            # Skip (don't append) — removed
        else:
            # Near-dup: replace with pointer
            new_seg = update_segment(seg, replacement)
            if in_prefix:
                prefix_bytes_mutated += len(seg.text.encode())
            touched.append(new_seg.segment_hash)
            result.append(new_seg)

    return result, touched, prefix_bytes_mutated


def _pick_representative(
    global_indices: list[int],
    offsets: list[int],
    prefix_limit: int,
) -> int:
    """
    CR-1: prefer the in-prefix copy; among ties, pick the earliest (lowest index).
    """
    in_prefix = [gi for gi in global_indices if offsets[gi] < prefix_limit]
    candidates = in_prefix if in_prefix else global_indices
    return min(candidates)


def _seg_prefix_safe(
    seg_idx: int,
    offsets: list[int],
    savings_tokens: int,
    ctx: OptimizationContext,
) -> bool:
    span_start = offsets[seg_idx]
    if span_start >= ctx.prefix_cacheable_span_bytes:
        return True
    if ctx.prefix_cacheable_span_bytes == 0 or ctx.provider_cache_value <= 0:
        return True
    savings_usd = savings_tokens * ctx.input_price_per_token
    return savings_usd > ctx.provider_cache_value


def _manifest_item_would_be_lost(
    removed_seg: Segment,
    kept_seg: Segment,
    manifest: ConstraintManifest,
) -> bool:
    """
    Return True if removing `removed_seg` would lose a manifest item whose
    value is present in `removed_seg` but NOT in `kept_seg`.
    """
    for item in manifest.items:
        if item.value and item.value in removed_seg.text:
            if item.value not in kept_seg.text:
                return True
    return False
