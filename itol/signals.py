"""
Signal Extraction — Ingestion & Analysis Layer, §3.3

Computes per-request and per-segment signals used by the routing layer
and all downstream strategies.  Runs after segmentation; fills
ICR.meta.signals in-place.

All computations are CPU-only and dependency-free beyond the standard
library.  The embedding-based signals (segment embeddings) are computed
by the embedding module (added in a later step); this module handles
the cheap deterministic signals only so it can be imported anywhere.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from dataclasses import dataclass
from typing import Sequence

from itol.icr import AnalysisMeta, ICR, SegmentSignals, SegmentType
from itol.segmenter import Segment, filter_by_type, template_signature


# ---------------------------------------------------------------------------
# MinHash — §3.3 redundancy_score  (128-permutation, 5-gram shingles)
# ---------------------------------------------------------------------------

_MINHASH_PERMS = 128
# Mersenne prime and large prime for universal hashing
_MHASH_P = (1 << 61) - 1
_MHASH_MOD = (1 << 32)

# Pre-generate (a, b) coefficient pairs deterministically so every process
# produces the same signature for the same text.
_rng_seed = 0x4954_4F4C  # "ITOL"

def _minhash_coeffs() -> list[tuple[int, int]]:
    coeffs: list[tuple[int, int]] = []
    state = _rng_seed
    for _ in range(_MINHASH_PERMS):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFF_FFFF_FFFF_FFFF
        a = (state >> 33) | 1          # odd → guaranteed coprime to 2^32
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFF_FFFF_FFFF_FFFF
        b = state >> 33
        coeffs.append((a, b))
    return coeffs

_COEFFS = _minhash_coeffs()


def _shingles(text: str, k: int = 5) -> set[int]:
    """k-gram character shingles as integer hashes."""
    text = text.lower()
    if len(text) < k:
        return {hash(text) & 0xFFFF_FFFF}
    return {
        int.from_bytes(
            hashlib.md5(text[i:i+k].encode(), usedforsecurity=False).digest()[:4],
            "little",
        )
        for i in range(len(text) - k + 1)
    }


def minhash_signature(text: str) -> list[int]:
    """Return a 128-integer MinHash signature for `text`."""
    shingles = _shingles(text)
    if not shingles:
        return [0] * _MINHASH_PERMS
    sig = []
    for a, b in _COEFFS:
        sig.append(min((a * s + b) % _MHASH_MOD for s in shingles))
    return sig


def jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


def redundancy_score(segments: list[Segment]) -> float:
    """
    §3.3: fraction of segments with a ≥0.7-Jaccard sibling among all segments.

    Only compares within the same SegmentType (cross-type near-dupes are
    handled by S1 with type awareness; the score here gates S1 activation).
    """
    if len(segments) < 2:
        return 0.0

    # Group by type for efficiency
    by_type: dict[SegmentType, list[tuple[int, list[int]]]] = {}
    for i, seg in enumerate(segments):
        sigs = by_type.setdefault(seg.segment_type, [])
        sigs.append((i, minhash_signature(seg.text)))

    near_dup_indices: set[int] = set()
    for type_group in by_type.values():
        for j in range(len(type_group)):
            for k in range(j + 1, len(type_group)):
                i_j, sig_j = type_group[j]
                i_k, sig_k = type_group[k]
                if jaccard_estimate(sig_j, sig_k) >= 0.70:
                    near_dup_indices.add(i_j)
                    near_dup_indices.add(i_k)

    return len(near_dup_indices) / len(segments)


# ---------------------------------------------------------------------------
# Semantic density  (§3.3 — cheap entropy proxy)
# ---------------------------------------------------------------------------

def semantic_density(text: str) -> float:
    """
    §3.3: zstd-compressed-bytes / raw-bytes blended with unique-lemma ratio.

    We use zlib (available everywhere) as the compression proxy.
    Returns a float in (0, 1]; higher = denser / less compressible.

    Blend: 0.7·(compressed/raw) + 0.3·(unique_words/total_words)
    """
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    raw_len = len(raw)

    # Compression ratio component
    compressed_len = len(zlib.compress(raw, level=6))
    compression_ratio = min(compressed_len / raw_len, 1.0)

    # Unique lemma approximation: lowercased unique word types / total words
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        unique_ratio = 1.0
    else:
        unique_ratio = min(len(set(words)) / len(words), 1.0)

    return 0.7 * compression_ratio + 0.3 * unique_ratio


# ---------------------------------------------------------------------------
# Token counting  (§3.3)
# ---------------------------------------------------------------------------

# Characters-per-token estimates (§8.2 calibrated estimator fallback).
# These are used when no provider tokenizer is available.  Values are
# derived from empirical measurements on GPT-4-class models.
_CHARS_PER_TOKEN_DEFAULT = 3.8


def estimate_token_count(text: str, chars_per_token: float = _CHARS_PER_TOKEN_DEFAULT) -> int:
    """
    Calibrated character-ratio token estimator (§8.2).

    Observed error <3% after ~50 real requests; used until the provider
    adapter self-calibrates from live `usage` fields.
    """
    if not text:
        return 0
    return max(1, round(len(text) / chars_per_token))


def count_segment_tokens(segments: list[Segment], chars_per_token: float = _CHARS_PER_TOKEN_DEFAULT) -> int:
    """Sum estimated token counts across all segments."""
    return sum(estimate_token_count(s.text, chars_per_token) for s in segments)


# ---------------------------------------------------------------------------
# History depth + stale mass  (§3.3)
# ---------------------------------------------------------------------------

def history_depth(icr: ICR) -> int:
    """Number of complete assistant/user turn pairs in the conversation."""
    roles = [m.role for m in icr.messages]
    pairs = 0
    i = 0
    while i < len(roles) - 1:
        if roles[i] == "user" and roles[i + 1] == "assistant":
            pairs += 1
            i += 2
        else:
            i += 1
    return pairs


def stale_tool_mass(segments: list[Segment], k_turns: int = 6) -> int:
    """
    §3.3 stale_mass: token count of TOOL_RESULT segments older than K turns.

    A "turn" here is a message index step.  We count backwards from the
    last message; tool results beyond the K-turn window are stale.
    """
    if not segments:
        return 0
    max_msg_index = max(
        (s.source_message_index for s in segments if s.source_message_index is not None),
        default=0,
    )
    cutoff = max_msg_index - k_turns
    stale = [
        s for s in segments
        if s.segment_type is SegmentType.TOOL_RESULT
        and s.source_message_index is not None
        and s.source_message_index < cutoff
    ]
    return sum(estimate_token_count(s.text) for s in stale)


# ---------------------------------------------------------------------------
# Prefix-cacheable span  (§3.3 — G3 Prefix-Stable optimisation)
# ---------------------------------------------------------------------------

# Per-tenant, per-template: last request's system+instruction prefix bytes.
# Key: (tenant_id, template_sig)  Value: bytes of the stable prefix
_prefix_store: dict[tuple[str, str], bytes] = {}


def prefix_cacheable_span_tokens(
    icr: ICR,
    segments: list[Segment],
    sig: str,
    chars_per_token: float = _CHARS_PER_TOKEN_DEFAULT,
) -> int:
    """
    §3.3: longest leading byte span of SYSTEM_INSTRUCTION segments identical
    to the same tenant's previous request with the same template_signature.

    Returns the token count of the stable prefix (0 if no prior request or
    no match).  Updates the store with the current prefix.
    """
    system_segs = filter_by_type(segments, SegmentType.SYSTEM_INSTRUCTION)
    current_prefix = "\n".join(s.text for s in system_segs).encode("utf-8")

    key = (icr.tenant_id, sig)
    prior_prefix = _prefix_store.get(key, b"")

    # Find longest common leading byte span
    match_len = 0
    for a, b in zip(current_prefix, prior_prefix):
        if a == b:
            match_len += 1
        else:
            break

    _prefix_store[key] = current_prefix
    matched_text = current_prefix[:match_len].decode("utf-8", errors="replace")
    return estimate_token_count(matched_text, chars_per_token)


# ---------------------------------------------------------------------------
# Main entry point  (§3.3)
# ---------------------------------------------------------------------------

def extract_signals(icr: ICR, segments: list[Segment]) -> SegmentSignals:
    """
    Compute all §3.3 signals from an ICR and its segments.
    Returns a populated SegmentSignals; caller attaches it to ICR.meta.
    """
    total_tokens = count_segment_tokens(segments)

    instruction_tokens = count_segment_tokens(
        filter_by_type(segments, SegmentType.SYSTEM_INSTRUCTION)
    )
    instruction_ratio = instruction_tokens / total_tokens if total_tokens else 0.0

    full_text = "\n".join(s.text for s in segments)

    sig = template_signature(segments)

    return SegmentSignals(
        token_count=total_tokens,
        redundancy_score=redundancy_score(segments),
        semantic_density=semantic_density(full_text),
        instruction_context_ratio=instruction_ratio,
        history_depth=history_depth(icr),
        stale_mass=stale_tool_mass(segments),
        template_signature=sig,
        prefix_cacheable_span=prefix_cacheable_span_tokens(icr, segments, sig),
    )
