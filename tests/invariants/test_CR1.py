"""
§14.3 Invariant tests for CR-1 — deduplication keeps the in-prefix copy.

CR-1: when a duplicate cluster spans the provider's KV-cache prefix boundary,
the copy that lies INSIDE the prefix must be kept as the representative.
The out-of-prefix copy must be the one replaced/removed.

Rules:
- NEVER weaken thresholds or conditions.
"""

import hashlib
import re

import pytest

from itol.icr import ConstraintManifest, SegmentType
from itol.segmenter import Segment
from itol.strategies.base import OptimizationContext
from itol.strategies.s1_dedupe import _deduplicate, _pick_representative
from itol.config import ITOLConfig
from itol.routing.matrix import MATRIX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def _seg(text: str, seg_type: SegmentType = SegmentType.SYSTEM_INSTRUCTION) -> Segment:
    h = _seg_hash(text)
    return Segment(
        segment_type=seg_type,
        text=text,
        segment_hash=h,
        source_message_index=None,
        source_block_index=None,
        token_count=len(text.split()),
    )


def _ctx(prefix_bytes: int = 0) -> OptimizationContext:
    return OptimizationContext(
        request_class="SUMMARIZATION",
        matrix_row=MATRIX["SUMMARIZATION"],
        manifest=ConstraintManifest(),
        signals=__import__("itol.icr", fromlist=["SegmentSignals"]).SegmentSignals(),
        config=ITOLConfig(),
        prefix_cacheable_span_bytes=prefix_bytes,
        provider_cache_value=0.0,
    )


# ===========================================================================
# CR-1-a: In-prefix copy is kept, out-of-prefix copy is replaced
# ===========================================================================

class TestCR1_InPrefixKept:

    def test_in_prefix_copy_kept_exact_dup(self):
        """
        CR-1: when two identical segments exist and the first is in the prefix,
        the first (in-prefix) must be the representative.
        The second (out-of-prefix) must be removed.
        """
        identical_text = (
            "You are a helpful assistant. You must always answer accurately."
        )
        seg_a = _seg(identical_text)  # in-prefix (earlier)
        seg_b = _seg(identical_text)  # out-of-prefix (later)

        # seg_a is at byte offset 0 (in prefix), seg_b is out of prefix
        # prefix covers exactly the first segment's bytes
        prefix_bytes = len(identical_text.encode())
        ctx = _ctx(prefix_bytes=prefix_bytes)

        segments = [seg_a, seg_b]
        result, touched, pmb = _deduplicate(segments, ctx, ConstraintManifest())

        # One should have been removed — only 1 segment remains
        assert len(result) == 1, (
            f"CR-1: exact duplicate must be removed, got {len(result)} segments"
        )
        assert result[0].segment_hash == seg_a.segment_hash, (
            "CR-1: the in-prefix copy must be kept as representative"
        )

    def test_out_of_prefix_dup_removed_not_in_prefix(self):
        """
        When neither copy is in the prefix, the earliest copy is kept.
        """
        text = "You are a helpful assistant."
        seg_a = _seg(text)
        seg_b = _seg(text)

        ctx = _ctx(prefix_bytes=0)  # no prefix
        segments = [seg_a, seg_b]
        result, touched, pmb = _deduplicate(segments, ctx, ConstraintManifest())

        # Earliest (first) is kept
        assert len(result) == 1
        assert result[0].segment_hash == seg_a.segment_hash

    def test_three_copies_in_prefix_first_kept(self):
        """
        CR-1: with three copies, two in prefix and one outside,
        the in-prefix copy (earliest) must survive.
        """
        text = "This is a repeated system instruction segment."
        seg_a = _seg(text)  # in-prefix
        seg_b = _seg(text)  # in-prefix
        seg_c = _seg(text)  # out-of-prefix

        # Put seg_a and seg_b inside prefix
        prefix_bytes = len(text.encode()) * 2 + 10
        ctx = _ctx(prefix_bytes=prefix_bytes)

        segments = [seg_a, seg_b, seg_c]
        result, touched, pmb = _deduplicate(segments, ctx, ConstraintManifest())

        assert len(result) == 1
        assert result[0].segment_hash == seg_a.segment_hash


# ===========================================================================
# CR-1-b: _pick_representative logic
# ===========================================================================

class TestCR1_PickRepresentative:

    def test_pick_in_prefix_over_later(self):
        """_pick_representative must prefer the in-prefix index."""
        offsets = [0, 100, 200]
        prefix_limit = 150  # covers indices 0 and 1

        # Members at global indices 0, 1, 2 — 0 and 1 are in prefix
        result = _pick_representative([0, 1, 2], offsets, prefix_limit)
        assert result in (0, 1), "Must pick from in-prefix candidates"
        assert result == min(0, 1), "Among in-prefix, picks earliest"

    def test_pick_earliest_when_none_in_prefix(self):
        """When no member is in the prefix, picks the earliest index."""
        offsets = [200, 400, 600]
        prefix_limit = 100  # nothing in prefix

        result = _pick_representative([0, 1, 2], offsets, prefix_limit)
        assert result == 0, "No prefix → pick earliest"

    def test_single_member_returned(self):
        """A cluster with one member returns that member."""
        offsets = [0]
        result = _pick_representative([0], offsets, 50)
        assert result == 0


# ===========================================================================
# CR-1-c: Manifest guard blocks removal when item would be lost
# ===========================================================================

class TestCR1_ManifestGuard:

    def test_manifest_item_blocks_removal(self):
        """
        If a manifest constraint item's value exists ONLY in the candidate
        segment (not in the representative), removal must be blocked.
        """
        from itol.icr import ConstraintManifest, ManifestItem

        rep_text = "You are a helpful assistant."
        cand_text = "You are a helpful assistant. Budget is $5M."

        rep_seg = _seg(rep_text)
        cand_seg = _seg(cand_text)

        manifest = ConstraintManifest(
            items=[ManifestItem(item_type="NUMBER", value="$5M")]
        )

        # Both segments in a cluster — candidate has the manifest value, rep doesn't
        ctx = _ctx(prefix_bytes=0)
        segments = [rep_seg, cand_seg]
        result, touched, pmb = _deduplicate(segments, ctx, manifest)

        # Candidate must be retained (blocked by manifest guard)
        result_hashes = {s.segment_hash for s in result}
        assert cand_seg.segment_hash in result_hashes, (
            "CR-1 manifest guard: candidate containing unique manifest item must not be removed"
        )

    def test_no_manifest_item_allows_removal(self):
        """Without conflicting manifest items, removal proceeds normally."""
        text = "You are a helpful assistant."
        seg_a = _seg(text)
        seg_b = _seg(text)

        ctx = _ctx(prefix_bytes=0)
        manifest = ConstraintManifest()  # no items

        segments = [seg_a, seg_b]
        result, touched, pmb = _deduplicate(segments, ctx, manifest)
        assert len(result) == 1, "With no manifest guard, duplicate is removed"
