"""
§14.3 Invariant tests for CR-21 and CR-22  (§15.1 relation-aware manifest checking)

CR-21: qualifier tokens from governing_span must survive within ±1 sentence of
       the manifest item in the optimised prompt.
CR-22: polarity_guard hash must be identical pre/post optimisation for every item.

Rules:
- NEVER weaken a threshold to make a test pass.
- Every assertion encodes a spec requirement cited in the comments.
"""

import pytest

from itol.icr import ConstraintManifest, ManifestItem
from itol.analysis.manifest import (
    compute_polarity_guard,
    extract_manifest,
    polarity_intact,
    split_sentences,
)
from itol.icr import ICR, ContentBlock, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _icr(system_text: str, user_text: str) -> ICR:
    return ICR.create(
        provider="openai",
        model="gpt-4o",
        system=[ContentBlock.text(system_text)] if system_text else [],
        messages=[Message.user(user_text)],
        raw={},
    )


def _item_with_span(value: str, span: str) -> ManifestItem:
    return ManifestItem(
        item_type=ManifestItem.ItemType.NUMBER,
        value=value,
        governing_span=span,
        polarity_guard=compute_polarity_guard(span),
    )


# ===========================================================================
# CR-21: qualifier orphan triggers rollback
# §15.1: "A kept $4.2M whose projected qualifier was dropped → uncovered → rollback"
# ===========================================================================

class TestCR21:

    def test_qualifier_in_adjacent_sentence_included_in_span(self):
        """
        When a qualifier token appears in the sentence before the item,
        it must be included in the governing_span (§15.1).
        """
        text = "These are projected figures. The target revenue is $4.2M. Final approval pending."
        icr = _icr(system_text="", user_text=text)
        manifest = extract_manifest(icr)

        money_items = [i for i in manifest.items if "$4.2M" in i.value]
        assert money_items, "Manifest must contain $4.2M as a NUMBER item"
        item = money_items[0]
        assert "projected" in item.governing_span.lower(), (
            "governing_span must include the preceding 'projected' sentence (§15.1)"
        )

    def test_qualifier_orphan_triggers_rollback(self):
        """
        CR-21: compress a prompt where a number's qualifier is in an adjacent
        sentence; dropping that qualifier sentence → coverage < 1.0.

        Original: "These figures are projected. Revenue target is $4.2M."
        Optimised: qualifier sentence dropped, number kept → coverage < 1.0.
        """
        original = "These figures are projected. Revenue target is $4.2M."
        icr = _icr(system_text=original, user_text="What is the revenue target?")
        manifest = extract_manifest(icr)

        money_items = [i for i in manifest.items if "$4.2M" in i.value]
        assert money_items, "Manifest must contain $4.2M"
        item = money_items[0]
        assert "projected" in item.governing_span.lower(), (
            "governing_span must capture the adjacent 'projected' qualifier"
        )

        # Simulate optimisation that drops the qualifier sentence but keeps the number
        optimised = "Revenue target is $4.2M."
        cov = manifest.coverage(optimised)
        assert cov < 1.0, (
            "CR-21: dropping the 'projected' qualifier while keeping $4.2M "
            "must produce coverage < 1.0 (§15.1)"
        )

    def test_no_qualifier_neighbour_coverage_unaffected(self):
        """
        When no qualifier token is in the adjacent sentences, the governing_span
        is just the item's own sentence and coverage depends only on the value.
        """
        original = "The contract value is $500,000. Payment is due on receipt."
        icr = _icr(system_text=original, user_text="What is the value?")
        manifest = extract_manifest(icr)

        money_items = [i for i in manifest.items if "$500,000" in i.value or "500,000" in i.value]
        assert money_items, "Manifest must contain the contract value"

        # Optimised keeps the value; no qualifier to orphan → coverage = 1.0 for this item
        optimised = "The contract value is $500,000."
        item = money_items[0]
        sub_manifest = ConstraintManifest(items=[item])
        assert sub_manifest.coverage(optimised) == pytest.approx(1.0), (
            "Without a qualifying neighbour, keeping the value must give coverage = 1.0"
        )

    def test_qualifier_in_following_sentence_also_captured(self):
        """
        The ±1 window is bidirectional: a qualifier in the NEXT sentence must
        also be included in the governing_span.
        """
        text = "Total cost is $12.5M. This figure is estimated and subject to change."
        icr = _icr(system_text=text, user_text="What is the total cost?")
        manifest = extract_manifest(icr)

        money_items = [i for i in manifest.items if "$12.5M" in i.value or "12.5M" in i.value]
        assert money_items, "Manifest must contain $12.5M"
        item = money_items[0]
        assert "estimated" in item.governing_span.lower() or "subject to" in item.governing_span.lower(), (
            "governing_span must capture the following 'estimated'/'subject to' qualifier"
        )

    def test_coverage_full_when_qualifier_retained(self):
        """
        Optimised text that retains BOTH the item and its qualifier →
        coverage must be 1.0 for that item.
        """
        original = "These are projected figures. Revenue is $4.2M."
        icr = _icr(system_text=original, user_text="What is the revenue?")
        manifest = extract_manifest(icr)

        money_items = [i for i in manifest.items if "$4.2M" in i.value]
        assert money_items
        item = money_items[0]
        sub_manifest = ConstraintManifest(items=[item])

        # Optimised keeps both the qualifier and the number in adjacent sentences
        optimised = "These are projected figures. Revenue is $4.2M."
        assert sub_manifest.coverage(optimised) == pytest.approx(1.0), (
            "Retaining the qualifier and the value must give coverage = 1.0 (§15.1 CR-21)"
        )


# ===========================================================================
# CR-22: polarity_guard mismatch triggers rollback
# §15.1: "polarity_guard hash must be identical pre- and post-optimization"
# ===========================================================================

class TestCR22:

    def test_polarity_guard_includes_not_token(self):
        """
        A span containing 'not projected' must produce a polarity_guard that
        differs from a span containing only 'projected'.
        """
        span_with_not = "The revenue is not projected to exceed $4.2M this quarter."
        span_without_not = "The revenue is projected to exceed $4.2M this quarter."

        guard_with = compute_polarity_guard(span_with_not)
        guard_without = compute_polarity_guard(span_without_not)

        assert guard_with != guard_without, (
            "CR-22: 'not projected' and 'projected' must produce different polarity_guards (§15.1)"
        )

    def test_polarity_change_triggers_rollback(self):
        """
        CR-22: flip 'not projected' → 'projected'; assert polarity_guard mismatch detected.

        This is the exact scenario specified in §14.3.
        """
        original = "The revenue figure is not projected to exceed $4.2M."
        icr = _icr(system_text=original, user_text="What is the revenue figure?")
        manifest = extract_manifest(icr)

        money_items = [i for i in manifest.items if "$4.2M" in i.value or "4.2M" in i.value]
        assert money_items, "Manifest must contain the $4.2M figure"
        item = money_items[0]
        assert item.polarity_guard, "polarity_guard must be populated for this item"

        # 'not' is in the governing_span → polarity_guard must capture it
        guard_has_not = "not" in item.governing_span.lower()
        assert guard_has_not, (
            "governing_span must contain 'not' so polarity_guard reflects negation"
        )

        # Simulate an optimisation that drops 'not' — flipping the polarity
        optimised = "The revenue figure is projected to exceed $4.2M."

        assert not polarity_intact(manifest, optimised), (
            "CR-22: flipping 'not projected' to 'projected' must be detected as "
            "polarity_guard mismatch → polarity_intact() returns False (§15.1)"
        )

    def test_polarity_intact_when_no_change(self):
        """
        If the negation context is preserved verbatim, polarity_intact() must
        return True.
        """
        original = "You must never delete user data without explicit confirmation."
        icr = _icr(system_text=original, user_text="What is the deletion policy?")
        manifest = extract_manifest(icr)

        # Optimised keeps the full sentence intact
        assert polarity_intact(manifest, original), (
            "CR-22: unchanged text must pass polarity_intact() check (§15.1)"
        )

    def test_polarity_intact_empty_manifest(self):
        """An empty manifest has no polarity guards to check — must return True."""
        assert polarity_intact(ConstraintManifest(), "any text") is True

    def test_polarity_guard_is_sha256_hex(self):
        """polarity_guard must be a 64-char sha256 hex digest (§15.1)."""
        span = "The amount is not projected to exceed $4.2M."
        guard = compute_polarity_guard(span)
        assert len(guard) == 64
        int(guard, 16)  # raises ValueError if not valid hex

    def test_polarity_guard_empty_span_is_stable(self):
        """Empty span → no negation tokens → deterministic empty hash."""
        g1 = compute_polarity_guard("")
        g2 = compute_polarity_guard("")
        assert g1 == g2
        assert len(g1) == 64

    def test_polarity_guard_symmetric_token_order(self):
        """
        Token order in the span must not affect the guard — only presence matters.
        Ensures the hash is computed over a SORTED set (§15.1 sha256 of all tokens).
        """
        span_a = "The value is not projected."
        span_b = "Projected figures are not final."
        # Both have "not" and "projected" — should produce the same guard
        assert compute_polarity_guard(span_a) == compute_polarity_guard(span_b), (
            "Same set of negation/modality tokens must yield same polarity_guard regardless of order"
        )

    def test_adding_negation_changes_guard(self):
        """Adding 'never' to a span must change the polarity_guard hash."""
        base = "The system will delete records."
        with_never = "The system will never delete records."
        assert compute_polarity_guard(base) != compute_polarity_guard(with_never)

    def test_cr22_not_triggered_by_unrelated_change(self):
        """
        Changing text that does NOT affect negation/modality tokens near the
        manifest item must not trigger a polarity_guard mismatch.
        """
        original = "Revenue is not projected to exceed $4.2M. The margin improved."
        icr = _icr(system_text=original, user_text="What is the revenue?")
        manifest = extract_manifest(icr)

        # Optimise: keep the $4.2M sentence intact, slightly rephrase the unrelated part
        optimised = "Revenue is not projected to exceed $4.2M. Margins improved."
        assert polarity_intact(manifest, optimised), (
            "Unrelated text change must not trigger CR-22 polarity mismatch"
        )
