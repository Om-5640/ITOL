"""
Manifest Extractor — §5.1 + §15.1

Extracts the Constraint Manifest from an ICR.  Every item records:
  value          — exact text that must survive optimisation
  governing_span — ±1-sentence qualifier window (§15.1)
  polarity_guard — sha256 of negation/modality tokens in governing_span (§15.1)

Also provides:
  polarity_intact(manifest, optimised_text) → bool   for CR-22 checking
  compute_polarity_guard(span) → str                 (public; tested directly)
  split_sentences(text) → list[str]                  (public; tested directly)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

from itol.icr import (
    ICR,
    ConstraintManifest,
    ManifestItem,
    SegmentType,
    _COREF_TOKENS,
    _NEGATION_MODALITY,
    _split_for_coverage,
)
from itol.segmenter import Segment, filter_by_type, segment_icr


# ---------------------------------------------------------------------------
# Qualifier lexicon  (§15.1 — loaded from data/qualifiers.txt)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"
_QUALIFIERS_FILE = _DATA_DIR / "qualifiers.txt"


def _load_qualifiers() -> frozenset[str]:
    try:
        lines = _QUALIFIERS_FILE.read_text(encoding="utf-8").splitlines()
        return frozenset(
            line.strip().lower()
            for line in lines
            if line.strip() and not line.startswith("#")
        )
    except FileNotFoundError:
        # Fallback to the hardcoded set in icr.py
        from itol.icr import _QUALIFIER_TOKENS
        return _QUALIFIER_TOKENS


QUALIFIER_TOKENS: frozenset[str] = _load_qualifiers()


# ---------------------------------------------------------------------------
# Sentence splitter  (robust version for extraction; icr.py has a minimal one)
# ---------------------------------------------------------------------------

# Abbreviations that should not trigger sentence splits
_ABBREVS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "cf", "al",
    "eg", "ie", "etc", "approx", "est", "fig", "no", "vol", "pp",
})


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Strategy: split on [.!?] followed by whitespace + capital letter, OR on
    blank lines (paragraph boundary).  Protect common abbreviations.
    """
    if not text:
        return []

    # Paragraph boundary is always a sentence boundary
    paragraphs = re.split(r"\n{2,}", text)
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Within a paragraph, split on sentence-ending punctuation followed
        # by whitespace + uppercase letter, but not after known abbreviations.
        # Use a lookahead so the split point is between characters.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", para)
        for part in parts:
            # Protect abbreviation false-splits: "Dr. Smith" → one sentence
            # by re-joining parts that end with a known abbreviation word
            cleaned = part.strip()
            if cleaned:
                # Check if the last token before '.' is an abbreviation
                m = re.match(r".*\b(\w+)\.\s*$", cleaned)
                if m and m.group(1).lower() in _ABBREVS and sentences:
                    sentences[-1] = sentences[-1] + " " + cleaned
                else:
                    sentences.append(cleaned)
        # Also split on newlines within a paragraph (list items, etc.)
    return sentences if sentences else [text]


# ---------------------------------------------------------------------------
# polarity_guard  (§15.1 — public so the Guarantor and tests can call it)
# ---------------------------------------------------------------------------

_NEGATION_SINGLE: frozenset[str] = frozenset(
    t for t in _NEGATION_MODALITY if " " not in t
)
_NEGATION_PHRASES: frozenset[str] = frozenset(
    t for t in _NEGATION_MODALITY if " " in t
)


def compute_polarity_guard(span: str) -> str:
    """
    §15.1: sha256 of all negation/modality tokens found in span.

    Tokens are collected as a sorted set (position-independent) then hashed,
    so the same tokens in any order produce the same guard.  Removing or
    adding any negation/modality token changes the hash — CR-22 detects this.
    """
    lower = span.lower()

    found: set[str] = set()

    # Single-word tokens (word-boundary match to avoid "notable" matching "not")
    for token in _NEGATION_SINGLE:
        if re.search(r"\b" + re.escape(token) + r"\b", lower):
            found.add(token)

    # Multi-word phrases (substring match)
    for phrase in _NEGATION_PHRASES:
        if phrase in lower:
            found.add(phrase)

    digest_input = " ".join(sorted(found)).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


# ---------------------------------------------------------------------------
# Governing span computation  (§15.1)
# ---------------------------------------------------------------------------

def _has_qualifier_or_coref(sentence: str) -> bool:
    """Return True if sentence contains a qualifier token or coreference pronoun."""
    lower = sentence.lower()
    words = set(re.findall(r"\b\w+\b", lower))

    # Check single-word qualifiers + coreference pronouns
    if words & (QUALIFIER_TOKENS | _COREF_TOKENS):
        return True

    # Check multi-word qualifier phrases
    for token in QUALIFIER_TOKENS:
        if " " in token and token in lower:
            return True

    return False


def _governing_span(
    item_value: str,
    sentences: list[str],
    item_sent_idx: int,
) -> str:
    """
    §15.1: build the governing span for item_value.

    Always includes the sentence at item_sent_idx.  Includes the preceding
    sentence if it contains a qualifier token or coreference pronoun; likewise
    for the following sentence.
    """
    window: list[str] = [sentences[item_sent_idx]]

    if item_sent_idx > 0:
        prev = sentences[item_sent_idx - 1]
        if _has_qualifier_or_coref(prev):
            window.insert(0, prev)

    if item_sent_idx < len(sentences) - 1:
        nxt = sentences[item_sent_idx + 1]
        if _has_qualifier_or_coref(nxt):
            window.append(nxt)

    return " ".join(window)


def _make_item(
    item_type: ManifestItem.ItemType,
    value: str,
    sentences: list[str],
    source_hash: str | None = None,
) -> ManifestItem:
    """
    Build a ManifestItem with governing_span and polarity_guard.

    Finds the first sentence containing value, computes the ±1 window,
    then hashes the negation/modality tokens in that window.
    """
    item_sent_idx: int | None = None
    for i, s in enumerate(sentences):
        if value in s:
            item_sent_idx = i
            break

    if item_sent_idx is None:
        # Value spans a sentence boundary — use the full text as the span
        span = " ".join(sentences)
    else:
        span = _governing_span(value, sentences, item_sent_idx)

    return ManifestItem(
        item_type=item_type,
        value=value,
        source_segment_hash=source_hash,
        governing_span=span,
        polarity_guard=compute_polarity_guard(span),
    )


# ---------------------------------------------------------------------------
# Entity extraction  (§5.1 — capitalized spans + code identifiers)
# ---------------------------------------------------------------------------

# Capitalized multi-word entity: at least one capital initial, not at sentence start
_CAPS_ENTITY = re.compile(r"(?<!\.\s)(?<!\n)(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")
# Acronyms: 2+ uppercase letters (not followed by lowercase to avoid "I")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
# Code identifiers: camelCase, PascalCase, SCREAMING_SNAKE, dotted names
_CODE_IDENT = re.compile(r"\b(?:[a-z]+[A-Z]\w*|[A-Z]+_[A-Z_]+|[a-z]+\.[a-z_]+)\b")


def _extract_entities(text: str, sentences: list[str], source_hash: str | None) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    seen: set[str] = set()

    for pattern in (_CAPS_ENTITY, _ACRONYM, _CODE_IDENT):
        for m in pattern.finditer(text):
            value = m.group(0).strip()
            if value and value not in seen and len(value) > 1:
                seen.add(value)
                items.append(_make_item(
                    ManifestItem.ItemType.ENTITY, value, sentences, source_hash
                ))

    return items


# ---------------------------------------------------------------------------
# Number / unit extraction  (§5.1)
# ---------------------------------------------------------------------------

# Currency amounts: $4.2M, €1,000, £50k, USD 3.5B, etc.
_CURRENCY = re.compile(
    r"(?:[$€£¥][\d,.]+[KMBTkmbt]?|(?:USD|EUR|GBP|JPY|CAD)\s*[\d,.]+[KMBTkmbt]?)"
)
# Percentages
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*%")
# Dates: YYYY-MM-DD, MM/DD/YYYY, "Q3 2024", "January 2024", etc.
_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|Q[1-4]\s+\d{4}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{4})\b",
    re.IGNORECASE,
)
# Version strings: v1.2.3, 3.11.0, etc.
_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)*\b")
# Plain numerals (integers and decimals, including comma-formatted)
_NUMERAL = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+\.\d+\b")


def _extract_numbers(text: str, sentences: list[str], source_hash: str | None) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    seen: set[str] = set()

    for pattern in (_CURRENCY, _PERCENT, _DATE, _VERSION, _NUMERAL):
        for m in pattern.finditer(text):
            value = m.group(0).strip()
            if value and value not in seen:
                seen.add(value)
                items.append(_make_item(
                    ManifestItem.ItemType.NUMBER, value, sentences, source_hash
                ))

    return items


# ---------------------------------------------------------------------------
# Normative clause extraction  (§5.1)
# ---------------------------------------------------------------------------

_NORMATIVE_CLAUSE = re.compile(
    r"[^.!?\n]*\b(?:must|never|always|only|exactly|do not|required|forbidden|"
    r"shall not|should not|at least|at most|you must|you should not|"
    r"is required|are required|is forbidden|are forbidden)\b[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)


def _extract_normative_clauses(
    text: str, sentences: list[str], source_hash: str | None
) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    seen: set[str] = set()

    for m in _NORMATIVE_CLAUSE.finditer(text):
        value = m.group(0).strip()
        if value and value not in seen and len(value) > 10:
            seen.add(value)
            items.append(_make_item(
                ManifestItem.ItemType.NORMATIVE, value, sentences, source_hash
            ))

    return items


# ---------------------------------------------------------------------------
# Format specifier extraction  (§5.1)
# ---------------------------------------------------------------------------

# Quoted literals (single or double quotes, at least 2 chars inside)
_QUOTED = re.compile(r'"[^"]{2,}"|\'[^\']{2,}\'')
# JSON keys: "key": pattern
_JSON_KEY = re.compile(r'"(\w[\w_-]*)"(?=\s*:)')
# Regex-like patterns: /pattern/ or common regex metachar sequences
_REGEX_PAT = re.compile(r"/[^/\s]{3,}/[gimsuy]*")


def _extract_format_specifiers(
    text: str, sentences: list[str], source_hash: str | None
) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    seen: set[str] = set()

    for pattern in (_QUOTED, _REGEX_PAT):
        for m in pattern.finditer(text):
            value = m.group(0).strip()
            if value and value not in seen:
                seen.add(value)
                items.append(_make_item(
                    ManifestItem.ItemType.FORMAT, value, sentences, source_hash
                ))

    for m in _JSON_KEY.finditer(text):
        value = m.group(1)
        if value and value not in seen:
            seen.add(value)
            items.append(_make_item(
                ManifestItem.ItemType.FORMAT, value, sentences, source_hash
            ))

    return items


# ---------------------------------------------------------------------------
# Query term extraction  (§5.1)
# ---------------------------------------------------------------------------

# Common English stop words to exclude from query terms
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "and", "or", "but", "not", "no",
    "if", "then", "else", "when", "where", "how", "what", "which", "who",
    "that", "this", "these", "those", "it", "its", "i", "you", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "our", "their",
    "up", "out", "about", "into", "through", "after", "before", "above",
    "below", "between", "over", "under", "so", "as", "just", "more",
    "also", "very", "too", "only", "than", "now", "here", "there",
})


def _extract_query_terms(query: str, sentences: list[str]) -> list[ManifestItem]:
    """§5.1: content words of the final user query."""
    items: list[ManifestItem] = []
    seen: set[str] = set()

    words = re.findall(r"\b[a-zA-Z_][\w]*\b", query)
    for word in words:
        lower = word.lower()
        if lower not in _STOP_WORDS and len(lower) > 2 and lower not in seen:
            seen.add(lower)
            items.append(_make_item(
                ManifestItem.ItemType.QUERY_TERM, word, sentences
            ))

    return items


# ---------------------------------------------------------------------------
# Main entry point  (§5.1 + §15.1)
# ---------------------------------------------------------------------------

def extract_manifest(icr: ICR) -> ConstraintManifest:
    """
    Extract a Constraint Manifest from the original ICR.

    The manifest is extracted from the ORIGINAL (pre-optimisation) text.
    Must be called before any strategy mutates the ICR.
    """
    segments = segment_icr(icr)

    # Instruction segments (system + normative-heavy content) for entities/numbers/normative
    instruction_segs = filter_by_type(
        segments,
        SegmentType.SYSTEM_INSTRUCTION,
        SegmentType.USER_QUERY,
        SegmentType.RETRIEVED_DOC,
        SegmentType.FEW_SHOT_EXAMPLE,
        SegmentType.STRUCTURED_DATA,
    )
    instruction_text = "\n".join(s.text for s in instruction_segs)
    instruction_sents = split_sentences(instruction_text) if instruction_text else []

    items: list[ManifestItem] = []

    # Per-segment extraction (preserves source_segment_hash for rollback targeting)
    for seg in instruction_segs:
        sents = split_sentences(seg.text)
        if not sents:
            continue
        items.extend(_extract_entities(seg.text, sents, seg.segment_hash))
        items.extend(_extract_numbers(seg.text, sents, seg.segment_hash))
        # Normative clauses only from SYSTEM_INSTRUCTION segments (§5.1)
        if seg.segment_type is SegmentType.SYSTEM_INSTRUCTION:
            items.extend(_extract_normative_clauses(seg.text, sents, seg.segment_hash))
        items.extend(_extract_format_specifiers(seg.text, sents, seg.segment_hash))

    # Query terms from the final user query.
    # Use instruction_sents (full context) so polarity_intact sees the same
    # sentence boundaries as governing_span computation did here.
    query = icr.final_user_query()
    if query:
        items.extend(_extract_query_terms(query, instruction_sents or split_sentences(query)))

    # De-duplicate by (item_type, value) — keep first occurrence
    seen_keys: set[tuple] = set()
    deduped: list[ManifestItem] = []
    for item in items:
        key = (item.item_type, item.value)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(item)

    from itol.signals import estimate_token_count
    return ConstraintManifest(
        items=deduped,
        source_token_count=estimate_token_count(instruction_text + "\n" + query),
    )


# ---------------------------------------------------------------------------
# CR-22 — polarity_guard integrity check  (§15.1)
# ---------------------------------------------------------------------------

def polarity_intact(manifest: ConstraintManifest, optimised_text: str) -> bool:
    """
    CR-22: for every manifest item with a polarity_guard, verify the guard hash
    is identical in the optimised text.

    Procedure: find the item's value in optimised_text, extract ±1 sentence
    window, recompute polarity_guard, compare to stored hash.
    Returns True only if ALL items pass (or have no guard to check).
    """
    for item in manifest.items:
        if not item.polarity_guard:
            continue
        if item.value not in optimised_text:
            # Item is missing entirely — coverage() catches that; not our check here
            continue

        sents = _split_for_coverage(optimised_text)
        item_idx: int | None = None
        for i, s in enumerate(sents):
            if item.value in s:
                item_idx = i
                break

        if item_idx is None:
            return False

        lo = max(0, item_idx - 1)
        hi = min(len(sents), item_idx + 2)
        opt_window = " ".join(sents[lo:hi])

        if compute_polarity_guard(opt_window) != item.polarity_guard:
            return False

    return True
