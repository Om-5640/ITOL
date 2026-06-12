"""
S5 Conversation History Distillation — LOSSY-BOUNDED (§4, execution order position 5).

Turns older than the last K=6 turn-pairs are candidates for distillation.
For each aging-out turn, extract verbatim:
  - sentences containing normative tokens or first-person commitments
  - all manifest entities/numbers occurring in that turn
  - final answers to sub-questions (last assistant turn before a topic shift)

Drop (from aging turns):
  - greetings/hedges lexicon
  - repeated reasoning (Jaccard ≥ 0.6 to an earlier sentence in the conversation)
  - superseded drafts (Jaccard ≥ 0.6 on an artifact span vs. a LATER assistant turn)

Output: «Conversation ledger: decisions=[...], facts=[...], open=[...]» block
        + last K turn-pairs verbatim.

CR-5 (resurrection): BEFORE distilling a turn, write its full text to docs table
        keyed by s5_turn:<turn_index>.  maybe_resurrect() returns the original
        turn text if cos(query_emb, turn_emb) > cos(query_emb, ledger_emb) + 0.45.

CR-6 (incremental): the ledger is cached on the conversations table.
        Only turns newly aging out (turn_index = current_turn - K, not yet
        in turn_hashes) are re-extracted.  If a turn's hash changes, invalidate
        and recompute from that turn forward.

Gentle variant (CHAT_OPEN / GENERATION_CREATIVE): also keep verbatim any
        assistant turn with lexical diversity (TTR) > 1.5 × conversation average —
        these "stylistic exemplars" inform the model's voice.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from itol.icr import ICR, SegmentType, StrategyReport
from itol.segmenter import Segment
from itol.signals import estimate_token_count, jaccard_estimate, minhash_signature
from itol.strategies.base import OptimizationContext, Strategy, update_segment

if TYPE_CHECKING:
    from itol.cache.store import Store


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GREETING_HEDGES = frozenset({
    "thanks", "thank you", "sure", "of course", "certainly",
    "i'd be happy to", "i would be happy to", "great question",
    "good question", "absolutely", "no problem", "no worries",
    "i hope that helps", "let me know if you have any questions",
})

_NORMATIVE_PATTERN = re.compile(
    r"\b(must|never|always|only|exactly|shall|do not|required|forbidden"
    r"|shall not|should not|need to|have to)\b",
    re.IGNORECASE,
)

_COMMITMENT_PATTERN = re.compile(
    r"\b(i want|use |in [a-z]+|format|write in|reply in|respond in)\b",
    re.IGNORECASE,
)

_ENTITY_PATTERN = re.compile(
    r"\$[\d,]+(?:\.\d+)?[MBKkm]?|"         # money: $1234, $5M
    r"\b\d+(?:[,\d]*\.?\d+)?(?:%|percent)?\b|"  # numbers/percentages
    r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b",       # identifiers like CRF-21
)

_SUPERSEDE_JACCARD = 0.60
_REPEAT_JACCARD    = 0.60
_GENTLE_TTR_FACTOR = 1.5


# ---------------------------------------------------------------------------
# Dataclass for the in-memory ledger
# ---------------------------------------------------------------------------

@dataclass
class _Ledger:
    decisions: list[str] = field(default_factory=list)  # normative/commitment sentences
    facts:     list[str] = field(default_factory=list)   # entity/number occurrences
    open:      list[str] = field(default_factory=list)   # unanswered sub-questions


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class S5DistillStrategy(Strategy):
    """
    Incremental conversation distillation.

    `store` is required for CR-5 (resurrection archive) and CR-6 (incremental
    ledger cache).  If store is None, distillation still works but persists
    nothing (suitable for unit-testing the extraction logic).
    """

    strategy_id = "S5"
    risk_class  = "LOSSY_BOUNDED"

    def __init__(self, store: "Store | None" = None) -> None:
        self._store = store

    def applies(self, icr: ICR, segments: list[Segment], ctx: OptimizationContext) -> bool:
        cfg = ctx.config.strategies
        return (
            ctx.signals.history_depth > cfg.s5_history_depth_gate
            or ctx.signals.token_count > cfg.s5_history_tokens_gate
        )

    def apply(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> tuple[list[Segment], StrategyReport]:
        snapshot = list(segments)
        tokens_before = sum(s.token_count or estimate_token_count(s.text) for s in segments)

        cls_cfg = ctx.config.class_configs.get(ctx.request_class)
        K = cls_cfg.s5_k_turns if cls_cfg else 6
        gentle = ctx.request_class in ("CHAT_OPEN", "GENERATION_CREATIVE")

        # Identify turn segments (user + assistant, in order)
        turn_pairs = _collect_turn_pairs(segments)
        if len(turn_pairs) <= K:
            report = self._make_report(
                snapshot, tokens_before, tokens_before, [], [], 0, 0, activated=False
            )
            return segments, report

        aging_pairs = turn_pairs[:-K]
        recent_pairs = turn_pairs[-K:]

        # Gentle: identify stylistic exemplar turns to keep
        exemplar_indices: set[int] = set()
        if gentle:
            exemplar_indices = _find_exemplar_turns(segments, turn_pairs)

        # Aging segments (all non-exemplar turns older than K pairs)
        aging_seg_indices: set[int] = set()
        aging_segs: list[tuple[int, Segment]] = []  # (original_index, seg)
        for pairs in aging_pairs:
            for seg_idx in pairs:
                if seg_idx not in exemplar_indices:
                    aging_seg_indices.add(seg_idx)
                    aging_segs.append((seg_idx, segments[seg_idx]))

        if not aging_segs:
            report = self._make_report(
                snapshot, tokens_before, tokens_before, [], [], 0, 0, activated=False
            )
            return segments, report

        conversation_id = icr.conversation_id or f"{icr.tenant_id}:{icr.request_id[:8]}"
        tenant_id = icr.tenant_id

        # Load existing ledger (CR-6 incremental)
        existing_conv = self._store.get_conversation(conversation_id, tenant_id) if self._store else None
        existing_hashes: list[str] = (existing_conv or {}).get("turn_hashes") or []
        existing_ledger: dict | None = (existing_conv or {}).get("ledger")

        # Detect hash changes → invalidate from changed turn forward
        aging_hashes = [s.segment_hash for _, s in aging_segs]
        first_changed = _first_changed_idx(aging_hashes, existing_hashes)

        # Determine which turns are newly aging out
        existing_hash_set = set(existing_hashes[:first_changed])
        newly_aging = [
            (turn_idx, (seg_idx, seg))
            for turn_idx, (seg_idx, seg) in enumerate(aging_segs)
            if turn_idx >= first_changed or seg.segment_hash not in existing_hash_set
        ]

        # CR-5: write original text to docs BEFORE building ledger
        if self._store:
            for turn_idx, (seg_idx, seg) in enumerate(aging_segs):
                doc_key = f"s5_turn:{turn_idx}"
                self._store.save_doc(
                    doc_key, tenant_id, conversation_id,
                    seg.text, seg.segment_hash,
                )

        # Build/extend ledger from newly-aging turns
        ledger = _load_ledger(existing_ledger) if (existing_ledger and first_changed > 0) else _Ledger()
        all_sentences: list[str] = _gather_all_sentences(segments, aging_seg_indices)

        for _turn_idx, (seg_idx, seg) in newly_aging:
            _extract_into_ledger(seg, ledger, all_sentences, segments)

        # Persist updated ledger (CR-6)
        if self._store:
            self._store.set_conversation(
                conversation_id, tenant_id,
                ledger=_ledger_to_dict(ledger),
                turn_hashes=aging_hashes,
            )

        # Reassemble: replace aging segment range with ledger block
        ledger_text = _format_ledger(ledger)
        new_segments = _replace_aging_with_ledger(
            segments, aging_seg_indices, ledger_text, aging_segs[0][1]
        )

        tokens_after = sum(s.token_count or estimate_token_count(s.text) for s in new_segments)
        touched = [segments[i].segment_hash for i in aging_seg_indices]
        report = self._make_report(
            snapshot, tokens_before, tokens_after,
            touched, [], 0, 0, activated=True,
        )
        return new_segments, report


# ---------------------------------------------------------------------------
# CR-5 resurrection
# ---------------------------------------------------------------------------

def maybe_resurrect(
    current_query_embedding,
    conversation_id: str,
    tenant_id: str,
    ledger_text: str,
    store: "Store",
    *,
    threshold: float = 0.45,
) -> str | None:
    """
    CR-5: if cos(query_emb, distilled_turn_emb) > cos(query_emb, ledger_emb) + threshold,
    return the original turn text.  Returns None if no match exceeds threshold.
    """
    from itol.embed.onnx_embedder import cosine, embed

    ledger_emb = embed([ledger_text[:512]], model="minilm")[0]
    base_cos = float(cosine(current_query_embedding, ledger_emb))

    best_text: str | None = None
    best_delta: float = threshold

    turn_idx = 0
    while True:
        doc_key = f"s5_turn:{turn_idx}"
        original = store.get_doc(doc_key, tenant_id, conversation_id)
        if original is None:
            break
        turn_emb = embed([original[:512]], model="minilm")[0]
        turn_cos = float(cosine(current_query_embedding, turn_emb))
        delta = turn_cos - base_cos
        if delta > best_delta:
            best_delta = delta
            best_text = original
        turn_idx += 1

    return best_text


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _collect_turn_pairs(segments: list[Segment]) -> list[list[int]]:
    """
    Group consecutive USER_QUERY+ASSISTANT_TURN pairs.
    Returns a list of groups, each group being a list of segment indices.
    Each group ends at an ASSISTANT_TURN; a USER_QUERY without a following
    ASSISTANT_TURN forms its own group.
    """
    pairs: list[list[int]] = []
    current: list[int] = []
    for i, seg in enumerate(segments):
        if seg.segment_type == SegmentType.USER_QUERY:
            if current:
                pairs.append(current)
            current = [i]
        elif seg.segment_type == SegmentType.ASSISTANT_TURN:
            current.append(i)
            pairs.append(current)
            current = []
    if current:
        pairs.append(current)
    return pairs


def _first_changed_idx(current_hashes: list[str], stored_hashes: list[str]) -> int:
    """Return the index of the first hash that differs between the two lists."""
    for i, (ch, sh) in enumerate(zip(current_hashes, stored_hashes)):
        if ch != sh:
            return i
    return len(stored_hashes)  # no change in stored range; new ones start here


def _is_greeting(sentence: str) -> bool:
    lower = sentence.strip().lower()
    return any(g in lower for g in _GREETING_HEDGES)


def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _ttr(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _find_exemplar_turns(
    segments: list[Segment],
    turn_pairs: list[list[int]],
) -> set[int]:
    """
    Gentle variant: identify assistant turn segment indices with TTR >
    1.5 × the conversation average TTR.
    """
    asst_turns = [
        (idx_list, segments[idx_list[-1]])
        for idx_list in turn_pairs
        if idx_list and segments[idx_list[-1]].segment_type == SegmentType.ASSISTANT_TURN
    ]
    if not asst_turns:
        return set()
    avg_ttr = sum(_ttr(seg.text) for _, seg in asst_turns) / len(asst_turns)
    threshold = _GENTLE_TTR_FACTOR * avg_ttr
    exemplars: set[int] = set()
    for idx_list, seg in asst_turns:
        if _ttr(seg.text) > threshold:
            for idx in idx_list:
                exemplars.add(idx)
    return exemplars


def _gather_all_sentences(
    segments: list[Segment], skip_indices: set[int]
) -> list[str]:
    """Collect all sentences from non-aged segments (for repeat detection)."""
    sentences: list[str] = []
    for i, seg in enumerate(segments):
        if i in skip_indices:
            continue
        sentences.extend(_sentence_split(seg.text))
    return sentences


def _extract_into_ledger(
    seg: Segment,
    ledger: "_Ledger",
    all_sentences: list[str],
    all_segments: list[Segment],
) -> None:
    """Extract key content from `seg` and fold into `ledger`."""
    sentences = _sentence_split(seg.text)
    all_sigs = [minhash_signature(s) for s in all_sentences]

    # Get signatures of later assistant turns for supersedence check
    seg_idx_in_all = next(
        (i for i, s in enumerate(all_segments) if s.segment_hash == seg.segment_hash),
        None,
    )
    later_asst_sigs = []
    if seg_idx_in_all is not None:
        for j, s in enumerate(all_segments):
            if j > seg_idx_in_all and s.segment_type == SegmentType.ASSISTANT_TURN:
                later_asst_sigs.append(minhash_signature(s.text))

    for sent in sentences:
        if not sent:
            continue
        if _is_greeting(sent):
            continue

        # Repeat detection
        sent_sig = minhash_signature(sent)
        if any(jaccard_estimate(sent_sig, s) >= _REPEAT_JACCARD for s in all_sigs):
            continue

        # Superseded draft detection
        if later_asst_sigs and any(
            jaccard_estimate(sent_sig, la) >= _SUPERSEDE_JACCARD
            for la in later_asst_sigs
        ):
            continue

        # Classify into ledger bucket
        if _NORMATIVE_PATTERN.search(sent) or _COMMITMENT_PATTERN.search(sent):
            if sent not in ledger.decisions:
                ledger.decisions.append(sent)
        elif _ENTITY_PATTERN.search(sent):
            if sent not in ledger.facts:
                ledger.facts.append(sent)
        elif sent.strip().endswith("?"):
            if sent not in ledger.open:
                ledger.open.append(sent)


def _format_ledger(ledger: "_Ledger") -> str:
    decisions = "; ".join(ledger.decisions) if ledger.decisions else "none"
    facts = "; ".join(ledger.facts) if ledger.facts else "none"
    open_q = "; ".join(ledger.open) if ledger.open else "none"
    return f"«Conversation ledger: decisions=[{decisions}], facts=[{facts}], open=[{open_q}]»"


def _ledger_to_dict(ledger: "_Ledger") -> dict:
    return {
        "decisions": ledger.decisions,
        "facts": ledger.facts,
        "open": ledger.open,
    }


def _load_ledger(d: dict | None) -> "_Ledger":
    if not d:
        return _Ledger()
    return _Ledger(
        decisions=list(d.get("decisions", [])),
        facts=list(d.get("facts", [])),
        open=list(d.get("open", [])),
    )


def _replace_aging_with_ledger(
    segments: list[Segment],
    aging_indices: set[int],
    ledger_text: str,
    first_aging_seg: Segment,
) -> list[Segment]:
    """Replace all aging segments with a single ledger block at the position of the first."""
    result: list[Segment] = []
    ledger_inserted = False
    for i, seg in enumerate(segments):
        if i not in aging_indices:
            result.append(seg)
        elif not ledger_inserted:
            # Create ledger segment from first aging segment's position metadata
            from dataclasses import replace as dc_replace
            from itol.signals import estimate_token_count as etc
            ledger_seg = dc_replace(
                first_aging_seg,
                segment_type=SegmentType.ASSISTANT_TURN,
                text=ledger_text,
                segment_hash=hashlib.sha256(ledger_text.encode()).hexdigest(),
                token_count=etc(ledger_text),
            )
            result.append(ledger_seg)
            ledger_inserted = True
        # else: drop aging segment
    return result
