"""
Internal Canonical Request (ICR) — provider-agnostic request representation.

Every inbound request (OpenAI, Anthropic, Mistral, …) is normalised into an ICR
before entering the ITOL pipeline.  Outbound, the provider adapter translates the
(possibly mutated) ICR back into a provider-native payload.

This module is import-only (no I/O, no models) and must remain dependency-free
beyond the standard library so it can be imported in any environment.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Segment types  (§3.2)
# ---------------------------------------------------------------------------

class SegmentType(str, Enum):
    SYSTEM_INSTRUCTION  = "SYSTEM_INSTRUCTION"
    TOOL_SCHEMA         = "TOOL_SCHEMA"
    FEW_SHOT_EXAMPLE    = "FEW_SHOT_EXAMPLE"
    RETRIEVED_DOC       = "RETRIEVED_DOC"
    USER_QUERY          = "USER_QUERY"
    ASSISTANT_TURN      = "ASSISTANT_TURN"
    TOOL_RESULT         = "TOOL_RESULT"
    STRUCTURED_DATA     = "STRUCTURED_DATA"
    CODE_BLOCK          = "CODE_BLOCK"
    UNKNOWN             = "UNKNOWN"


# ---------------------------------------------------------------------------
# Content blocks  (provider-agnostic building blocks of a Message)
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    TEXT        = "text"
    IMAGE_URL   = "image_url"
    IMAGE_BASE64 = "image_base64"
    TOOL_USE    = "tool_use"       # model requesting a tool call
    TOOL_RESULT = "tool_result"    # environment returning tool output


@dataclass
class ContentBlock:
    type: ContentType
    # TEXT / TOOL_RESULT
    text: str | None = None
    # TOOL_USE
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    # TOOL_RESULT
    tool_result_for_id: str | None = None
    is_error: bool = False
    # IMAGE
    image_url: str | None = None          # for IMAGE_URL
    image_data: bytes | None = None       # for IMAGE_BASE64
    image_media_type: str | None = None   # "image/png" etc.
    # Internal analysis metadata (populated by the segmenter, never sent to provider)
    segment_type: SegmentType = SegmentType.UNKNOWN
    segment_hash: str | None = None       # sha256 of normalised text
    token_count: int | None = None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def text(cls, content: str) -> "ContentBlock":
        return cls(type=ContentType.TEXT, text=content)

    @classmethod
    def tool_use(cls, tool_use_id: str, name: str, input: dict[str, Any]) -> "ContentBlock":
        return cls(
            type=ContentType.TOOL_USE,
            tool_use_id=tool_use_id,
            tool_name=name,
            tool_input=input,
        )

    @classmethod
    def tool_result(
        cls,
        for_id: str,
        content: str,
        is_error: bool = False,
    ) -> "ContentBlock":
        return cls(
            type=ContentType.TOOL_RESULT,
            tool_result_for_id=for_id,
            text=content,
            is_error=is_error,
        )

    def __post_init__(self) -> None:
        # Enforce minimal field presence for each type so broken blocks fail early.
        if self.type is ContentType.TEXT and self.text is None:
            raise ValueError("ContentBlock(TEXT) requires `text`")
        if self.type is ContentType.TOOL_USE:
            if not self.tool_use_id or not self.tool_name:
                raise ValueError("ContentBlock(TOOL_USE) requires tool_use_id and tool_name")
        if self.type is ContentType.TOOL_RESULT and self.tool_result_for_id is None:
            raise ValueError("ContentBlock(TOOL_RESULT) requires tool_result_for_id")


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: list[ContentBlock]
    # Optional provider-pass-through fields (not used by ITOL logic)
    name: str | None = None       # function/tool name for role=tool (OpenAI style)

    # Convenience: build a simple user/assistant text message
    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", content=[ContentBlock.text(text)])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls(role="assistant", content=[ContentBlock.text(text)])

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role="system", content=[ContentBlock.text(text)])

    def text_content(self) -> str:
        """Concatenate all text blocks; used by analysis/manifest extraction."""
        return "\n".join(
            b.text for b in self.content if b.type is ContentType.TEXT and b.text
        )


# ---------------------------------------------------------------------------
# Tool definition  (provider-agnostic JSON-schema style)
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema object
    # Internal tracking
    call_count_last_20: int = 0  # set by S6 tool-schema pruning (§4.S6)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolDef requires a non-empty name")


# ---------------------------------------------------------------------------
# Analysis metadata  (populated by the Ingestion & Analysis layer, §3)
# ---------------------------------------------------------------------------

@dataclass
class SegmentSignals:
    """Cheap per-request signals computed during ingestion (§3.3)."""
    token_count: int = 0
    redundancy_score: float = 0.0          # fraction of near-duplicate segments
    semantic_density: float = 1.0          # zstd/raw ratio proxy; higher = denser
    instruction_context_ratio: float = 0.0 # instruction tokens / total
    history_depth: int = 0                 # assistant/user turn pairs
    stale_mass: int = 0                    # tokens in tool results older than K turns
    template_signature: str | None = None  # sha256 of type-sequence + system hashes
    prefix_cacheable_span: int = 0         # tokens identical to prior request prefix


@dataclass
class ClassifierResult:
    """Output of the request-type classifier (§3.4)."""
    primary: str                        # one of the 8 class names
    distribution: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0             # max probability; <0.6 → AMBIGUOUS routing
    ambiguous: bool = False
    top2: list[str] = field(default_factory=list)  # top-2 classes for AMBIGUOUS routing (§3.4)

    REQUEST_CLASSES: tuple[str, ...] = field(default=(
        "EXTRACTION",
        "GENERATION_CREATIVE",
        "GENERATION_FACTUAL",
        "REASONING",
        "SUMMARIZATION",
        "CLASSIFICATION_SHORT",
        "AGENT_TOOL_LOOP",
        "CHAT_OPEN",
    ), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.confidence < 0.6:
            self.ambiguous = True


# ---------------------------------------------------------------------------
# §15.1 — Relation-aware manifest checking constants
# ---------------------------------------------------------------------------

# Qualifier tokens from data/qualifiers.txt (hardcoded here so icr.py stays
# dependency-free; the manifest extractor also loads the file for extensibility).
_QUALIFIER_TOKENS: frozenset[str] = frozenset({
    "projected", "estimated", "not", "except", "only",
    "excluding", "before", "after", "assuming", "unless",
    "approximately", "pending", "assumed",
    # multi-word qualifiers (checked as substrings)
    "as of", "up to", "at most", "at least", "subject to",
})

# Negation / modality tokens for polarity_guard (§15.1).
_NEGATION_MODALITY: frozenset[str] = frozenset({
    "not", "never", "no", "neither", "nor", "cannot", "can't", "won't",
    "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't", "hardly", "rarely", "seldom",
    "projected", "estimated", "approximately", "pending", "assumed",
    "subject to",
})

# Coreference pronouns that extend the governing span (§15.1).
# "the" is omitted: as a generic determiner it appears in virtually every sentence
# and cannot signal true coreference without full NLP; the spec's intent is
# anaphoric reference (it, they, this, that …), not bare article use.
_COREF_TOKENS: frozenset[str] = frozenset({
    "it", "its", "they", "their", "this", "that", "these", "those",
})

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_for_coverage(text: str) -> list[str]:
    """Minimal sentence splitter used for CR-21 coverage checks."""
    parts = _SENT_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _qualifier_tokens_in_span(span: str) -> list[str]:
    """Return qualifier tokens present in a governing_span (for CR-21 checks)."""
    lower = span.lower()
    found: list[str] = []
    for token in _QUALIFIER_TOKENS:
        if " " in token:
            if token in lower:
                found.append(token)
        elif re.search(r"\b" + re.escape(token) + r"\b", lower):
            found.append(token)
    return found


def _qualifiers_survive(item_value: str, governing_span: str, opt_text: str) -> bool:
    """
    CR-21: all qualifier tokens from governing_span must appear within
    ±1 sentence of item_value in opt_text.
    """
    qualifier_tokens = _qualifier_tokens_in_span(governing_span)
    if not qualifier_tokens:
        return True

    sents = _split_for_coverage(opt_text)
    item_idx: int | None = None
    for i, s in enumerate(sents):
        if item_value in s:
            item_idx = i
            break
    if item_idx is None:
        return False

    lo, hi = max(0, item_idx - 1), min(len(sents), item_idx + 2)
    window = " ".join(sents[lo:hi]).lower()
    for qt in qualifier_tokens:
        if " " in qt:
            if qt not in window:
                return False
        elif not re.search(r"\b" + re.escape(qt) + r"\b", window):
            return False
    return True


@dataclass
class ManifestItem:
    """A single entry in the Constraint Manifest (§5.1 + §15.1)."""
    class ItemType(str, Enum):
        ENTITY     = "entity"
        NUMBER     = "number"
        NORMATIVE  = "normative"   # must/never/always/only/exactly … clauses
        FORMAT     = "format"      # JSON keys, regexes, quoted literals
        QUERY_TERM = "query_term"  # content words of the final user query

    item_type: "ManifestItem.ItemType"
    value: str                          # exact text that must survive optimisation
    source_segment_hash: str | None = None
    # §15.1 relation-aware fields
    governing_span: str = ""            # ±1-sentence qualifier window (§15.1)
    polarity_guard: str = ""            # sha256 of negation/modality tokens in governing_span


@dataclass
class ConstraintManifest:
    """
    Machine-checkable losslessness contract extracted from the original prompt (§5.1 + §15.1).

    coverage(optimised_prompt) must equal 1.0 before dispatch.
    """
    items: list[ManifestItem] = field(default_factory=list)
    source_token_count: int = 0

    def coverage(self, optimised_text: str) -> float:
        """
        Fraction of manifest items fully preserved in optimised_text.

        CR-21: an item is 'found' only if BOTH its value AND the qualifier
        tokens from its governing_span survive within ±1 sentence of each
        other in optimised_text.  Items with empty governing_span fall back
        to the plain value-presence check (backward-compatible).
        """
        if not self.items:
            return 1.0
        found = 0
        for item in self.items:
            if item.value not in optimised_text:
                continue
            if item.governing_span and not _qualifiers_survive(
                item.value, item.governing_span, optimised_text
            ):
                continue
            found += 1
        return found / len(self.items)


@dataclass
class StrategyReport:
    """
    Record produced by each strategy execution; consumed by the Guarantor (§5.2)
    for per-segment rollback.
    """
    strategy_id: str          # "S1", "S2", …, "S7"
    tokens_removed: int = 0
    risk_class: str = "LOSSLESS"  # "LOSSLESS" | "NEAR-LOSSLESS" | "LOSSY-BOUNDED" | "LOSSY-AGGRESSIVE"
    manifest_touches: list[str] = field(default_factory=list)   # hashes of touched segments (legacy alias)
    segment_snapshot: list[Any] | None = None                   # segment list BEFORE this strategy ran
    # CR-14.2: precise mutation provenance used by rollback
    touched_segments: list[str] = field(default_factory=list)   # hashes of every segment mutated/removed
    removed_spans: list[tuple[int, int]] = field(default_factory=list)  # (start, end) char offsets removed


@dataclass
class AnalysisMeta:
    """All analysis data attached to an ICR after the ingestion stage."""
    signals: SegmentSignals = field(default_factory=SegmentSignals)
    classifier: ClassifierResult | None = None
    manifest: ConstraintManifest = field(default_factory=ConstraintManifest)
    strategy_reports: list[StrategyReport] = field(default_factory=list)
    qps: float | None = None          # Quality Preservation Score (§5.2); set post-pipeline
    cache_result: dict[str, Any] = field(default_factory=dict)   # L0/L1/L2/miss + similarity
    latency_ms: dict[str, float] = field(default_factory=dict)   # per-stage timing


# ---------------------------------------------------------------------------
# Internal Canonical Request  (§3.1)
# ---------------------------------------------------------------------------

@dataclass
class ICR:
    """
    Provider-agnostic representation of a single LLM inference request.

    Every field is either set at construction (from provider normalisation) or
    populated during pipeline execution.  `raw` is the original provider-native
    payload and is NEVER mutated — it is dispatched verbatim on rollback.
    """
    request_id: str
    tenant_id: str
    provider: str               # "openai" | "anthropic" | "mistral" | "cohere" | …
    model: str
    system: list[ContentBlock]  # provider-agnostic system content
    messages: list[Message]     # conversation turns
    tools: list[ToolDef]
    params: dict[str, Any]      # temperature, max_tokens, stop, etc.
    raw: dict[str, Any]         # original provider-native payload (immutable)
    meta: AnalysisMeta | None = None   # populated by the analysis stage

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("ICR requires a non-empty request_id")
        if not self.provider:
            raise ValueError("ICR requires a non-empty provider")
        if not self.model:
            raise ValueError("ICR requires a non-empty model")
        if self.raw is None:
            raise ValueError("ICR.raw must not be None (needed for rollback)")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        model: str,
        messages: list[Message],
        system: list[ContentBlock] | None = None,
        tools: list[ToolDef] | None = None,
        params: dict[str, Any] | None = None,
        tenant_id: str = "default",
        raw: dict[str, Any] | None = None,
    ) -> "ICR":
        """Factory used in tests and by adapters; auto-generates request_id."""
        return cls(
            request_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            provider=provider,
            model=model,
            system=system or [],
            messages=messages,
            tools=tools or [],
            params=params or {},
            raw=raw or {},
        )

    def all_text(self) -> str:
        """Full text content of the request (system + messages); used by manifest extraction."""
        parts: list[str] = [b.text for b in self.system if b.text]
        for msg in self.messages:
            parts.append(msg.text_content())
        return "\n".join(filter(None, parts))

    def final_user_query(self) -> str:
        """Text of the last user-role message; used by S3 scoring and cache lookup."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.text_content()
        return ""


# ---------------------------------------------------------------------------
# Response wrapper  (used by the provider adapter and telemetry)
# ---------------------------------------------------------------------------

@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0    # provider-reported cache read tokens
    cache_write_tokens: int = 0


@dataclass
class ICRResponse:
    request_id: str
    provider: str
    model: str
    content: list[ContentBlock]
    usage: UsageStats
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None          # non-None → error/refusal; must not be cached (CR-20)
