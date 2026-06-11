"""
§14.3 Invariant tests for CR-11 — Classifier feeds routing correctly.

CR-11: The classifier output drives the §7.2 compatibility matrix.
       S7 must be DISABLED for EXTRACTION and REASONING.
       L1 must be DISABLED for GENERATION_CREATIVE and AGENT_TOOL_LOOP.
       AMBIGUOUS class must produce the intersection of the top-2 classes.
       Tools present in ICR must always yield AGENT_TOOL_LOOP.

Rules:
- NEVER weaken a threshold to make a test pass.
- Every assertion cites the spec section or CR number.
"""

import pytest

from itol.icr import ContentBlock, ICR, Message, ToolDef
from itol.analysis.classifier import classify
from itol.routing.matrix import (
    MATRIX,
    L1Status,
    StrategyStatus,
    ambiguous_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _icr(
    system_text: str = "",
    user_text: str = "Hello.",
    tools: list | None = None,
    extra_messages: list | None = None,
) -> ICR:
    msgs = [Message.user(user_text)]
    if extra_messages:
        msgs = extra_messages + msgs
    return ICR.create(
        provider="openai",
        model="gpt-4o",
        system=[ContentBlock.text(system_text)] if system_text else [],
        messages=msgs,
        tools=tools or [],
        raw={},
    )


def _tool(name: str = "search") -> ToolDef:
    return ToolDef(
        name=name,
        description="A tool",
        parameters={"type": "object", "properties": {}},
    )


# ===========================================================================
# CR-11-a: S7 DISABLED for EXTRACTION and REASONING
# §7.2: "S7 disabled for EXTRACTION and REASONING"
# ===========================================================================

class TestCR11_S7:

    def test_s7_disabled_extraction(self):
        """§7.2: S7 must be DISABLED for EXTRACTION class."""
        assert MATRIX["EXTRACTION"].strategies["S7"] == StrategyStatus.DISABLED, (
            "CR-11: S7 must be DISABLED for EXTRACTION (§7.2)"
        )

    def test_s7_disabled_reasoning(self):
        """§7.2: S7 must be DISABLED for REASONING class."""
        assert MATRIX["REASONING"].strategies["S7"] == StrategyStatus.DISABLED, (
            "CR-11: S7 must be DISABLED for REASONING (§7.2)"
        )

    def test_s7_allowed_summarization(self):
        """§7.2: S7 is optional (ALLOWED) for SUMMARIZATION — must NOT be DISABLED."""
        assert MATRIX["SUMMARIZATION"].strategies["S7"] != StrategyStatus.DISABLED, (
            "§7.2: S7 is optional for SUMMARIZATION — must not be DISABLED"
        )

    def test_s7_disabled_generation_factual(self):
        """§7.2: S7 must be DISABLED for GENERATION_FACTUAL."""
        assert MATRIX["GENERATION_FACTUAL"].strategies["S7"] == StrategyStatus.DISABLED

    def test_s7_disabled_generation_creative(self):
        """§7.2: S7 must be DISABLED for GENERATION_CREATIVE."""
        assert MATRIX["GENERATION_CREATIVE"].strategies["S7"] == StrategyStatus.DISABLED

    def test_s7_disabled_classification_short(self):
        """§7.2: S7 must be DISABLED for CLASSIFICATION_SHORT."""
        assert MATRIX["CLASSIFICATION_SHORT"].strategies["S7"] == StrategyStatus.DISABLED

    def test_s7_disabled_agent_tool_loop(self):
        """§7.2: S7 must be DISABLED for AGENT_TOOL_LOOP."""
        assert MATRIX["AGENT_TOOL_LOOP"].strategies["S7"] == StrategyStatus.DISABLED


# ===========================================================================
# CR-11-b: L1 DISABLED for GENERATION_CREATIVE and AGENT_TOOL_LOOP
# §7.2: "L1 cache disabled for GENERATION_CREATIVE and AGENT_TOOL_LOOP"
# ===========================================================================

class TestCR11_L1:

    def test_l1_disabled_creative(self):
        """§7.2: L1 must be DISABLED for GENERATION_CREATIVE."""
        assert MATRIX["GENERATION_CREATIVE"].l1 == L1Status.DISABLED, (
            "CR-11: L1 must be DISABLED for GENERATION_CREATIVE (§7.2)"
        )

    def test_l1_disabled_agent_tool_loop(self):
        """§7.2: L1 must be DISABLED for AGENT_TOOL_LOOP."""
        assert MATRIX["AGENT_TOOL_LOOP"].l1 == L1Status.DISABLED, (
            "CR-11: L1 must be DISABLED for AGENT_TOOL_LOOP (§7.2)"
        )

    def test_l1_allowed_extraction(self):
        """§7.2: L1 is ALLOWED for EXTRACTION (τ=0.97)."""
        entry = MATRIX["EXTRACTION"]
        assert entry.l1 == L1Status.ALLOWED
        assert entry.l1_tau == pytest.approx(0.97)

    def test_l1_allowed_reasoning(self):
        """§7.2: L1 is ALLOWED for REASONING (τ=0.97)."""
        entry = MATRIX["REASONING"]
        assert entry.l1 == L1Status.ALLOWED
        assert entry.l1_tau == pytest.approx(0.97)

    def test_l1_tau_summarization(self):
        """§7.2: L1 τ for SUMMARIZATION must be 0.95."""
        assert MATRIX["SUMMARIZATION"].l1_tau == pytest.approx(0.95)

    def test_l1_tau_generation_factual(self):
        """§7.2: L1 τ for GENERATION_FACTUAL must be 0.96."""
        assert MATRIX["GENERATION_FACTUAL"].l1_tau == pytest.approx(0.96)

    def test_l1_tau_classification_short(self):
        """§7.2: L1 τ for CLASSIFICATION_SHORT must be 0.93."""
        assert MATRIX["CLASSIFICATION_SHORT"].l1_tau == pytest.approx(0.93)

    def test_l1_disabled_chat_open(self):
        """§7.2: L1 is DISABLED for CHAT_OPEN."""
        assert MATRIX["CHAT_OPEN"].l1 == L1Status.DISABLED


# ===========================================================================
# CR-11-c: AMBIGUOUS intersects top-2 classes
# §3.4: "AMBIGUOUS → intersection of top-2 classes' permitted strategy sets"
# ===========================================================================

class TestCR11_Ambiguous:

    def test_ambiguous_intersects_top2(self):
        """
        CR-11: ambiguous_matrix(A, B) must disable a strategy that is
        DISABLED in either A or B.

        EXTRACTION disables S7; CHAT_OPEN allows S7.
        Intersection must disable S7.
        """
        result = ambiguous_matrix("EXTRACTION", "CHAT_OPEN")
        assert result.strategies["S7"] == StrategyStatus.DISABLED, (
            "CR-11: S7 DISABLED in EXTRACTION must propagate to AMBIGUOUS intersection"
        )

    def test_ambiguous_l1_disabled_if_either_disabled(self):
        """
        §3.4: AMBIGUOUS L1 must be DISABLED if either class disables L1.
        AGENT_TOOL_LOOP disables L1; EXTRACTION allows L1.
        Intersection must disable L1.
        """
        result = ambiguous_matrix("AGENT_TOOL_LOOP", "EXTRACTION")
        assert result.l1 == L1Status.DISABLED, (
            "CR-11: L1 DISABLED in AGENT_TOOL_LOOP must propagate to AMBIGUOUS intersection"
        )

    def test_ambiguous_stricter_restriction_wins(self):
        """
        When both classes restrict S3, the HIGHER (stricter) threshold wins.
        EXTRACTION: S3 restricted at 0.97; REASONING: S3 restricted at 0.97.
        Intersection S3 threshold must be 0.97.
        """
        result = ambiguous_matrix("EXTRACTION", "REASONING")
        assert result.strategies["S3"] == StrategyStatus.RESTRICTED
        assert result.restrictions["S3"] == pytest.approx(0.97)

    def test_ambiguous_all_strategies_covered(self):
        """AMBIGUOUS intersection matrix must cover all 7 strategies."""
        result = ambiguous_matrix("EXTRACTION", "CHAT_OPEN")
        assert set(result.strategies.keys()) == {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}

    def test_ambiguous_allowed_when_both_allowed(self):
        """A strategy ALLOWED in both classes must be ALLOWED in the intersection."""
        result = ambiguous_matrix("SUMMARIZATION", "GENERATION_FACTUAL")
        assert result.strategies["S1"] == StrategyStatus.ALLOWED
        assert result.strategies["S2"] == StrategyStatus.ALLOWED


# ===========================================================================
# CR-11-d: Tools present in ICR always yields AGENT_TOOL_LOOP
# §3.4 Rule 1: "Tools present in ICR → AGENT_TOOL_LOOP (confidence 0.95)"
# ===========================================================================

class TestCR11_ToolsPresent:

    def test_tools_present_always_agent(self):
        """
        CR-11: an ICR with tools must always classify as AGENT_TOOL_LOOP,
        regardless of other signals in the query (§3.4 Rule 1 has highest precedence).
        """
        icr = _icr(
            user_text="summarize the document and extract all entities",
            tools=[_tool("search")],
        )
        result = classify(icr)
        assert result.primary == "AGENT_TOOL_LOOP", (
            "CR-11: tools-present rule must beat all other signals (§3.4 Rule 1)"
        )
        assert result.confidence == pytest.approx(0.95)

    def test_tools_present_beats_summarize(self):
        """Tools beat summarization signal — precedence test."""
        icr = _icr(
            user_text="tl;dr of this document",
            tools=[_tool("fetch")],
        )
        result = classify(icr)
        assert result.primary == "AGENT_TOOL_LOOP"

    def test_tools_present_beats_extract(self):
        """Tools beat extraction signal — precedence test."""
        icr = _icr(
            user_text="extract all names from the doc",
            tools=[_tool("lookup")],
        )
        result = classify(icr)
        assert result.primary == "AGENT_TOOL_LOOP"

    def test_no_tools_no_agent(self):
        """Without tools, AGENT_TOOL_LOOP must not be the result for a plain query."""
        icr = _icr(user_text="Tell me a joke.")
        result = classify(icr)
        assert result.primary != "AGENT_TOOL_LOOP"


# ===========================================================================
# Additional classifier rule tests — precedence + class correctness
# ===========================================================================

class TestClassifierRules:

    def test_default_fallback_chat_open(self):
        """When no rules fire, default must be CHAT_OPEN (confidence 0.65)."""
        icr = _icr(user_text="Hello there!")
        result = classify(icr)
        assert result.primary == "CHAT_OPEN"
        assert result.confidence == pytest.approx(0.65)

    def test_math_keyword_reasoning(self):
        """'calculate' must trigger REASONING."""
        icr = _icr(user_text="calculate the compound interest over 10 years")
        result = classify(icr)
        assert result.primary == "REASONING"

    def test_code_keyword_reasoning(self):
        """'write a function' must trigger REASONING (code generation)."""
        icr = _icr(user_text="write a function to sort a list in Python")
        result = classify(icr)
        assert result.primary == "REASONING"

    def test_creative_generation(self):
        """Creative trigger + marker must yield GENERATION_CREATIVE."""
        icr = _icr(user_text="write a short story about a lost robot")
        result = classify(icr)
        assert result.primary == "GENERATION_CREATIVE"
        assert result.confidence == pytest.approx(0.82)

    def test_factual_generation(self):
        """Factual trigger without creative marker must yield GENERATION_FACTUAL."""
        icr = _icr(user_text="write a report on quarterly sales performance")
        result = classify(icr)
        assert result.primary == "GENERATION_FACTUAL"

    def test_classify_keyword_short_query(self):
        """'classify' + short query must yield CLASSIFICATION_SHORT."""
        icr = _icr(user_text="classify this customer feedback as positive or negative")
        result = classify(icr)
        assert result.primary == "CLASSIFICATION_SHORT"
        assert result.confidence == pytest.approx(0.87)

    def test_ambiguous_flag_set_below_threshold(self):
        """confidence < 0.6 must set ambiguous=True on ClassifierResult."""
        from itol.icr import ClassifierResult
        result = ClassifierResult(primary="CHAT_OPEN", confidence=0.55)
        assert result.ambiguous is True

    def test_top2_has_two_entries(self):
        """top2 must always contain exactly 2 entries."""
        icr = _icr(user_text="Hello!")
        result = classify(icr)
        assert len(result.top2) == 2

    def test_top2_primary_is_first(self):
        """top2[0] must equal primary."""
        icr = _icr(user_text="calculate the area of a circle")
        result = classify(icr)
        assert result.top2[0] == result.primary

    def test_distribution_contains_primary(self):
        """distribution must contain the primary class."""
        icr = _icr(user_text="debug this code please")
        result = classify(icr)
        assert result.primary in result.distribution

    def test_s4_disabled_extraction(self):
        """§7.2: S4 must be DISABLED for EXTRACTION."""
        assert MATRIX["EXTRACTION"].strategies["S4"] == StrategyStatus.DISABLED

    def test_s3_restricted_extraction(self):
        """§7.2: S3 must be RESTRICTED for EXTRACTION with threshold 0.97."""
        entry = MATRIX["EXTRACTION"]
        assert entry.strategies["S3"] == StrategyStatus.RESTRICTED
        assert entry.restrictions["S3"] == pytest.approx(0.97)

    def test_matrix_covers_all_classes(self):
        """MATRIX must have entries for all 8 request classes."""
        expected = {
            "EXTRACTION", "REASONING", "SUMMARIZATION", "GENERATION_FACTUAL",
            "GENERATION_CREATIVE", "CLASSIFICATION_SHORT", "AGENT_TOOL_LOOP", "CHAT_OPEN",
        }
        assert set(MATRIX.keys()) == expected

    def test_all_matrix_entries_have_all_strategies(self):
        """Every matrix entry must define S1–S7."""
        for cls, entry in MATRIX.items():
            assert set(entry.strategies.keys()) == {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}, (
                f"Matrix entry for {cls} is missing strategy keys"
            )
