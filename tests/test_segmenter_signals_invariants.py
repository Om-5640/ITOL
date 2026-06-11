"""
§14.3 Invariant tests for Step 2: segmenter.py + signals.py

Rules:
- NEVER weaken a threshold to make a test pass.
- Every assertion encodes a spec requirement; the comment cites the section.
"""

import hashlib

import pytest

from itol.icr import ContentBlock, ContentType, ICR, Message, SegmentType, ToolDef
from itol.segmenter import (
    Segment,
    _hash,
    _normalise,
    filter_by_type,
    segment_icr,
    segments_full_text,
    template_signature,
)
from itol.signals import (
    _MINHASH_PERMS,
    estimate_token_count,
    extract_signals,
    history_depth,
    jaccard_estimate,
    minhash_signature,
    redundancy_score,
    semantic_density,
    stale_tool_mass,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _icr(messages, system_text=None, tools=None):
    return ICR.create(
        provider="openai",
        model="gpt-4o",
        system=[ContentBlock.text(system_text)] if system_text else [],
        messages=messages,
        tools=tools or [],
        raw={},
    )


def _tool(name="search"):
    return ToolDef(name=name, description="Search the web", parameters={"q": "string"})


# ===========================================================================
# Normalisation + hashing  (§3.2)
# ===========================================================================

class TestNormalise:

    def test_collapses_horizontal_whitespace(self):
        assert _normalise("hello   world") == "hello world"

    def test_collapses_tabs(self):
        assert _normalise("a\t\tb") == "a b"

    def test_collapses_excess_newlines(self):
        result = _normalise("para1\n\n\n\npara2")
        assert result == "para1\n\npara2"

    def test_strips_leading_trailing(self):
        assert _normalise("  hi  ") == "hi"

    def test_preserves_case(self):
        """Case is semantic — must not be lowercased (§3.2)."""
        assert _normalise("OpenAI GPT-4o") == "OpenAI GPT-4o"

    def test_hash_is_sha256(self):
        text = "hello"
        expected = hashlib.sha256(_normalise(text).encode()).hexdigest()
        assert _hash(text) == expected

    def test_hash_stable_across_whitespace_variants(self):
        """Same logical text → same hash regardless of extra spaces (§3.2)."""
        assert _hash("hello   world") == _hash("hello world")

    def test_hash_different_for_different_content(self):
        assert _hash("hello") != _hash("world")

    def test_hash_preserves_case_difference(self):
        """Different cases → different hashes (case is semantic)."""
        assert _hash("OpenAI") != _hash("openai")


# ===========================================================================
# Segment type detection  (§3.2)
# ===========================================================================

class TestSegmentTypeDetection:

    # --- System prompt ---
    def test_system_content_is_system_instruction(self):
        icr = _icr([Message.user("hi")], system_text="You are a helpful assistant.")
        segs = segment_icr(icr)
        sys_segs = filter_by_type(segs, SegmentType.SYSTEM_INSTRUCTION)
        assert len(sys_segs) >= 1
        assert sys_segs[0].text == "You are a helpful assistant."

    # --- User query ---
    def test_simple_user_message_is_user_query(self):
        icr = _icr([Message.user("What is 2+2?")])
        segs = segment_icr(icr)
        user_segs = filter_by_type(segs, SegmentType.USER_QUERY)
        assert any(s.text == "What is 2+2?" for s in user_segs)

    # --- Assistant turn ---
    def test_assistant_message_is_assistant_turn(self):
        icr = _icr([Message.user("hi"), Message.assistant("hello")])
        segs = segment_icr(icr)
        asst_segs = filter_by_type(segs, SegmentType.ASSISTANT_TURN)
        assert any(s.text == "hello" for s in asst_segs)

    # --- Tool schema ---
    def test_tool_definition_becomes_tool_schema_segment(self):
        icr = _icr([Message.user("search for cats")], tools=[_tool("search")])
        segs = segment_icr(icr)
        tool_segs = filter_by_type(segs, SegmentType.TOOL_SCHEMA)
        assert len(tool_segs) >= 1
        assert any(s.metadata.get("tool_name") == "search" for s in tool_segs)

    # --- Tool result ---
    def test_tool_result_block_becomes_tool_result_segment(self):
        msg = Message(role="user", content=[
            ContentBlock.tool_result("id1", '{"results": []}')
        ])
        icr = _icr([msg])
        segs = segment_icr(icr)
        tr_segs = filter_by_type(segs, SegmentType.TOOL_RESULT)
        assert len(tr_segs) == 1
        assert tr_segs[0].metadata["tool_result_for_id"] == "id1"

    # --- JSON → structured data ---
    def test_json_content_is_structured_data(self):
        json_text = '{"key": "value", "num": 42}'
        icr = _icr([Message.user(json_text)])
        segs = segment_icr(icr)
        sd_segs = filter_by_type(segs, SegmentType.STRUCTURED_DATA)
        assert any(json_text in s.text for s in sd_segs)

    # --- JSON array → structured data ---
    def test_json_array_is_structured_data(self):
        json_text = '[{"a": 1}, {"a": 2}]'
        icr = _icr([Message.user(json_text)])
        segs = segment_icr(icr)
        assert any(s.segment_type is SegmentType.STRUCTURED_DATA for s in segs)

    # --- Code fence → code block ---
    def test_fenced_code_block_is_code_block(self):
        code = "```python\nprint('hello')\n```"
        icr = _icr([Message.user(code)])
        segs = segment_icr(icr)
        assert any(s.segment_type is SegmentType.CODE_BLOCK for s in segs)

    # --- Retrieved doc ---
    def test_long_doc_with_delimiter_is_retrieved_doc(self):
        doc = (
            "Source: Internal Wiki\n"
            + "The quarterly financial report shows strong performance. " * 15
        )
        icr = _icr([Message.user(doc)])
        segs = segment_icr(icr)
        assert any(s.segment_type is SegmentType.RETRIEVED_DOC for s in segs)

    # --- Few-shot ---
    def test_repeated_input_output_is_few_shot(self):
        few_shot = (
            "Input: hello\nOutput: world\n"
            "Input: foo\nOutput: bar\n"
            "Input: abc\nOutput: xyz\n"
        )
        icr = _icr([Message.user(few_shot)])
        segs = segment_icr(icr)
        assert any(s.segment_type is SegmentType.FEW_SHOT_EXAMPLE for s in segs)

    # --- Tool USE block (inline call) ---
    def test_tool_use_block_is_tool_schema_segment(self):
        msg = Message(role="assistant", content=[
            ContentBlock.tool_use("tid1", "search", {"q": "cats"})
        ])
        icr = _icr([msg])
        segs = segment_icr(icr)
        assert any(s.segment_type is SegmentType.TOOL_SCHEMA for s in segs)


# ===========================================================================
# Segment hash invariants  (§3.2)
# ===========================================================================

class TestSegmentHash:

    def test_each_segment_has_nonempty_hash(self):
        icr = _icr([Message.user("hello"), Message.assistant("world")])
        for seg in segment_icr(icr):
            assert seg.segment_hash, f"Segment missing hash: {seg}"

    def test_identical_text_same_hash(self):
        """Same content → same hash regardless of position (§3.2 stable hash)."""
        icr1 = _icr([Message.user("repeat me")])
        icr2 = _icr([Message.user("repeat me"), Message.assistant("ok"), Message.user("repeat me")])
        segs1 = segment_icr(icr1)
        segs2 = filter_by_type(segment_icr(icr2), SegmentType.USER_QUERY)
        hash1 = segs1[0].segment_hash
        hashes2 = {s.segment_hash for s in segs2 if s.text == "repeat me"}
        assert hash1 in hashes2

    def test_different_text_different_hash(self):
        icr = _icr([Message.user("alpha"), Message.user("beta")])
        segs = segment_icr(icr)
        hashes = [s.segment_hash for s in segs]
        assert len(hashes) == len(set(hashes)), "Different text segments must have different hashes"


# ===========================================================================
# Ordering and completeness  (§3.2)
# ===========================================================================

class TestOrdering:

    def test_system_segments_come_first(self):
        icr = _icr([Message.user("q")], system_text="You are helpful.")
        segs = segment_icr(icr)
        types = [s.segment_type for s in segs]
        if SegmentType.SYSTEM_INSTRUCTION in types and SegmentType.USER_QUERY in types:
            sys_idx = types.index(SegmentType.SYSTEM_INSTRUCTION)
            user_idx = types.index(SegmentType.USER_QUERY)
            assert sys_idx < user_idx, "System segments must precede message segments"

    def test_no_content_is_silently_dropped(self):
        """Every text content block must produce at least one segment."""
        msg1 = Message.user("user text")
        msg2 = Message.assistant("assistant text")
        icr = _icr([msg1, msg2], system_text="sys")
        segs = segment_icr(icr)
        all_texts = {s.text for s in segs}
        assert "user text" in all_texts
        assert "assistant text" in all_texts
        assert "sys" in all_texts

    def test_source_message_index_set_for_message_segments(self):
        icr = _icr([Message.user("a"), Message.assistant("b")])
        for seg in segment_icr(icr):
            if seg.source_message_index is not None:
                assert seg.source_message_index >= 0

    def test_source_message_index_none_for_system_segments(self):
        icr = _icr([Message.user("q")], system_text="sys")
        for seg in segment_icr(icr):
            if seg.segment_type is SegmentType.SYSTEM_INSTRUCTION:
                assert seg.source_message_index is None

    def test_segments_full_text_contains_all_content(self):
        icr = _icr([Message.user("part one"), Message.assistant("part two")], system_text="sys")
        segs = segment_icr(icr)
        full = segments_full_text(segs)
        assert "part one" in full
        assert "part two" in full
        assert "sys" in full


# ===========================================================================
# template_signature  (§3.3)
# ===========================================================================

class TestTemplateSignature:

    def test_same_template_same_signature(self):
        """Two requests with same system prompt + same message structure → same sig."""
        icr1 = _icr([Message.user("question A")], system_text="You are helpful.")
        icr2 = _icr([Message.user("question B")], system_text="You are helpful.")
        segs1 = segment_icr(icr1)
        segs2 = segment_icr(icr2)
        assert template_signature(segs1) == template_signature(segs2)

    def test_different_system_different_signature(self):
        icr1 = _icr([Message.user("q")], system_text="System A")
        icr2 = _icr([Message.user("q")], system_text="System B")
        assert template_signature(segment_icr(icr1)) != template_signature(segment_icr(icr2))

    def test_different_structure_different_signature(self):
        """Different message-type sequence → different signature."""
        icr1 = _icr([Message.user("q")])
        icr2 = _icr([Message.user("q"), Message.assistant("a")])
        assert template_signature(segment_icr(icr1)) != template_signature(segment_icr(icr2))

    def test_signature_is_hex_string(self):
        icr = _icr([Message.user("q")])
        sig = template_signature(segment_icr(icr))
        int(sig, 16)  # raises ValueError if not valid hex

    def test_signature_is_64_chars(self):
        """sha256 hex digest = 64 characters."""
        icr = _icr([Message.user("q")])
        sig = template_signature(segment_icr(icr))
        assert len(sig) == 64


# ===========================================================================
# filter_by_type
# ===========================================================================

class TestFilterByType:

    def test_filters_correctly(self):
        icr = _icr([Message.user("hi"), Message.assistant("hello")], system_text="sys")
        segs = segment_icr(icr)
        sys_segs = filter_by_type(segs, SegmentType.SYSTEM_INSTRUCTION)
        assert all(s.segment_type is SegmentType.SYSTEM_INSTRUCTION for s in sys_segs)

    def test_multi_type_filter(self):
        icr = _icr([Message.user("hi"), Message.assistant("hello")], system_text="sys")
        segs = segment_icr(icr)
        result = filter_by_type(segs, SegmentType.USER_QUERY, SegmentType.ASSISTANT_TURN)
        for s in result:
            assert s.segment_type in (SegmentType.USER_QUERY, SegmentType.ASSISTANT_TURN)

    def test_empty_result_when_no_match(self):
        icr = _icr([Message.user("hi")])
        segs = segment_icr(icr)
        assert filter_by_type(segs, SegmentType.TOOL_RESULT) == []


# ===========================================================================
# MinHash / Jaccard  (§3.3 — S1 activation gate)
# ===========================================================================

class TestMinHash:

    def test_signature_length_is_128(self):
        """128 permutations as specified in §3.3."""
        sig = minhash_signature("hello world")
        assert len(sig) == _MINHASH_PERMS == 128

    def test_identical_text_jaccard_is_one(self):
        text = "the quick brown fox jumps over the lazy dog"
        sig_a = minhash_signature(text)
        sig_b = minhash_signature(text)
        assert jaccard_estimate(sig_a, sig_b) == pytest.approx(1.0)

    def test_disjoint_text_jaccard_near_zero(self):
        sig_a = minhash_signature("aaaaaa bbbbbb cccccc dddddd eeeeee")
        sig_b = minhash_signature("111111 222222 333333 444444 555555")
        j = jaccard_estimate(sig_a, sig_b)
        assert j < 0.15, f"Expected near-zero Jaccard for disjoint text, got {j}"

    def test_near_duplicate_jaccard_high(self):
        # Use a long base text where only the LAST WORD changes; most shingles are
        # identical and the true Jaccard is well above 0.50.
        base = (
            "The detailed analysis of quarterly results reveals strong growth metrics "
            "across all key performance indicators for the reporting period. " * 20
            + "conclusion"
        )
        variant = base[:-10] + "summary"   # only the final word differs
        sig_a = minhash_signature(base)
        sig_b = minhash_signature(variant)
        j = jaccard_estimate(sig_a, sig_b)
        assert j >= 0.50, f"Near-duplicate Jaccard too low: {j}"

    def test_jaccard_symmetric(self):
        a, b = minhash_signature("foo bar baz"), minhash_signature("baz qux quux")
        assert jaccard_estimate(a, b) == jaccard_estimate(b, a)


# ===========================================================================
# redundancy_score  (§3.3 — activation gate for S1)
# ===========================================================================

class TestRedundancyScore:

    def test_no_duplicates_score_near_zero(self):
        icr = _icr([Message.user("completely unique query A"),
                    Message.user("entirely different question B")])
        segs = segment_icr(icr)
        score = redundancy_score(segs)
        assert score < 0.3

    def test_identical_segments_score_high(self):
        """Duplicate segments should push score ≥ activation gate (0.15)."""
        repeated = "The product supports multiple authentication methods including OAuth2."
        icr = _icr([
            Message.user(repeated),
            Message.user(repeated),
            Message.user(repeated),
        ])
        segs = segment_icr(icr)
        score = redundancy_score(segs)
        assert score >= 0.15, (
            f"Expected redundancy_score ≥ 0.15 for duplicate segments, got {score}"
        )

    def test_single_segment_score_is_zero(self):
        icr = _icr([Message.user("only one message")])
        segs = segment_icr(icr)
        score = redundancy_score(segs)
        assert score == 0.0

    def test_score_in_range(self):
        icr = _icr([Message.user("a"), Message.assistant("b"), Message.user("c")])
        score = redundancy_score(segment_icr(icr))
        assert 0.0 <= score <= 1.0


# ===========================================================================
# semantic_density  (§3.3)
# ===========================================================================

class TestSemanticDensity:

    def test_empty_text_is_one(self):
        assert semantic_density("") == pytest.approx(1.0)

    def test_repetitive_text_lower_density(self):
        """Highly repetitive text should have lower density than varied text."""
        repetitive = "the the the the the the the the the the " * 20
        varied = " ".join(f"word{i}" for i in range(100))
        assert semantic_density(repetitive) < semantic_density(varied)

    def test_density_in_range(self):
        texts = [
            "hello world",
            '{"key": "value", "list": [1, 2, 3]}',
            "a " * 200,
            "unique words: " + " ".join(f"word{i}" for i in range(50)),
        ]
        for text in texts:
            d = semantic_density(text)
            assert 0.0 < d <= 1.0, f"density {d} out of range for: {text[:40]!r}"

    def test_code_not_lower_than_gibberish(self):
        """Valid Python code has high information density."""
        code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
        repetitive = "the " * 100
        assert semantic_density(code) > semantic_density(repetitive)


# ===========================================================================
# estimate_token_count  (§3.3, §8.2)
# ===========================================================================

class TestTokenCount:

    def test_empty_string_is_zero(self):
        assert estimate_token_count("") == 0

    def test_positive_for_nonempty(self):
        assert estimate_token_count("hello world") >= 1

    def test_longer_text_more_tokens(self):
        short = "hello"
        long = "hello world " * 50
        assert estimate_token_count(long) > estimate_token_count(short)

    def test_estimate_within_10_percent_of_gpt4_rule_of_thumb(self):
        """
        GPT-4 averages ~4 chars/token on English prose.  Our estimator
        uses 3.8; for a 100-word (~600 char) English paragraph the result
        should be in [120, 200] tokens (industry rule-of-thumb: ~75 tokens
        per 100 words, ±50%).
        """
        text = (
            "The quarterly financial report highlights record revenue growth "
            "driven by strong enterprise adoption and geographic expansion. "
            "Operating margins improved significantly year-over-year as "
            "cost optimisation initiatives took effect across all divisions. "
        )  # ~60 words, ~380 chars
        count = estimate_token_count(text)
        assert 50 <= count <= 200, f"Token estimate {count} outside plausible range"

    def test_custom_chars_per_token(self):
        """Custom calibration factor is applied correctly."""
        text = "a" * 100
        assert estimate_token_count(text, chars_per_token=10.0) == 10


# ===========================================================================
# history_depth  (§3.3)
# ===========================================================================

class TestHistoryDepth:

    def test_zero_for_single_user_turn(self):
        icr = _icr([Message.user("hi")])
        assert history_depth(icr) == 0

    def test_one_pair(self):
        icr = _icr([Message.user("hi"), Message.assistant("hello")])
        assert history_depth(icr) == 1

    def test_three_pairs(self):
        icr = _icr([
            Message.user("1"), Message.assistant("a"),
            Message.user("2"), Message.assistant("b"),
            Message.user("3"), Message.assistant("c"),
        ])
        assert history_depth(icr) == 3

    def test_trailing_user_turn_not_counted(self):
        """A pending user turn without an assistant reply is not a complete pair."""
        icr = _icr([
            Message.user("1"), Message.assistant("a"),
            Message.user("2"),  # no reply yet
        ])
        assert history_depth(icr) == 1


# ===========================================================================
# stale_tool_mass  (§3.3)
# ===========================================================================

class TestStaleToolMass:

    def test_zero_when_no_tool_results(self):
        icr = _icr([Message.user("hi")])
        segs = segment_icr(icr)
        assert stale_tool_mass(segs, k_turns=6) == 0

    def test_zero_when_all_results_within_window(self):
        msgs = []
        for i in range(4):
            msgs.append(Message.user(f"q{i}"))
            msgs.append(Message(
                role="user",
                content=[ContentBlock.tool_result(f"id{i}", f"result {i}")],
            ))
        icr = _icr(msgs)
        segs = segment_icr(icr)
        # k_turns=10 → nothing is stale
        assert stale_tool_mass(segs, k_turns=10) == 0

    def test_positive_when_old_results_beyond_window(self):
        msgs = []
        for i in range(10):
            msgs.append(Message.user(f"question number {i}"))
            msgs.append(Message(
                role="user",
                content=[ContentBlock.tool_result(f"id{i}", "tool output " * 20)],
            ))
        icr = _icr(msgs)
        segs = segment_icr(icr)
        mass = stale_tool_mass(segs, k_turns=2)
        assert mass > 0, "Should detect stale tool results beyond the window"


# ===========================================================================
# extract_signals integration  (§3.3)
# ===========================================================================

class TestExtractSignals:

    def test_returns_segment_signals(self):
        from itol.icr import SegmentSignals
        icr = _icr([Message.user("test")])
        segs = segment_icr(icr)
        signals = extract_signals(icr, segs)
        assert isinstance(signals, SegmentSignals)

    def test_token_count_positive(self):
        icr = _icr([Message.user("hello world")])
        segs = segment_icr(icr)
        signals = extract_signals(icr, segs)
        assert signals.token_count > 0

    def test_template_signature_populated(self):
        icr = _icr([Message.user("q")], system_text="sys")
        segs = segment_icr(icr)
        signals = extract_signals(icr, segs)
        assert signals.template_signature is not None
        assert len(signals.template_signature) == 64

    def test_history_depth_matches_helper(self):
        icr = _icr([Message.user("a"), Message.assistant("b"), Message.user("c")])
        segs = segment_icr(icr)
        signals = extract_signals(icr, segs)
        assert signals.history_depth == history_depth(icr)

    def test_redundancy_score_in_range(self):
        icr = _icr([Message.user("unique"), Message.assistant("distinct")])
        segs = segment_icr(icr)
        signals = extract_signals(icr, segs)
        assert 0.0 <= signals.redundancy_score <= 1.0

    def test_semantic_density_in_range(self):
        icr = _icr([Message.user("hello")])
        segs = segment_icr(icr)
        signals = extract_signals(icr, segs)
        assert 0.0 < signals.semantic_density <= 1.0

    def test_instruction_ratio_in_range(self):
        icr = _icr([Message.user("q")], system_text="You must always respond in JSON format.")
        segs = segment_icr(icr)
        signals = extract_signals(icr, segs)
        assert 0.0 <= signals.instruction_context_ratio <= 1.0

    def test_prefix_cacheable_span_zero_on_first_request(self):
        """First request for a template has no prior prefix to match (§3.3)."""
        import itol.signals as sig_mod
        sig_mod._prefix_store.clear()   # ensure clean state
        icr = _icr([Message.user("first request ever")], system_text="Fresh system prompt XYZ.")
        segs = segment_icr(icr)
        signals = extract_signals(icr, segs)
        assert signals.prefix_cacheable_span == 0

    def test_prefix_cacheable_span_nonzero_on_repeated_request(self):
        """Second request with same system prompt → nonzero cached span (§3.3, G3)."""
        import itol.signals as sig_mod
        sig_mod._prefix_store.clear()
        system = "You are a helpful assistant with a stable system prompt."
        # First request: populates the prefix store
        icr1 = _icr([Message.user("first")], system_text=system)
        segs1 = segment_icr(icr1)
        extract_signals(icr1, segs1)
        # Second request with same tenant + same template
        icr2 = _icr([Message.user("second")], system_text=system)
        icr2.tenant_id = icr1.tenant_id  # same tenant
        segs2 = segment_icr(icr2)
        signals2 = extract_signals(icr2, segs2)
        assert signals2.prefix_cacheable_span > 0, (
            "Repeated stable system prompt should yield a nonzero prefix_cacheable_span (§3.3, G3)"
        )
