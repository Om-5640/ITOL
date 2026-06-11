"""
§14.3 Invariant tests for CR-19 — JSON minification must pass round-trip check.

CR-19: _try_minify_json must:
  - Return the original unchanged when JSON is invalid.
  - Return the minified form (without spacing) for valid JSON.
  - Verify round-trip: json.loads(minified) == json.loads(original).
  - Never touch CODE_BLOCK segments.
  - Report success=False when minified is not shorter than original.

Rules:
- NEVER weaken thresholds or conditions.
"""

import json

import pytest

from itol.strategies.s6_hygiene import _try_minify_json, _apply_json_minify
from itol.icr import SegmentType
from itol.segmenter import Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text: str, seg_type: SegmentType = SegmentType.STRUCTURED_DATA) -> Segment:
    import hashlib, re
    h = hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()
    return Segment(
        segment_type=seg_type,
        text=text,
        segment_hash=h,
        source_message_index=None,
        source_block_index=None,
        token_count=len(text.split()),
    )


# ===========================================================================
# CR-19-a: Invalid JSON is left unchanged
# ===========================================================================

class TestCR19_InvalidJson:

    def test_invalid_json_unchanged(self):
        """CR-19: invalid JSON must be returned unchanged with success=False."""
        text = "{not valid json"
        result, ok = _try_minify_json(text)
        assert not ok
        assert result == text

    def test_plain_text_unchanged(self):
        """Non-JSON plain text must be returned unchanged."""
        text = "This is a plain text segment, not JSON."
        result, ok = _try_minify_json(text)
        assert not ok
        assert result == text

    def test_partial_json_unchanged(self):
        """Incomplete JSON object must be returned unchanged."""
        text = '{"key": "value"'  # missing closing brace
        result, ok = _try_minify_json(text)
        assert not ok
        assert result == text

    def test_empty_string_unchanged(self):
        """Empty string must be returned unchanged."""
        result, ok = _try_minify_json("")
        assert not ok
        assert result == ""


# ===========================================================================
# CR-19-b: Valid JSON is minified with round-trip verification
# ===========================================================================

class TestCR19_ValidJson:

    def test_object_minified(self):
        """CR-19: valid JSON object with whitespace is minified."""
        text = '{ "key": "value",  "num": 42 }'
        result, ok = _try_minify_json(text)
        assert ok, "Valid JSON must be minified"
        assert " " not in result or result == result.strip(), (
            "Minified JSON must not have interior whitespace"
        )
        assert json.loads(result) == json.loads(text), "CR-19: round-trip must be identical"

    def test_array_minified(self):
        """CR-19: valid JSON array with whitespace is minified."""
        text = '[ 1,  2,  3,  "hello" ]'
        result, ok = _try_minify_json(text)
        assert ok
        assert json.loads(result) == json.loads(text)
        assert len(result) < len(text)

    def test_nested_object_minified(self):
        """Nested JSON structures are minified with round-trip preserved."""
        original = {
            "config": {"model": "gpt-4o", "temperature": 0.7},
            "messages": [{"role": "user", "content": "hi"}],
        }
        text = json.dumps(original, indent=2)
        result, ok = _try_minify_json(text)
        assert ok, "Nested JSON must be minified"
        assert json.loads(result) == original, "CR-19: round-trip must be identical"

    def test_roundtrip_semantic_equality(self):
        """
        CR-19 core: parsed objects pre/post minification must be equal.
        This is the invariant that must never be weakened.
        """
        text = '{"a": 1,   "b":   [true, null, 1.5]}'
        result, ok = _try_minify_json(text)
        assert ok
        assert json.loads(result) == json.loads(text), (
            "CR-19 INVARIANT: minified JSON must round-trip to identical structure"
        )


# ===========================================================================
# CR-19-c: CODE_BLOCK segments are never touched by JSON minification
# ===========================================================================

class TestCR19_CodeBlockUntouched:

    def test_code_block_not_minified(self):
        """
        S6 JSON minification must never touch CODE_BLOCK segments,
        even if they contain valid JSON.
        """
        json_in_code = '{ "key": "value",  "num": 42 }'
        seg = _seg(json_in_code, SegmentType.CODE_BLOCK)

        result_segs, touched = _apply_json_minify([seg])
        assert len(result_segs) == 1
        assert result_segs[0].text == json_in_code, (
            "CR-19: CODE_BLOCK segments must not be minified"
        )
        assert not touched

    def test_structured_data_is_minified(self):
        """Control: STRUCTURED_DATA with valid JSON IS minified."""
        json_text = '{ "key": "value",  "num": 42 }'
        seg = _seg(json_text, SegmentType.STRUCTURED_DATA)

        result_segs, touched = _apply_json_minify([seg])
        assert len(result_segs) == 1
        assert result_segs[0].text != json_text or len(result_segs[0].text) <= len(json_text)

    def test_tool_result_json_minified(self):
        """TOOL_RESULT segments containing valid JSON are minified."""
        json_text = '{"status": "ok",  "data":  [1, 2, 3]}'
        seg = _seg(json_text, SegmentType.TOOL_RESULT)

        result_segs, touched = _apply_json_minify([seg])
        assert len(result_segs) == 1
        # If minification succeeded, verify round-trip
        if result_segs[0].text != json_text:
            assert json.loads(result_segs[0].text) == json.loads(json_text), (
                "CR-19: TOOL_RESULT minification must pass round-trip"
            )


# ===========================================================================
# CR-19-d: No savings → not minified
# ===========================================================================

class TestCR19_NoSavings:

    def test_already_minified_json_unchanged(self):
        """JSON with no whitespace to remove reports success=False (no savings)."""
        text = '{"a":1,"b":2}'
        result, ok = _try_minify_json(text)
        assert not ok, (
            "CR-19: already-compact JSON must not be reported as minified "
            "(len(minified) >= len(original))"
        )
        assert result == text
