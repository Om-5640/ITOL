"""
Tests for scripts/prepare_hotpotqa_slice.py — turn-2 question generation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from prepare_hotpotqa_slice import _generate_turn2, _enrich_turns

_SUPPORT = ["Scott Derrickson", "Ed Wood"]


# ---------------------------------------------------------------------------
# Unit tests for _generate_turn2
# ---------------------------------------------------------------------------

def test_yes_no_falls_back_to_entity():
    """Yes/no answers must reference a supporting entity, not 'yes' or 'no'."""
    t2 = _generate_turn2("yes", _SUPPORT, "comparison")
    assert "yes" not in t2.lower().split(), (
        f"Turn-2 should not contain bare 'yes': {t2!r}"
    )
    assert _SUPPORT[0] in t2

    t2 = _generate_turn2("no", _SUPPORT, "comparison")
    assert "no" not in t2.lower().split()
    assert _SUPPORT[0] in t2


def test_number_falls_back_to_entity():
    """Pure-number and number+unit answers must not produce digit-only follow-ups."""
    # Bare integer
    t2 = _generate_turn2("42", ["Androscoggin Bank Colisée", "Lewiston Maineiacs"], "bridge")
    assert "42" not in t2
    assert "Androscoggin Bank Colisée" in t2

    # Number with unit suffix (e.g. HotpotQA arena-capacity answer)
    t2 = _generate_turn2("3,677 seated", ["Androscoggin Bank Colisée", "Lewiston Maineiacs"], "bridge")
    assert "3,677" not in t2
    assert "Androscoggin Bank Colisée" in t2

    # Year as answer
    t2 = _generate_turn2("1999", ["Guns N' Roses", "End of Days"], "bridge")
    assert "1999" not in t2
    assert "Guns N' Roses" in t2


def test_entity_answer_used_directly():
    """A multi-word entity answer should be used verbatim in the follow-up."""
    t2 = _generate_turn2(
        "Chief of Protocol",
        ["Kiss and Tell (1945 film)", "Shirley Temple"],
        "bridge",
    )
    assert t2 == "Tell me more about Chief of Protocol."


def test_entity_answer_single_word_name():
    """A single-word entity name (not yes/no/number) is used as-is."""
    t2 = _generate_turn2("Animorphs", ["The Hork-Bajir Chronicles", "Animorphs"], "bridge")
    assert t2 == "Tell me more about Animorphs."


def test_empty_answer_falls_back_to_entity():
    """Empty or single-char answer falls back to supporting entity."""
    t2 = _generate_turn2("", _SUPPORT, "bridge")
    assert _SUPPORT[0] in t2

    t2 = _generate_turn2("A", _SUPPORT, "bridge")
    assert _SUPPORT[0] in t2


# ---------------------------------------------------------------------------
# Integration test: scan the bundled corpus
# ---------------------------------------------------------------------------

_BUNDLED = Path(__file__).parent.parent / "data" / "bench_corpora" / "hotpotqa_150_real.jsonl"


@pytest.mark.skipif(not _BUNDLED.exists(), reason="bundled corpus not generated yet")
def test_all_150_have_nonsense_free_turn2():
    """
    All examples in the bundled corpus must have a sensible turn-2 question —
    no 'Tell me more about yes/no/digits.'
    """
    bad: list[tuple[str, str]] = []
    with open(_BUNDLED, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                continue
            tqs = obj.get("turn_questions", [])
            if len(tqs) < 2:
                bad.append((obj["id"], "(missing turn_questions)"))
                continue
            t2 = tqs[1].strip().lower()
            # Must not be verbatim yes/no answer
            for bad_phrase in ("tell me more about yes.", "tell me more about no."):
                if t2 == bad_phrase:
                    bad.append((obj["id"], tqs[1]))
            # Must not be a pure-digit entity in the follow-up
            # e.g. "Tell me more about 42." or "Tell me more about 3677."
            import re
            m = re.match(r"tell me more about ([0-9,\s]+)\.", t2)
            if m:
                bad.append((obj["id"], tqs[1]))

    assert not bad, (
        f"{len(bad)} examples have nonsense turn-2 questions:\n"
        + "\n".join(f"  {eid}: {q!r}" for eid, q in bad[:10])
    )
