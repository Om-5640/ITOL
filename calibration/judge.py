"""
calibration/judge.py — deterministic quality judge + optional Ollama tiebreaker.

Primary judgement: ROUGE-L + constraint satisfaction (no external calls).
Tiebreaker: Ollama (local LLM) called only when ROUGE-L delta < 0.05.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from calibration.fit import rouge_l, _constraint_present_in_text

_log = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "mistral"
_TIEBREAK_THRESHOLD = 0.05   # call Ollama only when ROUGE-L delta < this


def deterministic_score(
    output: str,
    reference: str,
    constraints: list[dict],
) -> dict[str, float]:
    """
    Return {"rouge_l", "constraint_recall", "combined"}.
    combined = 0.6*rouge_l + 0.4*constraint_recall
    """
    rl = rouge_l(output, reference)
    if constraints:
        satisfied = sum(
            1 for c in constraints if _constraint_present_in_text(c, output)
        )
        cr = satisfied / len(constraints)
    else:
        cr = 1.0
    combined = 0.6 * rl + 0.4 * cr
    return {"rouge_l": round(rl, 4), "constraint_recall": round(cr, 4), "combined": round(combined, 4)}


def _ollama_prefer(a: str, b: str, context: str, timeout: int = 15) -> str | None:
    """
    Ask Ollama to pick the better completion (returns 'a', 'b', or None if unavailable).
    """
    prompt = (
        f"Given the instruction: {context}\n\n"
        f"Which output is better?\nA: {a}\nB: {b}\n\n"
        "Answer with just the letter A or B."
    )
    payload = json.dumps({
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        _OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            answer = data.get("response", "").strip().upper()
            if answer.startswith("A"):
                return "a"
            if answer.startswith("B"):
                return "b"
            return None
    except (urllib.error.URLError, Exception) as exc:
        _log.debug("Ollama tiebreaker unavailable: %s", exc)
        return None


def judge_pair(
    output_a: str,
    output_b: str,
    reference: str,
    constraints: list[dict],
    instruction: str = "",
    use_ollama: bool = True,
) -> dict[str, Any]:
    """
    Compare two outputs.

    Returns:
        {
          "winner": "a" | "b" | "tie",
          "scores_a": {...},
          "scores_b": {...},
          "tiebreaker_used": bool,
        }
    """
    scores_a = deterministic_score(output_a, reference, constraints)
    scores_b = deterministic_score(output_b, reference, constraints)
    delta = abs(scores_a["combined"] - scores_b["combined"])

    tiebreaker_used = False
    if delta >= _TIEBREAK_THRESHOLD:
        winner = "a" if scores_a["combined"] >= scores_b["combined"] else "b"
    else:
        if use_ollama:
            preferred = _ollama_prefer(output_a, output_b, instruction)
            if preferred is not None:
                winner = preferred
                tiebreaker_used = True
            else:
                winner = "tie"
        else:
            winner = "tie"

    return {
        "winner": winner,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "tiebreaker_used": tiebreaker_used,
    }
