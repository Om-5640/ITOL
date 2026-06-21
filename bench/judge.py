"""
Quality judging for benchmark results.

Priority:
  1. Exact Match (EM) — for RAG (HotpotQA has gold answers)
  2. Token-level F1 — for RAG fallback
  3. Response-pair Jaccard — for chat/agent (no gold; reuses parity.py logic)
  4. Entity coverage — for agent (final answer mentions correct entity)
  5. LLM tiebreaker — Groq llama-3.1-8b-instant, only when deterministic score
     is in the ambiguous range [0.85, 0.95]; documented in report methodology.

All scoring returns float in [0, 1]. Higher = better quality preservation.
"""
from __future__ import annotations

import json
import logging
import re
import string
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalization (following SQuAD / HotpotQA eval convention)
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles (HotpotQA standard)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize_answer(text).split()


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(gold) else 0.0


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

def token_f1(prediction: str, gold: str) -> float:
    pred_toks = _tokenize(prediction)
    gold_toks = _tokenize(gold)
    if not pred_toks or not gold_toks:
        return 1.0 if (not pred_toks and not gold_toks) else 0.0
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall    = len(common) / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Jaccard over n-gram sets (parity-style)
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def jaccard_parity(text_a: str, text_b: str) -> float:
    """Jaccard similarity on 2-gram token sets; used for chat/agent workloads."""
    ta = _tokenize(text_a)
    tb = _tokenize(text_b)
    if not ta and not tb:
        return 1.0
    a_set = _ngrams(ta, 2) if len(ta) >= 2 else set(tuple(t) for t in ta)
    b_set = _ngrams(tb, 2) if len(tb) >= 2 else set(tuple(t) for t in tb)
    if not a_set and not b_set:
        return 1.0
    union = a_set | b_set
    inter = a_set & b_set
    return len(inter) / len(union)


# ---------------------------------------------------------------------------
# Length ratio check
# ---------------------------------------------------------------------------

def length_ratio_ok(text_a: str, text_b: str) -> float:
    """1.0 if len ratio in [0.5, 2.0], else 0.0."""
    la, lb = max(1, len(text_a.split())), max(1, len(text_b.split()))
    ratio = la / lb
    return 1.0 if 0.5 <= ratio <= 2.0 else 0.0


# ---------------------------------------------------------------------------
# Entity coverage (agent workload)
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9_-]{2,}\b|"      # PascalCase / camelCase identifiers
    r"\b(?:tool|function|api|endpoint)\b", # tool-loop keywords
    re.IGNORECASE,
)


def entity_coverage(response: str, gold_entities: list[str]) -> float:
    """Check what fraction of expected entities appear in the response."""
    if not gold_entities:
        return 1.0
    resp_lower = response.lower()
    found = sum(1 for e in gold_entities if e.lower() in resp_lower)
    return found / len(gold_entities)


# ---------------------------------------------------------------------------
# Refusal match
# ---------------------------------------------------------------------------

_REFUSAL_RE = re.compile(
    r"\b(i cannot|i'm not able|i am not able|i apologize, but|i can't|"
    r"i'm unable|unable to|cannot assist|not able to help)\b",
    re.IGNORECASE,
)


def refusal_match(text_a: str, text_b: str) -> float:
    """1.0 if both refuse or both don't; 0.0 if one refuses and the other doesn't."""
    ra = bool(_REFUSAL_RE.search(text_a))
    rb = bool(_REFUSAL_RE.search(text_b))
    return 1.0 if ra == rb else 0.0


# ---------------------------------------------------------------------------
# Composite quality score per workload type
# ---------------------------------------------------------------------------

def score_rag(
    response_itol: str,
    response_baseline: str,
    gold_answer: Optional[str],
) -> tuple[float, str]:
    """
    RAG quality: EM + F1 on gold (if available); otherwise Jaccard parity.
    Returns (score, method_name).
    """
    if gold_answer:
        em = exact_match(response_itol, gold_answer)
        f1 = token_f1(response_itol, gold_answer)
        # Blend: prefer EM when it fires, F1 as fallback
        score = max(em, f1 * 0.9)
        return min(1.0, score), "em_f1"
    # No gold: use parity against baseline
    j = jaccard_parity(response_itol, response_baseline)
    lr = length_ratio_ok(response_itol, response_baseline)
    rm = refusal_match(response_itol, response_baseline)
    return (j * 0.6 + lr * 0.2 + rm * 0.2), "jaccard"


def score_agent(
    response_itol: str,
    response_baseline: str,
    gold_entities: Optional[list[str]] = None,
) -> tuple[float, str]:
    """Agent quality: entity coverage + Jaccard parity."""
    ec = entity_coverage(response_itol, gold_entities or [])
    j  = jaccard_parity(response_itol, response_baseline)
    lr = length_ratio_ok(response_itol, response_baseline)
    score = ec * 0.5 + j * 0.35 + lr * 0.15
    return min(1.0, score), "entity_jaccard"


def score_chat(
    response_itol: str,
    response_baseline: str,
) -> tuple[float, str]:
    """Chat quality: Jaccard parity + length ratio + refusal match."""
    j  = jaccard_parity(response_itol, response_baseline)
    lr = length_ratio_ok(response_itol, response_baseline)
    rm = refusal_match(response_itol, response_baseline)
    return (j * 0.6 + lr * 0.2 + rm * 0.2), "jaccard"


def score_faq(
    response_itol: str,
    response_baseline: str,
    cache_hit: bool = False,
) -> tuple[float, str]:
    """
    FAQ: exact match on cache hits (cached response should be identical).
    For cache misses (passthrough), use Jaccard parity.
    """
    if cache_hit:
        em = 1.0 if response_itol.strip() == response_baseline.strip() else 0.7
        return em, "cache_exact"
    j  = jaccard_parity(response_itol, response_baseline)
    return j, "jaccard"


def judge(
    workload: str,
    response_itol: str,
    response_baseline: str,
    gold_answer: Optional[str] = None,
    gold_entities: Optional[list[str]] = None,
    cache_hit: bool = False,
) -> tuple[float, str]:
    """
    Top-level quality judge dispatcher.
    Returns (score ∈ [0,1], method_name).
    """
    if workload == "rag":
        return score_rag(response_itol, response_baseline, gold_answer)
    elif workload == "agent":
        return score_agent(response_itol, response_baseline, gold_entities)
    elif workload == "chat":
        return score_chat(response_itol, response_baseline)
    elif workload == "faq":
        return score_faq(response_itol, response_baseline, cache_hit=cache_hit)
    else:
        return score_chat(response_itol, response_baseline)


# ---------------------------------------------------------------------------
# LLM tiebreaker (Groq llama-3.1-8b-instant, free)
# ---------------------------------------------------------------------------

_LLM_JUDGE_PROMPT = """\
You are a quality evaluator. Compare two responses to the same question.

Question: {question}

Response A (original):
{response_a}

Response B (optimized):
{response_b}

Is Response B materially worse than Response A in terms of correctness,
completeness, and helpfulness? Answer with exactly one word: YES or NO."""


async def llm_judge(
    question: str,
    response_original: str,
    response_optimized: str,
    provider_name: str = "groq",
    model: str = "llama-3.1-8b-instant",
    api_key: Optional[str] = None,
) -> tuple[float, str]:
    """
    Use a small free LLM as a tiebreaker judge.
    Returns (1.0 if not worse, 0.0 if worse), method="llm_judge".
    Only called when deterministic score is in ambiguous range [0.85, 0.95].
    """
    import os
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        logger.debug("LLM judge skipped: no GROQ_API_KEY")
        return 0.90, "llm_judge_skipped"

    prompt = _LLM_JUDGE_PROMPT.format(
        question=question[:300],
        response_a=response_original[:600],
        response_b=response_optimized[:600],
    )

    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
            )
        if resp.status_code != 200:
            return 0.90, "llm_judge_error"
        answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        score = 0.0 if answer.startswith("YES") else 1.0
        return score, "llm_judge"
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return 0.90, "llm_judge_error"


async def maybe_llm_tiebreak(
    det_score: float,
    workload: str,
    question: str,
    response_baseline: str,
    response_itol: str,
) -> tuple[float, str]:
    """Call LLM judge only when deterministic score is in the ambiguous band."""
    if 0.85 <= det_score <= 0.95:
        llm_score, method = await llm_judge(question, response_baseline, response_itol)
        # Blend: 70% deterministic, 30% LLM
        return det_score * 0.7 + llm_score * 0.3, f"blended_{method}"
    return det_score, "deterministic"
