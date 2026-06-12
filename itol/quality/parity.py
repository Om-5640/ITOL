"""
Parity scoring — §5.3.

compute_parity(resp_optimized, resp_original, manifest) -> float

Formula:
    0.5 × cos(emb(resp_opt), emb(resp_orig))
  + 0.3 × key_fact_overlap          (Jaccard of extracted entity/number sets)
  + 0.2 × task_checks               (mean of 3 binary checks)

key_fact_overlap: lightweight number/entity extraction (reuses regex grammars
from manifest.py) applied to both response texts; Jaccard of the two sets.

task_checks (each 0.0 or 1.0, averaged):
  1. JSON validity match — both valid or both invalid
  2. Length ratio in [0.5, 2.0]
  3. Refusal phrase match — both refuse or both don't refuse
"""

from __future__ import annotations

import json
import re

from itol.icr import ConstraintManifest, ICRResponse


# ---------------------------------------------------------------------------
# Entity / number extraction (subset of manifest.py grammars, no ICR needed)
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(
    r"\$[\d,]+(?:\.\d+)?[MBKkm]?|"          # money
    r"\b\d+(?:[,\d]*\.?\d+)?(?:%|percent)?\b|"  # numbers/percentages
    r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b",          # identifiers
)

_REFUSAL_RE = re.compile(
    r"\b(i cannot|i'm not able to|i am not able to|i apologize, but"
    r"|i'm unable to|i can't)\b",
    re.IGNORECASE,
)


def _extract_entities(text: str) -> frozenset[str]:
    return frozenset(_ENTITY_RE.findall(text))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    inter = a & b
    return len(inter) / len(union)


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _task_checks(text_opt: str, text_orig: str) -> float:
    scores: list[float] = []

    # 1. JSON validity match
    scores.append(1.0 if _is_valid_json(text_opt) == _is_valid_json(text_orig) else 0.0)

    # 2. Length ratio in [0.5, 2.0]
    len_orig = max(len(text_orig), 1)
    ratio = len(text_opt) / len_orig
    scores.append(1.0 if 0.5 <= ratio <= 2.0 else 0.0)

    # 3. Refusal phrase match
    ref_opt = bool(_REFUSAL_RE.search(text_opt))
    ref_orig = bool(_REFUSAL_RE.search(text_orig))
    scores.append(1.0 if ref_opt == ref_orig else 0.0)

    return sum(scores) / len(scores)


def _response_text(response: ICRResponse) -> str:
    parts: list[str] = []
    for block in response.content:
        if block.text:
            parts.append(block.text)
    return "\n".join(parts)


def compute_parity(
    resp_optimized: ICRResponse,
    resp_original: ICRResponse,
    manifest: ConstraintManifest,
) -> float:
    """
    Compute a [0, 1] parity score between the optimized and original responses.

    Higher = more similar.  Below 0.85 triggers circuit-breaker concern.
    """
    text_opt  = _response_text(resp_optimized)
    text_orig = _response_text(resp_original)

    # Cosine similarity of embeddings
    try:
        from itol.embed.onnx_embedder import cosine, embed
        vecs = embed([text_opt[:512], text_orig[:512]], model="minilm")
        cos_score = float(cosine(vecs[0], vecs[1]))
    except Exception:
        cos_score = 1.0 if text_opt == text_orig else 0.5

    # Key-fact Jaccard
    ents_opt  = _extract_entities(text_opt)
    ents_orig = _extract_entities(text_orig)
    fact_score = _jaccard(ents_opt, ents_orig)

    # Task checks
    task_score = _task_checks(text_opt, text_orig)

    return 0.5 * cos_score + 0.3 * fact_score + 0.2 * task_score
