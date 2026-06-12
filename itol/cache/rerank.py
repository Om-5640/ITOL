"""
Cross-encoder reranker — §6.1 L1 verification rerank.

Backend priority (first available wins):
  1. ONNX cross-encoder (onnxruntime + transformers, local model path from config)
  2. sentence-transformers CrossEncoder (CPU inference)
  3. Token-overlap Jaccard  (pure Python, no ML — always works in CI)

Acceptance rule: bi_cos ≥ τ_class AND cross_score ≥ 0.85.
The rerank() function returns a list of floats in [0, 1], one per candidate.
"""

from __future__ import annotations

import math
import re
import threading
from typing import Any


_CROSS_SCORE_FLOOR = 0.85

# ---------------------------------------------------------------------------
# Fallback — token-overlap Jaccard
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", text.lower()))


def _jaccard_overlap(a: str, b: str) -> float:
    ta = _tokenise(a)
    tb = _tokenise(b)
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class _JaccardBackend:
    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        return [_jaccard_overlap(query, c) for c in candidates]


# ---------------------------------------------------------------------------
# sentence-transformers CrossEncoder backend
# ---------------------------------------------------------------------------

class _STCrossEncoderBackend:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        pairs = [(query, c) for c in candidates]
        logits = self._model.predict(pairs)
        # Normalise logits to [0, 1] via sigmoid
        return [float(1.0 / (1.0 + math.exp(-float(x)))) for x in logits]


# ---------------------------------------------------------------------------
# ONNX cross-encoder backend
# ---------------------------------------------------------------------------

class _ONNXCrossEncoderBackend:
    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            model_path + "/model.onnx",
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        import numpy as np
        scores: list[float] = []
        for candidate in candidates:
            encoded = self._tokenizer(
                query, candidate,
                truncation=True, max_length=512,
                return_tensors="np", padding=True,
            )
            output = self._session.run(None, dict(encoded))
            logit = float(output[0].squeeze())
            scores.append(float(1.0 / (1.0 + math.exp(-logit))))
        return scores


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_backend: Any = None
_backend_lock = threading.Lock()
_rerank_model_path: str | None = None


def configure(model_path: str) -> None:
    """Set the cross-encoder model path before first rerank() call."""
    global _rerank_model_path
    _rerank_model_path = model_path


def _get_backend() -> Any:
    global _backend
    with _backend_lock:
        if _backend is not None:
            return _backend

        if _rerank_model_path:
            try:
                _backend = _ONNXCrossEncoderBackend(_rerank_model_path)
                return _backend
            except Exception:
                pass

        try:
            _backend = _STCrossEncoderBackend("cross-encoder/ms-marco-MiniLM-L6-v2")
            return _backend
        except Exception:
            pass

        _backend = _JaccardBackend()
        return _backend


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: list[str]) -> list[float]:
    """
    Score each candidate against query.
    Returns a list of floats in [0, 1]; higher = more relevant.
    """
    if not candidates:
        return []
    try:
        return _get_backend().rerank(query, candidates)
    except Exception:
        return [_jaccard_overlap(query, c) for c in candidates]
