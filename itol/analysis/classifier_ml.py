"""
Stage B ML classifier — §5.3.

Logistic regression over a 73-dimensional feature vector:
  [0:5]   5 signal scalars  (redundancy_score, semantic_density,
                              instruction_context_ratio, history_depth/20,
                              token_count/10000)
  [5:69]  64-dim mean-pooled query embedding  (minilm 384-dim → reshape(64,6) → mean)
  [69:73] 4 verb-lexicon counts  (EXTRACTION, GENERATION_CREATIVE, REASONING,
                                   SUMMARIZATION) normalised by total tokens

Stage B is only invoked when Stage A (rule-based) emits confidence < 0.6
(ambiguous=True).  If the calibration file is absent, predict() returns None
and the system falls back to Stage A.

Weight file: itol/data/calibration/classifier_ml.json
  {"classes": [...], "W": [[float,...], ...], "b": [float,...], "feature_names": [...]}
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from itol.icr import ICR, ClassifierResult

_CALIB_DEFAULT = Path(__file__).parent.parent / "data" / "calibration" / "classifier_ml.json"

# Verb lexicon categories in stable order (must match training script)
_LEXICON_KEYS = ("EXTRACTION", "GENERATION_CREATIVE", "REASONING", "SUMMARIZATION")

_lexicon_cache: dict[str, list[str]] | None = None


def _load_lexicon() -> dict[str, list[str]]:
    global _lexicon_cache
    if _lexicon_cache is None:
        lex_path = Path(__file__).parent.parent / "data" / "verb_lexicon.json"
        with open(lex_path, encoding="utf-8") as fh:
            _lexicon_cache = json.load(fh)
    return _lexicon_cache


def _count_verbs(text: str) -> np.ndarray:
    """Return per-category verb counts as a length-4 float32 array, normalised by token count."""
    lex = _load_lexicon()
    tokens = text.lower().split()
    total = max(len(tokens), 1)
    counts = np.zeros(4, dtype=np.float32)
    for i, key in enumerate(_LEXICON_KEYS):
        for verb in lex.get(key, []):
            if " " in verb:
                counts[i] += text.lower().count(verb)
            else:
                counts[i] += tokens.count(verb)
    return counts / total


def _pool_embedding(vec: np.ndarray) -> np.ndarray:
    """Compress a 384-dim minilm vector to 64 dims via reshape+mean."""
    if vec.ndim == 1 and vec.shape[0] == 384:
        # reshape to (64, 6) then mean over axis=1
        return vec.reshape(64, 6).mean(axis=1).astype(np.float32)
    # already correct size or wrong shape — pad/truncate to 64
    out = np.zeros(64, dtype=np.float32)
    n = min(len(vec), 64)
    out[:n] = vec[:n]
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


@dataclass
class ClassifierML:
    """
    Pure-numpy logistic regression inference.

    Parameters loaded from calibration JSON; inference is a single
    matrix-vector multiply + softmax — no sklearn at runtime.
    """

    classes: list[str]
    W: np.ndarray      # shape (n_classes, 73)
    b: np.ndarray      # shape (n_classes,)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ClassifierML | None":
        """Return a ClassifierML instance, or None if the calibration file is absent."""
        calib_path = Path(path) if path else _CALIB_DEFAULT
        if not calib_path.exists():
            return None
        with open(calib_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            classes=data["classes"],
            W=np.array(data["W"], dtype=np.float32),
            b=np.array(data["b"], dtype=np.float32),
        )

    def predict(self, features: np.ndarray) -> "ClassifierResult | None":
        """
        Return a ClassifierResult, or None if features shape is wrong.

        features must be shape (73,).
        """
        # Import here to avoid circular imports at module load time
        from itol.icr import ClassifierResult

        if features is None or features.shape != (73,):
            return None
        logits = self.W @ features + self.b
        probs = _softmax(logits)
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        primary = self.classes[top_idx]

        # Determine second class for ambiguous_matrix routing
        sorted_idx = np.argsort(probs)[::-1]
        second_class = self.classes[int(sorted_idx[1])] if len(self.classes) > 1 else primary

        return ClassifierResult(
            primary=primary,
            confidence=confidence,
            ambiguous=(confidence < 0.6),
            top2=(primary, second_class) if confidence < 0.6 else (primary, second_class),
        )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(icr: "ICR") -> np.ndarray:
    """Build the 73-dimensional feature vector from an ICR."""
    from itol.analysis.signals import compute_signals
    from itol.embeddings import embed

    sig = compute_signals(icr)
    history_depth = len([m for m in icr.messages if m.role == "user"]) - 1
    token_count = sum(
        len(c.text.split()) if c.text else 0
        for m in icr.messages
        for c in (m.content if hasattr(m.content, "__iter__") else [])
    )

    signal_vec = np.array([
        sig.redundancy_score,
        sig.semantic_density,
        sig.instruction_context_ratio,
        float(history_depth) / 20.0,
        float(token_count) / 10_000.0,
    ], dtype=np.float32)

    # Build query text for embedding (last user message)
    query_parts = []
    for m in reversed(icr.messages):
        if m.role == "user":
            for block in (m.content if hasattr(m.content, "__iter__") else []):
                if hasattr(block, "text") and block.text:
                    query_parts.append(block.text)
            break
    query_text = " ".join(query_parts)[:512]

    raw_emb = embed([query_text], model="minilm")
    if raw_emb is None or len(raw_emb) == 0:
        emb_vec = np.zeros(64, dtype=np.float32)
    else:
        emb_vec = _pool_embedding(raw_emb[0])

    verb_vec = _count_verbs(query_text)

    return np.concatenate([signal_vec, emb_vec, verb_vec])


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

_ml_classifier: ClassifierML | None | bool = False  # False = not yet loaded


def _get_classifier(path: Path | str | None = None) -> ClassifierML | None:
    global _ml_classifier
    if _ml_classifier is False:
        _ml_classifier = ClassifierML.load(path)
    return _ml_classifier  # type: ignore[return-value]


def classify_with_ml(icr: "ICR", path: Path | str | None = None) -> "ClassifierResult | None":
    """
    Run Stage B ML classifier.

    Returns ClassifierResult, or None if calibration is absent (caller should
    keep Stage A's result).
    """
    clf = _get_classifier(path)
    if clf is None:
        return None
    features = _extract_features(icr)
    return clf.predict(features)
