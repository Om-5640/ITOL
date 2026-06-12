"""
Train the Stage B ML classifier and export weights to JSON.

Usage:
    python scripts/train_classifier.py --data data/classifier_training.csv \
        --out itol/data/calibration/classifier_ml.json

CSV format: text,label
  text  — raw user query text
  label — one of EXTRACTION/GENERATION_CREATIVE/REASONING/SUMMARIZATION/etc.

Output JSON (itol/data/calibration/classifier_ml.json):
  {
    "classes": ["EXTRACTION", ...],
    "W": [[w00, w01, ...], ...],   # shape (n_classes, 73)
    "b": [b0, b1, ...],            # shape (n_classes,)
    "feature_names": [...]
  }

sklearn is required only for training (not at inference).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from itol.analysis.classifier_ml import (
    _LEXICON_KEYS,
    _count_verbs,
    _pool_embedding,
)

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import LabelEncoder       # type: ignore
except ImportError as e:
    sys.exit(f"sklearn required for training: {e}")


def _build_features(text: str) -> np.ndarray:
    """Build 73-dim feature vector from raw text (no ICR needed during training)."""
    from itol.embeddings import embed

    tokens = text.split()
    token_count = len(tokens)

    # Signals approximation from text alone (no full ICR available during training)
    # redundancy_score: 0.0 (unknown), semantic_density: token diversity ratio,
    # instruction_context_ratio: ratio of normative words, history_depth: 0, token_count
    unique_ratio = len(set(tokens)) / max(token_count, 1)
    normative = {"must", "always", "never", "only", "shall", "required", "forbidden"}
    norm_ratio = sum(1 for t in tokens if t.lower() in normative) / max(token_count, 1)

    signal_vec = np.array([
        0.0,                             # redundancy_score (unknown in training)
        float(unique_ratio),             # semantic_density proxy
        float(norm_ratio),               # instruction_context_ratio proxy
        0.0,                             # history_depth (no conversation context)
        float(token_count) / 10_000.0,  # token_count normalised
    ], dtype=np.float32)

    raw_emb = embed([text[:512]], model="minilm")
    if raw_emb is None or len(raw_emb) == 0:
        emb_vec = np.zeros(64, dtype=np.float32)
    else:
        emb_vec = _pool_embedding(raw_emb[0])

    verb_vec = _count_verbs(text)

    return np.concatenate([signal_vec, emb_vec, verb_vec])


def _feature_names() -> list[str]:
    names = [
        "redundancy_score", "semantic_density", "instruction_context_ratio",
        "history_depth_norm", "token_count_norm",
    ]
    names += [f"emb_{i}" for i in range(64)]
    names += [f"verb_{k.lower()}" for k in _LEXICON_KEYS]
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ITOL Stage B classifier")
    parser.add_argument("--data", required=True, help="Path to labeled CSV (text,label)")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "itol" / "data" / "calibration" / "classifier_ml.json"),
        help="Output JSON path",
    )
    parser.add_argument("--C", type=float, default=1.0, help="LR regularisation strength")
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    # Load data
    texts, labels = [], []
    with open(args.data, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])

    if len(texts) == 0:
        sys.exit("No training examples found in CSV.")

    print(f"Building features for {len(texts)} examples …")
    X = np.array([_build_features(t) for t in texts], dtype=np.float32)
    y = np.array(labels)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"Training LogisticRegression (C={args.C}, max_iter={args.max_iter}) …")
    lr = LogisticRegression(
        C=args.C, max_iter=args.max_iter, multi_class="multinomial", solver="lbfgs"
    )
    lr.fit(X, y_enc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": list(le.classes_),
        "W": lr.coef_.tolist(),
        "b": lr.intercept_.tolist(),
        "feature_names": _feature_names(),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Weights written to {out_path}  ({len(le.classes_)} classes, shape {lr.coef_.shape})")


if __name__ == "__main__":
    main()
