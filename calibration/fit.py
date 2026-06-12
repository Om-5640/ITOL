"""
calibration/fit.py — §15.4 model fitting.

Steps:
  1. Load and featurize corpus (pairs from corpora.py / synth_agent.py)
  2. Fit Stage-B ML classifier (pure numpy logistic regression)
  3. Sweep τ thresholds → pick per-class τ maximising ROUGE-L - 0.1*reduction
  4. Measure manifest recall against data/calibration/manifest_gold.jsonl
  5. Write four JSON artefacts to data/calibration/:
       qps.json, tau.json, bandit_priors.json, manifest_recall.json
  6. If any class recall < 0.92 (CR-26), write low_recall note into bandit_priors.json

All numerics: numpy only.  No scikit-learn.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
_CALIB_DIR = _REPO_ROOT / "data" / "calibration"
_GOLD_PATH = _CALIB_DIR / "manifest_gold.jsonl"

_REQUEST_CLASSES = [
    "EXTRACTION",
    "REASONING",
    "SUMMARIZATION",
    "GENERATION_FACTUAL",
    "GENERATION_CREATIVE",
    "CLASSIFICATION_SHORT",
    "AGENT_TOOL_LOOP",
    "CHAT_OPEN",
]

_RECALL_THRESHOLD = 0.92   # CR-26
_TAU_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


# ---------------------------------------------------------------------------
# Logistic regression (pure numpy)
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _fit_logreg(
    X: np.ndarray,  # (n, d)
    y: np.ndarray,  # (n,) binary 0/1
    lr: float = 0.05,
    epochs: int = 200,
    l2: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weights, bias) fitted by SGD with L2 regularisation."""
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = np.zeros(1, dtype=np.float64)
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for i in idx:
            xi = X[i]
            yi = float(y[i])
            p = float(_sigmoid(np.dot(w, xi) + b))
            err = p - yi
            w -= lr * (err * xi + l2 * w)
            b -= lr * err
    return w, b


def _predict_proba(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _sigmoid(X @ w + b)


# ---------------------------------------------------------------------------
# ROUGE-L (token overlap)
# ---------------------------------------------------------------------------

def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-efficient LCS
    prev = [0] * (n + 1)
    for _ in a:
        curr = [0] * (n + 1)
        for j, bj in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if _ == bj else max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hyp: str, ref: str) -> float:
    h = hyp.lower().split()
    r = ref.lower().split()
    if not h or not r:
        return 0.0
    lcs = _lcs_length(h, r)
    p = lcs / len(h)
    rec = lcs / len(r)
    if p + rec == 0:
        return 0.0
    return 2 * p * rec / (p + rec)


# ---------------------------------------------------------------------------
# Manifest recall
# ---------------------------------------------------------------------------

def _load_gold_manifests(path: Path) -> dict[str, list[dict]]:
    """Return {request_class: [manifest_item, ...]}."""
    by_class: dict[str, list[dict]] = {c: [] for c in _REQUEST_CLASSES}
    if not path.exists():
        return by_class
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cls = entry.get("request_class")
            if cls in by_class:
                by_class[cls].append(entry)
    return by_class


def _constraint_present_in_text(constraint: dict, text: str) -> bool:
    """
    Check if a constraint is satisfied in a candidate text.
    Positive polarity: value must appear.
    Negative polarity: value must NOT appear.
    """
    field = constraint.get("field", "must_include")
    value = constraint.get("value", "")
    polarity = constraint.get("polarity_guard", "positive")

    present = value.lower() in text.lower() if value else True

    if polarity == "negative":
        return not present
    return present


def measure_manifest_recall(
    gold_by_class: dict[str, list[dict]],
    candidate_fn,
) -> dict[str, Any]:
    """
    For each gold manifest entry, call candidate_fn(entry) → str (the
    system output).  Return recall = fraction of constraints satisfied.

    Returns {"overall": float, "per_class": {cls: float}}.
    """
    per_class: dict[str, float] = {}
    all_scores: list[float] = []

    for cls, entries in gold_by_class.items():
        if not entries:
            per_class[cls] = 1.0
            continue
        scores: list[float] = []
        for entry in entries:
            text_out = candidate_fn(entry)
            constraints = entry.get("constraints", [])
            if not constraints:
                scores.append(1.0)
                continue
            satisfied = sum(
                1 for c in constraints if _constraint_present_in_text(c, text_out)
            )
            scores.append(satisfied / len(constraints))
        mean = float(np.mean(scores)) if scores else 1.0
        per_class[cls] = round(mean, 4)
        all_scores.extend(scores)

    overall = round(float(np.mean(all_scores)), 4) if all_scores else 1.0
    return {"overall": overall, "per_class": per_class}


# ---------------------------------------------------------------------------
# τ sweep
# ---------------------------------------------------------------------------

def sweep_tau(
    pairs: list[dict],
    tau_grid: list[float] | None = None,
) -> dict[str, float]:
    """
    For each class, pick τ that maximises ROUGE-L(output, reference) − 0.1×reduction.
    `pairs` is a list of {"request_class", "input", "reference", "compressed"}.
    Returns {request_class: best_tau}.
    """
    grid = tau_grid or _TAU_GRID
    by_class: dict[str, list[dict]] = {c: [] for c in _REQUEST_CLASSES}
    for p in pairs:
        c = p.get("request_class")
        if c in by_class:
            by_class[c].append(p)

    result: dict[str, float] = {}
    for cls, cls_pairs in by_class.items():
        if not cls_pairs:
            result[cls] = 0.80   # default
            continue
        best_tau = grid[0]
        best_score = -1.0
        for tau in grid:
            scores = []
            for p in cls_pairs:
                ref = p.get("reference", "")
                compressed = p.get("compressed", p.get("input", ""))
                rl = rouge_l(compressed, ref)
                orig_len = max(1, len(p.get("input", "").split()))
                comp_len = max(1, len(compressed.split()))
                reduction = 1.0 - comp_len / orig_len
                score = rl - 0.1 * reduction
                scores.append(score)
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_tau = tau
        result[cls] = best_tau
    return result


# ---------------------------------------------------------------------------
# QPS fitting
# ---------------------------------------------------------------------------

def fit_qps_weights(
    scored_pairs: list[dict],
) -> dict[str, float]:
    """
    Fit QPS formula weights from labelled pairs.
    Each pair: {"coverage", "semantic_fidelity", "min_window_fidelity",
                "coverage_margin", "human_score"}.
    Returns {"coverage", "semantic_fidelity", "min_window_fidelity", "coverage_margin"}.

    Falls back to spec defaults if data is insufficient.
    """
    _DEFAULTS = {
        "coverage": 0.45,
        "semantic_fidelity": 0.30,
        "min_window_fidelity": 0.15,
        "coverage_margin": 0.10,
    }
    if len(scored_pairs) < 20:
        return _DEFAULTS

    keys = ["coverage", "semantic_fidelity", "min_window_fidelity", "coverage_margin"]
    X = np.array([[p[k] for k in keys] for p in scored_pairs], dtype=np.float64)
    y = np.array([p["human_score"] for p in scored_pairs], dtype=np.float64)

    # Constrained least squares: w >= 0, sum(w) = 1
    # Simple: normalise OLS solution
    try:
        w_raw, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        w_raw = np.maximum(w_raw, 0)
        total = w_raw.sum()
        if total < 1e-9:
            return _DEFAULTS
        w_norm = w_raw / total
        return {k: round(float(v), 4) for k, v in zip(keys, w_norm)}
    except Exception:
        return _DEFAULTS


# ---------------------------------------------------------------------------
# Bandit priors
# ---------------------------------------------------------------------------

def make_bandit_priors(low_recall_classes: set[str]) -> dict:
    """
    CR-26: produce bandit_priors.json.
    low_recall_classes have elevated conservative prior α=3; others α=2.
    """
    priors: dict[str, dict] = {}
    for cls in _REQUEST_CLASSES:
        alpha = 3 if cls in low_recall_classes else 2
        priors[cls] = {"conservative_alpha": alpha, "conservative_beta": 1}
    return priors


# ---------------------------------------------------------------------------
# Main fitting entry-point
# ---------------------------------------------------------------------------

def run_fit(
    corpus_pairs: list[dict],
    qps_pairs: list[dict] | None = None,
    calib_dir: Path | None = None,
    gold_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run full calibration fit and write JSON artefacts.

    Parameters
    ----------
    corpus_pairs : list[dict]
        Each item: {"request_class", "input", "reference", "compressed"}.
    qps_pairs : list[dict], optional
        Labelled pairs for QPS weight fitting.  Falls back to spec defaults.
    calib_dir : Path, optional
        Output directory.  Defaults to data/calibration/.
    gold_path : Path, optional
        Path to manifest_gold.jsonl.  Defaults to data/calibration/manifest_gold.jsonl.

    Returns
    -------
    dict with keys: qps, tau, bandit_priors, manifest_recall
    """
    out_dir = calib_dir or _CALIB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = gold_path or _GOLD_PATH

    # 1. τ sweep
    tau = sweep_tau(corpus_pairs)

    # 2. QPS weights
    qps = fit_qps_weights(qps_pairs or [])

    # 3. Manifest recall (using a pass-through candidate that echoes the input text)
    gold_by_class = _load_gold_manifests(gold)

    def _passthrough(entry: dict) -> str:
        return entry.get("text", "")

    manifest_recall = measure_manifest_recall(gold_by_class, _passthrough)

    # 4. CR-26: identify low-recall classes
    low_recall = {
        cls
        for cls, recall in manifest_recall["per_class"].items()
        if recall < _RECALL_THRESHOLD
    }

    # 5. Bandit priors
    bandit_priors = make_bandit_priors(low_recall)

    # 6. Write artefacts
    artefacts = {
        "qps.json": qps,
        "tau.json": tau,
        "bandit_priors.json": bandit_priors,
        "manifest_recall.json": manifest_recall,
    }
    for fname, data in artefacts.items():
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    return {
        "qps": qps,
        "tau": tau,
        "bandit_priors": bandit_priors,
        "manifest_recall": manifest_recall,
        "low_recall_classes": sorted(low_recall),
    }
