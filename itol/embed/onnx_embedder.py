"""
ONNX / sentence-transformers / BOW embedder — §3.3 + §6.1.

Backend priority (first available wins):
  1. onnxruntime + transformers   (int8 ONNX, downloaded lazily from HF Hub)
  2. sentence-transformers        (CPU inference, no ONNX needed)
  3. BOW cosine                   (pure numpy, always works — used in CI)

Public API
----------
embed(texts, model="minilm")   → np.ndarray  shape (N, D)
sliding_window_fidelity(...)   → float        §15.2 min cosine over 2-sentence windows
cosine(a, b)                   → float        utility

Embedding LRU cache
-------------------
maxsize=512, keyed by (model_alias, sha256(text))
Thread-safe via threading.Lock.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

_MODEL_REPOS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge":    "BAAI/bge-small-en-v1.5",
}

_ONNX_FILE = "onnx/model_quantized.onnx"   # standard path in HF repos with ONNX exports

ModelAlias = Literal["minilm", "bge"]


# ---------------------------------------------------------------------------
# LRU embedding cache
# ---------------------------------------------------------------------------

class _LRUCache:
    def __init__(self, maxsize: int = 512) -> None:
        self._cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, model: str, text: str) -> np.ndarray | None:
        key = (model, hashlib.sha256(text.encode()).hexdigest())
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key].copy()

    def put(self, model: str, text: str, vec: np.ndarray) -> None:
        key = (model, hashlib.sha256(text.encode()).hexdigest())
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
            self._cache[key] = vec.copy()


_embed_cache = _LRUCache(maxsize=512)


# ---------------------------------------------------------------------------
# Backend protocol & implementations
# ---------------------------------------------------------------------------

class _Backend:
    """Base class for embedding backends."""
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class _ONNXBackend(_Backend):
    """
    ONNX Runtime backend (int8 quantised models downloaded from HF Hub).
    Requires: onnxruntime, transformers, huggingface_hub, numpy.
    """
    def __init__(self, model_alias: str, data_dir: str | None = None) -> None:
        import os
        from pathlib import Path
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download

        repo_id = _MODEL_REPOS[model_alias]
        cache_dir = Path(data_dir) / "models" if data_dir else Path.home() / ".itol" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=_ONNX_FILE,
            cache_dir=str(cache_dir),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=str(cache_dir))
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._model_alias = model_alias

    def embed(self, texts: list[str]) -> np.ndarray:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        outputs = self._session.run(None, dict(encoded))
        # Mean-pool over token dimension (index 0 of outputs is last_hidden_state)
        token_embeddings = outputs[0]                         # (N, seq_len, D)
        attention_mask = encoded["attention_mask"]            # (N, seq_len)
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        vecs = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
        return _l2_normalize(vecs)


class _STBackend(_Backend):
    """sentence-transformers fallback backend."""
    def __init__(self, model_alias: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(_MODEL_REPOS[model_alias], device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)


class _BOWBackend(_Backend):
    """
    Bag-of-words cosine similarity fallback.
    Pure numpy; no ML libraries required.  Used in CI and for quick tests.
    """
    def embed(self, texts: list[str]) -> np.ndarray:
        return _bow_embed(texts)


# ---------------------------------------------------------------------------
# Backend registry (one per model alias, lazily initialised)
# ---------------------------------------------------------------------------

_backends: dict[str, _Backend] = {}
_backend_lock = threading.Lock()
_data_dir: str | None = None


def configure(data_dir: str) -> None:
    """Set the model download directory before first embed() call."""
    global _data_dir
    _data_dir = data_dir


def _get_backend(model_alias: str) -> _Backend:
    with _backend_lock:
        if model_alias in _backends:
            return _backends[model_alias]

        backend: _Backend
        try:
            import onnxruntime  # noqa: F401
            import transformers  # noqa: F401
            import huggingface_hub  # noqa: F401
            backend = _ONNXBackend(model_alias, data_dir=_data_dir)
        except Exception:
            try:
                import sentence_transformers  # noqa: F401
                backend = _STBackend(model_alias)
            except Exception:
                backend = _BOWBackend()

        _backends[model_alias] = backend
        return backend


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(texts: list[str], model: ModelAlias = "minilm") -> np.ndarray:
    """
    Embed a batch of texts.  Returns an L2-normalised (N, D) float32 array.
    Results are cached by (model, sha256(text)); maxsize=512.
    """
    results = []
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, t in enumerate(texts):
        cached = _embed_cache.get(model, t)
        if cached is not None:
            results.append(cached)
        else:
            results.append(None)  # type: ignore[arg-type]
            uncached_indices.append(i)
            uncached_texts.append(t)

    if uncached_texts:
        backend = _get_backend(model)
        vecs = backend.embed(uncached_texts)
        for idx, vec in zip(uncached_indices, vecs):
            _embed_cache.put(model, texts[idx], vec)
            results[idx] = vec

    return np.stack(results, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (already L2-normalised → just dot)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a / na, b / nb))


def sliding_window_fidelity(
    original_seg: str,
    optimized_seg: str,
    window_sentences: int = 2,
) -> float:
    """
    §15.2 min_window_fidelity.

    Splits both segments into sentences, then slides a `window_sentences`-wide
    window over the original, comparing each window (position i) with the
    aligned window (position i) in the optimized text.

    If the optimized text has fewer windows, the last window is reused for
    alignment past the end.

    Returns the minimum cosine across all aligned window pairs.
    Floor flag: values < 0.75 indicate a problematic window (not clamped here;
    the QPS gate uses the raw minimum).
    """
    orig_sents = _split_sentences(original_seg)
    opt_sents  = _split_sentences(optimized_seg)

    if len(orig_sents) < window_sentences or len(opt_sents) < window_sentences:
        vecs = embed([original_seg, optimized_seg])
        return cosine(vecs[0], vecs[1])

    orig_windows = [
        " ".join(orig_sents[i : i + window_sentences])
        for i in range(len(orig_sents) - window_sentences + 1)
    ]
    opt_windows = [
        " ".join(opt_sents[i : i + window_sentences])
        for i in range(len(opt_sents) - window_sentences + 1)
    ]

    all_windows = orig_windows + opt_windows
    all_vecs = embed(all_windows)
    orig_vecs = all_vecs[: len(orig_windows)]
    opt_vecs  = all_vecs[len(orig_windows) :]

    min_cos = 1.0
    for i, ov in enumerate(orig_vecs):
        aligned_idx = min(i, len(opt_vecs) - 1)
        c = cosine(ov, opt_vecs[aligned_idx])
        if c < min_cos:
            min_cos = c

    return min_cos


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\.)\s*\n+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (arr / norms).astype(np.float32)


def _bow_embed(texts: list[str]) -> np.ndarray:
    """Bag-of-words embedding, L2-normalised. O(V) per text."""
    def tokenize(t: str) -> list[str]:
        return re.findall(r"\b[a-z]+\b", t.lower())

    vocab: dict[str, int] = {}
    token_lists = []
    for t in texts:
        toks = tokenize(t)
        token_lists.append(toks)
        for w in toks:
            if w not in vocab:
                vocab[w] = len(vocab)

    if not vocab:
        return np.zeros((len(texts), 1), dtype=np.float32)

    vecs = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        for w in toks:
            vecs[i, vocab[w]] += 1.0

    return _l2_normalize(vecs)
