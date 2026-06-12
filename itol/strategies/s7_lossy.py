"""
S7 Lossy Token-Level Compression — LOSSY-AGGRESSIVE (§4).

OFF BY DEFAULT.  Must be explicitly opted in via class_configs[cls].s7_enabled=True.
Eligible classes (matrix opt-in): SUMMARIZATION, CHAT_OPEN.

Three-tier compression backend
-------------------------------
Same lazy-fallback pattern as onnx_embedder.py:

  Tier 1 (ONNX):      LLMLingua-2-xlm-roberta-large ONNX int8 checkpoint.
                       Requires: onnxruntime, transformers, huggingface_hub.
                       Downloads lazily from HF Hub on first use.
                       PRODUCTION RECOMMENDATION: run this tier for best quality.

  Tier 2 (Distilled):  Smaller sentence-transformers model used as a
                       token-salience scorer.  Requires: sentence-transformers.

  Tier 3 (Token-freq): Pure Python / numpy fallback.  Drops lowest-frequency
                       non-manifest words up to the 2x ratio cap.  Crude but
                       sufficient for integration tests and gating-logic
                       verification.  No ML dependency required.

Activation conditions (ALL must hold)
---------------------------------------
  1. class_configs[request_class].s7_enabled = True       (off by default)
  2. Matrix permits S7 (SUMMARIZATION, CHAT_OPEN have opt-in)
  3. S1–S6 left total token count above class s3_class_budget
  4. At least one eligible segment:
         per-segment semantic_density < s7_density_gate (0.45)

Hard constraints (invariants)
--------------------------------
  - Manifest item values are NEVER deleted (token-level mask)
  - Compression ratio never exceeds s7_max_ratio (2.0)
  - QPS floor tightens to qps_floor_with_s7 (0.99) when S7 participated
  - Shadow eval rate raises to shadow_eval_rate_s7 (5×) when S7 participated
"""

from __future__ import annotations

import logging
import re
import threading
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

from itol.icr import ICR, StrategyReport
from itol.routing.matrix import StrategyStatus
from itol.segmenter import Segment
from itol.signals import semantic_density
from itol.strategies.base import OptimizationContext, Strategy, update_segment

if TYPE_CHECKING:
    from itol.icr import ConstraintManifest

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest token extraction
# ---------------------------------------------------------------------------

def _manifest_protected_words(manifest: "ConstraintManifest", seg_text: str) -> frozenset[str]:
    """
    Return the set of whitespace-split words from seg_text that appear in any
    manifest item value (case-insensitive substring match both ways).

    This is the token-level mask applied before compression — any word in this
    set cannot be dropped by the token-frequency backend.
    """
    if not manifest.items:
        return frozenset()

    manifest_values_lower = [item.value.lower() for item in manifest.items]
    protected: set[str] = set()
    for word in seg_text.split():
        word_lower = word.lower().strip(".,;:!?\"'()")
        for val in manifest_values_lower:
            if word_lower in val or val in word_lower:
                protected.add(word)
                break
    return frozenset(protected)


# ---------------------------------------------------------------------------
# Tier 3 — Token-frequency fallback (always available)
# ---------------------------------------------------------------------------

def _token_freq_compress(
    text: str,
    protected: frozenset[str],
    target_ratio: float,
) -> str:
    """
    Drop lowest-frequency non-protected whitespace tokens up to target_ratio.

    target_ratio=2.0 means keep ~50% of tokens (crude approximation).

    Implementation note: this fallback is intentionally simple — it exists
    to exercise S7's gating logic in CI without requiring a GPU or large model.
    For production token savings, configure tier 1 (LLMLingua-2 ONNX).
    """
    words = text.split()
    if not words:
        return text

    n_original = len(words)
    target_count = max(1, int(n_original / target_ratio))
    n_to_remove = n_original - target_count

    if n_to_remove <= 0:
        return text

    freq = Counter(words)

    # Sort candidate words (not protected) by frequency ascending
    candidates = sorted(
        [(w, f) for w, f in freq.items() if w not in protected],
        key=lambda x: x[1],
    )

    to_drop: set[str] = set()
    removed = 0
    for word, cnt in candidates:
        if removed >= n_to_remove:
            break
        to_drop.add(word)
        removed += cnt

    kept = [w for w in words if w not in to_drop]
    return " ".join(kept) if kept else text


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class _S7Backend(ABC):
    @abstractmethod
    def compress(
        self,
        text: str,
        protected: frozenset[str],
        target_ratio: float,
    ) -> str: ...


class _TokenFreqBackend(_S7Backend):
    """Tier 3: always-available Python fallback."""

    def compress(self, text: str, protected: frozenset[str], target_ratio: float) -> str:
        return _token_freq_compress(text, protected, target_ratio)


class _LLMLingua2ONNXBackend(_S7Backend):
    """
    Tier 1: LLMLingua-2-xlm-roberta-large ONNX int8 token-classification model.

    Each token gets an importance score; we keep the top K (by score, in
    original order) where K = len(tokens) / target_ratio, never removing
    protected tokens regardless of their score.

    Requires: onnxruntime, transformers, huggingface_hub.
    Model checkpoint: microsoft/llmlingua-2-xlm-roberta-large-meetingbank
    (int8-quantised ONNX export at onnx/model_quantized.onnx in the repo).
    """

    _HF_REPO = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    _ONNX_FILE = "onnx/model_quantized.onnx"

    def __init__(self, data_dir: str | None = None) -> None:
        import os
        from pathlib import Path

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        cache_dir = Path(data_dir) if data_dir else Path.home() / ".itol" / "models"
        onnx_path = hf_hub_download(
            self._HF_REPO, self._ONNX_FILE, cache_dir=str(cache_dir)
        )
        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._HF_REPO, cache_dir=str(cache_dir)
        )

    def compress(self, text: str, protected: frozenset[str], target_ratio: float) -> str:
        words = text.split()
        if not words:
            return text

        target_count = max(1, int(len(words) / target_ratio))

        # Tokenize for model (subword), then map scores back to words
        enc = self._tokenizer(text, return_tensors="np", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0]
        logits = self._session.run(
            None, {k: enc[k] for k in ("input_ids", "attention_mask")}
        )[0][0]  # shape (seq_len, 2)

        # Map token importance scores back to original word indices
        word_scores = [0.0] * len(words)
        char_pos = 0
        word_boundaries: list[tuple[int, int]] = []
        for w in words:
            start = text.find(w, char_pos)
            end = start + len(w)
            word_boundaries.append((start, end))
            char_pos = end

        import numpy as np
        keep_scores = logits[:, 1] - logits[:, 0]  # logit difference: keep vs drop
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            for wi, (ws, we) in enumerate(word_boundaries):
                if tok_start >= ws and tok_end <= we:
                    word_scores[wi] = max(word_scores[wi], float(keep_scores[tok_idx]))
                    break

        # Sort by score descending; keep top K (always keep protected)
        protected_indices = {
            i for i, w in enumerate(words) if w in protected
        }
        ranked = sorted(
            range(len(words)),
            key=lambda i: (1 if i in protected_indices else 0, word_scores[i]),
            reverse=True,
        )
        keep_indices = set(ranked[:max(target_count, len(protected_indices))])
        kept = [w for i, w in enumerate(words) if i in keep_indices]
        # Restore original order
        kept_ordered = [w for i, w in enumerate(words) if i in keep_indices]
        return " ".join(kept_ordered)


class _DistilledBackend(_S7Backend):
    """
    Tier 2: Sentence-transformers token importance scorer.

    Approximates token salience using word ablation: drop each token and measure
    cosine drop from the full-sentence embedding.  Expensive but no ONNX needed.
    Requires: sentence-transformers, numpy.
    """

    def __init__(self, model_alias: str = "minilm") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _MODEL_MAP = {
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        }
        self._model = SentenceTransformer(
            _MODEL_MAP.get(model_alias, model_alias), device="cpu"
        )

    def compress(self, text: str, protected: frozenset[str], target_ratio: float) -> str:
        import numpy as np

        words = text.split()
        if not words:
            return text

        target_count = max(1, int(len(words) / target_ratio))
        if target_count >= len(words):
            return text

        base_vec = self._model.encode([text], normalize_embeddings=True)[0]

        # Score each word: cosine drop when word is ablated
        scores = []
        for i, w in enumerate(words):
            ablated = " ".join(words[:i] + words[i + 1:])
            abl_vec = self._model.encode([ablated], normalize_embeddings=True)[0]
            drop = 1.0 - float(np.dot(base_vec, abl_vec))
            scores.append((i, drop))

        # Highest drop = most important (keep); lowest drop = least important (remove)
        protected_indices = {i for i, w in enumerate(words) if w in protected}
        ranked = sorted(
            scores,
            key=lambda x: (1 if x[0] in protected_indices else 0, x[1]),
            reverse=True,
        )
        keep_indices = set(
            i for i, _ in ranked[:max(target_count, len(protected_indices))]
        )
        return " ".join(w for i, w in enumerate(words) if i in keep_indices)


# ---------------------------------------------------------------------------
# Backend registry (singleton, lazily initialised)
# ---------------------------------------------------------------------------

_backend: _S7Backend | None = None
_backend_lock = threading.Lock()
_data_dir: str | None = None


def configure_s7(data_dir: str) -> None:
    """Set the model download directory before first compress call."""
    global _data_dir
    _data_dir = data_dir


def _get_backend() -> _S7Backend:
    global _backend
    with _backend_lock:
        if _backend is not None:
            return _backend

        try:
            import onnxruntime  # noqa: F401
            import huggingface_hub  # noqa: F401
            import transformers  # noqa: F401
            _backend = _LLMLingua2ONNXBackend(data_dir=_data_dir)
            _log.info("S7: using LLMLingua-2 ONNX backend (tier 1)")
        except Exception:
            try:
                import sentence_transformers  # noqa: F401
                _backend = _DistilledBackend()
                _log.info("S7: using distilled ST backend (tier 2)")
            except Exception:
                _backend = _TokenFreqBackend()
                _log.info("S7: using token-frequency fallback backend (tier 3)")
    return _backend


# ---------------------------------------------------------------------------
# S7 Strategy
# ---------------------------------------------------------------------------

class S7LossyStrategy(Strategy):
    """
    LOSSY-AGGRESSIVE token-level compression.

    OFF BY DEFAULT.  Set class_configs[cls].s7_enabled=True to enable.

    Eligible request classes: SUMMARIZATION, CHAT_OPEN (matrix opt-in).

    Safety invariants (enforced unconditionally):
      - Manifest item values NEVER removed (token-level mask)
      - Ratio cap: 2x (s7_max_ratio)
      - QPS floor: 0.99 (qps_floor_with_s7 in QualityConfig)
      - Shadow rate: 5x (shadow_eval_rate_s7 in QualityConfig)
    """

    strategy_id = "S7"
    risk_class  = "LOSSY_AGGRESSIVE"

    def applies(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> bool:
        # 1. Class-config enable flag (OFF BY DEFAULT)
        cls_cfg = ctx.config.class_configs.get(ctx.request_class)
        if cls_cfg is None or not cls_cfg.s7_enabled:
            return False

        # 2. Matrix gate (only SUMMARIZATION and CHAT_OPEN have opt-in S7)
        if ctx.matrix_row.strategies.get("S7") == StrategyStatus.DISABLED:
            return False

        # 3. S1–S6 must have left the prompt above the class budget
        budget = cls_cfg.s3_class_budget
        total_tokens = sum(s.token_count or 0 for s in segments)
        if total_tokens <= budget:
            return False

        # 4. At least one segment with low semantic density
        density_gate = ctx.config.strategies.s7_density_gate
        return any(
            (s.token_count or 0) > 0
            and semantic_density(s.text) < density_gate
            for s in segments
        )

    def apply(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> tuple[list[Segment], StrategyReport]:
        segments_before = list(segments)
        tokens_before = sum(s.token_count or 0 for s in segments)

        if not self.applies(icr, segments, ctx):
            return segments, self._make_report(
                segments_before, tokens_before, tokens_before,
                [], [], 0, 0, activated=False,
            )

        density_gate = ctx.config.strategies.s7_density_gate
        max_ratio = ctx.config.strategies.s7_max_ratio
        manifest = ctx.manifest
        backend = _get_backend()

        new_segments = list(segments)
        touched: list[str] = []

        for seg_idx, seg in enumerate(segments):
            if (seg.token_count or 0) <= 0:
                continue
            seg_density = semantic_density(seg.text)
            if seg_density >= density_gate:
                continue

            protected = _manifest_protected_words(manifest, seg.text)
            # Cap the target ratio
            target_ratio = min(max_ratio, ctx.config.strategies.s7_max_ratio)

            try:
                compressed = backend.compress(seg.text, protected, target_ratio)
            except Exception as exc:
                _log.warning("S7: backend error on seg %d — skipping: %s", seg_idx, exc)
                continue

            # Enforce manifest invariant: verify ALL manifest values still present
            for item in manifest.items:
                if item.value and item.value not in compressed:
                    _log.warning(
                        "S7: manifest value %r dropped — reverting segment", item.value[:40]
                    )
                    compressed = seg.text
                    break

            # Enforce 2x ratio cap at token level (compressed must be >= 50% of original)
            orig_tokens = seg.token_count or 1
            from itol.signals import estimate_token_count
            comp_tokens = estimate_token_count(compressed)
            actual_ratio = orig_tokens / max(comp_tokens, 1)
            if actual_ratio > max_ratio:
                # Re-compress at exactly 2x (keep 50% by word count)
                words_to_keep = max(1, len(seg.text.split()) // 2)
                words = seg.text.split()
                reprotected = [w for w in words if w in protected]
                others = [w for w in words if w not in protected]
                n_other_keep = max(0, words_to_keep - len(reprotected))
                compressed = " ".join(reprotected + others[:n_other_keep])

            if compressed == seg.text:
                continue

            new_segments[seg_idx] = update_segment(seg, compressed)
            touched.append(seg.segment_hash)
            _log.debug("S7: compressed seg %d (density=%.2f)", seg_idx, seg_density)

        tokens_after = sum(s.token_count or 0 for s in new_segments)
        return new_segments, self._make_report(
            segments_before, tokens_before, tokens_after,
            touched, [], 0, 0,
            activated=len(touched) > 0,
            notes=f"compressed_segs={len(touched)}",
        )
