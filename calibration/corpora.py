"""
calibration/corpora.py — corpus loading for calibration.

Primary: synthetic data bundled in calibration/data/.
Optional: Hugging Face Hub datasets (requires internet + datasets package).

If HF is unavailable, falls back to synthetic data with a logged warning.

TODO: wire up real HF datasets when network is available:
  - "cnn_dailymail" for SUMMARIZATION pairs
  - "squad" for EXTRACTION
  - "commonsense_qa" for REASONING
  - "ag_news" for CLASSIFICATION_SHORT
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Generator

_log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"

_SYNTHETIC_CHAT = _DATA_DIR / "synthetic_chat.json"
_PARAPHRASE_PAIRS = _DATA_DIR / "paraphrase_pairs.json"


# ---------------------------------------------------------------------------
# Synthetic loaders
# ---------------------------------------------------------------------------

def load_synthetic_chat() -> list[dict]:
    """Load synthetic chat examples from calibration/data/synthetic_chat.json."""
    if not _SYNTHETIC_CHAT.exists():
        _log.warning("synthetic_chat.json not found at %s", _SYNTHETIC_CHAT)
        return []
    with open(_SYNTHETIC_CHAT, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def load_paraphrase_pairs() -> list[dict]:
    """Load paraphrase pairs from calibration/data/paraphrase_pairs.json."""
    if not _PARAPHRASE_PAIRS.exists():
        _log.warning("paraphrase_pairs.json not found at %s", _PARAPHRASE_PAIRS)
        return []
    with open(_PARAPHRASE_PAIRS, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# HF loader (stub — requires `pip install datasets` and internet access)
# ---------------------------------------------------------------------------

def _hf_available() -> bool:
    try:
        import importlib
        importlib.import_module("datasets")
        return True
    except ImportError:
        return False


def load_hf_pairs(
    dataset_name: str,
    split: str = "validation",
    request_class: str = "SUMMARIZATION",
    max_samples: int = 200,
) -> list[dict]:
    """
    Load pairs from a HuggingFace dataset.

    TODO: implement full HF integration per dataset_name.
    Currently only cnn_dailymail → SUMMARIZATION is sketched.
    """
    if not _hf_available():
        _log.warning(
            "HuggingFace `datasets` not installed. "
            "Run `pip install datasets` for real corpus pairs. "
            "Falling back to synthetic data."
        )
        return []

    try:
        from datasets import load_dataset  # type: ignore[import]

        if dataset_name == "cnn_dailymail":
            ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
            pairs = []
            for i, row in enumerate(ds):
                if i >= max_samples:
                    break
                pairs.append({
                    "request_class": request_class,
                    "input": row["article"],
                    "reference": row["highlights"],
                    "compressed": row["highlights"],
                })
            return pairs

        # TODO: add squad, commonsense_qa, ag_news handling here
        _log.warning("HF dataset %r not yet implemented; returning [].", dataset_name)
        return []

    except Exception as exc:
        _log.warning("HF load failed (%s); falling back to synthetic.", exc)
        return []


# ---------------------------------------------------------------------------
# Combined corpus loader
# ---------------------------------------------------------------------------

def build_corpus_pairs(
    offline: bool = True,
    max_hf_samples: int = 200,
) -> list[dict]:
    """
    Return list of corpus pairs for calibration.

    When offline=True or HF is unavailable, returns only synthetic data.
    """
    pairs: list[dict] = []

    # Always include synthetic
    pairs.extend(load_synthetic_chat())
    pairs.extend(load_paraphrase_pairs())

    if not offline and _hf_available():
        hf_requests: list[tuple[str, str, str]] = [
            ("cnn_dailymail", "validation", "SUMMARIZATION"),
            # TODO: add more dataset/split/class triples here
        ]
        for ds_name, split, req_class in hf_requests:
            hf_pairs = load_hf_pairs(
                ds_name, split=split,
                request_class=req_class,
                max_samples=max_hf_samples,
            )
            pairs.extend(hf_pairs)
            _log.info("Loaded %d pairs from %s/%s", len(hf_pairs), ds_name, split)

    _log.info("Total corpus pairs: %d", len(pairs))
    return pairs
