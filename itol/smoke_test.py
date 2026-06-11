"""
Phase 1 acceptance gate smoke test — §14.1 Gate F.

Run with:
    python -m itol.smoke_test

Steps
-----
1. Build an ICR manually.
2. Run segmenter + signals + manifest + classifier.
3. Run QPS gate (expect pass at 1.0).
4. Write to L0 cache and read back.
5. Record to telemetry (SQLite + jsonl).

Exits 0 and prints "PHASE 1 GATE: PASS" on success.
Exits 1 and prints the failing step on any assertion error.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def run() -> None:
    failures: list[str] = []

    # ------------------------------------------------------------------
    # Step 1 — Build ICR manually
    # ------------------------------------------------------------------
    from itol.icr import ICR, ContentBlock, Message
    icr = ICR.create(
        provider="openai",
        model="gpt-4o",
        system=[ContentBlock.text(
            "You are a helpful assistant. "
            "The project budget is $1.2M. "
            "This figure is estimated and subject to revision."
        )],
        messages=[Message.user("Summarize the key points.")],
        raw={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Summarize the key points."}],
        },
    )
    print("[1/5] ICR built")

    # ------------------------------------------------------------------
    # Step 2 — segmenter + signals + manifest + classifier
    # ------------------------------------------------------------------
    from itol.segmenter import segment_icr, segments_full_text
    from itol.signals import extract_signals
    from itol.analysis.manifest import extract_manifest
    from itol.analysis.classifier import classify

    segments   = segment_icr(icr)
    signals    = extract_signals(icr, segments)
    manifest   = extract_manifest(icr)
    cls_result = classify(icr)

    assert len(segments) >= 1, "Must produce at least 1 segment"
    assert signals.token_count > 0, "token_count must be > 0"
    assert len(manifest.items) >= 1, "Manifest must contain at least 1 item"
    assert cls_result.primary in (
        "GENERATION_FACTUAL", "SUMMARIZATION", "CHAT_OPEN", "REASONING",
        "EXTRACTION", "CLASSIFICATION_SHORT",
    ), f"Unexpected class: {cls_result.primary}"

    print(
        f"[2/5] Pipeline OK — {len(segments)} segments, "
        f"class={cls_result.primary} ({cls_result.confidence:.2f}), "
        f"{len(manifest.items)} manifest items"
    )

    # ------------------------------------------------------------------
    # Step 3 — QPS gate (expect pass at 1.0 on unmodified text)
    # ------------------------------------------------------------------
    from itol.config import ITOLConfig
    from itol.quality.qps import compute_qps

    cfg      = ITOLConfig().quality
    full_text = segments_full_text(segments)
    qps_result = compute_qps(manifest, full_text, cfg)

    assert qps_result.passed, (
        f"QPS should pass for unmodified text, got qps={qps_result.qps:.4f} "
        f"(floor={qps_result.floor_used})"
    )
    print(f"[3/5] QPS gate PASS — qps={qps_result.qps:.4f}")

    # ------------------------------------------------------------------
    # Step 4 — L0 cache round-trip
    # ------------------------------------------------------------------
    from itol.cache.store import Store
    from itol.cache.l0_exact import L0Cache
    from itol.icr import ICRResponse, UsageStats

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = Store(tmpdir)
        l0    = L0Cache(store)

        fake_response = ICRResponse(
            request_id=icr.request_id,
            provider="openai",
            model="gpt-4o",
            content=[ContentBlock.text("The project budget is $1.2M (estimated).")],
            usage=UsageStats(input_tokens=50, output_tokens=12),
            finish_reason="stop",
        )

        key = l0.make_key(icr)
        assert key is not None, "make_key must return a valid key for this ICR"
        assert len(key) == 64, "Cache key must be a 64-char sha256 hex string"

        l0.set(key, icr.tenant_id, fake_response, ttl_seconds=3600)
        retrieved = l0.get(key, icr.tenant_id)

        assert retrieved is not None, "L0 cache miss after set()"
        assert retrieved.request_id == fake_response.request_id
        assert retrieved.content[0].text == fake_response.content[0].text
        print("[4/5] L0 cache round-trip PASS")

        # ------------------------------------------------------------------
        # Step 5 — Telemetry record
        # ------------------------------------------------------------------
        from itol.telemetry.recorder import Recorder

        recorder = Recorder(store, data_dir=tmpdir)
        recorder.record(
            request_id=icr.request_id,
            tenant_id=icr.tenant_id,
            provider="openai",
            model="gpt-4o",
            request_class=cls_result.primary,
            classifier_conf=cls_result.confidence,
            tokens_in_original=signals.token_count,
            tokens_in_optimized=signals.token_count,
            tokens_saved=0,
            gross_cost_saved_usd=0.0,
            shadow_cost_usd=0.0,
            provider_usage={"prompt_tokens": 50, "completion_tokens": 12},
            qps=qps_result.qps,
        )

        # Verify SQLite record
        db_row = store.get_request(icr.request_id)
        assert db_row is not None, "Telemetry record must appear in SQLite"
        assert db_row["request_id"] == icr.request_id

        # Verify JSON-lines record
        jsonl_path = Path(tmpdir) / "telemetry.jsonl"
        assert jsonl_path.exists(), "telemetry.jsonl must be created"
        with jsonl_path.open() as f:
            jsonl_record = json.loads(f.readline())
        assert jsonl_record["request_id"] == icr.request_id

        print("[5/5] Telemetry PASS — SQLite + jsonl written")

        # Close SQLite connections before TemporaryDirectory cleanup (Windows file locking)
        store.close()

    print()
    print("PHASE 1 GATE: PASS")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"\nPHASE 1 GATE: FAIL — {exc}", file=sys.stderr)
        sys.exit(1)
