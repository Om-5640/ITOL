"""
Persistent SQLite store — §6 + §9.

One file at <data_dir>/itol.db.  WAL mode enabled; connections are per-thread
via threading.local.

Tables
------
requests     — telemetry record per optimisation pass (§9.1)
l0_cache     — exact-match response cache (§6.1)
templates    — S2 template reuse tracking
docs         — RACR doc store for S4
conversations — S5 conversation ledger (CR-6)

CR-17
-----
record_request() accepts provider_usage: dict as the authoritative source for
tokens_out — never uses an estimate.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


_local = threading.local()

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS requests (
    request_id           TEXT PRIMARY KEY,
    tenant_id            TEXT NOT NULL,
    ts                   REAL NOT NULL,
    provider             TEXT,
    model                TEXT,
    request_class        TEXT,
    classifier_conf      REAL,
    template_sig         TEXT,
    tokens_in_original   INTEGER,
    tokens_in_optimized  INTEGER,
    tokens_out           INTEGER,
    tokens_saved         INTEGER,
    est_cost_saved_usd   REAL,
    shadow_cost_usd      REAL DEFAULT 0,
    provider_cache_read_tokens  INTEGER,
    strategies_applied   TEXT,
    strategy_savings     TEXT,
    qps                  REAL,
    rollback_stage       TEXT,
    cache_result         TEXT,
    latency_ms           TEXT,
    shadow_sampled       INTEGER DEFAULT 0,
    shadow_parity        REAL,
    error                TEXT
);

CREATE TABLE IF NOT EXISTS l0_cache (
    cache_key    TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at   REAL NOT NULL,
    expires_at   REAL NOT NULL,
    tokens_saved INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS templates (
    template_sig          TEXT NOT NULL,
    tenant_id             TEXT NOT NULL,
    reuse_count           INTEGER DEFAULT 1,
    last_seen             REAL NOT NULL,
    compressed_instruction TEXT,
    compression_verified  INTEGER DEFAULT 0,
    PRIMARY KEY (template_sig, tenant_id)
);

CREATE TABLE IF NOT EXISTS docs (
    doc_hash        TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    content         TEXT NOT NULL,
    version_hash    TEXT NOT NULL,
    created_at      REAL NOT NULL,
    PRIMARY KEY (doc_hash, tenant_id, conversation_id)
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    ledger_json     TEXT,
    turn_hashes     TEXT,
    updated_at      REAL NOT NULL,
    PRIMARY KEY (conversation_id, tenant_id)
);
"""


class Store:
    """
    Persistent store for all ITOL state.

    Thread-safe: each thread gets its own sqlite3.Connection via threading.local.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "itol.db"
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if getattr(_local, "conn", None) is None or _local.db_path != str(self.db_path):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            _local.conn = conn
            _local.db_path = str(self.db_path)
        return _local.conn  # type: ignore[return-value]

    def _init_schema(self) -> None:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    def close(self) -> None:
        """Close the current thread's connection (important on Windows before dir cleanup)."""
        conn = getattr(_local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            _local.conn = None

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def record_request(
        self,
        *,
        request_id: str,
        tenant_id: str,
        provider: str | None = None,
        model: str | None = None,
        request_class: str | None = None,
        classifier_conf: float | None = None,
        template_sig: str | None = None,
        tokens_in_original: int = 0,
        tokens_in_optimized: int = 0,
        tokens_saved: int = 0,
        est_cost_saved_usd: float = 0.0,
        shadow_cost_usd: float = 0.0,
        provider_usage: dict[str, Any] | None = None,  # CR-17: authoritative token source
        qps: float | None = None,
        rollback_stage: str | None = None,
        cache_result: dict[str, Any] | None = None,
        latency_ms: dict[str, float] | None = None,
        strategies_applied: list[str] | None = None,
        strategy_savings: dict[str, Any] | None = None,
        shadow_sampled: bool = False,
        shadow_parity: float | None = None,
        error: str | None = None,
    ) -> None:
        # CR-17: tokens_out comes from provider_usage, never from estimates
        tokens_out = None
        provider_cache_read = None
        if provider_usage:
            tokens_out = provider_usage.get("completion_tokens") or provider_usage.get("output_tokens")
            provider_cache_read = provider_usage.get("prompt_tokens_details", {}).get("cached_tokens") if isinstance(provider_usage.get("prompt_tokens_details"), dict) else None

        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO requests (
                request_id, tenant_id, ts, provider, model, request_class,
                classifier_conf, template_sig, tokens_in_original, tokens_in_optimized,
                tokens_out, tokens_saved, est_cost_saved_usd, shadow_cost_usd,
                provider_cache_read_tokens, strategies_applied, strategy_savings,
                qps, rollback_stage, cache_result, latency_ms,
                shadow_sampled, shadow_parity, error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                request_id, tenant_id, time.time(), provider, model, request_class,
                classifier_conf, template_sig, tokens_in_original, tokens_in_optimized,
                tokens_out, tokens_saved, est_cost_saved_usd, shadow_cost_usd,
                provider_cache_read,
                json.dumps(strategies_applied) if strategies_applied is not None else None,
                json.dumps(strategy_savings) if strategy_savings is not None else None,
                qps, rollback_stage,
                json.dumps(cache_result) if cache_result is not None else None,
                json.dumps(latency_ms) if latency_ms is not None else None,
                1 if shadow_sampled else 0, shadow_parity, error,
            ),
        )
        conn.commit()

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # L0 cache
    # ------------------------------------------------------------------

    def get_l0(self, cache_key: str, tenant_id: str) -> str | None:
        """Return cached response_json if the entry exists and has not expired."""
        row = self._conn().execute(
            "SELECT response_json FROM l0_cache WHERE cache_key=? AND tenant_id=? AND expires_at>?",
            (cache_key, tenant_id, time.time()),
        ).fetchone()
        return row[0] if row else None

    def set_l0(
        self,
        cache_key: str,
        tenant_id: str,
        response_json: str,
        ttl_seconds: int,
        tokens_saved: int = 0,
    ) -> None:
        now = time.time()
        self._conn().execute(
            """INSERT OR REPLACE INTO l0_cache
               (cache_key, tenant_id, response_json, created_at, expires_at, tokens_saved)
               VALUES (?,?,?,?,?,?)""",
            (cache_key, tenant_id, response_json, now, now + ttl_seconds, tokens_saved),
        )
        self._conn().commit()

    # ------------------------------------------------------------------
    # Template tracking  (S2)
    # ------------------------------------------------------------------

    def increment_template(
        self,
        template_sig: str,
        tenant_id: str,
        compressed_instruction: str | None = None,
    ) -> int:
        conn = self._conn()
        now = time.time()
        row = conn.execute(
            "SELECT reuse_count FROM templates WHERE template_sig=? AND tenant_id=?",
            (template_sig, tenant_id),
        ).fetchone()
        if row:
            new_count = row[0] + 1
            conn.execute(
                "UPDATE templates SET reuse_count=?, last_seen=? WHERE template_sig=? AND tenant_id=?",
                (new_count, now, template_sig, tenant_id),
            )
        else:
            new_count = 1
            conn.execute(
                "INSERT INTO templates (template_sig, tenant_id, reuse_count, last_seen, compressed_instruction) VALUES (?,?,?,?,?)",
                (template_sig, tenant_id, 1, now, compressed_instruction),
            )
        conn.commit()
        return new_count

    # ------------------------------------------------------------------
    # Conversation ledger  (S5 / CR-6)
    # ------------------------------------------------------------------

    def get_conversation(self, conversation_id: str, tenant_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT ledger_json, turn_hashes FROM conversations WHERE conversation_id=? AND tenant_id=?",
            (conversation_id, tenant_id),
        ).fetchone()
        if not row:
            return None
        return {
            "ledger": json.loads(row[0]) if row[0] else None,
            "turn_hashes": json.loads(row[1]) if row[1] else [],
        }

    def set_conversation(
        self,
        conversation_id: str,
        tenant_id: str,
        ledger: Any | None = None,
        turn_hashes: list[str] | None = None,
    ) -> None:
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO conversations
               (conversation_id, tenant_id, ledger_json, turn_hashes, updated_at)
               VALUES (?,?,?,?,?)""",
            (
                conversation_id, tenant_id,
                json.dumps(ledger) if ledger is not None else None,
                json.dumps(turn_hashes or []),
                time.time(),
            ),
        )
        conn.commit()
