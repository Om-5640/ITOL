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

CREATE TABLE IF NOT EXISTS l1_cache (
    entry_id         TEXT NOT NULL,
    namespace        TEXT NOT NULL,
    response_json    TEXT NOT NULL,
    query_text       TEXT NOT NULL DEFAULT '',
    manifest_numbers TEXT NOT NULL DEFAULT '[]',
    tokens_saved     INTEGER NOT NULL DEFAULT 0,
    last_access_1    REAL NOT NULL DEFAULT 0,
    last_access_2    REAL NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    PRIMARY KEY (entry_id, namespace)
);
CREATE INDEX IF NOT EXISTS idx_l1_cache_ns ON l1_cache (namespace);

CREATE TABLE IF NOT EXISTS l2_plans (
    template_sig           TEXT NOT NULL,
    tenant_id              TEXT NOT NULL,
    plan_json              TEXT NOT NULL,
    depends_on_doc_versions TEXT NOT NULL DEFAULT '[]',
    created_at             REAL NOT NULL,
    PRIMARY KEY (template_sig, tenant_id)
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

    # ------------------------------------------------------------------
    # L1 semantic cache (§6.1)
    # ------------------------------------------------------------------

    def get_l1_entry(self, entry_id: str, namespace: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT entry_id, namespace, response_json, query_text, manifest_numbers, tokens_saved "
            "FROM l1_cache WHERE entry_id=? AND namespace=?",
            (entry_id, namespace),
        ).fetchone()
        return dict(row) if row else None

    def set_l1_entry(
        self,
        entry_id: str,
        namespace: str,
        response_json: str,
        query_text: str,
        manifest_numbers_json: str,
        tokens_saved: int = 0,
    ) -> None:
        now = time.time()
        self._conn().execute(
            """INSERT OR REPLACE INTO l1_cache
               (entry_id, namespace, response_json, query_text, manifest_numbers,
                tokens_saved, last_access_1, last_access_2, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (entry_id, namespace, response_json, query_text,
             manifest_numbers_json, tokens_saved, now, now, now),
        )
        self._conn().commit()

    def update_l1_access(self, entry_id: str, namespace: str, now: float) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE l1_cache SET last_access_1 = last_access_2, last_access_2 = ? "
            "WHERE entry_id=? AND namespace=?",
            (now, entry_id, namespace),
        )
        conn.commit()

    def delete_l1_entry(self, entry_id: str, namespace: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM l1_cache WHERE entry_id=? AND namespace=?", (entry_id, namespace))
        conn.commit()

    def count_l1_entries(self, namespace: str | None = None) -> int:
        if namespace is not None:
            row = self._conn().execute(
                "SELECT COUNT(*) FROM l1_cache WHERE namespace=?", (namespace,)
            ).fetchone()
        else:
            row = self._conn().execute("SELECT COUNT(*) FROM l1_cache").fetchone()
        return row[0] if row else 0

    def evict_l1_lru_k(self, max_entries: int, savings_weight: float = 1e-4) -> int:
        """
        LRU-K(K=2) eviction: remove lowest-priority entries until count ≤ max_entries.
        Priority = last_access_1 + tokens_saved * savings_weight (higher = keep longer).
        """
        total = self.count_l1_entries()
        if total <= max_entries:
            return 0
        n_evict = total - max_entries
        conn = self._conn()
        rows = conn.execute(
            "SELECT entry_id, namespace FROM l1_cache "
            "ORDER BY (last_access_1 + tokens_saved * ?) ASC LIMIT ?",
            (savings_weight, n_evict),
        ).fetchall()
        evicted = 0
        for entry_id, namespace in rows:
            conn.execute(
                "DELETE FROM l1_cache WHERE entry_id=? AND namespace=?",
                (entry_id, namespace),
            )
            evicted += 1
        conn.commit()
        return evicted

    # ------------------------------------------------------------------
    # L2 template plan cache (§6.1)
    # ------------------------------------------------------------------

    def get_l2_plan(self, template_sig: str, tenant_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT plan_json, depends_on_doc_versions FROM l2_plans "
            "WHERE template_sig=? AND tenant_id=?",
            (template_sig, tenant_id),
        ).fetchone()
        return {"plan_json": row[0], "depends_on_doc_versions": row[1]} if row else None

    def set_l2_plan(
        self,
        template_sig: str,
        tenant_id: str,
        plan_json: str,
        depends_on_doc_versions: str,
    ) -> None:
        self._conn().execute(
            """INSERT OR REPLACE INTO l2_plans
               (template_sig, tenant_id, plan_json, depends_on_doc_versions, created_at)
               VALUES (?,?,?,?,?)""",
            (template_sig, tenant_id, plan_json, depends_on_doc_versions, time.time()),
        )
        self._conn().commit()

    def invalidate_l2_by_doc(self, doc_hash: str) -> int:
        """
        CR-9: delete all L2 plans that depend on doc_hash.
        Returns the number of plans tombstoned.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT template_sig, tenant_id, depends_on_doc_versions FROM l2_plans"
        ).fetchall()
        tombstoned = 0
        for template_sig, tenant_id, dep_json in rows:
            try:
                deps = json.loads(dep_json or "[]")
            except (json.JSONDecodeError, TypeError):
                deps = []
            if doc_hash in deps:
                conn.execute(
                    "DELETE FROM l2_plans WHERE template_sig=? AND tenant_id=?",
                    (template_sig, tenant_id),
                )
                tombstoned += 1
        conn.commit()
        return tombstoned
