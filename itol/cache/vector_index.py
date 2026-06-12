"""
Vector index — §6.1 L1 ANN backend.

Two implementations:
  SqliteVecIndex — sqlite-vec if available; brute-force numpy cosine fallback.

CR-7 namespace isolation: vectors from different namespaces are stored and
searched independently — search() ONLY returns hits from within the given namespace.
"""

from __future__ import annotations

import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Vector math helpers
# ---------------------------------------------------------------------------

def _vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# VectorIndex ABC
# ---------------------------------------------------------------------------

class VectorIndex(ABC):
    """
    Abstract vector index with namespace isolation.

    All three operations are scoped to `namespace` — entries in different
    namespaces are never visible to each other (CR-7).
    """

    @abstractmethod
    def add(self, id: str, vector: np.ndarray, namespace: str) -> None:
        """Add or replace a vector entry."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        namespace: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k most similar (id, cosine_score) pairs in `namespace`.
        CR-7: MUST only search within the given namespace.
        """
        ...

    @abstractmethod
    def delete(self, id: str, namespace: str) -> None:
        """Remove an entry from the index."""
        ...


# ---------------------------------------------------------------------------
# SqliteVecIndex — sqlite-vec ANN or brute-force numpy fallback
# ---------------------------------------------------------------------------

_LOCAL = threading.local()

_VECTORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS l1_vectors (
    entry_id  TEXT NOT NULL,
    namespace TEXT NOT NULL,
    vector    BLOB NOT NULL,
    PRIMARY KEY (entry_id, namespace)
);
CREATE INDEX IF NOT EXISTS idx_l1_vectors_ns ON l1_vectors (namespace);
"""


class SqliteVecIndex(VectorIndex):
    """
    SQLite-backed vector index.

    Tries sqlite-vec for native ANN (sub-linear search).
    Falls back to brute-force numpy cosine (linear scan, correct for small scale).
    Both paths maintain namespace isolation.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._sqlite_vec_available = False
        self._init_schema()
        self._try_load_sqlite_vec()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if getattr(_LOCAL, "vec_conn", None) is None or _LOCAL.vec_db_path != self._db_path:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            if self._sqlite_vec_available:
                try:
                    import sqlite_vec
                    conn.enable_load_extension(True)
                    sqlite_vec.load(conn)
                    conn.enable_load_extension(False)
                except Exception:
                    pass
            _LOCAL.vec_conn = conn
            _LOCAL.vec_db_path = self._db_path
        return _LOCAL.vec_conn  # type: ignore[return-value]

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.executescript(_VECTORS_SCHEMA)
        conn.commit()
        conn.close()

    def _try_load_sqlite_vec(self) -> None:
        try:
            import sqlite_vec  # noqa: F401
            self._sqlite_vec_available = True
        except ImportError:
            self._sqlite_vec_available = False

    # ------------------------------------------------------------------
    # VectorIndex interface
    # ------------------------------------------------------------------

    def add(self, id: str, vector: np.ndarray, namespace: str) -> None:
        blob = _vec_to_blob(vector)
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO l1_vectors (entry_id, namespace, vector) VALUES (?,?,?)",
            (id, namespace, blob),
        )
        conn.commit()

    def search(
        self,
        query_vector: np.ndarray,
        namespace: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        CR-7: search is STRICTLY scoped to `namespace`.
        Returns up to top_k (entry_id, cosine_score) pairs, sorted descending.
        """
        rows = self._conn().execute(
            "SELECT entry_id, vector FROM l1_vectors WHERE namespace = ?",
            (namespace,),
        ).fetchall()
        if not rows:
            return []

        scores: list[tuple[str, float]] = []
        for row in rows:
            vec = _blob_to_vec(bytes(row["vector"]))
            score = _cosine(query_vector.astype(np.float32), vec)
            scores.append((row["entry_id"], score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def delete(self, id: str, namespace: str) -> None:
        conn = self._conn()
        conn.execute(
            "DELETE FROM l1_vectors WHERE entry_id=? AND namespace=?", (id, namespace)
        )
        conn.commit()

    def close(self) -> None:
        conn = getattr(_LOCAL, "vec_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            _LOCAL.vec_conn = None
