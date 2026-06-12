"""
Seed 120 realistic fake request rows into the ITOL SQLite store for demo/dev.

Usage:
    python scripts/seed_dashboard_demo.py [--data-dir ./data]

Generates:
  - 120 requests spread over 24 h ending now
  - lognormal token savings  (mean ~200, max ~2000)
  - realistic class distribution
  - cache hit rates: ~10% L0, ~25% L1, ~10% L2, ~55% miss
  - traffic peak at 18:00 local time
  - random strategies from S1–S7
"""

from __future__ import annotations

import argparse
import math
import random
import sqlite3
import time
import uuid
from pathlib import Path

CLASSES     = ["simple", "complex", "code", "creative", "factual"]
CLASS_WEIGHTS = [0.30, 0.25, 0.20, 0.10, 0.15]

MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
    "claude-opus-4-8-20251101", "claude-sonnet-4-6-20251022",
]

STRATEGIES = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]

CACHE_LEVELS  = ["l0", "l1", "l2", "miss"]
CACHE_WEIGHTS = [0.10, 0.25, 0.10, 0.55]

NUM_ROWS = 120
WINDOW_H = 24

SCHEMA_REQUESTS = """
CREATE TABLE IF NOT EXISTS requests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT    NOT NULL,
    tenant_id    TEXT    NOT NULL DEFAULT 'default',
    ts           REAL    NOT NULL,
    model        TEXT,
    request_class TEXT,
    tokens_saved  INTEGER NOT NULL DEFAULT 0,
    cost_saved_usd REAL   NOT NULL DEFAULT 0.0,
    qps           REAL,
    cache_result  TEXT    NOT NULL DEFAULT 'miss',
    strategies    TEXT    NOT NULL DEFAULT '[]',
    rollback_stage TEXT,
    raw_request   TEXT,
    raw_response  TEXT
)
"""


def traffic_weight(hour: float) -> float:
    """Gaussian-ish bump peaking at 18:00."""
    center = 18.0
    sigma  = 4.0
    base   = 0.15
    return base + (1.0 - base) * math.exp(-0.5 * ((hour - center) / sigma) ** 2)


def lognormal_tokens() -> int:
    """lognormal with mean ~200, hard-capped at 2000."""
    mu, sigma = math.log(200) - 0.5 * 0.9**2, 0.9
    val = int(random.lognormvariate(mu, sigma))
    return min(max(val, 0), 2000)


def pick_strategies(cls: str) -> list[str]:
    """Return 1–3 strategies; complex/code get more."""
    n = 3 if cls in ("complex", "code") else (2 if cls == "creative" else 1)
    return random.sample(STRATEGIES, k=min(n, len(STRATEGIES)))


def cost_per_token(model: str, tokens: int) -> float:
    prices = {
        "gpt-4o":          0.000015,
        "gpt-4o-mini":     0.0000006,
        "gpt-3.5-turbo":   0.000002,
        "claude-opus-4-8-20251101":   0.000015,
        "claude-sonnet-4-6-20251022": 0.000003,
    }
    return tokens * prices.get(model, 0.00001)


def sample_timestamps(n: int, window_h: int) -> list[float]:
    """Sample n timestamps over [now-window_h*3600, now] using traffic_weight."""
    now = time.time()
    start = now - window_h * 3600
    samples: list[float] = []
    while len(samples) < n:
        t = random.uniform(start, now)
        hour = (t % 86400) / 3600
        if random.random() < traffic_weight(hour):
            samples.append(t)
    return sorted(samples)


def seed(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "itol.db"
    con = sqlite3.connect(str(db_path))
    con.execute(SCHEMA_REQUESTS)
    con.commit()

    timestamps = sample_timestamps(NUM_ROWS, WINDOW_H)
    rows = []
    for ts in timestamps:
        cls      = random.choices(CLASSES, weights=CLASS_WEIGHTS)[0]
        model    = random.choice(MODELS)
        cache    = random.choices(CACHE_LEVELS, weights=CACHE_WEIGHTS)[0]
        strats   = pick_strategies(cls)
        tokens   = lognormal_tokens() if cache == "miss" else int(lognormal_tokens() * 0.4)
        cost     = cost_per_token(model, tokens)
        qps_val  = round(random.uniform(0.88, 1.0), 4)
        rollback = None
        if qps_val < 0.93 and random.random() < 0.3:
            rollback = random.choice(["s4", "s5"])
            tokens   = 0
            cost     = 0.0
        rows.append((
            str(uuid.uuid4()),
            "default",
            ts,
            model,
            cls,
            tokens,
            cost,
            qps_val,
            cache,
            str(strats),
            rollback,
        ))

    con.executemany("""
        INSERT INTO requests
          (request_id, tenant_id, ts, model, request_class, tokens_saved,
           cost_saved_usd, qps, cache_result, strategies, rollback_stage)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    con.commit()
    con.close()

    print(f"Seeded {len(rows)} rows into {db_path}")
    cache_counts = {}
    for r in rows:
        c = r[8]
        cache_counts[c] = cache_counts.get(c,0) + 1
    for k,v in sorted(cache_counts.items()):
        print(f"  {k}: {v} ({v/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", help="ITOL data directory")
    args = parser.parse_args()
    seed(Path(args.data_dir))
