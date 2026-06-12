"""
Dashboard stats aggregation — queries the requests table.

All queries are read-only; no writes here.
"""

from __future__ import annotations

import json
import time
from typing import Any


def _ts_floor(window: str) -> float:
    now = time.time()
    windows = {"24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}
    return now - windows.get(window, 86400)


def get_stats(store: Any, tenant_id: str = "default", window: str = "24h") -> dict:
    """
    Return a dashboard stats snapshot for the given tenant + time window.

    Returns a dict with:
      total_requests, tokens_saved, cost_saved_usd, avg_qps, rollbacks,
      cache_breakdown (l0/l1/l2/miss counts),
      strategy_breakdown (strategy_id → count),
      timeseries (list of {ts_hour, tokens_saved, cost_saved} per hour),
      recent (last 20 activity records)
    """
    ts_floor = _ts_floor(window)

    conn = store._conn()
    cur = conn.cursor()

    # Summary aggregates
    cur.execute(
        """
        SELECT
            COUNT(*)                     AS total_requests,
            COALESCE(SUM(tokens_saved), 0)      AS tokens_saved,
            COALESCE(SUM(est_cost_saved_usd), 0) AS cost_saved,
            COALESCE(AVG(qps), 0)               AS avg_qps,
            COALESCE(SUM(CASE WHEN rollback_stage IS NOT NULL THEN 1 ELSE 0 END), 0) AS rollbacks
        FROM requests
        WHERE ts >= ? AND tenant_id = ?
        """,
        (ts_floor, tenant_id),
    )
    row = cur.fetchone()
    summary = {
        "total_requests": row[0] or 0,
        "tokens_saved": row[1] or 0,
        "cost_saved_usd": round(row[2] or 0.0, 6),
        "avg_qps": round(row[3] or 0.0, 4),
        "rollbacks": row[4] or 0,
    }

    # Cache breakdown
    cur.execute(
        "SELECT cache_result FROM requests WHERE ts >= ? AND tenant_id = ? AND cache_result IS NOT NULL",
        (ts_floor, tenant_id),
    )
    cache_breakdown = {"l0": 0, "l1": 0, "l2": 0, "miss": 0}
    for (cr_json,) in cur.fetchall():
        try:
            cr = json.loads(cr_json) if isinstance(cr_json, str) else cr_json
            level = (cr.get("level") or cr.get("cache") or "miss").lower()
            if level in cache_breakdown:
                cache_breakdown[level] += 1
            else:
                cache_breakdown["miss"] += 1
        except Exception:
            cache_breakdown["miss"] += 1

    # Strategy breakdown
    cur.execute(
        "SELECT strategies_applied FROM requests WHERE ts >= ? AND tenant_id = ? AND strategies_applied IS NOT NULL",
        (ts_floor, tenant_id),
    )
    strategy_breakdown: dict[str, int] = {}
    for (sa_json,) in cur.fetchall():
        try:
            strategies = json.loads(sa_json) if isinstance(sa_json, str) else []
            for s in (strategies or []):
                strategy_breakdown[s] = strategy_breakdown.get(s, 0) + 1
        except Exception:
            pass

    # Hourly time series
    cur.execute(
        """
        SELECT
            CAST(ts / 3600 AS INTEGER) AS ts_hour,
            COALESCE(SUM(tokens_saved), 0) AS tokens_saved,
            COALESCE(SUM(est_cost_saved_usd), 0) AS cost_saved
        FROM requests
        WHERE ts >= ? AND tenant_id = ?
        GROUP BY ts_hour
        ORDER BY ts_hour ASC
        """,
        (ts_floor, tenant_id),
    )
    timeseries = [
        {"ts_hour": r[0], "tokens_saved": r[1], "cost_saved": round(r[2], 6)}
        for r in cur.fetchall()
    ]

    # Recent activity (last 20)
    cur.execute(
        """
        SELECT request_id, ts, model, request_class, tokens_saved, qps, cache_result,
               strategies_applied, rollback_stage, error
        FROM requests
        WHERE tenant_id = ?
        ORDER BY ts DESC
        LIMIT 20
        """,
        (tenant_id,),
    )
    recent = []
    for r in cur.fetchall():
        recent.append({
            "request_id": r[0],
            "ts": r[1],
            "model": r[2],
            "request_class": r[3],
            "tokens_saved": r[4] or 0,
            "qps": round(r[5], 4) if r[5] is not None else None,
            "cache": _parse_cache_level(r[6]),
            "strategies": _safe_json_list(r[7]),
            "rollback": r[8],
            "error": r[9],
        })

    return {
        **summary,
        "window": window,
        "cache_breakdown": cache_breakdown,
        "strategy_breakdown": strategy_breakdown,
        "timeseries": timeseries,
        "recent": recent,
        "generated_at": time.time(),
    }


def _parse_cache_level(cr_json: str | None) -> str:
    if not cr_json:
        return "miss"
    try:
        cr = json.loads(cr_json) if isinstance(cr_json, str) else cr_json
        return (cr.get("level") or cr.get("cache") or "miss").lower()
    except Exception:
        return "miss"


def _safe_json_list(v: str | None) -> list:
    if not v:
        return []
    try:
        result = json.loads(v)
        return result if isinstance(result, list) else []
    except Exception:
        return []
