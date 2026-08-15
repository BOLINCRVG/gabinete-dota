"""Roster continuity and transfer signal helpers (pure ML functions)."""
from __future__ import annotations

import math
from typing import Any, Callable


def transfer_signal_from_details(details: list[dict[str, Any]] | None, *, tier_priority_fn: Callable[[str], Any]) -> float:
    if not isinstance(details, list) or not details:
        return 0.0
    good = 0.0
    total = 0.0
    for row in details:
        if not isinstance(row, dict) or not bool(row.get("used")):
            continue
        for side in ("blue_dbg", "red_dbg"):
            dbg = row.get(side) or {}
            if not isinstance(dbg, dict):
                continue
            last_tier = str(dbg.get("last_tier") or "").strip()
            cur_tier = str(dbg.get("cur_tier") or "").strip()
            if not last_tier or not cur_tier:
                continue
            total += 1.0
            try:
                if int(tier_priority_fn(last_tier)) < int(tier_priority_fn(cur_tier)):
                    good += 1.0
            except Exception:
                continue
    return max(0.0, min(1.0, good / total)) if total > 0 else 0.0


def lineup_continuity_from_details(details: list[dict[str, Any]] | None, *, freshness_avg: float, cov_ratio: float) -> float:
    try:
        fresh = max(0.0, min(1.0, float(freshness_avg if math.isfinite(float(freshness_avg)) else 0.0)))
    except Exception:
        fresh = 0.0
    try:
        cov = max(0.0, min(1.0, float(cov_ratio if math.isfinite(float(cov_ratio)) else 0.0)))
    except Exception:
        cov = 0.0
    if not isinstance(details, list) or not details:
        return float(0.5 * fresh + 0.5 * cov)
    vals: list[float] = []
    for row in details:
        if not isinstance(row, dict) or not bool(row.get("used")):
            continue
        for side in ("blue_dbg", "red_dbg"):
            dbg = row.get(side) or {}
            if not isinstance(dbg, dict):
                continue
            gp = float(dbg.get("gp", 0.0) or 0.0)
            tscale = float(dbg.get("transfer_scale", 1.0) or 1.0)
            gp_sig = max(0.0, min(1.0, gp / 12.0))
            tr_sig = max(0.0, min(1.0, tscale))
            vals.append(0.55 * gp_sig + 0.45 * tr_sig)
    base = float(sum(vals) / len(vals)) if vals else 0.5
    cont = 0.55 * base + 0.25 * fresh + 0.20 * cov
    return float(max(0.0, min(1.0, cont)))
