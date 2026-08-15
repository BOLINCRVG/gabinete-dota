"""Calibration helpers — measure how well past picks landed in each prob range.

When a new pick predicts p_hist=0.65, we want to answer:
  "When you bet with p_hist between 0.60 and 0.70 historically, you hit X%"

That answer can only come from the tracker's resolved picks (the user's own
history). For brand-new accounts, sample is tiny — caller should degrade
gracefully (e.g., "calibração indisponível — só 2 picks nessa faixa").
"""
from __future__ import annotations

from typing import Optional, Sequence

from app_core.picks_tracker_core import Pick


def calibration_in_range(
    picks: Sequence[Pick],
    *,
    p_center: float,
    half_window_pp: float = 5.0,
    metric_key: Optional[str] = None,
) -> dict:
    """Return hit rate of settled picks whose p_hist sits inside the window.

    p_center / half_window_pp are in 0-1 scale (e.g., p_center=0.65,
    half_window_pp=0.05 → window [0.60, 0.70]).

    metric_key filters to a single market when provided. None = aggregate
    across all markets.

    Returns:
      {n: int, wins: int, hit_pct: float | None,
       p_min: float, p_max: float,
       confidence: "low"|"medium"|"high"}
    """
    p_min = max(0.0, float(p_center) - float(half_window_pp))
    p_max = min(1.0, float(p_center) + float(half_window_pp))
    n = 0; wins = 0
    for p in picks:
        if p.status not in ("won", "lost"):  # void picks skipped
            continue
        if p.p_hist is None:
            continue
        if metric_key and str(p.metric_key) != str(metric_key):
            continue
        if not (p_min <= float(p.p_hist) <= p_max):
            continue
        n += 1
        if p.status == "won":
            wins += 1
    if n == 0:
        return {"n": 0, "wins": 0, "hit_pct": None,
                "p_min": p_min, "p_max": p_max, "confidence": "low"}
    hit_pct = 100.0 * wins / n
    # Confidence by sample size:
    if n >= 20:
        confidence = "high"
    elif n >= 8:
        confidence = "medium"
    else:
        confidence = "low"
    return {"n": n, "wins": wins, "hit_pct": hit_pct,
            "p_min": p_min, "p_max": p_max, "confidence": confidence}


def calibration_summary(picks: Sequence[Pick]) -> list[dict]:
    """Bucket all settled picks into 10pp bins and report hit rate per bin.

    Returns a list of {p_min, p_max, n, wins, hit_pct} for bins
    [0.40,0.50), [0.50,0.60), ..., [0.80,0.90].
    """
    bins = [(round(0.40 + 0.10 * i, 2), round(0.50 + 0.10 * i, 2)) for i in range(5)]
    out: list[dict] = []
    for lo, hi in bins:
        n = 0; wins = 0
        for p in picks:
            if p.status not in ("won", "lost") or p.p_hist is None:
                continue
            if lo <= float(p.p_hist) < hi:
                n += 1
                if p.status == "won":
                    wins += 1
        out.append({
            "p_min": lo, "p_max": hi, "n": n, "wins": wins,
            "hit_pct": (100.0 * wins / n) if n > 0 else None,
        })
    return out


__all__ = ["calibration_in_range", "calibration_summary"]
