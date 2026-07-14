"""Compute live ROI maps from the tracked picks (xlsx backfill + manual).

Turns the user's real settled history into:
  - by_market: {(metric_key, side_key): roi_pct}  with sample counts
  - by_league: {LEAGUE: roi_pct}                  with sample counts

These replace the hardcoded tables in picks_anti_loss_core at runtime, so the
"ROI hist" shown on cards and the demotion thresholds reflect the actual
spreadsheet history instead of a frozen snapshot.

Pure + deterministic: takes a list of Pick, returns plain dicts.
"""
from __future__ import annotations

from typing import Iterable, Optional


def _roi_from(profit_units: float, risked_units: float) -> Optional[float]:
    if risked_units <= 0:
        return None
    return 100.0 * profit_units / risked_units


def compute_roi_maps(
    picks: Iterable,
    *,
    min_n_market: int = 30,
    min_n_league: int = 100,
) -> dict:
    """Aggregate settled picks into ROI maps.

    Only settled picks (won/lost/void) count. Void contributes 0 profit and is
    excluded from the risked denominator (no risk on a cashout/void).

    Markets below min_n_market (or leagues below min_n_league) are dropped —
    small samples are noise, same rule used for the original hardcoded tables.

    Returns:
      {
        "by_market": {(metric, side): roi_pct},
        "n_by_market": {(metric, side): n_settled},
        "by_league": {LEAGUE_UPPER: roi_pct},
        "n_by_league": {LEAGUE_UPPER: n_settled},
        "n_total": int,
      }
    """
    mkt_profit: dict[tuple, float] = {}
    mkt_risk: dict[tuple, float] = {}
    mkt_n: dict[tuple, int] = {}
    lg_profit: dict[str, float] = {}
    lg_risk: dict[str, float] = {}
    lg_n: dict[str, int] = {}
    n_total = 0

    for p in picks:
        status = getattr(p, "status", "")
        if status not in ("won", "lost", "void"):
            continue
        n_total += 1
        profit = float(getattr(p, "profit_units", 0.0) or 0.0)
        stake = float(getattr(p, "stake_units", 0.0) or 0.0)
        risk = 0.0 if status == "void" else stake

        metric = str(getattr(p, "metric_key", "") or "")
        side = str(getattr(p, "side_key", "") or "")
        # ML side A/B isn't an Over/Under market — group under ("ml","").
        if metric == "ml":
            mkey = ("ml", "")
        else:
            mkey = (metric, side)
        mkt_profit[mkey] = mkt_profit.get(mkey, 0.0) + profit
        mkt_risk[mkey] = mkt_risk.get(mkey, 0.0) + risk
        mkt_n[mkey] = mkt_n.get(mkey, 0) + 1

        league = str(getattr(p, "league", "") or "").strip().upper()
        if league:
            lg_profit[league] = lg_profit.get(league, 0.0) + profit
            lg_risk[league] = lg_risk.get(league, 0.0) + risk
            lg_n[league] = lg_n.get(league, 0) + 1

    by_market: dict[tuple, float] = {}
    n_by_market: dict[tuple, int] = {}
    for k, n in mkt_n.items():
        if n < int(min_n_market):
            continue
        roi = _roi_from(mkt_profit.get(k, 0.0), mkt_risk.get(k, 0.0))
        if roi is not None:
            by_market[k] = roi
            n_by_market[k] = n

    by_league: dict[str, float] = {}
    n_by_league: dict[str, int] = {}
    for k, n in lg_n.items():
        if n < int(min_n_league):
            continue
        roi = _roi_from(lg_profit.get(k, 0.0), lg_risk.get(k, 0.0))
        if roi is not None:
            by_league[k] = roi
            n_by_league[k] = n

    return {
        "by_market": by_market,
        "n_by_market": n_by_market,
        "by_league": by_league,
        "n_by_league": n_by_league,
        "n_total": n_total,
    }


__all__ = ["compute_roi_maps"]
