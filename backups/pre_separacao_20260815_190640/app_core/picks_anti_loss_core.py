"""Demote picks based on the user's own historical ROI by market and league.

Backed by analysis of 6,151 LOL picks (2025 + 2026):
  - Some markets bleed money no matter the matchup (e.g., Over Torres lost
    52% ROI across 44 picks).
  - Some leagues underperform consistently (LEC, LFL, EMEA).

Rather than hard-blocking, this module:
  1. Demotes the pick's category (recommend → watch → avoid) when its market
     has a strongly negative historical ROI.
  2. Surfaces a warning text the UI can render inline ("⚠ Mercado com ROI -X%
     no seu histórico").

Side categorization (Over vs Under) matters — Under Torres is profitable
(+18.81%) while Over Torres bleeds (-52.45%). We keep the markets keyed by
(metric_key, side_key) so the rule is precise.
"""
from __future__ import annotations

from typing import Optional


# ── Fallback snapshot (frozen 2025+2026 analysis) ───────────────────────────
# Used when the live tracker hasn't been loaded yet. Once the xlsx backfill
# lands in ops/picks_tracker.json, the live overlay below supersedes these.
# Only markets with n >= 30 picks are included — smaller samples are noise.
BAD_MARKET_ROI: dict[tuple[str, str], float] = {
    ("towers", "Over"):    -52.45,   # n=44   — severe
    ("dragons", "Over"):   -20.33,   # n=76   — severe
    ("inhibitors", "Over"):  -7.75,  # n=39   — mild (under threshold)
}

# League warnings — these don't demote on their own; they surface a caption.
# Only leagues with n >= 100 picks are included.
BAD_LEAGUE_ROI: dict[str, float] = {
    "LFL":   -9.61,   # n=147
    "EMEA":  -5.12,   # n=182
    "LEC":   -3.99,   # n=720
}

# Markets where the user has a strong positive edge — we surface this too,
# as a positive caption when the matchup hits one of these.
STRONG_MARKET_ROI: dict[tuple[str, str], float] = {
    ("towers", "Under"):     +18.81,  # n=866
    ("dragons", "Under"):    +14.64,  # n=242
    ("kills", "Over"):       +12.66,  # n=286
    ("inhibitors", "Under"): +11.50,  # n=116
    ("time", "Under"):        +8.88,  # n=229
    ("kills", "Under"):       +8.33,  # n=2025
}

# ── Live overlay (populated from ops/picks_tracker.json) ────────────────────
# {(metric_key, side_key): roi_pct}. None until loaded. When present, lookups
# prefer this over the frozen snapshot above.
_LIVE_MARKET_ROI: Optional[dict[tuple[str, str], float]] = None
_LIVE_LEAGUE_ROI: Optional[dict[str, float]] = None


def set_live_roi(by_market: Optional[dict], by_league: Optional[dict]) -> None:
    """Install the live ROI overlay computed from the tracker.

    Pass None/empty to clear and fall back to the frozen snapshot.
    """
    global _LIVE_MARKET_ROI, _LIVE_LEAGUE_ROI
    _LIVE_MARKET_ROI = dict(by_market) if by_market else None
    _LIVE_LEAGUE_ROI = dict(by_league) if by_league else None


def load_live_roi_from_tracker(tracker_path: str = "ops/picks_tracker.json") -> bool:
    """Read the tracker, compute ROI maps, install them as the live overlay.

    Returns True when an overlay was installed (tracker had settled picks),
    False otherwise (caller keeps using the frozen snapshot). Never raises.
    """
    try:
        from app_core.picks_tracker_core import load_picks
        from app_core.picks_roi_stats_core import compute_roi_maps
        picks = load_picks(tracker_path)
        if not picks:
            return False
        maps = compute_roi_maps(picks)
        by_market = maps.get("by_market") or {}
        by_league = maps.get("by_league") or {}
        if not by_market and not by_league:
            return False
        set_live_roi(by_market, by_league)
        return True
    except Exception:
        return False


def _market_roi_lookup(key: tuple[str, str]) -> Optional[float]:
    """ROI for (metric, side): live overlay first, frozen snapshot as fallback."""
    if _LIVE_MARKET_ROI is not None and key in _LIVE_MARKET_ROI:
        return _LIVE_MARKET_ROI[key]
    if key in STRONG_MARKET_ROI:
        return STRONG_MARKET_ROI[key]
    if key in BAD_MARKET_ROI:
        return BAD_MARKET_ROI[key]
    return None


def _league_roi_lookup(league_upper: str) -> Optional[float]:
    if _LIVE_LEAGUE_ROI is not None and league_upper in _LIVE_LEAGUE_ROI:
        return _LIVE_LEAGUE_ROI[league_upper]
    return BAD_LEAGUE_ROI.get(league_upper)


def _normalize_league(league: Optional[str]) -> str:
    return str(league or "").strip().upper()


def demote_classification(
    classification: str,
    *,
    metric_key: str,
    side_key: str,
) -> tuple[str, Optional[str]]:
    """Return (new_classification, warning_text).

    Demotion thresholds:
      ROI <= -20%  → 2 levels down (recommend → avoid)
      ROI <= -10%  → 1 level down  (recommend → watch, watch → avoid)
      otherwise    → no demotion, no warning
    """
    key = (str(metric_key).lower().strip(), str(side_key).strip())
    roi = _market_roi_lookup(key)
    if roi is None or roi > -10.0:
        return classification, None

    severity_text = f"Mercado com ROI {roi:+.0f}% no seu histórico"
    if roi <= -20.0:
        # Severe: any pick goes to avoid.
        if classification in ("recommend", "watch"):
            return "avoid", f"⛔ {severity_text} (severo)."
        return classification, f"⛔ {severity_text} (severo)."
    # Mild (between -10% and -20%): one level down.
    if classification == "recommend":
        return "watch", f"⚠ {severity_text}."
    if classification == "watch":
        return "avoid", f"⚠ {severity_text}."
    return classification, f"⚠ {severity_text}."


def league_caption(league: Optional[str]) -> Optional[str]:
    """If the league has a known negative ROI, return a warning caption.

    None when the league is unknown or has acceptable historical ROI.
    """
    lg = _normalize_league(league)
    if not lg:
        return None
    roi = _league_roi_lookup(lg)
    # Only warn on negative ROI (live overlay may carry positive leagues too).
    if roi is None or roi >= 0.0:
        return None
    return f"⚠ Liga **{lg}** tem ROI {roi:+.1f}% no seu histórico — atenção redobrada."


def market_strength_caption(metric_key: str, side_key: str) -> Optional[str]:
    """If the (metric, side) is in the strong set, return a positive caption.

    Used in the v3 redesigned Resumo to put a green note on profitable markets.
    """
    key = (str(metric_key).lower().strip(), str(side_key).strip())
    roi = _market_roi_lookup(key)
    # Positive caption only when the market is actually profitable (>= +5%).
    if roi is None or roi < 5.0:
        return None
    return f"🟢 {key[1]} {key[0]} tem ROI **{roi:+.1f}%** no seu histórico — ponto forte."


def market_historical_roi(metric_key: str, side_key: str) -> Optional[float]:
    """Single lookup that returns the user's historical ROI for any tracked
    (metric_key, side_key), or None when there's no entry.

    Used by the simplified Resumo calculator: when you preview a pick, we
    show 'this market gave you +18% ROI in 2025-2026' or '-52% ROI (avoid)'.
    """
    key = (str(metric_key).lower().strip(), str(side_key).strip())
    return _market_roi_lookup(key)


__all__ = [
    "BAD_LEAGUE_ROI",
    "BAD_MARKET_ROI",
    "STRONG_MARKET_ROI",
    "demote_classification",
    "league_caption",
    "load_live_roi_from_tracker",
    "market_historical_roi",
    "market_strength_caption",
    "set_live_roi",
]
