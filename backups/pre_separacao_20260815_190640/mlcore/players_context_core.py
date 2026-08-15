from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from mlcore.utils import clip_safe as _clip, safe_float as _f, wmean as _wmean


def _coerce_dt(v: Any) -> pd.Timestamp:
    if isinstance(v, pd.Timestamp):
        return v.tz_localize(None) if v.tzinfo is not None else v
    if isinstance(v, datetime):
        return pd.Timestamp(v).tz_localize(None)
    if isinstance(v, date):
        return pd.Timestamp(v)
    dt = pd.to_datetime(v, errors="coerce")
    if isinstance(dt, pd.Timestamp) and dt.tzinfo is not None:
        dt = dt.tz_localize(None)
    if not isinstance(dt, pd.Timestamp) or pd.isna(dt):
        return pd.Timestamp.utcnow().tz_localize(None)
    return dt


def _as_filter_field(filters: Any, key: str) -> Any:
    if isinstance(filters, Mapping):
        return filters.get(key)
    return getattr(filters, key, None)


def _wrate(mask: np.ndarray, y01: np.ndarray, w: np.ndarray) -> float:
    mm = mask & np.isfinite(y01) & np.isfinite(w) & (w > 0.0)
    if int(np.sum(mm)) <= 0:
        return float("nan")
    return _wmean(y01[mm], w[mm])


def _safe_col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.full((len(df),), np.nan, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def _team_quality_profile(
    df: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    half_life_days: float,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "n_games": 0,
            "quality_score": 0.0,
            "reliability": 0.0,
            "data_reliability": 0.0,
        }

    dt = pd.to_datetime(df.get("date"), errors="coerce")
    age_days = ((as_of - dt).dt.total_seconds() / (24.0 * 3600.0)).fillna(0.0).to_numpy(dtype=float)
    age_days = np.maximum(age_days, 0.0)
    hl = float(max(5.0, _f(half_life_days, 45.0)))
    w = np.power(0.5, age_days / hl)
    if not np.isfinite(w).any() or float(np.sum(w)) <= 1e-12:
        w = np.ones((len(df),), dtype=float)

    win = np.where(_safe_col(df, "win") > 0.5, 1.0, 0.0)
    kills_for = _safe_col(df, "kills_for")
    kills_against = _safe_col(df, "kills_against")
    towers_for = _safe_col(df, "towers_for")
    towers_against = _safe_col(df, "towers_against")
    dragons_for = _safe_col(df, "dragons_for")
    dragons_against = _safe_col(df, "dragons_against")
    barons_for = _safe_col(df, "barons_for")
    barons_against = _safe_col(df, "barons_against")
    inhib_for = _safe_col(df, "inhibitors_for")
    inhib_against = _safe_col(df, "inhibitors_against")
    game_time = _safe_col(df, "game_time_min")

    gd15 = _safe_col(df, "golddiffat15_for")
    if not np.isfinite(gd15).any():
        gd15 = _safe_col(df, "golddiffat15")
    lead_n = np.clip(gd15 / 4500.0, -2.0, 2.0)

    kills_den = np.maximum(1.0, np.abs(kills_for) + np.abs(kills_against))
    kill_control = (kills_for - kills_against) / kills_den
    towers_den = np.maximum(1.0, np.abs(towers_for) + np.abs(towers_against))
    tower_control = (towers_for - towers_against) / towers_den
    dragons_den = np.maximum(1.0, np.abs(dragons_for) + np.abs(dragons_against))
    dragon_control = (dragons_for - dragons_against) / dragons_den
    barons_den = np.maximum(1.0, np.abs(barons_for) + np.abs(barons_against))
    baron_control = (barons_for - barons_against) / barons_den
    inhib_den = np.maximum(1.0, np.abs(inhib_for) + np.abs(inhib_against))
    inhib_control = (inhib_for - inhib_against) / inhib_den

    objective_control = (
        0.32 * tower_control
        + 0.22 * dragon_control
        + 0.24 * baron_control
        + 0.10 * inhib_control
        + 0.12 * kill_control
    )
    net_control = 0.55 * np.clip(lead_n, -1.0, 1.0) + 0.45 * objective_control

    lead_mask = np.isfinite(lead_n) & (lead_n > 0.18)
    behind_mask = np.isfinite(lead_n) & (lead_n < -0.18)
    loss = np.where(win > 0.5, 0.0, 1.0)
    closeout_mask = lead_mask & np.isfinite(game_time) & (game_time <= np.nanmedian(game_time))
    stomp_mask = lead_mask & (tower_control >= 0.18) & (win > 0.5)
    suffer_mask = behind_mask & (tower_control <= -0.15) & (loss > 0.5)

    closeout_rate = _wrate(closeout_mask, win, w)
    throw_rate = _wrate(lead_mask, loss, w)
    comeback_rate = _wrate(behind_mask, win, w)
    suffer_rate = _wrate(suffer_mask, loss, w)
    stomp_rate = _wrate(stomp_mask, win, w)

    objective_control_avg = _wmean(objective_control, w)
    net_control_avg = _wmean(net_control, w)
    containment_avg = _wmean(-np.clip(kills_against / np.maximum(1.0, kills_for + 1.0), 0.0, 2.5) + 1.0, w)

    # Composite in [-1, 1]: rewards controlled wins/comebacks and penalizes throws/suffer losses.
    quality_score = (
        0.26 * (objective_control_avg if math.isfinite(objective_control_avg) else 0.0)
        + 0.16 * (net_control_avg if math.isfinite(net_control_avg) else 0.0)
        + 0.18 * ((closeout_rate if math.isfinite(closeout_rate) else 0.0) - (throw_rate if math.isfinite(throw_rate) else 0.0))
        + 0.16 * ((comeback_rate if math.isfinite(comeback_rate) else 0.0) - (suffer_rate if math.isfinite(suffer_rate) else 0.0))
        + 0.14 * (stomp_rate if math.isfinite(stomp_rate) else 0.0)
        + 0.10 * (containment_avg if math.isfinite(containment_avg) else 0.0)
    )
    quality_score = _clip(quality_score, -1.0, 1.0)

    field_list = [
        "win",
        "game_time_min",
        "towers_for",
        "towers_against",
        "dragons_for",
        "dragons_against",
        "barons_for",
        "barons_against",
        "inhibitors_for",
        "inhibitors_against",
        "kills_for",
        "kills_against",
        "golddiffat15_for",
    ]
    present = 0
    total = max(1, len(field_list))
    for c in field_list:
        if c in df.columns:
            arr = _safe_col(df, c)
            if np.isfinite(arr).any():
                present += 1
    data_reliability = float(present / total)

    n_games = len(df)
    n_lead = int(np.sum(lead_mask))
    n_behind = int(np.sum(behind_mask))
    sample_rel = float(n_games / (n_games + 12.0))
    situation_rel = float(min(1.0, (n_lead + n_behind) / max(1.0, 0.60 * n_games)))
    reliability = _clip((0.50 * sample_rel) + (0.30 * data_reliability) + (0.20 * situation_rel), 0.0, 1.0)

    return {
        "n_games": int(n_games),
        "n_lead": int(n_lead),
        "n_behind": int(n_behind),
        "quality_score": float(quality_score),
        "reliability": float(reliability),
        "data_reliability": float(data_reliability),
        "terms": {
            "objective_control": float(objective_control_avg if math.isfinite(objective_control_avg) else 0.0),
            "net_control": float(net_control_avg if math.isfinite(net_control_avg) else 0.0),
            "closeout_rate": float(closeout_rate if math.isfinite(closeout_rate) else 0.0),
            "throw_rate": float(throw_rate if math.isfinite(throw_rate) else 0.0),
            "comeback_rate": float(comeback_rate if math.isfinite(comeback_rate) else 0.0),
            "suffer_rate": float(suffer_rate if math.isfinite(suffer_rate) else 0.0),
            "stomp_rate": float(stomp_rate if math.isfinite(stomp_rate) else 0.0),
            "containment": float(containment_avg if math.isfinite(containment_avg) else 0.0),
        },
    }


def compute_players_context_quality_signal(
    *,
    team_games: Any,
    team_a: str,
    team_b: str,
    filters: Any = None,
    as_of_date: Any = None,
    enabled: bool = True,
    lookback_games: int = 24,
    half_life_days: float = 45.0,
    min_games: int = 6,
    strength: float = 0.35,
    cap: float = 0.22,
    min_reliability: float = 0.28,
    interregional: bool = False,
    interregional_mult: float = 0.88,
    league_hint_a: str = "",
    league_hint_b: str = "",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "",
        "delta": 0.0,
        "edge_score": 0.0,
        "edge_reliability": 0.0,
        "n_min": 0,
        "team_a": {"n_games": 0, "quality_score": 0.0, "reliability": 0.0},
        "team_b": {"n_games": 0, "quality_score": 0.0, "reliability": 0.0},
    }
    if not bool(enabled):
        out["reason"] = "disabled"
        return out
    if not isinstance(team_games, pd.DataFrame) or team_games.empty:
        out["reason"] = "no_team_games"
        return out

    tcol = "team" if "team" in team_games.columns else ("teamname" if "teamname" in team_games.columns else "")
    if not tcol:
        out["reason"] = "missing_team_column"
        return out

    df = team_games.copy()
    if "date" not in df.columns:
        out["reason"] = "missing_date_column"
        return out
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        out["reason"] = "no_valid_dates"
        return out

    f_year = _as_filter_field(filters, "year")
    f_split = _as_filter_field(filters, "split")
    f_playoffs = _as_filter_field(filters, "playoffs")
    f_league = _as_filter_field(filters, "league")
    if f_year is not None and "year" in df.columns:
        try:
            df = df[pd.to_numeric(df["year"], errors="coerce").fillna(-1).astype(int) == int(f_year)].copy()
        except Exception:
            pass
    if f_split and str(f_split).strip().upper() != "ALL" and "split" in df.columns:
        df = df[df["split"].fillna("").astype(str) == str(f_split)].copy()
    if f_playoffs is not None and "playoffs" in df.columns:
        try:
            po_val = bool(int(f_playoffs)) if isinstance(f_playoffs, (int, float, str)) else bool(f_playoffs)
            df = df[df["playoffs"] == po_val].copy()
        except Exception:
            pass
    if f_league and str(f_league).strip().upper() != "ALL" and "league" in df.columns:
        df = df[df["league"].fillna("").astype(str) == str(f_league)].copy()

    if df.empty:
        out["reason"] = "empty_after_filters"
        return out

    as_of = _coerce_dt(as_of_date)
    lookback_n = int(max(4, min(80, int(lookback_games or 24))))
    t_a = str(team_a or "").strip()
    t_b = str(team_b or "").strip()
    df_a = df[df[tcol].astype(str) == t_a].copy().sort_values("date").tail(lookback_n)
    df_b = df[df[tcol].astype(str) == t_b].copy().sort_values("date").tail(lookback_n)
    if "league" in df.columns:
        lha = str(league_hint_a or "").strip()
        lhb = str(league_hint_b or "").strip()
        if lha and len(df_a) >= 2:
            _tmp = df_a[df_a["league"].fillna("").astype(str) == lha].copy()
            if len(_tmp) >= 2:
                df_a = _tmp
        if lhb and len(df_b) >= 2:
            _tmp = df_b[df_b["league"].fillna("").astype(str) == lhb].copy()
            if len(_tmp) >= 2:
                df_b = _tmp

    pa = _team_quality_profile(df_a, as_of=as_of, half_life_days=float(half_life_days))
    pb = _team_quality_profile(df_b, as_of=as_of, half_life_days=float(half_life_days))
    out["team_a"] = pa
    out["team_b"] = pb
    out["n_min"] = int(min(int(pa.get("n_games", 0) or 0), int(pb.get("n_games", 0) or 0)))

    n_min = int(out["n_min"])
    if n_min < int(max(2, int(min_games or 6))):
        out["reason"] = "low_sample"
        return out

    qa = _f(pa.get("quality_score"), 0.0)
    qb = _f(pb.get("quality_score"), 0.0)
    edge = float(_clip(qa - qb, -2.0, 2.0))
    rel = float(
        _clip(
            min(
                _f(pa.get("reliability"), 0.0),
                _f(pb.get("reliability"), 0.0),
            ),
            0.0,
            1.0,
        )
    )
    out["edge_score"] = float(edge)
    out["edge_reliability"] = float(rel)

    if rel < float(max(0.05, min(0.95, _f(min_reliability, 0.28)))):
        out["reason"] = "low_reliability"
        return out

    s = float(max(0.0, min(2.0, _f(strength, 0.35))))
    c = float(max(0.02, min(0.75, _f(cap, 0.22))))
    raw = float(s * edge)
    shaped = float(c * math.tanh(raw / max(0.05, c)))
    if bool(interregional):
        shaped *= float(max(0.60, min(1.00, _f(interregional_mult, 0.88))))
    delta = float(shaped * rel)

    out["applied"] = bool(abs(delta) > 1e-9)
    out["reason"] = "ok" if out["applied"] else "zero_delta"
    out["delta"] = float(delta)
    out["weights"] = {
        "strength": float(s),
        "cap": float(c),
        "half_life_days": float(max(5.0, _f(half_life_days, 45.0))),
        "lookback_games": int(lookback_n),
        "min_games": int(max(2, int(min_games or 6))),
        "min_reliability": float(max(0.05, min(0.95, _f(min_reliability, 0.28)))),
        "interregional": bool(interregional),
        "interregional_mult": float(max(0.60, min(1.00, _f(interregional_mult, 0.88)))),
    }
    out["raw"] = {"edge_raw": float(edge), "delta_raw": float(raw), "delta_shaped": float(shaped)}
    return out
