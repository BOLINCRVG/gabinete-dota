"""Pure statistical and formatting helpers for live-bet draft analysis.

Extracted from live_bet_draft_core.py to reduce that module's complexity.
All functions here are stateless — no Streamlit, no session state.
"""
from __future__ import annotations

from html import escape
from typing import Any, Optional

import pandas as pd

from mlcore.utils import safe_float_or_none as _safe_float


def _draft_wr_parts(df_scope_players: pd.DataFrame, champ_name: str, player_name: str | None) -> dict[str, Any]:
    """Compute Laplace-smoothed win-rate stats for a champion (and optionally a player) by side."""
    d_ch = df_scope_players[df_scope_players["champ"].astype(str) == str(champ_name)].copy()
    n_ch = int(d_ch.shape[0])
    w_ch = float(pd.to_numeric(d_ch.get("result"), errors="coerce").fillna(0).sum()) if n_ch > 0 else 0.0
    p_ch = (w_ch + 2.0) / (n_ch + 4.0) if n_ch > 0 else 0.5
    n_pl = 0
    p_pl = p_ch
    if player_name:
        d_pl = d_ch[d_ch["player"].astype(str) == str(player_name)].copy()
        n_pl = int(d_pl.shape[0])
        if n_pl > 0:
            w_pl = float(pd.to_numeric(d_pl.get("result"), errors="coerce").fillna(0).sum())
            p_pl = (w_pl + 1.0) / (n_pl + 2.0)
    d_ch_blue = d_ch[d_ch["side"].astype(str).str.title() == "Blue"].copy() if "side" in d_ch.columns else pd.DataFrame([])
    d_ch_red = d_ch[d_ch["side"].astype(str).str.title() == "Red"].copy() if "side" in d_ch.columns else pd.DataFrame([])
    n_ch_blue = int(d_ch_blue.shape[0])
    n_ch_red = int(d_ch_red.shape[0])
    p_ch_blue = (
        float((pd.to_numeric(d_ch_blue.get("result"), errors="coerce").fillna(0).sum() + 1.0) / (n_ch_blue + 2.0))
        if n_ch_blue > 0
        else 0.5
    )
    p_ch_red = (
        float((pd.to_numeric(d_ch_red.get("result"), errors="coerce").fillna(0).sum() + 1.0) / (n_ch_red + 2.0))
        if n_ch_red > 0
        else 0.5
    )
    d_pl_all = d_ch[d_ch["player"].astype(str) == str(player_name)].copy() if player_name else pd.DataFrame([])
    d_pl_blue = d_pl_all[d_pl_all["side"].astype(str).str.title() == "Blue"].copy() if ("side" in d_pl_all.columns and not d_pl_all.empty) else pd.DataFrame([])
    d_pl_red = d_pl_all[d_pl_all["side"].astype(str).str.title() == "Red"].copy() if ("side" in d_pl_all.columns and not d_pl_all.empty) else pd.DataFrame([])
    n_pl_blue = int(d_pl_blue.shape[0])
    n_pl_red = int(d_pl_red.shape[0])
    p_pl_blue = (
        float((pd.to_numeric(d_pl_blue.get("result"), errors="coerce").fillna(0).sum() + 1.0) / (n_pl_blue + 2.0))
        if n_pl_blue > 0
        else p_ch_blue
    )
    p_pl_red = (
        float((pd.to_numeric(d_pl_red.get("result"), errors="coerce").fillna(0).sum() + 1.0) / (n_pl_red + 2.0))
        if n_pl_red > 0
        else p_ch_red
    )
    w_player = float(n_pl) / float(n_pl + 8.0) if n_pl > 0 else 0.0
    p_blend = (1.0 - w_player) * float(p_ch) + w_player * float(p_pl)
    return {
        "p_league": float(p_ch),
        "p_player": float(p_pl),
        "p_blend": float(p_blend),
        "n_league": int(n_ch),
        "n_player": int(n_pl),
        "p_league_blue": float(p_ch_blue),
        "p_league_red": float(p_ch_red),
        "n_league_blue": int(n_ch_blue),
        "n_league_red": int(n_ch_red),
        "p_player_blue": float(p_pl_blue),
        "p_player_red": float(p_pl_red),
        "n_player_blue": int(n_pl_blue),
        "n_player_red": int(n_pl_red),
    }


def _league_alert_label(
    *,
    current_wr: Optional[float],
    current_sample: int,
    other_wr: Optional[float],
    other_sample: int,
) -> str:
    """Return a contextual label when a champion's WR diverges significantly across leagues."""
    if current_wr is None or other_wr is None or int(current_sample or 0) < 3 or int(other_sample or 0) < 5:
        return ""
    delta = float(current_wr) - float(other_wr)
    if abs(delta) < 8.0:
        return ""
    if delta >= 12.0:
        return "Campeonato puxa pra cima"
    if delta <= -12.0:
        return "Campeonato puxa pra baixo"
    return "Sensivel ao campeonato"


def _laplace_win_pct(df_in: pd.DataFrame) -> Optional[float]:
    """Return Laplace-smoothed win percentage (0-100) from a player/champion dataframe."""
    if df_in is None or df_in.empty or "result" not in df_in.columns:
        return None
    wins = float(pd.to_numeric(df_in.get("result"), errors="coerce").fillna(0).sum())
    n_games = int(df_in.shape[0])
    if n_games <= 0:
        return None
    return 100.0 * float((wins + 1.0) / (n_games + 2.0))


def _mean_per_min(df_in: pd.DataFrame, value_col: str, time_col: str = "gamelength") -> Optional[float]:
    """Compute average per-minute rate for a stat column divided by game length in seconds."""
    if df_in is None or df_in.empty or value_col not in df_in.columns or time_col not in df_in.columns:
        return None
    value = pd.to_numeric(df_in.get(value_col), errors="coerce")
    time_value = pd.to_numeric(df_in.get(time_col), errors="coerce")
    if value is None or time_value is None:
        return None
    time_min = time_value / 60.0
    out = value / time_min.replace(0, pd.NA)
    out = pd.to_numeric(out, errors="coerce").dropna()
    if out.empty:
        return None
    return _safe_float(out.mean())


def _scale_0_100(value: Any, *, target: float) -> float:
    """Scale value to [0, 100] relative to target. Returns 0 if target <= 0."""
    raw = float(_safe_float(value) or 0.0)
    if target <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * raw / float(target)))


def _safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    """Divide two values safely; returns None when denominator is zero or inputs are invalid."""
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if num is None or den is None or den <= 0.0:
        return None
    return float(num) / float(den)


def _series_mean_safe(df_in: pd.DataFrame, column: str) -> Optional[float]:
    """Return mean of a numeric column from a dataframe, or None if empty or missing."""
    if df_in is None or df_in.empty or column not in df_in.columns:
        return None
    return _safe_float(pd.to_numeric(df_in.get(column), errors="coerce").mean())


def _wr_side_span(label: str, value: Any, games: int, color: str) -> str:
    """Render a colored HTML span for a side-specific win-rate value."""
    text = str(value or "-").strip()
    games_text = f" ({games})" if int(games or 0) > 0 else ""
    return (
        f'<span style="display:inline-block;margin-left:6px;font-size:12px;'
        f'color:{color};font-weight:600;">{escape(label)} {escape(text)}%{escape(games_text)}</span>'
        if text and text != "-"
        else f'<span style="display:inline-block;margin-left:6px;font-size:12px;color:{color};opacity:0.72;">{escape(label)} -</span>'
    )


def _wr_dual_html(*, blue_value: Any, blue_games: int, red_value: Any, red_games: int) -> str:
    """Render blue-side and red-side win-rate spans side by side."""
    return _wr_side_span("B", blue_value, blue_games, "#6ec1ff") + _wr_side_span("R", red_value, red_games, "#ff7a86")
