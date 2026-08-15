"""Renderer for the new 'Mercado v2' panel.

Layout:
  - Header with team logos + names + leagues
  - ML probability strip (Mapa / MD3 / MD5)
  - Per-market block:
      - editable line(s), one row per line with "+" to add
      - stats table per window (Ano todo / Últimos 10 / Últimos 5) with
        semaphore-colored hit-rate cells
      - context alerts (impressive hits, weak hits, surprising misses)

All ML probability computations are delegated — this module only handles
display. The pure-data layer lives in market_window_analysis_core.
"""
from __future__ import annotations

import math
from html import escape
from typing import Any, List, MutableMapping, Optional

import pandas as pd

from app_core.market_window_analysis_core import (
    WINDOWS_DEFAULT,
    build_alert_lines,
    compute_window_stats,
    highlight_context_samples,
)


# Color palette for team badges — picked by deterministic hash of team name.
_BADGE_COLORS = (
    "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
    "#3b82f6", "#6366f1", "#a855f7", "#ec4899", "#0ea5e9",
)


def _team_badge_html(team_name: str) -> str:
    """Build a tiny inline badge (initials + colored circle) — no external dep."""
    name = str(team_name or "").strip()
    if not name:
        return ""
    parts = [p for p in re.split(r"[\s\.\-]+", name) if p]
    if not parts:
        initials = name[:2].upper()
    elif len(parts) == 1:
        initials = parts[0][:2].upper()
    else:
        initials = (parts[0][0] + parts[1][0]).upper()
    color_idx = sum(ord(c) for c in name) % len(_BADGE_COLORS)
    bg = _BADGE_COLORS[color_idx]
    return (
        f"<div style='display:inline-flex;align-items:center;justify-content:center;"
        f"width:56px;height:56px;border-radius:50%;background:{bg};"
        f"color:white;font-weight:800;font-size:1.1em;"
        f"box-shadow:inset 0 1px 0 rgba(255,255,255,0.18);"
        f"border:1px solid rgba(255,255,255,0.15);'>"
        f"{escape(initials)}</div>"
    )


import re


_TONE_COLOR = {
    "green":  "#22c55e",
    "yellow": "#facc15",
    "red":    "#ef4444",
    "neutral": "#9ca3af",
}

_TONE_BG = {
    "green":  "rgba(34,197,94,0.15)",
    "yellow": "rgba(250,204,21,0.18)",
    "red":    "rgba(239,68,68,0.15)",
    "neutral": "rgba(156,163,175,0.10)",
}

_ALERT_TONE_BG = {
    "good": "rgba(34,197,94,0.12)",
    "warn": "rgba(250,204,21,0.16)",
    "info": "rgba(96,165,250,0.12)",
}
_ALERT_TONE_BORDER = {
    "good": "rgba(34,197,94,0.45)",
    "warn": "rgba(250,204,21,0.45)",
    "info": "rgba(96,165,250,0.45)",
}
_ALERT_TONE_ICON = {
    "good": "✓",
    "warn": "⚠",
    "info": "ℹ",
}


def _fmt_pct(v: float, decimals: int = 0) -> str:
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        return "—"
    return f"{float(v):.{int(decimals)}f}%"


def _fmt_num(v: float, decimals: int = 1) -> str:
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        return "—"
    return f"{float(v):.{int(decimals)}f}"


def _odds_from_prob(p: float) -> float:
    if not (isinstance(p, (int, float)) and 0.0 < float(p) < 1.0 and math.isfinite(float(p))):
        return float("nan")
    return 1.0 / float(p)


def render_team_header(
    *,
    st: Any,
    team_a: str,
    team_b: str,
    league_a: str = "",
    league_b: str = "",
) -> None:
    """Header row: [badge_a] team_a (liga_a)   VS   team_b (liga_b) [badge_b].

    Uses lightweight text-only badges (no external image deps) so the header
    never breaks due to CDN/hotlink issues.
    """
    badge_a = _team_badge_html(team_a)
    badge_b = _team_badge_html(team_b)
    league_a_html = (
        f"<div style='opacity:0.7;font-size:0.85em;margin-top:2px;'>{escape(str(league_a))}</div>"
        if str(league_a).strip() else ""
    )
    league_b_html = (
        f"<div style='opacity:0.7;font-size:0.85em;margin-top:2px;'>{escape(str(league_b))}</div>"
        if str(league_b).strip() else ""
    )
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:18px;padding:8px 4px;'>"
        f"<div style='flex:0 0 auto;'>{badge_a}</div>"
        f"<div style='flex:1 1 auto;'>"
        f"<div style='font-size:1.4em;font-weight:700;line-height:1.1;'>{escape(str(team_a))}</div>"
        f"{league_a_html}"
        f"</div>"
        f"<div style='flex:0 0 auto;font-size:1.3em;opacity:0.5;font-weight:600;'>VS</div>"
        f"<div style='flex:1 1 auto;text-align:right;'>"
        f"<div style='font-size:1.4em;font-weight:700;line-height:1.1;'>{escape(str(team_b))}</div>"
        f"{league_b_html}"
        f"</div>"
        f"<div style='flex:0 0 auto;'>{badge_b}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_ml_strip(
    *,
    st: Any,
    team_a: str,
    team_b: str,
    p_map: Optional[float] = None,
    p_md3: Optional[float] = None,
    p_md5: Optional[float] = None,
) -> None:
    """Display Mapa/MD3/MD5 probabilities as a compact strip."""
    rows = [
        ("Mapa", p_map),
        ("MD3",  p_md3),
        ("MD5",  p_md5),
    ]
    cols = st.columns(len(rows))
    for col, (label, p) in zip(cols, rows):
        with col:
            if p is None or not (isinstance(p, (int, float)) and 0.0 < float(p) < 1.0):
                st.markdown(f"**{label}** — predição indisponível")
                continue
            p_a = float(p)
            p_b = 1.0 - p_a
            odd_a = _odds_from_prob(p_a)
            odd_b = _odds_from_prob(p_b)
            st.markdown(
                f"**{label}**  \n"
                f"{escape(team_a)}: **{odd_a:.2f}** ({p_a*100:.1f}%)  \n"
                f"{escape(team_b)}: **{odd_b:.2f}** ({p_b*100:.1f}%)"
            )


def _stat_cell_html(stat) -> str:
    """Render a stat cell (n, hits, rate%, mean) with semaphore color."""
    color = _TONE_COLOR.get(stat.tone, _TONE_COLOR["neutral"])
    bg = _TONE_BG.get(stat.tone, _TONE_BG["neutral"])
    rate_str = _fmt_pct(stat.hit_rate_pct, 0)
    mean_str = _fmt_num(stat.mean_value, 1)
    opp_str = _fmt_num(stat.opp_avg_baseline, 1)
    return (
        f"<div style='background:{bg};border-left:3px solid {color};"
        f"padding:6px 10px;border-radius:6px;line-height:1.35;'>"
        f"<div style='font-weight:700;font-size:1.1em;color:{color};'>"
        f"{stat.hits}/{stat.n} · {rate_str}</div>"
        f"<div style='opacity:0.7;font-size:0.78em;'>méd {mean_str} | adv. baseline {opp_str}</div>"
        f"</div>"
    )


def render_market_block(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    market_label: str,
    market_key: str,
    metric_col: str,
    direction: str,
    default_line: float,
    team_a: str,
    team_b: str,
    df_team_a: pd.DataFrame,
    df_team_b: pd.DataFrame,
    league_scope_df: pd.DataFrame,
) -> None:
    """Render one market block: line input + per-team stats table + alerts."""

    st.markdown(f"#### {escape(market_label)}")

    # ----- Line inputs: one + button to add more -------------------
    lines_state_key = f"mv2_lines_{market_key}_{team_a}_{team_b}"
    if lines_state_key not in session_state:
        session_state[lines_state_key] = [float(default_line)]
    lines: List[float] = list(session_state.get(lines_state_key) or [float(default_line)])

    # Build line rows
    # %g = no trailing zeros, dot as separator (27.5 stays "27.5", 28 stays "28")
    line_format = "%g"
    new_lines: List[float] = []
    for i, line_val in enumerate(lines):
        line_cols = st.columns([3, 0.5])
        with line_cols[0]:
            raw = st.number_input(
                f"Linha #{i+1}", min_value=0.0, max_value=200.0, step=0.5,
                value=float(line_val),
                key=f"{lines_state_key}__val_{i}",
                format=line_format,
                label_visibility="collapsed" if i > 0 else "visible",
            )
        with line_cols[1]:
            keep = True
            if len(lines) > 1:
                if st.button("✕", key=f"{lines_state_key}__rm_{i}", help="Remover esta linha"):
                    keep = False
            if keep:
                new_lines.append(float(raw))
            else:
                # Skipping this line; do not append to new_lines
                pass

    add_col, _ = st.columns([1, 5])
    with add_col:
        if st.button("+ Adicionar linha", key=f"{lines_state_key}__add"):
            new_lines.append(float(new_lines[-1] if new_lines else default_line) + 1.0)

    # Persist (deduped while preserving order)
    seen = set()
    dedup = []
    for v in new_lines or [float(default_line)]:
        if v not in seen:
            seen.add(v)
            dedup.append(v)
    session_state[lines_state_key] = dedup

    # ----- For each line, render team-by-window stats ------------------
    for line_val in dedup:
        st.markdown(
            f"<div style='margin-top:6px;opacity:0.78;'>"
            f"<b>{escape(direction)} {line_val:g}</b></div>",
            unsafe_allow_html=True,
        )

        stats_a = compute_window_stats(
            team_df=df_team_a, league_scope_df=league_scope_df,
            metric_col=metric_col, line=float(line_val), direction=direction,
        )
        stats_b = compute_window_stats(
            team_df=df_team_b, league_scope_df=league_scope_df,
            metric_col=metric_col, line=float(line_val), direction=direction,
        )

        # Render: 1 col team label, then one col per window
        header_cells = ["<div></div>"] + [
            f"<div style='font-weight:600;opacity:0.7;font-size:0.85em;text-align:center;'>{escape(label)}</div>"
            for label, _ in WINDOWS_DEFAULT
        ]
        grid_cols = st.columns([1.2] + [1.0] * len(WINDOWS_DEFAULT))
        for col, html in zip(grid_cols, header_cells):
            with col:
                st.markdown(html, unsafe_allow_html=True)
        for team_name, team_stats in ((team_a, stats_a), (team_b, stats_b)):
            grid_cols = st.columns([1.2] + [1.0] * len(team_stats))
            with grid_cols[0]:
                st.markdown(f"<div style='padding-top:8px;font-weight:600;'>{escape(team_name)}</div>",
                            unsafe_allow_html=True)
            for col, stat in zip(grid_cols[1:], team_stats):
                with col:
                    st.markdown(_stat_cell_html(stat), unsafe_allow_html=True)

        # ----- Context alerts ----------------------------------------
        alerts: List[dict] = []
        for team_name, team_df in ((team_a, df_team_a), (team_b, df_team_b)):
            samples = highlight_context_samples(
                team_df=team_df, league_scope_df=league_scope_df,
                metric_col=metric_col, line=float(line_val), direction=direction,
                last_n_to_inspect=10, max_samples_per_side=2,
            )
            alerts.extend(build_alert_lines(team_name=team_name, samples=samples,
                                              direction=direction))
        if alerts:
            for alert in alerts[:6]:
                tone = str(alert.get("tone") or "info")
                text = str(alert.get("text") or "")
                icon = _ALERT_TONE_ICON.get(tone, "ℹ")
                bg = _ALERT_TONE_BG.get(tone, _ALERT_TONE_BG["info"])
                border = _ALERT_TONE_BORDER.get(tone, _ALERT_TONE_BORDER["info"])
                st.markdown(
                    f"<div style='background:{bg};border-left:3px solid {border};"
                    f"padding:6px 10px;border-radius:6px;margin-top:6px;font-size:0.92em;'>"
                    f"{icon} {escape(text)}</div>",
                    unsafe_allow_html=True,
                )


def render_market_v2_panel(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    team_a: str,
    team_b: str,
    league_a: str = "",
    league_b: str = "",
    df_team_a: pd.DataFrame,
    df_team_b: pd.DataFrame,
    league_scope_df: pd.DataFrame,
    p_map: Optional[float] = None,
    p_md3: Optional[float] = None,
    p_md5: Optional[float] = None,
    markets: Optional[List[dict]] = None,
) -> None:
    """Top-level renderer. `markets` is a list of dicts:
        {label, key, metric_col, direction, default_line}.
    """
    if not str(team_a).strip() or not str(team_b).strip():
        st.info("Selecione os dois times para abrir a visão de mercados v2.")
        return

    render_team_header(st=st, team_a=team_a, team_b=team_b,
                        league_a=league_a, league_b=league_b)
    st.markdown("---")
    render_ml_strip(st=st, team_a=team_a, team_b=team_b,
                     p_map=p_map, p_md3=p_md3, p_md5=p_md5)
    st.markdown("---")

    if not markets:
        markets = [{
            "label": "Total de kills",
            "key": "kills",
            "metric_col": "kills_for",
            "direction": "Over",
            "default_line": 27.5,
        }]
    for mkt in markets:
        render_market_block(
            st=st, session_state=session_state,
            market_label=str(mkt.get("label") or mkt.get("key") or "Mercado"),
            market_key=str(mkt.get("key") or "mkt"),
            metric_col=str(mkt.get("metric_col") or "kills_for"),
            direction=str(mkt.get("direction") or "Over"),
            default_line=float(mkt.get("default_line") or 0.0),
            team_a=str(team_a), team_b=str(team_b),
            df_team_a=df_team_a, df_team_b=df_team_b,
            league_scope_df=league_scope_df,
        )
        st.markdown(" ")


__all__ = [
    "render_market_block",
    "render_market_v2_panel",
    "render_ml_strip",
    "render_team_header",
]
