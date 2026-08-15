"""Renderer for the new per-market sub-tab inside 'Análise de Linha' (v3).

Layout (flat — no nested expanders):
  1. Header: badges + team names + leagues
  2. ML strip: Mapa / MD3 / MD5
  3. Context bar: toggles (perspectiva, mapa) + line input
  4. Análise histórica em destaque (windows + H2H, big cards)
  5. Por mapa (panoramic table M1..M5)
  6. Predição ML (smaller, complementary)
  7. Perfil de adversário (league-wide table)
  8. Jogos recentes contextualizados
  9. Alertas

Designed to be reused across markets (Kills, Torres, Drakes...) by passing
different metric_col / market_label.
"""
from __future__ import annotations

import math
from html import escape
from typing import Any, Dict, List, Optional

import pandas as pd

from app_core.market_v3_analysis_core import (
    MapStats,
    OpponentProfileRow,
    RecentGameRow,
    WindowStats,
)
# Canonical design primitives live in ui_design_core; alias to the private
# names this module already uses so every internal callsite stays unchanged.
from app_core.ui_design_core import (
    TONE_BG as _TONE_BG,
    TONE_BORDER as _TONE_BORDER,
    TONE_HTML_COLOR as _TONE_HTML_COLOR,
    TONE_TEXT as _TONE_TEXT,
    team_badge_html as _team_badge_html,
    tone_for_pct as _tone_for_pct,
)

_STRENGTH_LABEL = {
    "very_under": "muito under",
    "under": "under",
    "avg": "neutro",
    "over": "over",
    "very_over": "muito over",
}

_TREND_ARROW = {
    "up": "↑",
    "down": "↓",
    "flat": "─",
    "na": "—",
}


def _fmt_metric_v3(value: float, *, is_time: bool, decimals: int = 1) -> str:
    """Format a metric value. For time markets, render as mm:ss; else N.d.

    `value` is in minutes (decimal) when is_time=True (e.g., 32.5 → "32:30").
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x):
        return "-"
    if is_time:
        sec = round(x * 60.0)
        return f"{sec // 60:02d}:{sec % 60:02d}"
    return f"{x:.{int(decimals)}f}"


def _fmt_pct(v: float, dec: int = 0) -> str:
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        return "—"
    return f"{float(v):.{int(dec)}f}%"


def _fmt_num(v: float, dec: int = 1) -> str:
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        return "—"
    return f"{float(v):.{int(dec)}f}"


def render_team_header(
    *, st: Any, team_a: str, team_b: str,
    league_a: str = "", league_b: str = "",
) -> None:
    badge_a = _team_badge_html(team_a, size=56)
    badge_b = _team_badge_html(team_b, size=56)
    la_html = (f"<div style='opacity:0.65;font-size:0.85em;margin-top:2px;'>{escape(str(league_a))}</div>"
               if str(league_a).strip() else "")
    lb_html = (f"<div style='opacity:0.65;font-size:0.85em;margin-top:2px;'>{escape(str(league_b))}</div>"
               if str(league_b).strip() else "")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:18px;padding:8px 4px;'>"
        f"<div>{badge_a}</div>"
        f"<div style='flex:1 1 auto;'>"
        f"<div style='font-size:1.5em;font-weight:700;line-height:1.1;'>{escape(team_a)}</div>{la_html}"
        f"</div>"
        f"<div style='font-size:1.3em;opacity:0.5;font-weight:600;'>VS</div>"
        f"<div style='flex:1 1 auto;text-align:right;'>"
        f"<div style='font-size:1.5em;font-weight:700;line-height:1.1;'>{escape(team_b)}</div>{lb_html}"
        f"</div>"
        f"<div>{badge_b}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_ml_strip(
    *, st: Any, team_a: str, team_b: str,
    p_map: Optional[float] = None,
    p_md3: Optional[float] = None,
    p_md5: Optional[float] = None,
) -> None:
    """Compact ML probabilities for Mapa/MD3/MD5."""
    rows = [("Mapa", p_map), ("MD3", p_md3), ("MD5", p_md5)]
    cols = st.columns(len(rows))
    for col, (label, p) in zip(cols, rows):
        with col:
            if p is None or not (isinstance(p, (int, float)) and 0.0 < float(p) < 1.0):
                st.markdown(f"**{label}**  \n_predição indisponível_")
                continue
            p_a = float(p); p_b = 1.0 - p_a
            odd_a = 1.0 / p_a if p_a > 0 else float("nan")
            odd_b = 1.0 / p_b if p_b > 0 else float("nan")
            st.markdown(
                f"**{label}** &nbsp; "
                f"{escape(team_a)} <b>{odd_a:.2f}</b> "
                f"<span style='opacity:0.7;'>({p_a*100:.1f}%)</span> · "
                f"{escape(team_b)} <b>{odd_b:.2f}</b> "
                f"<span style='opacity:0.7;'>({p_b*100:.1f}%)</span>",
                unsafe_allow_html=True,
            )


def _historic_card_html(team_name: str, ws: WindowStats, line: float, line_label: str) -> str:
    tone = _tone_for_pct(ws.over_pct)
    bg = _TONE_BG[tone]; border = _TONE_BORDER[tone]; color = _TONE_TEXT[tone]
    arrow = _TREND_ARROW.get(ws.trend, "—")
    arrow_color = ("#22c55e" if ws.trend == "up" else
                   "#ef4444" if ws.trend == "down" else "#9ca3af")
    if ws.n == 0:
        body = "<div style='opacity:0.5;'>sem jogos</div>"
    else:
        body = (
            f"<div style='display:flex;align-items:baseline;gap:12px;'>"
            f"<div style='font-size:1.8em;font-weight:700;color:{color};line-height:1;'>"
            f"{ws.hits_over}/{ws.n}</div>"
            f"<div style='font-size:1.2em;font-weight:600;color:{color};'>{ws.over_pct:.0f}%</div>"
            f"<div style='font-size:1.2em;color:{arrow_color};'>{arrow}</div>"
            f"</div>"
            f"<div style='opacity:0.7;font-size:0.85em;margin-top:6px;'>"
            f"méd <b>{ws.mean:.1f}</b> · under {ws.hits_under}/{ws.n} "
            f"({ws.under_pct:.0f}%)</div>"
        )
    return (
        f"<div style='background:{bg};border:1px solid {border};border-radius:10px;"
        f"padding:12px 14px;margin-bottom:6px;'>"
        f"<div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;"
        f"letter-spacing:0.06em;margin-bottom:4px;'>{escape(ws.label)} · {escape(team_name)}</div>"
        f"{body}"
        f"</div>"
    )


def render_historic_block(
    *, st: Any,
    team_a: str, team_b: str,
    stats_a: List[WindowStats], stats_b: List[WindowStats],
    h2h_a: WindowStats,
    line: float, line_label: str,
    perspective: str = "ambos",  # "a", "b", "ambos"
) -> None:
    st.markdown("### Análise histórica")
    st.caption(f"Hit rate de **Over {line_label}** por janela. Setas mostram tendência ao longo dos jogos.")

    if perspective == "a":
        cols = st.columns(len(stats_a))
        for col, ws in zip(cols, stats_a):
            with col:
                st.markdown(_historic_card_html(team_a, ws, line, line_label), unsafe_allow_html=True)
    elif perspective == "b":
        cols = st.columns(len(stats_b))
        for col, ws in zip(cols, stats_b):
            with col:
                st.markdown(_historic_card_html(team_b, ws, line, line_label), unsafe_allow_html=True)
    else:
        # Ambos: side-by-side rows
        st.markdown(f"**{escape(team_a)}**")
        cols_a = st.columns(len(stats_a))
        for col, ws in zip(cols_a, stats_a):
            with col:
                st.markdown(_historic_card_html(team_a, ws, line, line_label), unsafe_allow_html=True)
        st.markdown(f"**{escape(team_b)}**")
        cols_b = st.columns(len(stats_b))
        for col, ws in zip(cols_b, stats_b):
            with col:
                st.markdown(_historic_card_html(team_b, ws, line, line_label), unsafe_allow_html=True)

    # H2H
    if h2h_a.n > 0:
        tone = _tone_for_pct(h2h_a.over_pct)
        bg = _TONE_BG[tone]; border = _TONE_BORDER[tone]; color = _TONE_TEXT[tone]
        st.markdown(
            f"<div style='background:{bg};border:1px solid {border};border-radius:10px;"
            f"padding:12px 14px;margin-top:10px;'>"
            f"<div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;"
            f"letter-spacing:0.06em;margin-bottom:4px;'>H2H · {escape(team_a)} vs {escape(team_b)}</div>"
            f"<div style='display:flex;align-items:baseline;gap:12px;'>"
            f"<div style='font-size:1.8em;font-weight:700;color:{color};line-height:1;'>"
            f"{h2h_a.hits_over}/{h2h_a.n}</div>"
            f"<div style='font-size:1.2em;font-weight:600;color:{color};'>{h2h_a.over_pct:.0f}%</div>"
            f"</div>"
            f"<div style='opacity:0.7;font-size:0.85em;margin-top:6px;'>"
            f"méd <b>{h2h_a.mean:.1f}</b> · under {h2h_a.hits_under}/{h2h_a.n} "
            f"({h2h_a.under_pct:.0f}%)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_map_block(
    *, st: Any,
    team_name: str,
    map_stats: List[MapStats],
    line: float, line_label: str,
    is_time_metric: bool = False,
) -> None:
    """Per-map card with OVER + UNDER balanced (no direction bias)."""
    if not map_stats:
        return
    st.markdown(f"### Por mapa · {escape(team_name)}")
    cols = st.columns(len(map_stats))
    for col, ms in zip(cols, map_stats):
        with col:
            over_color = _TONE_TEXT[_tone_for_pct(ms.over_pct)]
            under_color = _TONE_TEXT[_tone_for_pct(ms.under_pct)]
            arrow = _TREND_ARROW.get(ms.trend, "—")
            arrow_color = ("#22c55e" if ms.trend == "up" else
                           "#ef4444" if ms.trend == "down" else "#9ca3af")
            # Slope per game (round to 1 decimal). Skip if not finite.
            slope_html = ""
            if isinstance(ms.trend_value, (int, float)) and math.isfinite(float(ms.trend_value)) and ms.trend != "na":
                sign = "+" if ms.trend_value >= 0 else "−"
                slope_unit = "s/jogo" if is_time_metric else "/jogo"
                slope_val = abs(ms.trend_value) * 60.0 if is_time_metric else abs(ms.trend_value)
                slope_html = (f" <span style='opacity:0.6;font-size:0.78em;'>"
                              f"{sign}{slope_val:.0f}{slope_unit}</span>")
            note_html = (f"<span style='opacity:0.6;'>{escape(ms.note)}</span>"
                         if ms.note else "")
            tendencia_html = (f"<span style='color:{arrow_color};'>tend {arrow}</span>{slope_html}")
            sd_text = (f"{round(ms.sd * 60.0)}s" if is_time_metric else f"{ms.sd:.1f}")
            sd_html = (f" <span style='opacity:0.6;'>σ ±{sd_text}</span>"
                       if ms.sd > 0 else "")
            mean_text = _fmt_metric_v3(ms.mean, is_time=is_time_metric)

            st.markdown(
                f"<div style='background:rgba(45,170,196,0.05);"
                f"border:1px solid rgba(45,170,196,0.3);border-radius:10px;"
                f"padding:10px 12px;'>"
                # Header: M1/M2/etc + nº jogos
                f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
                f"margin-bottom:6px;'>"
                f"<span style='opacity:0.75;font-size:0.85em;font-weight:600;'>M{ms.map_n}</span>"
                f"<span style='opacity:0.55;font-size:0.78em;'>{ms.games} jogos</span>"
                f"</div>"
                # Over + Under lado a lado
                f"<div style='display:flex;justify-content:space-around;gap:6px;"
                f"padding:6px 0;border-top:1px solid rgba(255,255,255,0.06);"
                f"border-bottom:1px solid rgba(255,255,255,0.06);'>"
                f"  <div style='flex:1;text-align:center;'>"
                f"    <div style='opacity:0.55;font-size:0.72em;'>OVER</div>"
                f"    <div style='font-size:1.15em;font-weight:700;color:{over_color};'>"
                f"      {ms.over_pct:.0f}%</div>"
                f"    <div style='opacity:0.55;font-size:0.72em;'>{ms.hits_over}/{ms.games}</div>"
                f"  </div>"
                f"  <div style='width:1px;background:rgba(255,255,255,0.08);'></div>"
                f"  <div style='flex:1;text-align:center;'>"
                f"    <div style='opacity:0.55;font-size:0.72em;'>UNDER</div>"
                f"    <div style='font-size:1.15em;font-weight:700;color:{under_color};'>"
                f"      {ms.under_pct:.0f}%</div>"
                f"    <div style='opacity:0.55;font-size:0.72em;'>{ms.hits_under}/{ms.games}</div>"
                f"  </div>"
                f"</div>"
                # Footer: méd / sd / tendência / desvio
                f"<div style='font-size:0.78em;padding-top:6px;'>"
                f"  <div>méd <b>{mean_text}</b>{sd_html}</div>"
                f"  <div style='margin-top:2px;'>{tendencia_html}"
                + (f" · {note_html}" if note_html else "")
                + "</div>"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def render_ml_block(
    *, st: Any,
    p_over: Optional[float], p_under: Optional[float],
    mean_pred: Optional[float], sigma: Optional[float],
    market_label: str, line: float, line_label: str,
) -> None:
    st.markdown("### Predição do modelo (complemento)")
    if mean_pred is None or p_over is None:
        st.caption("_Predição ML indisponível neste recorte._")
        return
    p_o = float(p_over); p_u = float(p_under) if p_under is not None else (1.0 - p_o)
    tone_o = _tone_for_pct(p_o * 100.0)
    color_o = _TONE_TEXT[tone_o]
    cols = st.columns([1.2, 1.2, 1.6])
    with cols[0]:
        st.markdown(
            f"<div style='padding:8px;'><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;'>"
            f"Mean previsto</div>"
            f"<div style='font-size:1.6em;font-weight:700;margin-top:4px;'>{mean_pred:.1f}</div>"
            f"<div style='opacity:0.65;font-size:0.82em;'>σ {sigma:.1f}</div></div>"
            if sigma is not None and math.isfinite(float(sigma)) else
            f"<div style='padding:8px;'><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;'>"
            f"Mean previsto</div><div style='font-size:1.6em;font-weight:700;margin-top:4px;'>"
            f"{mean_pred:.1f}</div></div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"<div style='padding:8px;'><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;'>"
            f"P(Over {line_label})</div>"
            f"<div style='font-size:1.6em;font-weight:700;color:{color_o};margin-top:4px;'>"
            f"{p_o*100:.1f}%</div>"
            f"<div style='opacity:0.7;font-size:0.85em;'>P(Under): {p_u*100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            "<div style='padding:8px;opacity:0.8;font-size:0.88em;'>"
            "Modelo calibrado (Fase 7). Comparar com o histórico — quando "
            "concordam é sinal mais forte.</div>",
            unsafe_allow_html=True,
        )


_ARCH_LABEL = {
    "very_over": "muito over",
    "over": "over",
    "neutral": "neutro",
    "under": "under",
    "very_under": "muito under",
}
_ARCH_COLOR = {
    "very_over": "#22c55e",
    "over": "#86efac",
    "neutral": "#9ca3af",
    "under": "#fca5a5",
    "very_under": "#ef4444",
}


def render_contextual_opponent_profile(
    *,
    st: Any,
    team_a: str, team_b: str,
    archetype_a: Any,  # TeamArchetype
    archetype_b: Any,  # TeamArchetype
    profile_a: Any,    # ContextualProfileRow — team_a vs opposite-of-A opponents
    profile_b: Any,    # ContextualProfileRow — team_b vs opposite-of-B opponents
    line: float, line_label: str,
    is_time_metric: bool = False,
) -> None:
    """Render the smart contextual opponent profile.

    Shows each team's archetype (over/under/neutral) + how they perform vs
    OPPOSITE-archetype opponents — the most informative cross-comparison."""
    def _archetype_chip(arch: Any, team_name: str) -> str:
        a_label = _ARCH_LABEL.get(arch.archetype, "—")
        a_color = _ARCH_COLOR.get(arch.archetype, "#9ca3af")
        diff_html = ""
        if math.isfinite(arch.diff_pct):
            sign = "+" if arch.diff_pct >= 0 else "−"
            diff_html = (f" <span style='opacity:0.65;font-size:0.78em;'>"
                         f"({sign}{abs(arch.diff_pct)*100:.0f}% vs liga)</span>")
        mean_html = (f" <span style='opacity:0.7;font-size:0.82em;'>"
                     f"méd <b>{_fmt_metric_v3(arch.mean, is_time=is_time_metric)}</b> · "
                     f"liga {_fmt_metric_v3(arch.league_mean, is_time=is_time_metric)}</span>"
                     if math.isfinite(arch.mean) and math.isfinite(arch.league_mean) else "")
        return (
            f"<div style='padding:6px 0;'>"
            f"<span style='font-weight:700;color:#2daac4;'>{escape(team_name)}</span>"
            f"<span style='margin-left:8px;display:inline-block;padding:2px 10px;"
            f"border-radius:10px;background:rgba(45,170,196,0.15);color:{a_color};"
            f"font-weight:600;font-size:0.85em;letter-spacing:0.02em;'>{escape(a_label)}</span>"
            f"{diff_html}{mean_html}"
            f"</div>"
        )

    def _profile_card(focal_team: str, focal_arch: Any, prof: Any) -> str:
        """Build the right-side profile card (focal team vs opponent group)."""
        # Label must mirror the ACTUAL opponent group selected (prof.opponent_archetype,
        # e.g. "over/very_over", "neutral"), not the team's archetype-vs-league.
        # Otherwise "Neutro" (which contains neither "over" nor "under") would
        # wrongly fall through to the "de extremo" label.
        arch_str = str(getattr(prof, "opponent_archetype", "") or "").lower()
        tokens = {t for t in arch_str.split("/") if t}
        has_over = bool(tokens & {"over", "very_over"})
        has_under = bool(tokens & {"under", "very_under"})
        if has_over and has_under:
            opposite_label = "times de extremo (over/under)"
        elif tokens == {"very_over"}:
            opposite_label = "times muito over"
        elif has_over:
            opposite_label = "times over"
        elif tokens == {"very_under"}:
            opposite_label = "times muito under"
        elif has_under:
            opposite_label = "times under"
        elif "neutral" in tokens:
            opposite_label = "times neutros"
        else:
            opposite_label = "este grupo de adversários"

        if int(getattr(prof, "n_opponents", 0)) == 0:
            # No team of this archetype exists in the league at all — alert.
            body = (f"<div style='background:rgba(250,204,21,0.12);border:1px solid "
                    f"rgba(250,204,21,0.45);border-radius:8px;padding:10px 12px;"
                    f"color:#facc15;font-size:0.9em;'>"
                    f"⚠️ Não há times <b>{escape(opposite_label)}</b> nesta liga/recorte. "
                    f"Selecione outro grupo de adversário acima.</div>")
        elif prof.games_played == 0:
            body = (f"<div style='opacity:0.55;padding:10px 0;'>"
                    f"{focal_team} não enfrentou nenhum dos {prof.n_opponents} time(s) "
                    f"{escape(opposite_label)} neste recorte "
                    f"({', '.join(prof.opponents[:5])}"
                    + ("..." if len(prof.opponents) > 5 else "") + ").</div>")
        else:
            over_color = _TONE_TEXT[_tone_for_pct(prof.over_pct)]
            under_color = _TONE_TEXT[_tone_for_pct(prof.under_pct)]
            sd_text = (f"{round(prof.sd * 60.0)}s" if is_time_metric else f"{prof.sd:.1f}")
            sd_html = (f" <span style='opacity:0.6;'>σ ±{sd_text}</span>"
                       if prof.sd > 0 else "")
            prof_mean_text = _fmt_metric_v3(prof.mean, is_time=is_time_metric)
            opps_html = (f"<div style='opacity:0.55;font-size:0.78em;margin-top:4px;'>"
                         f"Contra: {escape(', '.join(prof.opponents[:6]))}"
                         + ("..." if len(prof.opponents) > 6 else "")
                         + "</div>")
            body = (
                f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
                f"margin:6px 0;'>"
                f"<span style='opacity:0.7;font-size:0.85em;'>{prof.games_played} jogos</span>"
                f"<span>méd <b>{prof_mean_text}</b>{sd_html}</span>"
                f"</div>"
                f"<div style='display:flex;justify-content:space-around;gap:8px;"
                f"padding:8px 0;border-top:1px solid rgba(255,255,255,0.06);"
                f"border-bottom:1px solid rgba(255,255,255,0.06);'>"
                f"  <div style='flex:1;text-align:center;'>"
                f"    <div style='opacity:0.55;font-size:0.72em;'>OVER</div>"
                f"    <div style='font-size:1.3em;font-weight:700;color:{over_color};'>"
                f"      {prof.over_pct:.0f}%</div>"
                f"    <div style='opacity:0.55;font-size:0.72em;'>{prof.hits_over}/{prof.games_played}</div>"
                f"  </div>"
                f"  <div style='width:1px;background:rgba(255,255,255,0.08);'></div>"
                f"  <div style='flex:1;text-align:center;'>"
                f"    <div style='opacity:0.55;font-size:0.72em;'>UNDER</div>"
                f"    <div style='font-size:1.3em;font-weight:700;color:{under_color};'>"
                f"      {prof.under_pct:.0f}%</div>"
                f"    <div style='opacity:0.55;font-size:0.72em;'>{prof.hits_under}/{prof.games_played}</div>"
                f"  </div>"
                f"</div>"
                f"{opps_html}"
            )

        return (
            f"<div style='background:rgba(45,170,196,0.05);"
            f"border:1px solid rgba(45,170,196,0.3);border-radius:10px;"
            f"padding:12px 14px;'>"
            f"<div style='font-weight:600;color:#2daac4;font-size:1.05em;margin-bottom:6px;'>"
            f"{escape(focal_team)} vs {escape(opposite_label)}"
            f"</div>"
            f"{body}"
            f"</div>"
        )

    # Archetype chips (header)
    st.markdown(_archetype_chip(archetype_a, team_a), unsafe_allow_html=True)
    st.markdown(_archetype_chip(archetype_b, team_b), unsafe_allow_html=True)
    # Side-by-side profile cards
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown(_profile_card(team_a, archetype_a, profile_a),
                     unsafe_allow_html=True)
    with c_b:
        st.markdown(_profile_card(team_b, archetype_b, profile_b),
                     unsafe_allow_html=True)


def render_v3_stats_table(
    *,
    st: Any,
    team_a: str, team_b: str,
    rows: List[Any],  # List[StatsRow]
    scope_label: str = "",
    cross_league: bool = False,
    league_label_a: str = "Liga",
    league_label_b: str = "Liga",
) -> None:
    """Render the stats comparative table (v3 style) — Team A | Team B | Liga.

    Uses inline HTML so we can color individual cells without fighting pandas
    styler quirks. Layout is a grid: label / A / B / Liga. In cross-league
    matchups it becomes label / A / B / Liga A / Liga B, so each team is shown
    against its OWN league reference instead of a single (mixed) one.

    Percentage rows are colored with the shared 4-band scale (0-40 red,
    41-60 yellow, 61-80 green, 81-100 elite/cyan); other rows stay neutral.
    """
    if not rows:
        st.markdown("<div style='opacity:0.55;'>— sem dados de stats —</div>",
                      unsafe_allow_html=True)
        return

    # Two Liga columns only when cross-league AND a row actually carries a
    # second reference (defensive: avoids an empty column if flag is stale).
    show_two_leagues = bool(cross_league) and any(
        str(getattr(r, "league_value_b", "") or "") for r in rows
    )
    grid_cols = ("1.7fr 1fr 1fr 1fr 1fr" if show_two_leagues
                 else "1.7fr 1fr 1fr 1fr")

    def _pct_color(value: str) -> str:
        """Color a percentage display string ('75%') via the 4-band scale."""
        try:
            num = float(str(value).strip().rstrip("%"))
        except (TypeError, ValueError):
            return _TONE_HTML_COLOR["neutral"]
        return _TONE_TEXT[_tone_for_pct(num)]

    def _cell(value: str, color: str, *, bold: bool = False, rank: str = "") -> str:
        weight = "700" if bold else "500"
        # Rank badge (↓N most-under / ↑N most-over in the league) next to value.
        rank_html = ""
        if rank:
            rank_html = (f" <span style='font-size:0.78em;opacity:0.7;font-weight:600;"
                         f"color:#2daac4;'>{escape(rank)}</span>")
        return (f"<div style='text-align:center;color:{color};font-weight:{weight};"
                f"font-size:0.98em;padding:6px 8px;'>"
                f"{escape(value)}{rank_html}</div>")

    _neutral = _TONE_HTML_COLOR["neutral"]

    # Table header
    liga_headers = (
        f"<div style='text-align:center;opacity:0.7;'>{escape(league_label_a)}</div>"
        f"<div style='text-align:center;opacity:0.7;'>{escape(league_label_b)}</div>"
        if show_two_leagues else
        "<div style='text-align:center;opacity:0.7;'>Liga</div>"
    )
    header_html = (
        f"<div style='display:grid;grid-template-columns:{grid_cols};"
        "background:rgba(45,170,196,0.12);border-radius:8px 8px 0 0;"
        "padding:6px 8px;font-weight:700;opacity:0.85;font-size:0.78em;"
        "text-transform:uppercase;letter-spacing:0.05em;'>"
        f"<div>Métrica</div>"
        f"<div style='text-align:center;color:#2daac4;'>{escape(team_a)}</div>"
        f"<div style='text-align:center;color:#2daac4;'>{escape(team_b)}</div>"
        f"{liga_headers}"
        "</div>"
    )
    # Body
    body_parts: List[str] = []
    for i, r in enumerate(rows):
        bg = "rgba(255,255,255,0.02)" if (i % 2 == 0) else "rgba(255,255,255,0.04)"
        # Color only percentage rows; absolute-value rows (Game Time, Kills
        # feitas/sofridas, Combined) stay neutral per user preference.
        _is_pct = bool(getattr(r, "is_pct", False))
        color_a = _pct_color(r.team_a_value) if _is_pct else _neutral
        color_b = _pct_color(r.team_b_value) if _is_pct else _neutral
        liga_cells = _cell(r.league_value, _neutral)
        if show_two_leagues:
            liga_cells += _cell(getattr(r, "league_value_b", "") or "—", _neutral)
        body_parts.append(
            f"<div style='display:grid;grid-template-columns:{grid_cols};"
            f"background:{bg};border-bottom:1px solid rgba(255,255,255,0.04);'>"
            f"<div style='padding:8px 10px;opacity:0.85;'>{escape(r.label)}</div>"
            f"{_cell(r.team_a_value, color_a, bold=True, rank=getattr(r, 'rank_a', ''))}"
            f"{_cell(r.team_b_value, color_b, bold=True, rank=getattr(r, 'rank_b', ''))}"
            f"{liga_cells}"
            "</div>"
        )
    body_html = "".join(body_parts)
    if scope_label:
        st.caption(f"Recorte: {scope_label}")
    st.markdown(
        f"<div style='border:1px solid rgba(45,170,196,0.25);border-radius:10px;"
        f"overflow:hidden;'>{header_html}{body_html}</div>",
        unsafe_allow_html=True,
    )


def render_opponent_profile_table(
    *, st: Any,
    team_a: str, team_b: str,
    profile: List[OpponentProfileRow],
    line: float, line_label: str,
) -> None:
    if not profile:
        return
    st.markdown("### Perfil de adversário (toda a liga)")
    st.caption("Como cada time da liga performa neste mercado. Use pra contextualizar quem vai puxar over/under.")
    rows = []
    for p in profile:
        rows.append({
            "Time": p.team,
            "Jogos (liga)": p.games_vs_league,
            "Méd": f"{p.mean_vs_league:.1f}",
            f"Over {line_label}": f"{p.over_pct_vs_league:.0f}%",
            "Perfil": _STRENGTH_LABEL.get(p.strength_tag, p.strength_tag),
            f"H2H vs {team_a}": (f"{p.mean_in_matchup:.1f} ({p.games_in_matchup}j)"
                                 if p.games_in_matchup > 0 else "—"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render_recent_games_table(
    *, st: Any,
    team_name: str,
    recent: List[RecentGameRow],
    line: float, line_label: str,
) -> None:
    if not recent:
        return
    st.markdown(f"### Recentes · {escape(team_name)}")
    rows = []
    for g in recent:
        sign = "+" if g.delta_from_line >= 0 else "−"
        rows.append({
            "Data": g.date,
            "Vs": g.opponent,
            "M": g.game_map if g.game_map > 0 else "—",
            "Valor": f"{g.value:.0f}",
            "Δ linha": f"{sign}{abs(g.delta_from_line):.1f}",
            "Hit": "✓" if g.hit else "✗",
            "Contexto adv.": _STRENGTH_LABEL.get(g.opponent_strength, g.opponent_strength),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render_alerts_block(*, st: Any, alerts: List[Dict[str, str]]) -> None:
    if not alerts:
        return
    st.markdown("### Alertas")
    tone_icon = {"good": "✓", "warn": "⚠", "info": "ℹ", "alert": "🔔"}
    tone_bg = {
        "good":  "rgba(34,197,94,0.12)",
        "warn":  "rgba(250,204,21,0.15)",
        "info":  "rgba(96,165,250,0.12)",
        "alert": "rgba(239,68,68,0.13)",
    }
    tone_border = {
        "good":  "rgba(34,197,94,0.45)",
        "warn":  "rgba(250,204,21,0.45)",
        "info":  "rgba(96,165,250,0.45)",
        "alert": "rgba(239,68,68,0.45)",
    }
    for a in alerts[:8]:
        tone = a.get("tone", "info")
        icon = tone_icon.get(tone, "ℹ")
        bg = tone_bg.get(tone, tone_bg["info"])
        border = tone_border.get(tone, tone_border["info"])
        st.markdown(
            f"<div style='background:{bg};border-left:3px solid {border};"
            f"padding:8px 12px;border-radius:6px;margin-top:4px;font-size:0.95em;'>"
            f"{icon} {escape(str(a.get('text') or ''))}</div>",
            unsafe_allow_html=True,
        )


__all__ = [
    "render_alerts_block",
    "render_historic_block",
    "render_map_block",
    "render_ml_block",
    "render_ml_strip",
    "render_opponent_profile_table",
    "render_recent_games_table",
    "render_team_header",
]
