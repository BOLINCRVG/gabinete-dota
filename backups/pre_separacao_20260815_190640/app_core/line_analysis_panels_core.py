# -*- coding: utf-8 -*-
"""Helpers extraídos do plays_app (fatiamento F3, 2026-07-18).

Funções PURAS (não referenciam nada definido no plays_app; sem __file__).
O plays_app importa todas de volta, então o namespace que alimenta o
ctx das abas (globals()->build_render_ctx) segue idêntico.
"""
from __future__ import annotations

from html import escape
from mlcore.bo import prob_win_series
from typing import Any


def _render_ml_chip_vencedor(
    *, st: Any, team_a: str, team_b: str,
    p_map: float | None, p_series: float | None = None,
) -> None:
    """Compact ML probability chip for Mapa/MD3/MD5.

    MD3 e MD5 são derivados de p_map via prob_win_series(p_map, 3) e (p_map, 5).
    Antes ambos usavam o MESMO p_series (a prob do bo SELECIONADO), então MD3 e
    MD5 saíam idênticos — bug. p_series é aceito por compat mas não é usado.
    """
    def _series(bo: int) -> float | None:
        if not (isinstance(p_map, (int, float)) and 0 < float(p_map) < 1):
            return None
        try:
            v = float(prob_win_series(float(p_map), bo))
            return v if 0.0 < v < 1.0 else None
        except Exception:
            return None

    def _odds(p: float) -> float:
        return (1.0 / p) if (p and 0 < p < 1) else float("nan")

    def _cell(label: str, p: float | None) -> str:
        if p is None or not (isinstance(p, (int, float)) and 0 < p < 1):
            return (f"<div style='flex:1;text-align:center;padding:8px;'>"
                    f"<div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;'>{label}</div>"
                    f"<div style='opacity:0.5;'>—</div></div>")
        p_a = float(p); p_b = 1.0 - p_a
        odd_a = _odds(p_a); odd_b = _odds(p_b)
        c_a = "#22c55e" if p_a >= 0.55 else "#facc15" if p_a >= 0.45 else "#ef4444"
        return (
            f"<div style='flex:1;text-align:center;padding:8px;'>"
            f"<div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em;'>{label}</div>"
            f"<div style='font-size:1.0em;margin-top:4px;'>"
            f"  <span style='color:#2daac4;'>{team_a}</span> "
            f"  <b style='color:{c_a};'>{odd_a:.2f}</b> "
            f"  <span style='opacity:0.6;'>({p_a*100:.1f}%)</span>"
            f"</div>"
            f"<div style='font-size:1.0em;margin-top:2px;'>"
            f"  <span style='color:#2daac4;'>{team_b}</span> "
            f"  <b>{odd_b:.2f}</b> "
            f"  <span style='opacity:0.6;'>({p_b*100:.1f}%)</span>"
            f"</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;background:rgba(45,170,196,0.08);"
        f"border:1px solid rgba(45,170,196,0.3);border-radius:10px;margin:10px 0;'>"
        f"{_cell('Mapa', p_map)}"
        f"<div style='width:1px;background:rgba(255,255,255,0.08);'></div>"
        f"{_cell('MD3', _series(3))}"
        f"<div style='width:1px;background:rgba(255,255,255,0.08);'></div>"
        f"{_cell('MD5', _series(5))}"
        f"</div>",
        unsafe_allow_html=True,
    )

def _render_balanced_line_analysis(
    *,
    st: Any,
    team_a: str, team_b: str,
    stats_a: list, stats_b: list, h2h: Any,
    line: float,
) -> None:
    """Render hit-rate per window WITHOUT direction bias: Over and Under side-by-side."""
    # Use the shared 4-band % scale (0-40 red, 41-60 yellow, 61-80 green,
    # 81-100 elite/cyan) so these cards match the rest of the app.
    from app_core.ui_design_core import TONE_TEXT, tone_for_pct

    def _cell_html(label: str, n: int, hits_over: int, hits_under: int,
                    over_pct: float, under_pct: float, mean: float) -> str:
        if n == 0:
            body = "<div style='opacity:0.45;padding:6px 0;'>— sem jogos —</div>"
        else:
            # Color both sides equally, no preference
            over_color = TONE_TEXT[tone_for_pct(over_pct)]
            under_color = TONE_TEXT[tone_for_pct(under_pct)]
            body = (
                f"<div style='display:flex;justify-content:space-around;gap:10px;padding:6px 0;'>"
                f"  <div style='text-align:center;flex:1;'>"
                f"    <div style='opacity:0.6;font-size:0.74em;'>OVER</div>"
                f"    <div style='font-size:1.35em;font-weight:700;color:{over_color};'>"
                f"      {hits_over}/{n}</div>"
                f"    <div style='font-weight:600;color:{over_color};'>{over_pct:.0f}%</div>"
                f"  </div>"
                f"  <div style='width:1px;background:rgba(255,255,255,0.08);'></div>"
                f"  <div style='text-align:center;flex:1;'>"
                f"    <div style='opacity:0.6;font-size:0.74em;'>UNDER</div>"
                f"    <div style='font-size:1.35em;font-weight:700;color:{under_color};'>"
                f"      {hits_under}/{n}</div>"
                f"    <div style='font-weight:600;color:{under_color};'>{under_pct:.0f}%</div>"
                f"  </div>"
                f"</div>"
                f"<div style='text-align:center;opacity:0.6;font-size:0.8em;padding-top:4px;"
                f"border-top:1px solid rgba(255,255,255,0.06);'>"
                f"méd <b>{mean:.1f}</b></div>"
            )
        return (
            f"<div style='border:1px solid rgba(45,170,196,0.25);border-radius:8px;"
            f"padding:8px 10px;margin-bottom:6px;background:rgba(45,170,196,0.04);'>"
            f"<div style='opacity:0.7;font-size:0.78em;text-transform:uppercase;"
            f"letter-spacing:0.06em;text-align:center;margin-bottom:4px;'>{escape(label)}</div>"
            f"{body}"
            f"</div>"
        )

    # Team A row
    st.markdown(
        f"<div style='font-weight:600;color:#2daac4;margin-top:8px;font-size:1.05em;'>"
        f"{escape(team_a)}</div>",
        unsafe_allow_html=True,
    )
    cols_a = st.columns(len(stats_a))
    for col, ws in zip(cols_a, stats_a):
        with col:
            st.markdown(
                _cell_html(ws.label, ws.n, ws.hits_over, ws.hits_under,
                            ws.over_pct, ws.under_pct, ws.mean),
                unsafe_allow_html=True,
            )

    # Team B row
    st.markdown(
        f"<div style='font-weight:600;color:#2daac4;margin-top:14px;font-size:1.05em;'>"
        f"{escape(team_b)}</div>",
        unsafe_allow_html=True,
    )
    cols_b = st.columns(len(stats_b))
    for col, ws in zip(cols_b, stats_b):
        with col:
            st.markdown(
                _cell_html(ws.label, ws.n, ws.hits_over, ws.hits_under,
                            ws.over_pct, ws.under_pct, ws.mean),
                unsafe_allow_html=True,
            )

    # H2H — single centered card
    if h2h is not None and getattr(h2h, "n", 0) > 0:
        st.markdown(
            f"<div style='font-weight:600;color:#2daac4;margin-top:14px;font-size:1.05em;'>"
            f"H2H · {escape(team_a)} vs {escape(team_b)}</div>",
            unsafe_allow_html=True,
        )
        _c_left, c_mid, _c_right = st.columns([1, 2, 1])
        with c_mid:
            st.markdown(
                _cell_html("H2H", h2h.n, h2h.hits_over, h2h.hits_under,
                            h2h.over_pct, h2h.under_pct, h2h.mean),
                unsafe_allow_html=True,
            )

