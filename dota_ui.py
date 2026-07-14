# -*- coding: utf-8 -*-
"""Identidade visual do Gabinete Dota = a MESMA do Gabinete LoL, com o acento
trocado de ciano (#2daac4) para vermelho Dota (#c94a38). Reusa os primitivos
puros do app do LoL (tons, badge, escala de %) e re-tema so o que e cor.
"""
from __future__ import annotations

from typing import Any

from dota_paths import ensure_lol_on_path

LOL_REPO = ensure_lol_on_path()

# primitivos identicos (puros, sem cor de acento embutida)
from app_core.ui_design_core import (  # noqa: F401,E402
    TONE_BG, TONE_BORDER, TONE_TEXT, TONE_HTML_COLOR,
    tone_for_pct, team_badge_html,
)

RED = "#c94a38"        # acento Dota (Radiant vermelho classico)
RED_DARK = "#9c3527"   # gradiente inferior


def inject_global_ui_css(st: Any) -> None:
    """CSS global identico ao do LoL (global_css_core), acento vermelho."""
    st.markdown(
        f"""
<style>
  :root {{
    --gb-accent: {RED};
    --gb-panel: rgba(255,255,255,0.035);
    --gb-panel-border: rgba(255,255,255,0.12);
  }}
  .main .block-container {{ padding-top: 1.2rem; padding-bottom: 1.2rem; }}
  .gb-hero {{
    border: 1px solid var(--gb-panel-border);
    background: linear-gradient(135deg, rgba(201,74,56,0.12), rgba(255,255,255,0.02));
    border-radius: 14px; padding: 12px 14px; margin-bottom: 10px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  }}
  .gb-hero .title {{ font-size: 1.55rem; font-weight: 800; letter-spacing: 0.01em; line-height: 1.15; }}
  .gb-hero .sub {{ opacity: 0.80; margin-top: 4px; font-size: 0.92rem; }}
  .gb-eyebrow {{
    display: inline-block; margin-bottom: 6px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase; color: rgba(255,255,255,0.66);
  }}
  .gb-chip-row {{ display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 10px; }}
  .gb-chip {{
    display: inline-flex; align-items: center; min-height: 26px; padding: 0.2rem 0.6rem;
    border-radius: 999px; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10); font-size: 0.78rem; line-height: 1;
    color: rgba(255,255,255,0.88);
  }}
  .gb-section-head {{ margin: 0.4rem 0 0.75rem 0; }}
  .gb-section-title {{ font-size: 1rem; font-weight: 760; line-height: 1.15; letter-spacing: 0.01em; }}
  .gb-section-sub {{ margin-top: 4px; font-size: 0.88rem; color: rgba(255,255,255,0.72); }}
  div[data-testid="stTabs"] button[role="tab"] {{
    border-radius: 10px; padding: 0.35rem 0.75rem; margin-right: 0.25rem;
    transition: background 120ms ease, border-color 120ms ease;
  }}
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background: rgba(201,74,56,0.16);
    border-bottom-color: var(--gb-accent) !important;
    box-shadow: inset 0 0 0 1px rgba(201,74,56,0.25);
  }}
  .stButton > button {{ border-radius: 10px; }}
  div[data-testid="stExpander"] {{
    background: var(--gb-panel); border: 1px solid var(--gb-panel-border); border-radius: 10px;
  }}
  div[data-testid="stMetric"] {{
    border: 1px solid var(--gb-panel-border); border-radius: 10px;
    padding: 8px 10px; background: rgba(255,255,255,0.02);
  }}
</style>
        """,
        unsafe_allow_html=True,
    )


def red_banner(title: str) -> str:
    """Mesmo banner-header de card do LoL (cyan_banner), em vermelho."""
    return (f"<div style='background:linear-gradient(180deg, {RED} 0%, {RED_DARK} 100%);"
            f"padding:8px 12px;border-radius:8px 8px 0 0;text-align:center;"
            f"font-weight:700;letter-spacing:0.05em;font-size:0.92em;color:white;"
            f"box-shadow:inset 0 1px 0 rgba(255,255,255,0.18);'>{title}</div>")


def red_section_header(st: Any, title: str) -> None:
    """Mesmo section header full-width do LoL (cyan_section_header), vermelho."""
    st.markdown(
        f"<div style='margin-top:18px;background:linear-gradient(180deg, {RED} 0%, {RED_DARK} 100%);"
        f"padding:8px 14px;border-radius:8px;font-weight:700;letter-spacing:0.05em;"
        f"font-size:0.92em;color:white;text-transform:uppercase;'>"
        f"{title}</div>",
        unsafe_allow_html=True,
    )


def pct_cell_html(pct: float, n: int) -> str:
    """Celula de % com tom (mesma escala 4 faixas do LoL)."""
    import math
    if not (isinstance(pct, (int, float)) and math.isfinite(float(pct))):
        return "<span style='color:#9ca3af'>—</span>"
    tone = tone_for_pct(pct)
    return (f"<span style='display:inline-block;min-width:64px;text-align:center;"
            f"padding:2px 8px;border-radius:6px;background:{TONE_BG[tone]};"
            f"border:1px solid {TONE_BORDER[tone]};color:{TONE_TEXT[tone]};"
            f"font-weight:700;'>{pct:.0f}%</span>"
            f"<span style='opacity:0.6;font-size:0.85em;'> ({n})</span>")
