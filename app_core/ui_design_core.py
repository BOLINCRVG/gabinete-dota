"""Shared visual design primitives for the Streamlit app.

Single source of truth for the "cyan" look established in the Resumo and
Análise de Linha tabs: gradient banners, section headers, the team badge,
the tone color palette and the 4-band percentage color scale.

This module is pure presentation — it must NOT import other app_core modules,
so any renderer can depend on it without creating import cycles. The HTML
strings here are byte-for-byte the ones that previously lived inline in the
tab renderers; moving them here changes nothing on screen, it just lets every
tab reuse the exact same look.
"""
from __future__ import annotations

import math
import re
from html import escape
from typing import Any

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------

CYAN = "#2daac4"        # primary accent
CYAN_DARK = "#1f8ba1"   # gradient bottom stop

# Deterministic palette for team-initials badges.
BADGE_COLORS = (
    "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
    "#3b82f6", "#6366f1", "#a855f7", "#ec4899", "#0ea5e9",
)

# Tone palette used by stat cards / percentage cells.
TONE_BG = {
    "elite":  "rgba(34,211,238,0.15)",
    "green":  "rgba(34,197,94,0.13)",
    "yellow": "rgba(250,204,21,0.16)",
    "red":    "rgba(239,68,68,0.13)",
    "neutral": "rgba(156,163,175,0.10)",
}
TONE_BORDER = {
    "elite":  "rgba(34,211,238,0.50)",
    "green":  "rgba(34,197,94,0.45)",
    "yellow": "rgba(250,204,21,0.45)",
    "red":    "rgba(239,68,68,0.45)",
    "neutral": "rgba(156,163,175,0.35)",
}
TONE_TEXT = {
    "elite":  "#22d3ee",
    "green":  "#22c55e",
    "yellow": "#facc15",
    "red":    "#ef4444",
    "neutral": "#9ca3af",
}

# Tones used by the comparative stats table cells (good/warn/bad semantics).
TONE_HTML_COLOR = {
    "good":    "#22c55e",
    "warn":    "#facc15",
    "bad":     "#ef4444",
    "neutral": "rgba(255,255,255,0.85)",
}


# ---------------------------------------------------------------------------
# Percentage color scale (4 bands)
# ---------------------------------------------------------------------------

def tone_for_pct(pct: float, *, direction: str = "Over") -> str:
    """Color bands (set 2026-06-02), aligned to the displayed integer %:
    0-40 red, 41-60 yellow, 61-80 green, 81-100 elite (cyan).
    """
    if not (isinstance(pct, (int, float)) and math.isfinite(float(pct))):
        return "neutral"
    pr = round(float(pct))
    if pr >= 81:
        return "elite"
    if pr >= 61:
        return "green"
    if pr >= 41:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# Banners / section headers
# ---------------------------------------------------------------------------

def cyan_banner(title: str) -> str:
    """Centered gradient banner with rounded top corners (card header style)."""
    return (f"<div style='background:linear-gradient(180deg, #2daac4 0%, #1f8ba1 100%);"
            f"padding:8px 12px;border-radius:8px 8px 0 0;text-align:center;"
            f"font-weight:700;letter-spacing:0.05em;font-size:0.92em;color:white;"
            f"box-shadow:inset 0 1px 0 rgba(255,255,255,0.18);'>{title}</div>")


def cyan_section_header(st: Any, title: str) -> None:
    """Full-width uppercase gradient section header, rendered via st.markdown."""
    st.markdown(
        f"<div style='margin-top:18px;background:linear-gradient(180deg, #2daac4 0%, #1f8ba1 100%);"
        f"padding:8px 14px;border-radius:8px;font-weight:700;letter-spacing:0.05em;"
        f"font-size:0.92em;color:white;text-transform:uppercase;'>"
        f"{title}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Team badge (initials in a colored circle)
# ---------------------------------------------------------------------------

def team_badge_html(team_name: str, size: int = 56) -> str:
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
    color_idx = sum(ord(c) for c in name) % len(BADGE_COLORS)
    bg = BADGE_COLORS[color_idx]
    # <span>, não <div>: o badge é usado dentro de <span> flex (ex.: header da
    # Visão Geral) e div-dentro-de-span é HTML inválido — o navegador quebra o
    # span pai pra reacomodar o div, destruindo o alinhamento logo+nome.
    return (
        f"<span style='display:inline-flex;align-items:center;justify-content:center;"
        f"width:{size}px;height:{size}px;border-radius:50%;background:{bg};"
        f"color:white;font-weight:800;font-size:1.05em;flex:0 0 auto;"
        f"box-shadow:inset 0 1px 0 rgba(255,255,255,0.18);"
        f"border:1px solid rgba(255,255,255,0.15);'>"
        f"{escape(initials)}</span>"
    )
