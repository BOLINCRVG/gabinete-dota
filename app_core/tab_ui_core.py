"""Shared UI helpers for extracted tab renderers."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable


def _clean_text(value: Any) -> str:
    return escape(str(value or "").strip())


def build_tab_hero_html(
    *,
    title: str,
    subtitle: str,
    eyebrow: str = "",
    chips: Iterable[str] | None = None,
) -> str:
    eyebrow_html = f'<div class="gb-eyebrow">{_clean_text(eyebrow)}</div>' if str(eyebrow or "").strip() else ""
    chip_rows = [f'<span class="gb-chip">{_clean_text(chip)}</span>' for chip in (chips or ()) if str(chip or "").strip()]
    chips_html = f'<div class="gb-chip-row">{"".join(chip_rows)}</div>' if chip_rows else ""
    return f"""
<div class="gb-hero gb-hero-tab" style="margin-top:4px;">
  {eyebrow_html}
  <div class="title" style="font-size:1.1rem;">{_clean_text(title)}</div>
  <div class="sub">{_clean_text(subtitle)}</div>
  {chips_html}
</div>
    """


def render_tab_hero(
    st: Any,
    *,
    title: str,
    subtitle: str,
    eyebrow: str = "",
    chips: Iterable[str] | None = None,
) -> None:
    st.markdown(
        build_tab_hero_html(title=title, subtitle=subtitle, eyebrow=eyebrow, chips=chips),
        unsafe_allow_html=True,
    )


def build_tab_section_html(*, title: str, subtitle: str = "", eyebrow: str = "") -> str:
    eyebrow_html = f'<div class="gb-eyebrow">{_clean_text(eyebrow)}</div>' if str(eyebrow or "").strip() else ""
    subtitle_html = f'<div class="gb-section-sub">{_clean_text(subtitle)}</div>' if str(subtitle or "").strip() else ""
    return f"""
<div class="gb-section-head">
  {eyebrow_html}
  <div class="gb-section-title">{_clean_text(title)}</div>
  {subtitle_html}
</div>
    """


def render_tab_section(st: Any, *, title: str, subtitle: str = "", eyebrow: str = "") -> None:
    st.markdown(
        build_tab_section_html(title=title, subtitle=subtitle, eyebrow=eyebrow),
        unsafe_allow_html=True,
    )
