# -*- coding: utf-8 -*-
"""Global UI CSS injection for the Plays app."""
from __future__ import annotations
from typing import Any


def inject_global_ui_css(st: Any) -> None:
    st.markdown(
        """
<style>
  :root {
    --gb-accent: #2daac4;
    --gb-panel: rgba(255,255,255,0.035);
    --gb-panel-border: rgba(255,255,255,0.12);
  }
  .main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
  }
  .gb-hero {
    border: 1px solid var(--gb-panel-border);
    background: linear-gradient(135deg, rgba(45,170,196,0.10), rgba(255,255,255,0.02));
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .gb-hero-tab {
    background:
      radial-gradient(circle at top right, rgba(45,170,196,0.14), transparent 32%),
      linear-gradient(135deg, rgba(45,170,196,0.10), rgba(255,255,255,0.02));
  }
  .gb-hero .title {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.01em;
    line-height: 1.15;
  }
  .gb-hero .sub {
    opacity: 0.80;
    margin-top: 4px;
    font-size: 0.92rem;
  }
  .gb-eyebrow {
    display: inline-block;
    margin-bottom: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.66);
  }
  .gb-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 10px;
  }
  .gb-chip {
    display: inline-flex;
    align-items: center;
    min-height: 26px;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    font-size: 0.78rem;
    line-height: 1;
    color: rgba(255,255,255,0.88);
  }
  .gb-section-head {
    margin: 0.4rem 0 0.75rem 0;
  }
  .gb-section-title {
    font-size: 1rem;
    font-weight: 760;
    line-height: 1.15;
    letter-spacing: 0.01em;
  }
  .gb-section-sub {
    margin-top: 4px;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.72);
  }
  div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 10px;
    padding: 0.35rem 0.75rem;
    margin-right: 0.25rem;
    transition: background 120ms ease, border-color 120ms ease;
  }
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(45,170,196,0.14);
    border-bottom-color: var(--gb-accent) !important;
    box-shadow: inset 0 0 0 1px rgba(45,170,196,0.22);
  }
  .stButton > button {
    border-radius: 10px;
  }
  div[data-testid="stExpander"] {
    background: var(--gb-panel);
    border: 1px solid var(--gb-panel-border);
    border-radius: 10px;
  }
  div[data-testid="stMetric"] {
    border: 1px solid var(--gb-panel-border);
    border-radius: 10px;
    padding: 8px 10px;
    background: rgba(255,255,255,0.02);
  }
</style>
        """,
        unsafe_allow_html=True,
    )
