# -*- coding: utf-8 -*-
"""Odds cards UI renderer — Over/Under and ML+Totals card views."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app_core.display_tables_core import (
    df_contract_view as _df_contract_view,
    display_ou_table as _display_ou_table,
    display_ml_totals_table as _display_ml_totals_table,
)
from app_core.ui_labels import fmt_num as _fmt_num, fmt_prob_pct as _fmt_prob_pct


def _inject_css(st: Any) -> None:
    st.markdown(
        """
<style>
  .b2t-row { margin: 0.25rem 0 0.65rem 0; }
  .b2t-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 14px 14px;
  }
  .b2t-card.fav {
    background: rgba(255,255,255,0.06);
    border: 2px solid rgba(255,255,255,0.32);
  }
  .b2t-title { font-size: 0.85rem; opacity: 0.85; margin-bottom: 6px; }
  .b2t-big { font-size: 1.55rem; font-weight: 750; line-height: 1.15; }
  .b2t-mid { font-size: 1.10rem; font-weight: 650; line-height: 1.15; }
  .b2t-sub { font-size: 0.92rem; opacity: 0.75; margin-top: 6px; }
  .b2t-mono { font-variant-numeric: tabular-nums; }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_odds_cards(
    df_raw: pd.DataFrame,
    mode: str = "ou",
    table_label: str = "Ver tabela",
    *,
    st: Any,
) -> None:
    """Renderiza odds em cards (neutro), destacando o lado com maior probabilidade.

    mode:
      - 'ou': Acima/Abaixo (p_over/p_under)
      - 'ml_totals': ML & Acima / ML & Abaixo (p_ML+OVER / p_ML+UNDER)
    """
    if df_raw is None or df_raw.empty:
        st.info("Nenhuma linha informada.")
        return

    _inject_css(st)

    def _contract_view_fn(df, required_cols, **kw):
        return _df_contract_view(df, required_cols, warning_fn=st.warning, **kw)

    if mode == "ml_totals":
        label_a, label_b = "ML & Acima", "ML & Abaixo"
        p_a_col, odd_a_col = "p_ML+OVER(map/map)", "odd_ML+OVER(map/map)"
        p_b_col, odd_b_col = "p_ML+UNDER(map/map)", "odd_ML+UNDER(map/map)"
        table_df = _display_ml_totals_table(
            df_raw,
            df_contract_view_fn=_contract_view_fn,
            format_pct_fn=_fmt_prob_pct,
            format_num_fn=_fmt_num,
        )
    else:
        label_a, label_b = "Acima", "Abaixo"
        p_a_col, odd_a_col = "p_over", "odd_over"
        p_b_col, odd_b_col = "p_under", "odd_under"
        table_df = _display_ou_table(
            df_raw,
            df_contract_view_fn=_contract_view_fn,
            format_pct_fn=_fmt_prob_pct,
            format_num_fn=_fmt_num,
        )

    def _safe_f(x: Any) -> float:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return float("nan")
        return xf if math.isfinite(xf) else float("nan")

    for _, row in df_raw.iterrows():
        line = row.get("line")
        p_a = _safe_f(row.get(p_a_col))
        p_b = _safe_f(row.get(p_b_col))
        odd_a = row.get(odd_a_col)
        odd_b = row.get(odd_b_col)

        fav_a = None
        if math.isfinite(p_a) and math.isfinite(p_b):
            fav_a = bool(p_a >= p_b)

        c1, c2, c3 = st.columns([1.05, 1.55, 1.55], gap="small")

        with c1:
            st.markdown(
                f"""<div class='b2t-row'><div class='b2t-card b2t-mono'>
                    <div class='b2t-title'>Linha</div>
                    <div class='b2t-mid'>{line}</div>
                </div></div>""",
                unsafe_allow_html=True,
            )

        with c2:
            cls = "b2t-card fav" if fav_a is True else "b2t-card"
            st.markdown(
                f"""<div class='b2t-row'><div class='{cls} b2t-mono'>
                    <div class='b2t-title'>{label_a}</div>
                    <div class='b2t-big'>{_fmt_num(odd_a, 2)}</div>
                    <div class='b2t-sub'>{_fmt_prob_pct(p_a)}</div>
                </div></div>""",
                unsafe_allow_html=True,
            )

        with c3:
            cls = "b2t-card fav" if fav_a is False else "b2t-card"
            st.markdown(
                f"""<div class='b2t-row'><div class='{cls} b2t-mono'>
                    <div class='b2t-title'>{label_b}</div>
                    <div class='b2t-big'>{_fmt_num(odd_b, 2)}</div>
                    <div class='b2t-sub'>{_fmt_prob_pct(p_b)}</div>
                </div></div>""",
                unsafe_allow_html=True,
            )

    with st.expander(str(table_label)):
        st.dataframe(table_df, width="stretch")
