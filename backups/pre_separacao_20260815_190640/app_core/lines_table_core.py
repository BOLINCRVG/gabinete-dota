# -*- coding: utf-8 -*-
"""Builds Over/Under and ML+Totals odds tables from a totals distribution object."""
from __future__ import annotations

import math
from typing import Any, List

import pandas as pd

from app_core.market import odd_from_prob as _odd_from_p, combine_ml_totals_probs as _combine_ml_totals_probs
from app_core.ml_session_runtime_core import get_ml_scope as _get_ml_scope, set_ml_scope as _set_ml_scope
from app_core.ui_labels import min_to_mmss as _min_to_mmss
from mlcore.lines_engine import total_over_prob


def lines_table(metric: str, lines_list: List[float], total_obj: Any, *, st: Any) -> pd.DataFrame:
    """Generates a fair-odds table (Over/Under) and, if available, ML+Over/Under combos."""
    lines_list = list(lines_list or [])
    if not lines_list:
        return pd.DataFrame(columns=[
            "line", "p_over", "odd_over", "p_under", "odd_under",
            "p_ML+OVER(map/map)", "odd_ML+OVER(map/map)",
            "p_ML+UNDER(map/map)", "odd_ML+UNDER(map/map)",
        ])

    p_ml = _get_ml_scope(st.session_state)
    rows = []
    for ln in lines_list:
        try:
            line = float(ln)
        except (TypeError, ValueError):
            continue

        p_over = float(total_over_prob(total_obj, line))
        p_under = float(1.0 - p_over) if math.isfinite(p_over) else float("nan")

        row = {
            "line": _min_to_mmss(line) if metric == "time" else line,
            "p_over": p_over,
            "odd_over": _odd_from_p(p_over),
            "p_under": p_under,
            "odd_under": _odd_from_p(p_under),
            "p_ML+OVER(map/map)": float("nan"),
            "odd_ML+OVER(map/map)": float("nan"),
            "p_ML+UNDER(map/map)": float("nan"),
            "odd_ML+UNDER(map/map)": float("nan"),
        }

        if math.isfinite(p_ml):
            p_ml_over, p_ml_under = _combine_ml_totals_probs(
                float(p_ml),
                float(p_over),
                dependence=0.0,
            )
            row["p_ML+OVER(map/map)"] = p_ml_over
            row["odd_ML+OVER(map/map)"] = _odd_from_p(p_ml_over)
            row["p_ML+UNDER(map/map)"] = p_ml_under
            row["odd_ML+UNDER(map/map)"] = _odd_from_p(p_ml_under)

        rows.append(row)

    df = pd.DataFrame(rows)
    if metric != "time" and "line" in df.columns:
        df = df.sort_values("line", ascending=True).reset_index(drop=True)
    return df


def lines_table_with_p_ml(
    metric: str,
    lines_list: List[float],
    total_obj: Any,
    p_ml_value: float,
    *,
    st: Any,
) -> pd.DataFrame:
    """Builds the odds table using a specific p(ML) without mutating the session state permanently."""
    old = _get_ml_scope(st.session_state)
    try:
        _set_ml_scope(st.session_state, p_ml_value)
        return lines_table(metric, lines_list, total_obj, st=st)
    finally:
        _set_ml_scope(st.session_state, old)
