# -*- coding: utf-8 -*-
"""Builds the series totals table (Bo3/Bo5) via Monte Carlo simulation."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app_core.ui_labels import min_to_mmss as _min_to_mmss

_METRIC_PT = {
    "kills": "kills",
    "towers": "torres",
    "dragons": "dragões",
    "dragons_elemental": "dragões elementais",
    "barons": "barões",
    "inhibitors": "inibidores",
    "time": "tempo",
}


def series_totals_table(tot_avg: Any, *, st: Any, ctx: Mapping[str, Any]) -> pd.DataFrame:
    """Tabela de totais da série via MC. ctx supplies: _series_sims_cached, totals_mode, bo, p_map_cal, n_sims."""
    _series_sims_cached = ctx.get("_series_sims_cached")
    totals_mode = ctx.get("totals_mode")
    bo = ctx.get("bo")
    p_map_cal = ctx.get("p_map_cal")
    n_sims = ctx.get("n_sims")

    rows = []
    use_series = (totals_mode == "series" and int(bo) > 1 and math.isfinite(float(p_map_cal)))

    for k in ["kills", "towers", "dragons", "dragons_elemental", "barons", "inhibitors", "time"]:
        if k not in tot_avg:
            continue
        t = tot_avg[k]
        mean = getattr(t, "mean", None)
        sd = getattr(t, "sd", None)
        dist = getattr(t, "dist", None)

        if use_series:
            sims = _series_sims_cached(
                int(bo),
                float(p_map_cal),
                t,
                int(n_sims),
                int(st.session_state.get("_sim_seed_series", 1337) or 1337),
            )
            mean = float(np.mean(sims)) if len(sims) else float("nan")
            sd = float(np.std(sims, ddof=1)) if len(sims) > 1 else 0.0
            dist = "series_mc"

        try:
            mean_f = float(mean)
        except (TypeError, ValueError):
            mean_f = float("nan")
        try:
            sd_f = float(sd)
        except (TypeError, ValueError):
            sd_f = float("nan")

        rows.append({"métrica": _METRIC_PT.get(k, k), "média": mean_f, "dp": sd_f, "dist": dist})

    df = pd.DataFrame(rows)

    if not df.empty and "métrica" in df.columns:
        mask = df["métrica"] == "tempo"
        if mask.any():
            df.loc[mask, "média"] = df.loc[mask, "média"].apply(_min_to_mmss)
            df.loc[mask, "dp"] = df.loc[mask, "dp"].apply(_min_to_mmss)

    return df
