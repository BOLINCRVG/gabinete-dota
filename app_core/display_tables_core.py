"""Core helpers for safe DataFrame contract views and odds display tables."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import pandas as pd

from app_core.ui_labels import fmt_num, min_to_mmss


def df_contract_view(
    df: pd.DataFrame | None,
    required_cols: list[str],
    *,
    optional_cols: list[str] | None = None,
    context: str = "",
    fill_value: Any = "-",
    warn_on_missing: bool = True,
    warning_fn: Optional[Callable[[str], Any]] = None,
) -> pd.DataFrame:
    """Guarantee required columns exist before rendering a table."""
    optional_cols = list(optional_cols or [])
    if df is None or (not isinstance(df, pd.DataFrame)):
        out = pd.DataFrame(columns=list(required_cols) + list(optional_cols))
        for c in required_cols:
            out[c] = pd.Series(dtype=object)
        return out[list(required_cols)]

    out = df.copy()
    missing = [c for c in required_cols if c not in out.columns]
    for c in missing:
        out[c] = fill_value

    if warn_on_missing and missing and (not out.empty) and callable(warning_fn):
        ctx = f"{context}: " if str(context).strip() else ""
        warning_fn(f"{ctx}colunas ausentes preenchidas com fallback: {', '.join(missing)}")

    cols = list(required_cols) + [c for c in optional_cols if c in out.columns]
    return out[[c for c in cols if c in out.columns]]


def display_ou_table(
    df: pd.DataFrame,
    *,
    df_contract_view_fn: Callable[..., pd.DataFrame],
    format_pct_fn: Callable[[Any], str],
    format_num_fn: Callable[[Any, int], str],
) -> pd.DataFrame:
    """Build display table for Over/Under odds."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Linha", "Acima (%)", "Odd Acima", "Abaixo (%)", "Odd Abaixo"])
    src = df_contract_view_fn(
        df,
        ["line", "p_over", "odd_over", "p_under", "odd_under"],
        context="Consistência > tabela OU",
        fill_value=float("nan"),
    )
    return pd.DataFrame(
        {
            "Linha": src.get("line"),
            "Acima (%)": src.get("p_over").apply(format_pct_fn),
            "Odd Acima": src.get("odd_over").apply(lambda x: format_num_fn(x, 2)),
            "Abaixo (%)": src.get("p_under").apply(format_pct_fn),
            "Odd Abaixo": src.get("odd_under").apply(lambda x: format_num_fn(x, 2)),
        }
    )


def display_ml_totals_table(
    df: pd.DataFrame,
    *,
    df_contract_view_fn: Callable[..., pd.DataFrame],
    format_pct_fn: Callable[[Any], str],
    format_num_fn: Callable[[Any, int], str],
    team_label: Optional[str] = None,
) -> pd.DataFrame:
    """Build display table for ML + Totals block."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Linha", "ML & Acima (%)", "Odd ML & Acima", "ML & Abaixo (%)", "Odd ML & Abaixo"])
    src = df_contract_view_fn(
        df,
        ["line", "p_ML+OVER(map/map)", "odd_ML+OVER(map/map)", "p_ML+UNDER(map/map)", "odd_ML+UNDER(map/map)"],
        context="Consistência > tabela ML+Totais",
        fill_value=float("nan"),
    )
    out = pd.DataFrame(
        {
            "Linha": src.get("line"),
            "ML & Acima (%)": src.get("p_ML+OVER(map/map)").apply(format_pct_fn),
            "Odd ML & Acima": src.get("odd_ML+OVER(map/map)").apply(lambda x: format_num_fn(x, 2)),
            "ML & Abaixo (%)": src.get("p_ML+UNDER(map/map)").apply(format_pct_fn),
            "Odd ML & Abaixo": src.get("odd_ML+UNDER(map/map)").apply(lambda x: format_num_fn(x, 2)),
        }
    )
    if team_label:
        out.insert(0, "Time", str(team_label))
    return out


_TOTALS_METRIC_PT = {
    "kills": "Kills",
    "towers": "Torres",
    "dragons": "Dragões",
    "dragons_elemental": "Dragões Elementais",
    "barons": "Barões",
    "inhibitors": "Inibidores",
    "time": "Tempo",
}


def totals_table(tot: Any) -> pd.DataFrame:
    """Tabela simples (média/DP) por métrica. `time` é exibido como MM:SS."""
    rows = []
    for k in ["kills", "towers", "dragons", "dragons_elemental", "barons", "inhibitors", "time"]:
        if k not in tot:
            continue
        t = tot[k]
        rows.append(
            {
                "Métrica": _TOTALS_METRIC_PT.get(k, k),
                "Média": getattr(t, "mean", None),
                "DP": getattr(t, "sd", None),
                "Dist": getattr(t, "dist", None),
            }
        )

    df = pd.DataFrame(rows)

    def _clean(x: Any) -> Any:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        return None if (not math.isfinite(xf)) else xf

    if not df.empty:
        if "Média" in df.columns:
            df["Média"] = df["Média"].apply(_clean)
        if "DP" in df.columns:
            df["DP"] = df["DP"].apply(_clean)

        if "Métrica" in df.columns:
            mask = df["Métrica"] == "Tempo"
            if mask.any():
                df["Média"] = df["Média"].astype(object)
                df["DP"] = df["DP"].astype(object)
                df.loc[mask, "Média"] = df.loc[mask, "Média"].apply(min_to_mmss)
                df.loc[mask, "DP"] = df.loc[mask, "DP"].apply(min_to_mmss)

        mask_not_time = df.get("Métrica", "") != "Tempo"
        if "Média" in df.columns:
            df.loc[mask_not_time, "Média"] = df.loc[mask_not_time, "Média"].apply(
                lambda x: fmt_num(x, 2) if x is not None else "-"
            )
        if "DP" in df.columns:
            df.loc[mask_not_time, "DP"] = df.loc[mask_not_time, "DP"].apply(
                lambda x: fmt_num(x, 2) if x is not None else "-"
            )

    return df


def team_profile_table(p: Any) -> pd.DataFrame:
    """Tabela estilo gol.gg: métricas 'for' e 'against' do time."""
    rows = [
        {"metric": "kills", "for": p.kills_for, "against": p.kills_against, "sd_for": p.kills_for_sd, "sd_against": p.kills_against_sd},
        {"metric": "towers", "for": p.towers_for, "against": p.towers_against, "sd_for": p.towers_for_sd, "sd_against": p.towers_against_sd},
        {"metric": "dragons", "for": p.dragons_for, "against": p.dragons_against, "sd_for": p.dragons_for_sd, "sd_against": p.dragons_against_sd},
        {"metric": "barons", "for": p.barons_for, "against": p.barons_against, "sd_for": p.barons_for_sd, "sd_against": p.barons_against_sd},
        {"metric": "inhibitors", "for": p.inhib_for, "against": p.inhib_against, "sd_for": p.inhib_for_sd, "sd_against": p.inhib_against_sd},
        {"metric": "time", "for": p.time_min, "against": None, "sd_for": p.time_min_sd, "sd_against": None},
    ]
    df = pd.DataFrame(rows)

    def _clean(x: Any) -> Any:
        try:
            return None if (x is None or not math.isfinite(float(x))) else float(x)
        except (TypeError, ValueError):
            return None

    for c in ["for", "against", "sd_for", "sd_against"]:
        df[c] = df[c].apply(_clean)

    disp = df.copy()
    disp["for"] = disp.apply(lambda r: min_to_mmss(r["for"]) if r["metric"] == "time" else r["for"], axis=1)
    disp["against"] = disp.apply(lambda r: min_to_mmss(r["against"]) if r["metric"] == "time" else r["against"], axis=1)
    disp["sd_for"] = disp.apply(lambda r: min_to_mmss(r["sd_for"]) if r["metric"] == "time" else r["sd_for"], axis=1)
    disp["sd_against"] = disp.apply(lambda r: min_to_mmss(r["sd_against"]) if r["metric"] == "time" else r["sd_against"], axis=1)

    return disp[["metric", "for", "against", "sd_for", "sd_against"]]

