from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from app_core.championship_stats_core import STATS_MARKET_CONFIG
from mlcore.utils import safe_float_or_none as _safe_float


FRIENDLY_MARKET_CONFIG: dict[str, dict[str, str]] = {
    "Kills": dict(STATS_MARKET_CONFIG["Kills"]),
    "Torres": dict(STATS_MARKET_CONFIG["Torres"]),
    "Dragões": {
        "metric_key": "dragons",
        "combined_col": "total_dragons",
        "threshold_key": "sc_th_dragons",
        "default_thresholds": "4.5,5.5",
        "pct_prefix": "% Dragons > ",
    },
    "Barões": {
        "metric_key": "barons",
        "combined_col": "total_nashors",
        "threshold_key": "sc_th_barons",
        "default_thresholds": "1.5",
        "pct_prefix": "% Nashors > ",
    },
    "Inibidores": dict(STATS_MARKET_CONFIG["Inibidores"]),
    "Tempo": dict(STATS_MARKET_CONFIG["Tempo"]),
}

PROFILE_BUCKET_ORDER = ["Muito Under", "Under", "Neutro", "Over", "Muito Over"]
PHASE_LABEL_ORDER = ["Final", "0-15", "15-20", "20+"]


def resolve_market_config(market_label: str) -> dict[str, str]:
    return dict(FRIENDLY_MARKET_CONFIG.get(str(market_label), FRIENDLY_MARKET_CONFIG["Kills"]))


def available_market_labels() -> list[str]:
    return list(FRIENDLY_MARKET_CONFIG.keys())


def metric_series(df_scope: pd.DataFrame | None, metric_key: str) -> pd.Series:
    cfg = next((v for v in FRIENDLY_MARKET_CONFIG.values() if str(v.get("metric_key")) == str(metric_key)), None)
    if not cfg or df_scope is None or df_scope.empty:
        return pd.Series(dtype=float)
    col_name = str(cfg.get("combined_col") or "")
    if not col_name or col_name not in df_scope.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df_scope[col_name], errors="coerce")


def phase_options_for_market(df_scope: pd.DataFrame | None, market_label: str) -> list[str]:
    cfg = resolve_market_config(market_label)
    if str(cfg.get("metric_key") or "") != "kills":
        return ["Final"]
    if df_scope is None or df_scope.empty:
        return ["Final"]
    cols = set(df_scope.columns)
    options = ["Final"]
    if "total_kills_at15" in cols:
        options.append("0-15")
    if "total_kills_15_20" in cols:
        options.append("15-20")
    if "total_kills_20_plus" in cols:
        options.append("20+")
    return options


def _phase_metric_col(metric_key: str, phase_label: str, default_col: str) -> str:
    if str(metric_key) != "kills":
        return str(default_col)
    phase = str(phase_label or "Final")
    if phase == "0-15":
        return "total_kills_at15"
    if phase == "15-20":
        return "total_kills_15_20"
    if phase == "20+":
        return "total_kills_20_plus"
    return str(default_col)


def _bucket_from_score(score: float) -> str:
    if score <= -1.0:
        return "Muito Under"
    if score <= -0.35:
        return "Under"
    if score < 0.35:
        return "Neutro"
    if score < 1.0:
        return "Over"
    return "Muito Over"


def _confidence_from_games(n_games: int) -> tuple[str, float]:
    n = max(int(n_games or 0), 0)
    if n <= 0:
        return ("Sem amostra", 0.0)
    score = 100.0 * float(n) / float(n + 4.0)
    if n == 1:
        return ("Muito fraca", score)
    if n <= 3:
        return ("Fraca", score)
    if n <= 6:
        return ("Média", score)
    if n <= 9:
        return ("Boa", score)
    return ("Alta", score)


def _sample_risk_from_games(n_games: int) -> str:
    n = max(int(n_games or 0), 0)
    if n <= 0:
        return "Sem base"
    if n <= 2:
        return "Muito sensível"
    if n <= 4:
        return "Baixa amostra"
    if n <= 7:
        return "Atenção"
    return "Estável"


def build_opponent_reference_table(
    df_scope: pd.DataFrame | None,
    market_label: str,
    *,
    phase_label: str = "Final",
) -> pd.DataFrame:
    cfg = resolve_market_config(market_label)
    metric_key = str(cfg.get("metric_key") or "kills")
    combined_col = _phase_metric_col(metric_key, phase_label, str(cfg.get("combined_col") or ""))
    if df_scope is None or df_scope.empty or "team" not in df_scope.columns or combined_col not in df_scope.columns:
        return pd.DataFrame(columns=["Team", "Games", "Metric Mean", "Metric Z", "Bucket"])

    grouped = (
        df_scope.groupby("team", dropna=False)
        .agg(
            Games=(combined_col, "size"),
            MetricMean=(combined_col, lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
        .rename(columns={"team": "Team"})
    )
    mean_global = _safe_float(grouped["MetricMean"].mean()) or 0.0
    std_global = _safe_float(grouped["MetricMean"].std(ddof=0)) or 0.0
    if std_global <= 1e-9:
        grouped["Metric Z"] = 0.0
    else:
        grouped["Metric Z"] = (pd.to_numeric(grouped["MetricMean"], errors="coerce") - float(mean_global)) / float(std_global)
    grouped["Metric Mean"] = pd.to_numeric(grouped["MetricMean"], errors="coerce")
    grouped["Bucket"] = grouped["Metric Z"].map(lambda value: _bucket_from_score(float(value) if pd.notna(value) else 0.0))
    grouped = grouped.drop(columns=["MetricMean"])
    return grouped[["Team", "Games", "Metric Mean", "Metric Z", "Bucket"]].sort_values(["Metric Mean", "Games"], ascending=[False, False])


def _signal_threshold(metric_key: str, baseline_std: float) -> float:
    floors = {
        "kills": 2.0,
        "towers": 0.8,
        "dragons": 0.45,
        "barons": 0.25,
        "inhibitors": 0.2,
        "time": 1.5,
    }
    floor = float(floors.get(str(metric_key), 0.5))
    return max(float(baseline_std or 0.0) * 0.35, floor)


def describe_profile_signal(delta: Optional[float], metric_key: str, baseline_std: float) -> str:
    dv = _safe_float(delta)
    if dv is None:
        return "-"
    threshold = _signal_threshold(metric_key, baseline_std)
    if dv >= threshold:
        return "Acelera"
    if dv <= -threshold:
        return "Desacelera"
    return "Mantém"


def _build_examples(df_in: pd.DataFrame, max_rows: int = 4) -> str:
    if df_in is None or df_in.empty:
        return "-"
    rows: list[str] = []
    for _, row in df_in.sort_values("date").tail(int(max_rows)).iterrows():
        opp = str(row.get("opponent", "") or "-")
        league = str(row.get("league", "") or "").strip()
        win_val = _safe_float(row.get("win"))
        wl = "W" if win_val is not None and win_val > 0.5 else "L"
        suffix = f" | {league}" if league else ""
        rows.append(f"{opp} {wl}{suffix}")
    return " ; ".join(rows) if rows else "-"


def _build_examples_frame(df_in: pd.DataFrame, *, combined_col: str) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=["Data", "Oponente", "Campeonato", "Resultado", "Valor"])
    view = df_in.copy()
    if "date" in view.columns:
        sort_dates = pd.to_datetime(view["date"], errors="coerce")
        view["Data"] = sort_dates.dt.strftime("%Y-%m-%d").fillna("-")
        view["_sort_date"] = sort_dates
    else:
        view["Data"] = "-"
        view["_sort_date"] = pd.NaT
    view["Oponente"] = view.get("opponent", "-").astype(str)
    view["Campeonato"] = view.get("league", "-").astype(str) if "league" in view.columns else "-"
    win_val = pd.to_numeric(view.get("win"), errors="coerce")
    view["Resultado"] = win_val.map(lambda value: "W" if pd.notna(value) and float(value) > 0.5 else "L")
    view["Valor"] = pd.to_numeric(view.get(combined_col), errors="coerce")
    out = (
        view[["Data", "Oponente", "Campeonato", "Resultado", "Valor", "_sort_date"]]
        .sort_values(["_sort_date", "Oponente"], ascending=[False, True])
        .drop(columns=["_sort_date"], errors="ignore")
        .reset_index(drop=True)
    )
    return out


def build_team_opponent_profile(
    df_scope: pd.DataFrame | None,
    *,
    team: str,
    opponent_team: str,
    market_label: str,
    line_val: Optional[float] = None,
    reference_table: pd.DataFrame | None = None,
    phase_label: str = "Final",
) -> dict[str, Any]:
    cfg = resolve_market_config(market_label)
    metric_key = str(cfg.get("metric_key") or "kills")
    combined_col = _phase_metric_col(metric_key, phase_label, str(cfg.get("combined_col") or "total_kills"))
    empty_rows = pd.DataFrame(
        columns=[
            "Atual",
            "Perfil do Oponente",
            "Jogos no bucket",
            "Amostra",
            "Risco",
            "Confiança",
            "Média",
            "Delta",
            "Delta vs Liga",
            "% Over",
            "% Under",
            "Leitura",
            "Exemplos",
        ]
    )
    if (
        df_scope is None
        or df_scope.empty
        or "team" not in df_scope.columns
        or "opponent" not in df_scope.columns
        or combined_col not in df_scope.columns
    ):
        return {
            "team": str(team),
            "opponent_team": str(opponent_team),
            "team_bucket": "-",
            "opponent_bucket": "-",
            "phase_label": str(phase_label),
            "baseline_mean": None,
            "baseline_std": None,
            "baseline_over": None,
            "baseline_under": None,
            "rows": empty_rows,
            "focus_row": None,
            "focus_games": pd.DataFrame(columns=["Data", "Oponente", "Campeonato", "Resultado", "Valor"]),
            "bucket_games": {},
        }

    ref = reference_table if reference_table is not None else build_opponent_reference_table(df_scope, market_label, phase_label=phase_label)
    if ref is None or ref.empty:
        return {
            "team": str(team),
            "opponent_team": str(opponent_team),
            "team_bucket": "-",
            "opponent_bucket": "-",
            "phase_label": str(phase_label),
            "baseline_mean": None,
            "baseline_std": None,
            "baseline_over": None,
            "baseline_under": None,
            "rows": empty_rows,
            "focus_row": None,
            "focus_games": pd.DataFrame(columns=["Data", "Oponente", "Campeonato", "Resultado", "Valor"]),
            "bucket_games": {},
        }

    bucket_by_team = {str(r["Team"]): str(r["Bucket"]) for _, r in ref.iterrows()}
    mean_by_team = {str(r["Team"]): _safe_float(r["Metric Mean"]) for _, r in ref.iterrows()}
    z_by_team = {str(r["Team"]): _safe_float(r["Metric Z"]) for _, r in ref.iterrows()}
    df_team = df_scope[df_scope["team"].astype(str) == str(team)].copy()
    if df_team.empty:
        return {
            "team": str(team),
            "opponent_team": str(opponent_team),
            "team_bucket": str(bucket_by_team.get(str(team), "-")),
            "opponent_bucket": str(bucket_by_team.get(str(opponent_team), "-")),
            "phase_label": str(phase_label),
            "baseline_mean": None,
            "baseline_std": None,
            "baseline_over": None,
            "baseline_under": None,
            "rows": empty_rows,
            "focus_row": None,
            "focus_games": pd.DataFrame(columns=["Data", "Oponente", "Campeonato", "Resultado", "Valor"]),
            "bucket_games": {},
        }

    team_series = pd.to_numeric(df_team[combined_col], errors="coerce").dropna()
    baseline_mean = _safe_float(team_series.mean())
    baseline_std = _safe_float(team_series.std(ddof=0)) or 0.0
    league_mean = _safe_float(pd.to_numeric(df_scope[combined_col], errors="coerce").mean())
    baseline_over = None
    baseline_under = None
    phase_supports_line = str(phase_label) == "Final"
    if phase_supports_line and _safe_float(line_val) is not None and not team_series.empty:
        lv = float(line_val)
        baseline_over = 100.0 * float((team_series > lv).mean())
        baseline_under = 100.0 * float((team_series <= lv).mean())

    df_team["opponent_bucket"] = df_team["opponent"].astype(str).map(lambda value: bucket_by_team.get(str(value), "Neutro"))
    df_team["opponent_metric_mean"] = df_team["opponent"].astype(str).map(lambda value: mean_by_team.get(str(value)))

    focus_bucket = str(bucket_by_team.get(str(opponent_team), "-"))
    rows: list[dict[str, Any]] = []
    bucket_games: dict[str, pd.DataFrame] = {}
    for bucket in PROFILE_BUCKET_ORDER:
        subset = df_team[df_team["opponent_bucket"].astype(str) == str(bucket)].copy()
        bucket_games[str(bucket)] = _build_examples_frame(subset, combined_col=combined_col)
        metric_values = pd.to_numeric(subset.get(combined_col), errors="coerce").dropna()
        metric_mean = _safe_float(metric_values.mean())
        delta = (float(metric_mean) - float(baseline_mean)) if (metric_mean is not None and baseline_mean is not None) else None
        delta_vs_league = (float(metric_mean) - float(league_mean)) if (metric_mean is not None and league_mean is not None) else None
        over_pct = None
        under_pct = None
        if phase_supports_line and _safe_float(line_val) is not None and not metric_values.empty:
            lv = float(line_val)
            over_pct = 100.0 * float((metric_values > lv).mean())
            under_pct = 100.0 * float((metric_values <= lv).mean())
        sample_label, confidence_score = _confidence_from_games(int(subset.shape[0]))
        rows.append(
            {
                "Atual": "Atual" if str(bucket) == focus_bucket else "",
                "Perfil do Oponente": bucket,
                "Jogos no bucket": int(subset.shape[0]),
                "Amostra": sample_label,
                "Risco": _sample_risk_from_games(int(subset.shape[0])),
                "Confiança": confidence_score,
                "Média": metric_mean,
                "Delta": delta,
                "Delta vs Liga": delta_vs_league,
                "% Over": over_pct,
                "% Under": under_pct,
                "Leitura": describe_profile_signal(delta, metric_key, float(baseline_std)),
                "Exemplos": _build_examples(subset),
            }
        )

    rows_df = pd.DataFrame(rows)
    focus_df = rows_df[rows_df["Perfil do Oponente"].astype(str) == focus_bucket].copy()
    focus_row = focus_df.iloc[0].to_dict() if not focus_df.empty else None
    focus_games = _build_examples_frame(
        df_team[df_team["opponent_bucket"].astype(str) == focus_bucket].copy(),
        combined_col=combined_col,
    )
    return {
        "team": str(team),
        "opponent_team": str(opponent_team),
        "team_bucket": str(bucket_by_team.get(str(team), "-")),
        "opponent_bucket": focus_bucket,
        "team_games_total": int(df_team.shape[0]),
        "opponent_metric_mean": mean_by_team.get(str(opponent_team)),
        "opponent_metric_z": z_by_team.get(str(opponent_team)),
        "league_mean": league_mean,
        "phase_label": str(phase_label),
        "baseline_mean": baseline_mean,
        "baseline_std": float(baseline_std),
        "baseline_over": baseline_over,
        "baseline_under": baseline_under,
        "rows": rows_df,
        "focus_row": focus_row,
        "focus_games": focus_games,
        "bucket_games": bucket_games,
    }


def build_matchup_focus_table(team_a_payload: Mapping[str, Any], team_b_payload: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in (team_a_payload, team_b_payload):
        focus = dict(payload.get("focus_row") or {})
        rows.append(
            {
                "Time": str(payload.get("team") or "-"),
                "Fase": str(payload.get("phase_label") or "Final"),
                "Adversário Atual": str(payload.get("opponent_team") or "-"),
                "Perfil Atual": str(payload.get("opponent_bucket") or "-"),
                "Jogos no bucket": int(focus.get("Jogos no bucket") or 0),
                "Jogos do time": int(payload.get("team_games_total") or 0),
                "Amostra": str(focus.get("Amostra") or "-"),
                "Risco": str(focus.get("Risco") or "-"),
                "Confiança": focus.get("Confiança"),
                "Score Adv.": payload.get("opponent_metric_mean"),
                "Z Adv.": payload.get("opponent_metric_z"),
                "Média": focus.get("Média"),
                "Delta": focus.get("Delta"),
                "Delta vs Liga": focus.get("Delta vs Liga"),
                "% Over": focus.get("% Over"),
                "% Under": focus.get("% Under"),
                "Leitura": str(focus.get("Leitura") or "-"),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "PHASE_LABEL_ORDER",
    "PROFILE_BUCKET_ORDER",
    "available_market_labels",
    "build_matchup_focus_table",
    "build_opponent_reference_table",
    "build_team_opponent_profile",
    "describe_profile_signal",
    "phase_options_for_market",
    "resolve_market_config",
]
