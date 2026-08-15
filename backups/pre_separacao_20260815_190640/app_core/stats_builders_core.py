"""Table building and metric computation for championship stats."""
from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping, Optional

import pandas as pd

from app_core.shared_formatters import fmt_value
from app_core.stats_config import STATS_MARKET_CONFIG
from mlcore.utils import safe_float_or_none as _safe_float


def _safe_pct(series_bool: pd.Series) -> float:
    try:
        return 100.0 * float(series_bool.mean())
    except Exception:
        return float("nan")


def _dedupe_lines(values: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for value in values:
        fv = _safe_float(value)
        if fv is None:
            continue
        key = round(float(fv), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(fv))
    return sorted(out)


def build_threshold_lines_by_metric(
    *,
    session_state: MutableMapping[str, Any],
    parse_lines_fn: Callable[[str], list[float]],
    parse_time_lines_fn: Callable[[str], list[float]],
    custom_metric_label: str = "",
    extra_lines: list[float] | None = None,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for market_label, cfg in STATS_MARKET_CONFIG.items():
        threshold_key = str(cfg["threshold_key"])
        default_text = str(cfg["default_thresholds"])
        raw_text = str(session_state.get(threshold_key, default_text) or default_text)
        parser = parse_time_lines_fn if str(cfg["metric_key"]) == "time" else parse_lines_fn
        lines = list(parser(raw_text))
        if str(market_label) == str(custom_metric_label) and extra_lines:
            lines.extend(list(extra_lines))
        out[str(cfg["metric_key"])] = _dedupe_lines(lines)
    return out


def apply_scope_filters(
    df_scope: pd.DataFrame | None,
    *,
    side_filter: str = "All",
    win_loss_filter: str = "All",
) -> pd.DataFrame:
    if df_scope is None or df_scope.empty:
        return pd.DataFrame([])
    out = df_scope.copy()
    side_key = str(side_filter or "All").strip().lower()
    if side_key in {"blue", "red"} and "side" in out.columns:
        out = out[out["side"].astype(str).str.strip().str.lower() == side_key].copy()
    wl_key = str(win_loss_filter or "All").strip().lower()
    if wl_key in {"wins", "losses"} and "win" in out.columns:
        wins = pd.to_numeric(out["win"], errors="coerce")
        if wl_key == "wins":
            out = out[wins > 0.5].copy()
        else:
            out = out[wins <= 0.5].copy()
    return out


def _metric_series(df_scope: pd.DataFrame, metric_key: str) -> pd.Series:
    cfg = next((v for v in STATS_MARKET_CONFIG.values() if str(v.get("metric_key")) == str(metric_key)), None)
    if not cfg:
        return pd.Series(dtype=float)
    col_name = str(cfg.get("combined_col") or "")
    if not col_name or col_name not in df_scope.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df_scope[col_name], errors="coerce")


def compute_team_metric_pct(
    df_scope: pd.DataFrame | None,
    *,
    team: str,
    metric_key: str,
    side_key: str,
    line_val: float,
) -> Optional[float]:
    if df_scope is None or df_scope.empty or "team" not in df_scope.columns:
        return None
    team_df = df_scope[df_scope["team"].astype(str) == str(team)].copy()
    if team_df.empty:
        return None
    series = _metric_series(team_df, metric_key)
    if series.empty:
        return None
    line_f = _safe_float(line_val)
    if line_f is None:
        return None
    side_norm = str(side_key or "").strip().lower()
    if side_norm == "under":
        return _safe_pct(series <= float(line_f))
    return _safe_pct(series > float(line_f))


def build_championship_stats_table(
    df_scope: pd.DataFrame | None,
    *,
    threshold_lines_by_metric: Mapping[str, list[float]] | None = None,
) -> pd.DataFrame:
    if df_scope is None or df_scope.empty:
        return pd.DataFrame([])

    thresholds = dict(threshold_lines_by_metric or {})
    rows: list[dict[str, Any]] = []

    for team_name, group in df_scope.groupby("team"):
        group = group.copy()
        games_n = int(group.shape[0])
        win_rate = 100.0 * float(pd.to_numeric(group.get("win"), errors="coerce").mean()) if "win" in group.columns else float("nan")
        row: dict[str, Any] = {
            "Team": str(team_name),
            "# Games": games_n,
            "Win Rate": win_rate,
            "Game Time": pd.to_numeric(group.get("game_time_min"), errors="coerce").mean() if "game_time_min" in group.columns else float("nan"),
            "FB%": _safe_pct(pd.to_numeric(group.get("firstblood"), errors="coerce") > 0.5) if "firstblood" in group.columns else float("nan"),
            "FRH%": _safe_pct(pd.to_numeric(group.get("firstherald"), errors="coerce") > 0.5) if "firstherald" in group.columns else float("nan"),
            "FT%": _safe_pct(pd.to_numeric(group.get("firsttower"), errors="coerce") > 0.5) if "firsttower" in group.columns else float("nan"),
            "FD%": _safe_pct(pd.to_numeric(group.get("firstdragon"), errors="coerce") > 0.5) if "firstdragon" in group.columns else float("nan"),
            "FNASH%": _safe_pct(pd.to_numeric(group.get("firstbaron"), errors="coerce") > 0.5) if "firstbaron" in group.columns else float("nan"),
            "Gold": pd.to_numeric(group.get("gold_for"), errors="coerce").mean() if "gold_for" in group.columns else float("nan"),
            "Kills": pd.to_numeric(group.get("kills_for"), errors="coerce").mean() if "kills_for" in group.columns else float("nan"),
            "Deaths": pd.to_numeric(group.get("deaths_for"), errors="coerce").mean() if "deaths_for" in group.columns else float("nan"),
            "Towers": pd.to_numeric(group.get("towers_for"), errors="coerce").mean() if "towers_for" in group.columns else float("nan"),
            "C. Gold": pd.to_numeric(group.get("total_gold"), errors="coerce").mean() if "total_gold" in group.columns else float("nan"),
            "C. Kills": pd.to_numeric(group.get("total_kills"), errors="coerce").mean() if "total_kills" in group.columns else float("nan"),
            "C. Towers": pd.to_numeric(group.get("total_towers"), errors="coerce").mean() if "total_towers" in group.columns else float("nan"),
            "C. Dragons": pd.to_numeric(group.get("total_dragons"), errors="coerce").mean() if "total_dragons" in group.columns else float("nan"),
            "C. Nashors": pd.to_numeric(group.get("total_nashors"), errors="coerce").mean() if "total_nashors" in group.columns else float("nan"),
            "C. Inhib": pd.to_numeric(group.get("total_inhibitors"), errors="coerce").mean() if "total_inhibitors" in group.columns else float("nan"),
        }

        for _market_label, cfg in STATS_MARKET_CONFIG.items():
            metric_key = str(cfg["metric_key"])
            combined_col = str(cfg["combined_col"])
            pct_prefix = str(cfg["pct_prefix"])
            if combined_col not in group.columns:
                continue
            series = pd.to_numeric(group[combined_col], errors="coerce")
            for line_val in list(thresholds.get(metric_key, []) or []):
                fv = _safe_float(line_val)
                if fv is None:
                    continue
                row[f"{pct_prefix}{fv:g}"] = _safe_pct(series > float(fv))

        rows.append(row)

    return pd.DataFrame(rows)


def build_stats_table_view(
    df_stats_raw: pd.DataFrame | None,
    *,
    min_to_mmss_fn: Callable[[Any], str],
    format_num_fn: Callable[[float, int], str] | None = None,
    priority_teams: list[str] | None = None,
) -> pd.DataFrame:
    if df_stats_raw is None or df_stats_raw.empty:
        return pd.DataFrame([])

    out = df_stats_raw.copy()
    if priority_teams:
        order_map = {str(name): idx for idx, name in enumerate(priority_teams)}
        out["_prio"] = out["Team"].astype(str).map(lambda name: order_map.get(name, len(order_map)))
        out = out.sort_values(["_prio", "Team"]).drop(columns=["_prio"], errors="ignore").reset_index(drop=True)
    else:
        out = out.sort_values(["Team"]).reset_index(drop=True)

    if "Game Time" in out.columns:
        out["Game Time"] = out["Game Time"].apply(lambda x: min_to_mmss_fn(x) if _safe_float(x) is not None else "-")

    pct_cols = [col for col in out.columns if "%" in str(col)]
    for col in pct_cols:
        out[col] = out[col].apply(fmt_value)

    for col in out.columns:
        if col in {"Team", "Game Time"} or "%" in str(col):
            continue
        if col == "# Games":
            out[col] = out[col].apply(lambda x: str(int(_safe_float(x) or 0)))
            continue
        out[col] = out[col].apply(fmt_value)

    return out


def format_matchup_stats_detail(*, team_a: str, team_b: str, pct_a: Any, pct_b: Any) -> str:
    fa = _safe_float(pct_a)
    fb = _safe_float(pct_b)
    left = "-" if fa is None else f"{round(fa)}%"
    right = "-" if fb is None else f"{round(fb)}%"
    return f"{team_a} {left} | {team_b} {right}"


def enrich_resumo_rows_with_stats_detail(
    rows: list[dict[str, Any]] | None,
    *,
    df_scope: pd.DataFrame | None,
    team_a: str,
    team_b: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        metric_key = str(item.get("_metric") or "").strip()
        line_val = _safe_float(item.get("_line"))
        side = str(item.get("Lado") or "").strip().lower()
        if metric_key and line_val is not None:
            side_key = "under" if side in {"abaixo", "under"} else "over"
            pct_a = compute_team_metric_pct(df_scope, team=str(team_a), metric_key=metric_key, side_key=side_key, line_val=float(line_val))
            pct_b = compute_team_metric_pct(df_scope, team=str(team_b), metric_key=metric_key, side_key=side_key, line_val=float(line_val))
            item["Stats Detail"] = format_matchup_stats_detail(team_a=str(team_a), team_b=str(team_b), pct_a=pct_a, pct_b=pct_b)
        out.append(item)
    return out
