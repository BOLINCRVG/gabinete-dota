"""Helpers for Consistencia de Mercado panel data/modeling (UI-agnostic)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import pandas as pd


def panel_title_with_suffix(panel_title: str, suffix: str = "(combined)") -> str:
    base_title = str(panel_title or "").strip() or "Mercado"
    return base_title if base_title.lower().endswith(str(suffix).lower()) else f"{base_title} {suffix}"


def fmt_metric_value(
    col_name: str,
    value: Any,
    decimals: int = 1,
    time_formatter: Callable[[float], str] | None = None,
) -> str:
    try:
        x = float(value)
        if not math.isfinite(x):
            return "-"
        if str(col_name) == "game_time_min" and callable(time_formatter):
            return str(time_formatter(float(x)))
        # Spread columns (handicaps): show explicit + sign for positive values
        # so the user immediately sees direction (vencer por X / perder por X).
        if str(col_name).startswith("mv2_spread_"):
            if x > 0:
                return f"+{x:.{int(decimals)}f}"
            if x < 0:
                return f"−{abs(x):.{int(decimals)}f}"
            return f"{x:.{int(decimals)}f}"
        return f"{x:.{int(decimals)}f}"
    except (TypeError, ValueError):
        return "-"


def team_metric_stats(df_in: pd.DataFrame, col_name: str) -> Dict[str, Any]:
    if df_in is None or df_in.empty or str(col_name) not in df_in.columns:
        return {"games": 0, "avg": float("nan"), "med": float("nan"), "min": float("nan"), "max": float("nan")}
    x = pd.to_numeric(df_in[str(col_name)], errors="coerce").dropna()
    if x.empty:
        return {"games": 0, "avg": float("nan"), "med": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "games": int(x.shape[0]),
        "avg": float(x.mean()),
        "med": float(x.median()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def last_rows_metric(
    df_in: pd.DataFrame,
    col_name: str,
    n: int,
    metric_formatter: Callable[[str, Any, int], str],
) -> List[str]:
    if df_in is None or df_in.empty or str(col_name) not in df_in.columns:
        return []
    d = df_in.sort_values("date").tail(int(n))
    out: List[str] = []
    for _, row in d.iterrows():
        opp = str(row.get("opponent", "") or "-")
        league_name = str(row.get("league", "") or "").strip()
        v = pd.to_numeric(pd.Series([row.get(col_name)]), errors="coerce").iloc[0]
        suffix = f" | {league_name}" if league_name else ""
        out.append(f"{opp} - {metric_formatter(str(col_name), v, 0)}{suffix}")
    return out


_PAIR_METRIC_COLUMNS: Dict[str, Tuple[str, str]] = {
    "total_kills": ("kills_for", "kills_against"),
    "total_towers": ("towers_for", "towers_against"),
    "total_dragons": ("dragons_for", "dragons_against"),
    "total_elementaldrakes": ("elementaldrakes_for", "elementaldrakes_against"),
    "total_nashors": ("barons_for", "barons_against"),
    "total_inhibitors": ("inhibitors_for", "inhibitors_against"),
}


def recent_rows_with_hc_markers(
    df_in: pd.DataFrame,
    *,
    shown_team: str,
    hc_team_selected: str,
    hc_lines: List[float] | None,
    fallback_col_name: str,
    metric_formatter: Callable[[str, Any, int], str],
    n: int = 10,
) -> Tuple[List[str], List[float]]:
    if df_in is None or df_in.empty:
        return ([], [])
    if "kills_for" not in df_in.columns or "kills_against" not in df_in.columns:
        return (last_rows_metric(df_in, fallback_col_name, n, metric_formatter), [])
    applied_source = [float(x) for x in list(hc_lines or [])]
    if not applied_source:
        return (last_rows_metric(df_in, fallback_col_name, n, metric_formatter), [])

    applied_lines = list(applied_source) if str(shown_team) == str(hc_team_selected) else [(-float(x)) for x in applied_source]
    d = df_in.sort_values("date").tail(int(n))
    out: List[str] = []
    for _, row in d.iterrows():
        opp = str(row.get("opponent", "") or "-")
        league_name = str(row.get("league", "") or "").strip()
        v_for = pd.to_numeric(pd.Series([row.get("kills_for")]), errors="coerce").iloc[0]
        v_against = pd.to_numeric(pd.Series([row.get("kills_against")]), errors="coerce").iloc[0]
        suffix = f" | {league_name}" if league_name else ""
        base = f"{opp} - {metric_formatter('kills_for', v_for, 0)}{suffix}"
        if pd.isna(v_for) or pd.isna(v_against):
            out.append(f"{base} | HC: -")
            continue
        diff = float(v_for) - float(v_against)
        marks = ["V" if (diff + float(line)) > 0.0 else "X" for line in applied_lines]
        out.append(f"{base} | HC: {'/'.join(marks)}")
    return (out, applied_lines)


def h2h_rows_verbose(
    df_in: pd.DataFrame,
    *,
    metric_col: str,
    team_a: str,
    metric_formatter: Callable[[str, Any, int], str],
    n: int = 10,
) -> List[str]:
    if df_in is None or df_in.empty or str(metric_col) not in df_in.columns:
        return []
    d = df_in.sort_values("date").tail(int(n))
    out: List[str] = []
    for _, row in d.iterrows():
        opp = str(row.get("opponent", "") or "-")
        league_name = str(row.get("league", "") or "").strip()
        suffix = f" | {league_name}" if league_name else ""
        if metric_col in _PAIR_METRIC_COLUMNS:
            c_for, c_against = _PAIR_METRIC_COLUMNS[metric_col]
            v_tot = metric_formatter(metric_col, row.get(metric_col), 0)
            v_for = metric_formatter(metric_col, row.get(c_for), 0)
            v_against = metric_formatter(metric_col, row.get(c_against), 0)
            out.append(f"{team_a} vs {opp} -> total={v_tot} ({team_a}={v_for} | {opp}={v_against}){suffix}")
            continue
        if metric_col.endswith("_for"):
            c_against = metric_col[:-4] + "_against"
            v_for = metric_formatter(metric_col, row.get(metric_col), 0)
            if c_against in d.columns:
                v_against = metric_formatter(metric_col, row.get(c_against), 0)
                out.append(f"{team_a} vs {opp} -> {team_a}={v_for} | {opp}={v_against}{suffix}")
            else:
                out.append(f"{team_a} vs {opp} -> {team_a}={v_for}{suffix}")
            continue
        if metric_col.endswith("_against"):
            c_for = metric_col[:-8] + "_for"
            v_against = metric_formatter(metric_col, row.get(metric_col), 0)
            if c_for in d.columns:
                v_for = metric_formatter(metric_col, row.get(c_for), 0)
                out.append(f"{team_a} vs {opp} -> {team_a}={v_for} | {opp}={v_against}{suffix}")
            else:
                out.append(f"{team_a} vs {opp} -> {opp}={v_against}{suffix}")
            continue
        if metric_col == "game_time_min":
            out.append(f"{team_a} vs {opp} -> tempo={metric_formatter(metric_col, row.get(metric_col), 0)}{suffix}")
            continue
        out.append(f"{team_a} vs {opp} -> valor={metric_formatter(metric_col, row.get(metric_col), 0)}{suffix}")
    return out


def has_metric_history(*frames: pd.DataFrame, metric_col: str) -> bool:
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or frame.empty or str(metric_col) not in frame.columns:
            continue
        vals = pd.to_numeric(frame[str(metric_col)], errors="coerce")
        if bool(vals.notna().any()):
            return True
    return False


def _safe_table_team_labels(team_a: str, team_b: str) -> Tuple[str, str]:
    a_raw = str(team_a or "").strip()
    b_raw = str(team_b or "").strip()
    a = a_raw or "Time A"
    b = b_raw or "Time B"
    if a == b:
        if not a_raw and not b_raw:
            return "Time A", "Time B"
        if not b_raw:
            b = "Time B"
            if b == a:
                b = f"{a} (2)"
        else:
            b = f"{b} (2)"
    return a, b


def build_overall_tables(
    team_a: str,
    team_b: str,
    col_name: str,
    s_a_cur: Dict[str, Any],
    s_b_cur: Dict[str, Any],
    s_a_all: Dict[str, Any],
    s_b_all: Dict[str, Any],
    num_formatter: Callable[[Any, int], str],
    metric_formatter: Callable[[str, Any, int], str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    team_a_col, team_b_col = _safe_table_team_labels(team_a, team_b)
    cols = ["Métrica", team_a_col, team_b_col]
    rows_names = ["# Games", "Avg", "Median", "Min", "Max"]

    cur = pd.DataFrame(
        {
            "Métrica": rows_names,
            team_a_col: [
                num_formatter(s_a_cur["games"], 0),
                metric_formatter(col_name, s_a_cur["avg"], 1),
                metric_formatter(col_name, s_a_cur["med"], 1),
                metric_formatter(col_name, s_a_cur["min"], 0),
                metric_formatter(col_name, s_a_cur["max"], 0),
            ],
            team_b_col: [
                num_formatter(s_b_cur["games"], 0),
                metric_formatter(col_name, s_b_cur["avg"], 1),
                metric_formatter(col_name, s_b_cur["med"], 1),
                metric_formatter(col_name, s_b_cur["min"], 0),
                metric_formatter(col_name, s_b_cur["max"], 0),
            ],
        },
        columns=cols,
    )

    all_df = pd.DataFrame(
        {
            "Métrica": rows_names,
            team_a_col: [
                num_formatter(s_a_all["games"], 0),
                metric_formatter(col_name, s_a_all["avg"], 1),
                metric_formatter(col_name, s_a_all["med"], 1),
                metric_formatter(col_name, s_a_all["min"], 0),
                metric_formatter(col_name, s_a_all["max"], 0),
            ],
            team_b_col: [
                num_formatter(s_b_all["games"], 0),
                metric_formatter(col_name, s_b_all["avg"], 1),
                metric_formatter(col_name, s_b_all["med"], 1),
                metric_formatter(col_name, s_b_all["min"], 0),
                metric_formatter(col_name, s_b_all["max"], 0),
            ],
        },
        columns=cols,
    )
    return cur, all_df


def h2h_numeric_series(df_h2h: pd.DataFrame, col_name: str) -> pd.Series:
    if df_h2h is None or df_h2h.empty or str(col_name) not in df_h2h.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df_h2h.get(col_name), errors="coerce").dropna()


def normalize_p_ml_scope(p_ml_scope: Any) -> float:
    try:
        p = float(p_ml_scope)
    except (TypeError, ValueError):
        return 0.0
    if not (0.0 <= p <= 1.0):
        return 0.0
    return float(p)


def build_ml_totals_model_payload(
    team_a: str,
    team_b: str,
    totals_for_lines: Dict[str, Any],
    combo_kills_lines: List[float],
    combo_time_lines: List[float],
    p_ml_scope: Any,
    lines_table_with_p_ml_fn: Callable[[str, List[float], Any, float], pd.DataFrame],
    display_ml_totals_table_fn: Callable[[pd.DataFrame, str], pd.DataFrame],
) -> Dict[str, Any]:
    metrics = [
        ("kills", "Kills", list(combo_kills_lines or []), "ML + Totais (Kills)", "total_kills"),
        ("time", "Tempo", list(combo_time_lines or []), "ML + Totais (Tempo)", "game_time_min"),
    ]
    has_any_lines = any(lines for _, _, lines, _, _ in metrics)
    p_ml_combo = normalize_p_ml_scope(p_ml_scope)
    out_tabs: List[Dict[str, Any]] = []

    for metric_key, label, lines_list, context_title, context_col in metrics:
        if not lines_list:
            out_tabs.append(
                {
                    "metric": metric_key,
                    "label": label,
                    "has_lines": False,
                    "lines": [],
                    "team_a_label": f"Time 1: {team_a}",
                    "team_b_label": f"Time 2: {team_b}",
                    "table_a": pd.DataFrame(),
                    "table_b": pd.DataFrame(),
                    "context_title": context_title,
                    "context_col": context_col,
                }
            )
            continue

        total_obj = totals_for_lines.get(metric_key)
        df_full_a = lines_table_with_p_ml_fn(metric_key, lines_list, total_obj, p_ml_combo)
        df_full_b = lines_table_with_p_ml_fn(metric_key, lines_list, total_obj, 1.0 - p_ml_combo)
        out_tabs.append(
            {
                "metric": metric_key,
                "label": label,
                "has_lines": True,
                "lines": list(lines_list),
                "team_a_label": f"Time 1: {team_a}",
                "team_b_label": f"Time 2: {team_b}",
                "table_a": display_ml_totals_table_fn(df_full_a, f"Time 1 ({team_a})"),
                "table_b": display_ml_totals_table_fn(df_full_b, f"Time 2 ({team_b})"),
                "context_title": context_title,
                "context_col": context_col,
            }
        )

    return {
        "has_any_lines": bool(has_any_lines),
        "p_ml_combo": float(p_ml_combo),
        "tabs": out_tabs,
    }


def build_ml_totals_laplace_payload(
    team_a: str,
    team_b: str,
    combo_kills_lines: List[float],
    combo_time_lines: List[float],
    df_a_all: pd.DataFrame,
    df_b_all: pd.DataFrame,
    df_h2h_a: pd.DataFrame,
    df_h2h_b: pd.DataFrame,
    event_block_fn: Callable[..., pd.DataFrame],
    floor_odd: float = 1.10,
) -> Dict[str, Any]:
    def _ev_factory(value_col: str, line: float, want_over: bool):
        def _ev(df: pd.DataFrame):
            win = pd.to_numeric(df.get("win"), errors="coerce") > 0.5
            val = pd.to_numeric(df.get(value_col), errors="coerce")
            if want_over:
                return win & (val > float(line))
            return win & (val < float(line))

        return _ev

    combos: List[Tuple[str, str, float]] = []
    combos.extend([("ML + Kills", "total_kills", float(ln)) for ln in list(combo_kills_lines or [])])
    combos.extend([("ML + Tempo", "game_time_min", float(ln)) for ln in list(combo_time_lines or [])])

    out_rows: List[Dict[str, Any]] = []
    for title, value_col, line in combos:
        ev_over = _ev_factory(value_col, float(line), True)
        ev_under = _ev_factory(value_col, float(line), False)

        row = {
            "title": str(title),
            "line": float(line),
            "team_a_label": f"Time de referência (Time 1): {team_a}",
            "team_b_label": f"Time de referência (Time 2): {team_b}",
            "tbl_over_a": event_block_fn(
                df_a_all,
                df_a_all.tail(15),
                df_a_all.tail(10),
                df_a_all.tail(5),
                df_h2h_a,
                df_h2h_a.tail(15),
                df_h2h_a.tail(10),
                df_h2h_a.tail(5),
                event_fn=ev_over,
                label_yes="ML&Over",
                label_no="Outros",
                floor_odd=float(floor_odd),
            ),
            "tbl_under_a": event_block_fn(
                df_a_all,
                df_a_all.tail(15),
                df_a_all.tail(10),
                df_a_all.tail(5),
                df_h2h_a,
                df_h2h_a.tail(15),
                df_h2h_a.tail(10),
                df_h2h_a.tail(5),
                event_fn=ev_under,
                label_yes="ML&Under",
                label_no="Outros",
                floor_odd=float(floor_odd),
            ),
            "tbl_over_b": event_block_fn(
                df_b_all,
                df_b_all.tail(15),
                df_b_all.tail(10),
                df_b_all.tail(5),
                df_h2h_b,
                df_h2h_b.tail(15),
                df_h2h_b.tail(10),
                df_h2h_b.tail(5),
                event_fn=ev_over,
                label_yes="ML&Over",
                label_no="Outros",
                floor_odd=float(floor_odd),
            ),
            "tbl_under_b": event_block_fn(
                df_b_all,
                df_b_all.tail(15),
                df_b_all.tail(10),
                df_b_all.tail(5),
                df_h2h_b,
                df_h2h_b.tail(15),
                df_h2h_b.tail(10),
                df_h2h_b.tail(5),
                event_fn=ev_under,
                label_yes="ML&Under",
                label_no="Outros",
                floor_odd=float(floor_odd),
            ),
        }
        out_rows.append(row)

    return {"has_any_lines": bool(out_rows), "rows": out_rows}


def build_audit_base_options(
    team_a: str,
    team_b: str,
    df_league_year: pd.DataFrame,
    df_league_15: pd.DataFrame,
    df_league_10: pd.DataFrame,
    df_league_5: pd.DataFrame,
    df_h2h_year: pd.DataFrame,
    df_h2h_15: pd.DataFrame,
    df_h2h_10: pd.DataFrame,
    df_h2h_5: pd.DataFrame,
    df_a_all: pd.DataFrame,
    df_b_all: pd.DataFrame,
) -> List[Tuple[str, pd.DataFrame]]:
    return [
        ("Liga - Ano todo (união)", df_league_year),
        ("Liga - Últimos 15 (15+15)", df_league_15),
        ("Liga - Últimos 10 (10+10)", df_league_10),
        ("Liga - Últimos 5 (5+5)", df_league_5),
        ("H2H - Ano todo", df_h2h_year),
        ("H2H - Últimos 15", df_h2h_15),
        ("H2H - Últimos 10", df_h2h_10),
        ("H2H - Últimos 5", df_h2h_5),
        (f"{team_a} - Ano todo (todos confrontos)", df_a_all),
        (f"{team_b} - Ano todo (todos confrontos)", df_b_all),
    ]


def build_audit_market_options(
    team_a: str,
    team_b: str,
    kills: List[float],
    towers: List[float],
    dragons: List[float],
    barons: List[float],
    inhib: List[float],
    time_m: List[float],
    team_a_kills_lines: List[float],
    team_b_kills_lines: List[float],
) -> Dict[str, Tuple[str, List[float]]]:
    return {
        "Total Kills": ("total_kills", list(kills or [])),
        "Total Torres": ("total_towers", list(towers or [])),
        "Total Dragões": ("total_dragons", list(dragons or [])),
        "Total Barões": ("total_nashors", list(barons or [])),
        "Total Inibidores": ("total_inhibitors", list(inhib or [])),
        "Tempo": ("game_time_min", list(time_m or [])),
        "Total Dragões Elementais": ("total_elementaldrakes", list(dragons or [])),
        f"Kills {team_a}": ("kills_for", list(team_a_kills_lines or [])),
        f"Kills {team_b}": ("kills_for", list(team_b_kills_lines or [])),
    }


def build_audit_view(
    df_base: pd.DataFrame,
    col_name: str | None,
    line_val: float | None,
    time_formatter: Callable[[float], str],
) -> pd.DataFrame:
    if df_base is None or not isinstance(df_base, pd.DataFrame) or df_base.empty:
        return pd.DataFrame()
    view = df_base.copy()
    cols_show = ["date", "league", "split", "playoffs", "team", "opponent"]
    if col_name:
        cols_show.append(str(col_name))
    cols_show = [c for c in cols_show if c in view.columns]

    val = None
    if col_name and col_name in view.columns:
        val = pd.to_numeric(view[col_name], errors="coerce")

    if line_val is not None and col_name and col_name in view.columns and val is not None:
        view["OU"] = "Push"
        view.loc[val > float(line_val), "OU"] = "Acima"
        view.loc[val < float(line_val), "OU"] = "Abaixo"
        cols_show = [*cols_show, "OU"]

    if col_name == "game_time_min" and col_name in view.columns and val is not None:
        view[col_name] = val.apply(lambda x: time_formatter(float(x)) if pd.notna(x) else "-")

    if "date" in view.columns:
        view = view.sort_values("date", ascending=False)
    out = view[cols_show].copy()
    return out.rename(columns={"league": "Campeonato"})


def build_simple_audit_view(
    df_base: pd.DataFrame,
    col_name: str | None,
    line_val: float | None,
    time_formatter: Callable[[float], str],
) -> pd.DataFrame:
    if df_base is None or not isinstance(df_base, pd.DataFrame) or df_base.empty:
        return pd.DataFrame()
    view = df_base.copy()
    cols_show = ["date", "league", "team", "opponent", "side", "win"]
    if col_name:
        cols_show.append(str(col_name))
    cols_show = [c for c in cols_show if c in view.columns]

    val = None
    if col_name and col_name in view.columns:
        val = pd.to_numeric(view[col_name], errors="coerce")

    if line_val is not None and col_name and col_name in view.columns and val is not None:
        view["OU"] = "Push"
        view.loc[val > float(line_val), "OU"] = "Acima"
        view.loc[val < float(line_val), "OU"] = "Abaixo"
        cols_show = [*cols_show, "OU"]

    if "win" in view.columns:
        win_s = pd.to_numeric(view["win"], errors="coerce")
        view["win"] = win_s.map(lambda x: "Vitória" if pd.notna(x) and float(x) > 0.5 else ("Derrota" if pd.notna(x) else "-"))
    if "date" in view.columns:
        dt = pd.to_datetime(view["date"], errors="coerce")
        view["date"] = dt.map(lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "-")
    if col_name == "game_time_min" and col_name in view.columns and val is not None:
        view[col_name] = val.apply(lambda x: time_formatter(float(x)) if pd.notna(x) else "-")

    rename_map = {
        "date": "Data",
        "league": "Campeonato",
        "team": "Time",
        "opponent": "Adversário",
        "side": "Lado",
        "win": "Resultado",
    }
    if col_name:
        rename_map[str(col_name)] = "Valor"
    out = view[cols_show].copy()
    if "Data" not in out.columns and "date" in out.columns:
        out = out.sort_values("date", ascending=False)
    return out.rename(columns=rename_map)


def build_quick_metric_options(
    team_a: str,
    team_b: str,
    kills: List[float],
    towers: List[float],
    dragons: List[float],
    barons: List[float],
    inhib: List[float],
    time_m: List[float],
    team_a_kills_lines: List[float],
    team_b_kills_lines: List[float],
    team_sel: str,
) -> Dict[str, Tuple[str, str, List[float]]]:
    return {
        "Kills (Total)": ("kills", "total_kills", list(kills or [])),
        "Torres (Total)": ("towers", "total_towers", list(towers or [])),
        "Dragões (Total)": ("dragons", "total_dragons", list(dragons or [])),
        "Barões (Total)": ("barons", "total_nashors", list(barons or [])),
        "Inibidores (Total)": ("inhibitors", "total_inhibitors", list(inhib or [])),
        "Tempo": ("time", "game_time_min", list(time_m or [])),
        "Dragões Elementais": ("dragons_elemental", "total_elementaldrakes", list(dragons or [])),
        "Kills do Time selecionado": (
            "team_kills",
            "kills_for",
            list(team_a_kills_lines if str(team_sel) == str(team_a) else team_b_kills_lines or []),
        ),
    }


def quick_team_frame(team_is_a: bool, df_a_all: pd.DataFrame, df_b_all: pd.DataFrame, df_h2h_a: pd.DataFrame, df_h2h_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (df_a_all, df_h2h_a) if bool(team_is_a) else (df_b_all, df_h2h_b)


def quick_outcome_probs(
    p_ml_scope: float,
    team_is_a: bool,
    want_win: Optional[bool],
    df_team: pd.DataFrame,
    side_key: Optional[str],
) -> Tuple[float, Optional[float]]:
    p_ml_base = float(p_ml_scope) if math.isfinite(float(p_ml_scope)) else 0.5
    p_team_win = p_ml_base if bool(team_is_a) else (1.0 - p_ml_base)
    if want_win is None:
        p_outcome = 1.0
    else:
        p_outcome = p_team_win if bool(want_win) else (1.0 - p_team_win)

    p_side_ctx: Optional[float] = None
    side_key_norm = str(side_key or "").strip().lower()
    if "side" in df_team.columns and side_key_norm in {"blue", "red"}:
        side_ser = df_team["side"].astype(str).str.strip().str.lower()
        side_n = int((side_ser.isin(["blue", "red"])).sum())
        side_k = int((side_ser == side_key_norm).sum())
        if side_n > 0:
            p_side_ctx = (float(side_k) + 1.0) / (float(side_n) + 2.0)
    return float(p_outcome), p_side_ctx


def build_quick_selector_rows(
    df_team: pd.DataFrame,
    df_h2h_team: pd.DataFrame,
    value_col: str,
    line_f: float,
    side_key: Optional[str],
    want_win: Optional[bool],
    p_outcome: float,
    p_side_ctx: Optional[float],
    p_over: Optional[float],
    laplace_pick: str,
    mix_w_model: float,
    mix_w_laplace: float,
    mix_w_sc: float,
    metric_key: str,
    format_num_fn: Callable[[Any, int], str],
    line_formatter_fn: Callable[[str, float], str],
) -> List[Dict[str, Any]]:
    def _tail_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
        if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
            return pd.DataFrame([])
        return df.tail(int(n)).copy()

    def _win_series(df: pd.DataFrame) -> pd.Series:
        if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
            return pd.Series(dtype=float)
        if "win" in df.columns:
            w = pd.to_numeric(df.get("win"), errors="coerce")
            if w.notna().any():
                return w
        return pd.to_numeric(df.get("result"), errors="coerce")

    def _ctx_counts(df: pd.DataFrame, is_over: bool) -> Tuple[int, int]:
        if df is None or (not isinstance(df, pd.DataFrame)) or df.empty or value_col not in df.columns:
            return 0, 0
        val = pd.to_numeric(df.get(value_col), errors="coerce")
        win = _win_series(df)
        mask = val.notna() & win.notna()
        side_key_norm = str(side_key or "").strip().lower()
        if "side" in df.columns and side_key_norm in {"blue", "red"}:
            s = df["side"].astype(str).str.strip().str.lower()
            mask &= (s == side_key_norm)
        if want_win is True:
            mask &= (win > 0.5)
        elif want_win is False:
            mask &= (win <= 0.5)
        n = int(mask.sum())
        if n <= 0:
            return 0, 0
        if bool(is_over):
            k = int(((val > float(line_f)) & mask).sum())
        else:
            k = int(((val < float(line_f)) & mask).sum())
        return k, n

    def _lap_odds(k: int, n: int) -> Optional[float]:
        if n <= 0:
            return None
        return (float(n) + 2.0) / (float(k) + 1.0)

    def _fmt_kn(k: int, n: int) -> str:
        if n <= 0:
            return "-"
        pct = 100.0 * (float(k) / float(n))
        return f"{k}/{n} ({pct:.0f}%)"

    def _odd_to_p(odd: Any) -> Optional[float]:
        try:
            o = float(odd)
            if not math.isfinite(o) or o <= 1.0:
                return None
            return 1.0 / o
        except (TypeError, ValueError):
            return None

    def _p_to_odd(p: Any) -> Optional[float]:
        try:
            pp = float(p)
            if not math.isfinite(pp) or pp <= 0.0 or pp >= 1.0:
                return None
            return 1.0 / pp
        except (TypeError, ValueError):
            return None

    def _pick_hist_odd(side_token: str, hist_map: Dict[str, Optional[float]]) -> Optional[float]:
        pick = str(laplace_pick or "Liga 15").strip()

        def _get(scope: str, rec: str) -> Optional[float]:
            return hist_map.get(f"{scope}_{rec}_{side_token}")

        if pick.startswith("Liga "):
            rec = pick.split(" ", 1)[1].strip()
            return _get("Liga", rec)
        if pick.startswith("H2H "):
            rec = pick.split(" ", 1)[1].strip()
            return _get("H2H", rec)
        if pick.startswith("Média "):
            rec = pick.split(" ", 1)[1].split()[-1].strip()
            o1 = _get("Liga", rec)
            o2 = _get("H2H", rec)
            p1 = _odd_to_p(o1)
            p2 = _odd_to_p(o2)
            if p1 is None and p2 is None:
                return None
            if p1 is None:
                return o2
            if p2 is None:
                return o1
            return _p_to_odd(0.5 * (p1 + p2))
        return _get("Liga", "15")

    def _mix_odd(odd_model: Any, odd_hist: Any, p_stats: Any) -> Optional[float]:
        src: List[Tuple[float, float]] = []
        p_m = _odd_to_p(odd_model)
        p_h = _odd_to_p(odd_hist)
        if p_m is not None:
            src.append((p_m, float(mix_w_model)))
        if p_h is not None:
            src.append((p_h, float(mix_w_laplace)))
        try:
            if p_stats is not None and math.isfinite(float(p_stats)):
                src.append((float(p_stats), float(mix_w_sc)))
        except (TypeError, ValueError):
            pass
        if not src:
            return None
        total_w = sum(max(0.0, w) for _, w in src)
        if total_w <= 0:
            total_w = float(len(src))
            src = [(p, 1.0) for p, _ in src]
        p_mix = sum(p * max(0.0, w) for p, w in src) / float(total_w)
        return _p_to_odd(p_mix)

    recortes = [
        ("Liga", "Ano", df_team),
        ("Liga", "15", _tail_df(df_team, 15)),
        ("Liga", "10", _tail_df(df_team, 10)),
        ("Liga", "5", _tail_df(df_team, 5)),
        ("H2H", "Ano", df_h2h_team),
        ("H2H", "15", _tail_df(df_h2h_team, 15)),
        ("H2H", "10", _tail_df(df_h2h_team, 10)),
        ("H2H", "5", _tail_df(df_h2h_team, 5)),
    ]

    result_rows: List[Dict[str, Any]] = []
    for ou_key, ou_label in [("Over", "Acima"), ("Under", "Abaixo")]:
        is_over = ou_key == "Over"
        hist_map: Dict[str, Any] = {}
        for scope, rec, df_rec in recortes:
            k, n = _ctx_counts(df_rec, is_over=is_over)
            hist_map[f"{scope}_{rec}_{ou_key}"] = _lap_odds(k, n)
            hist_map[f"{scope}_{rec}_{ou_key}_k"] = int(k)
            hist_map[f"{scope}_{rec}_{ou_key}_n"] = int(n)

        k_l_ano = int(hist_map.get(f"Liga_Ano_{ou_key}_k", 0) or 0)
        n_l_ano = int(hist_map.get(f"Liga_Ano_{ou_key}_n", 0) or 0)
        k_h_ano = int(hist_map.get(f"H2H_Ano_{ou_key}_k", 0) or 0)
        n_h_ano = int(hist_map.get(f"H2H_Ano_{ou_key}_n", 0) or 0)

        odd_hist = _pick_hist_odd(ou_key, hist_map)
        stats_probs: List[float] = []
        if n_l_ano > 0:
            stats_probs.append(float(k_l_ano) / float(n_l_ano))
        if n_h_ano > 0:
            stats_probs.append(float(k_h_ano) / float(n_h_ano))
        p_stats = float(sum(stats_probs) / float(len(stats_probs))) if stats_probs else None

        odd_model = None
        if p_over is not None and math.isfinite(float(p_over)):
            p_metric = float(p_over) if is_over else float(max(0.0, 1.0 - float(p_over)))
            p_evt = float(p_metric) * float(max(0.0, min(1.0, float(p_outcome))))
            if p_side_ctx is not None and math.isfinite(float(p_side_ctx)):
                p_evt *= float(max(0.0, min(1.0, float(p_side_ctx))))
            odd_model = _p_to_odd(p_evt)

        odd_mix = _mix_odd(odd_model, odd_hist, p_stats)
        line_disp = line_formatter_fn(metric_key, float(line_f))

        result_rows.append(
            {
                "Linha": line_disp,
                "Lado": ou_label,
                "Stats (Liga Ano)": _fmt_kn(k_l_ano, n_l_ano),
                "Stats (H2H Ano)": _fmt_kn(k_h_ano, n_h_ano),
                "Odd (Modelo)": format_num_fn(odd_model, 2) if odd_model is not None and math.isfinite(float(odd_model)) else "-",
                "Odd (Hist)": format_num_fn(odd_hist, 2) if odd_hist is not None and math.isfinite(float(odd_hist)) else "-",
                "Odd (Mix)": format_num_fn(odd_mix, 2) if odd_mix is not None and math.isfinite(float(odd_mix)) else "-",
            }
        )
    return result_rows
