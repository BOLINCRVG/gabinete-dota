"""Runtime helpers for Resumo tab orchestration."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from app_core.stats_config import (
    combined_col_map as _stats_combined_col_map,
    default_lines_map as _stats_default_lines_map,
)


# Derivado da fonte única STATS_MARKET_CONFIG (app_core/stats_config.py) — antes
# era duplicado aqui. Adicionar/remover mercado de stats = editar lá, só.
SC_METRIC_COL: dict[str, str] = _stats_combined_col_map()


# Standard default lines the Resumo shows when the user has NOT typed a custom
# line for a metric. Constantes fixas de propósito: campo vazio sempre resolve
# pra estas, nunca pra uma linha antiga de outro confronto (mata o "fantasma
# 25.5"). Uma linha digitada para o confronto atual substitui a padrão da métrica.
# DERIVADO da fonte única STATS_MARKET_CONFIG (default_line por mercado).
RESUMO_DEFAULT_LINES: dict[str, list[float]] = _stats_default_lines_map()


# Ponte Análise de Linha (v3) → Resumo. A aba "Análise de Linha" salva as linhas
# que o usuário adiciona em session_state["v3kills_lines_{mercado}_{timeA}_{timeB}"]
# (lista de floats). O Resumo, por legado, lia de outro lugar (_lines_store, campos
# lines_kills…) que só era preenchido pela tela de linhas DESATIVADA — por isso as
# linhas adicionadas nunca chegavam ao Resumo. Mapa de métrica-do-Resumo → mercado-v3
# (atenção: inibidores é "inibs" na v3, "inhibitors" no Resumo).
_V3_MARKET_FOR_RESUMO_METRIC: dict[str, str] = {
    "kills": "kills",
    "towers": "towers",
    "dragons": "dragons",
    "barons": "barons",
    "inhibitors": "inibs",
    "time": "time",
}


def v3_lines_for_metric(session_state: Any, metric_key: str, team_a: Any, team_b: Any) -> list[float]:
    """Linhas que o usuário adicionou na aba Análise de Linha (v3) para o confronto."""
    v3m = _V3_MARKET_FOR_RESUMO_METRIC.get(str(metric_key or "").strip().lower())
    if not v3m:
        return []
    ta = str(team_a or "").strip()
    tb = str(team_b or "").strip()
    val = session_state.get(f"v3kills_lines_{v3m}_{ta}_{tb}") if session_state is not None else None
    out: list[float] = []
    if isinstance(val, list):
        for x in val:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
    return out


def merge_with_default_lines(metric_key: str, parsed: list[float]) -> list[float]:
    """Linhas a analisar para a métrica: PADRÃO + digitadas, JUNTAS.

    Regra do usuário (2026-06-11): as linhas padrão (kills 27.5, torres 11.5,
    dragões 4.5, tempo 32.5, barões/inibidores 1.5) aparecem SEMPRE, e qualquer
    linha digitada para o confronto SOMA a elas (digitou 28.5 → mostra 27.5 E
    28.5). Como o padrão é constante, um campo vazio nunca ressuscita uma linha
    de um confronto anterior (sem "fantasma 25.5").
    """
    cleaned: list[float] = [
        float(x) for x in RESUMO_DEFAULT_LINES.get(str(metric_key).strip().lower(), [])
    ]
    for v in (parsed or []):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not any(abs(fv - float(e)) < 1e-9 for e in cleaned):
            cleaned.append(fv)
    return sorted(cleaned)


def select_h2h_games(
    team_games: pd.DataFrame | None,
    *,
    team_a: str,
    team_b: str,
    year: Any = None,
    norm_team_fn: Callable[[Any], str],
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select head-to-head games between two teams, NOT limited by split/phase.

    The Resumo H2H must span the whole season: a meeting from Split 1 still
    counts in Split 2 (and playoffs vs regular don't hide each other). So the
    only optional narrowing is by ``year``; split/playoffs filters are never
    applied here. Name matching is normalized so mojibake/casing don't drop
    real meetings. Returns ``(df_h2h_a, df_h2h_b)`` from each team's side.
    """
    empty = pd.DataFrame([])
    if not isinstance(team_games, pd.DataFrame) or team_games.empty:
        return empty, empty
    src = team_games
    if year and "year" in src.columns:
        src = src[pd.to_numeric(src["year"], errors="coerce") == int(year)]
    if not {"team", "opponent"}.issubset(src.columns):
        return empty, empty
    ka, kb = norm_team_fn(team_a), norm_team_fn(team_b)
    ta_key = src["team"].astype(str).map(norm_team_fn)
    opp_key = src["opponent"].astype(str).map(norm_team_fn)
    df_a = src[(ta_key == ka) & (opp_key == kb)].copy()
    df_b = src[(ta_key == kb) & (opp_key == ka)].copy()
    if date_col in df_a.columns:
        df_a = df_a.sort_values(date_col)
    if date_col in df_b.columns:
        df_b = df_b.sort_values(date_col)
    return df_a, df_b


def build_statscamps_scope_frame(
    *,
    enabled: bool,
    session_state: Mapping[str, Any],
    team_games: pd.DataFrame | None,
    filters_cls: Any,
    active_filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    ml_info: Mapping[str, Any] | None,
    map_mode_ui: Any,
    team_a: str,
    team_b: str,
) -> pd.DataFrame | None:
    if not enabled:
        return None
    try:
        sc_league = str(session_state.get("sc_league", "") or "").strip()
        sc_year = str(session_state.get("sc_year", "") or "").strip()
        sc_split = str(session_state.get("sc_split", "") or "").strip()
        sc_playoffs = str(session_state.get("sc_playoffs", "") or "").strip()
        sc_map = str(session_state.get("sc_map", "All maps") or "All maps").strip()

        df_sc_base = None
        if sc_league and sc_year and sc_split:
            sc_filters = filters_cls(
                year=int(sc_year),
                league=sc_league,
                split=sc_split,
                playoffs=None if not sc_playoffs else (True if sc_playoffs == "Playoffs" else False),
            )
            df_sc_base = apply_filters_fn(team_games, sc_filters).copy()
            if sc_map != "All maps" and df_sc_base is not None and "map_number" in df_sc_base.columns:
                try:
                    map_number = int(sc_map.split()[-1])
                    df_sc_base = df_sc_base[df_sc_base["map_number"] == map_number].copy()
                except (IndexError, TypeError, ValueError):
                    pass
        else:
            league_a = ""
            league_b = ""
            if isinstance(ml_info, Mapping):
                league_a = str(((ml_info.get("blue") or {}).get("league")) or "").strip()
                league_b = str(((ml_info.get("red") or {}).get("league")) or "").strip()
            league_fb = getattr(active_filters, "league", None)
            if not league_fb and league_a and league_b and league_a == league_b:
                league_fb = league_a
            sc_filters = filters_cls(
                year=getattr(active_filters, "year", None),
                league=league_fb,
                split=getattr(active_filters, "split", None),
                playoffs=getattr(active_filters, "playoffs", None),
            )
            df_sc_base = apply_filters_fn(team_games, sc_filters).copy()
            if (
                map_mode_ui
                and str(map_mode_ui).startswith("map")
                and df_sc_base is not None
                and "map_number" in df_sc_base.columns
            ):
                try:
                    map_number = int(str(map_mode_ui).replace("map", "").strip())
                    df_sc_base = df_sc_base[df_sc_base["map_number"] == map_number].copy()
                except (TypeError, ValueError):
                    pass

        if df_sc_base is None or df_sc_base.empty or "team" not in df_sc_base.columns:
            return None

        df_sc_ab = df_sc_base[df_sc_base["team"].astype(str).isin([str(team_a), str(team_b)])].copy()
        if "gameid" in df_sc_ab.columns:
            df_sc_ab = df_sc_ab.drop_duplicates(subset=["gameid"], keep="last")
        elif "gameId" in df_sc_ab.columns:
            df_sc_ab = df_sc_ab.drop_duplicates(subset=["gameId"], keep="last")
        return df_sc_ab
    except (TypeError, ValueError):
        return None


def compute_statscamps_pct(
    *,
    enabled: bool,
    df_sc_ab: pd.DataFrame | None,
    metric_key: str,
    side_key: str,
    line_val: float,
    metric_col_map: Mapping[str, str] | None = None,
) -> Optional[float]:
    if (not enabled) or df_sc_ab is None or (not isinstance(df_sc_ab, pd.DataFrame)) or df_sc_ab.empty:
        return None
    col_map = dict(metric_col_map or SC_METRIC_COL)
    col_sc = col_map.get(str(metric_key).strip().lower())
    if not col_sc:
        return None
    series = pd.to_numeric(df_sc_ab.get(col_sc), errors="coerce").dropna()
    if int(series.shape[0]) <= 0:
        return None
    try:
        line_float = float(line_val)
    except (TypeError, ValueError):
        return None
    side_low = str(side_key).strip().lower()
    if side_low == "over":
        pct = 100.0 * float((series > line_float).mean())
    else:
        pct = 100.0 * float((series < line_float).mean())
    return float(pct) if math.isfinite(float(pct)) else None


def format_statscamps_badge(pct: Optional[float], *, threshold: float) -> str:
    if pct is None:
        return "-"
    icon = "OK" if float(pct) >= float(threshold) else "-"
    return f"{icon} {float(pct):.0f}%"


def ensure_metric_lines(
    metric_key: str,
    values: Any,
    *,
    parse_lines_fn: Callable[[str], list[float]],
    parse_time_lines_fn: Callable[[str], list[float]],
) -> list[float]:
    if values is None:
        return []
    if isinstance(values, str):
        return parse_time_lines_fn(values) if str(metric_key).strip().lower() == "time" else parse_lines_fn(values)
    if isinstance(values, (list, tuple, set)):
        out: list[float] = []
        for value in values:
            try:
                float_value = float(value)
                if math.isfinite(float_value):
                    out.append(float_value)
            except (TypeError, ValueError):
                pass
        return out
    try:
        float_value = float(values)
        return [float_value] if math.isfinite(float_value) else []
    except (TypeError, ValueError):
        return []


def infer_matchup_league(*, df_league_year: pd.DataFrame | None, ml_info: Mapping[str, Any] | None) -> str:
    league_from_scope = ""
    try:
        league_vals = pd.Series(df_league_year.get("league")).dropna().astype(str).str.strip()
        if not league_vals.empty:
            league_from_scope = str(league_vals.mode().iloc[0]).upper()
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        league_from_scope = ""
    if league_from_scope:
        return league_from_scope
    try:
        league_blue = str((((ml_info or {}).get("blue") or {}).get("league") or "")).strip().upper()
        league_red = str((((ml_info or {}).get("red") or {}).get("league") or "")).strip().upper()
        return league_blue if league_blue == league_red else (league_blue or league_red)
    except (AttributeError, TypeError, ValueError):
        return ""


def build_market_trace_rows(rows: list[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        try:
            out.append(
                {
                    "mercado": str(row.get("Mercado", "")),
                    "lado": str(row.get("Lado", "")),
                    "linha": str(row.get("Linha", "")),
                    "odd_model": (float(row.get("Odd (Modelo)")) if row.get("Odd (Modelo)") is not None else None),
                    "odd_hist": (float(row.get("Odd (Hist)")) if row.get("Odd (Hist)") is not None else None),
                    "odd_mix": (float(row.get("Odd (Mix)")) if row.get("Odd (Mix)") is not None else None),
                    "obs_mix": str(row.get("Obs Mix", "")),
                    "calib_reason": str(row.get("_calib_reason", "")),
                    "calib_source": str(row.get("_calib_source", "")),
                }
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return out
