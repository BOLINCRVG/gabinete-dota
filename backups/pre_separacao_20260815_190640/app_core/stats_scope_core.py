"""Scope resolution helpers for championship stats views."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import pandas as pd


def _resolve_scope_filters(
    *,
    session_state: Mapping[str, Any],
    filters_cls: Any,
    active_filters: Any,
    ml_info: Mapping[str, Any] | None,
    map_mode_ui: Any,
) -> dict[str, Any]:
    sc_league = str(session_state.get("sc_league", "") or "").strip()
    sc_year = str(session_state.get("sc_year", "") or "").strip()
    sc_split = str(session_state.get("sc_split", "") or "").strip()
    sc_playoffs = str(session_state.get("sc_playoffs", "") or "").strip()
    sc_map = str(session_state.get("sc_map", "All maps") or "All maps").strip()

    if sc_league and sc_year and sc_split:
        return {
            "filters": filters_cls(
                year=int(sc_year),
                league=sc_league,
                split=sc_split,
                playoffs=None if not sc_playoffs else (True if sc_playoffs == "Playoffs" else False),
            ),
            "map_selector": sc_map,
            "manual": True,
            "league": sc_league,
            "year": sc_year,
            "split": sc_split,
            "playoffs": sc_playoffs,
        }

    league_a = ""
    league_b = ""
    try:
        if ml_info is not None:
            league_a = str(((ml_info.get("blue") or {}).get("league")) or "").strip()
            league_b = str(((ml_info.get("red") or {}).get("league")) or "").strip()
    except Exception:
        league_a = ""
        league_b = ""

    league_fb = getattr(active_filters, "league", None)
    if not league_fb and league_a and league_b and league_a == league_b:
        league_fb = league_a

    return {
        "filters": filters_cls(
            year=getattr(active_filters, "year", None),
            league=league_fb,
            split=getattr(active_filters, "split", None),
            playoffs=getattr(active_filters, "playoffs", None),
        ),
        "map_selector": str(map_mode_ui or "avg"),
        "manual": False,
        "league": str(league_fb or ""),
        "year": str(getattr(active_filters, "year", "") or ""),
        "split": str(getattr(active_filters, "split", "") or ""),
        "playoffs": (
            ""
            if getattr(active_filters, "playoffs", None) is None
            else ("Playoffs" if bool(active_filters.playoffs) else "Regular")
        ),
    }


def build_championship_scope_frame(
    *,
    session_state: Mapping[str, Any],
    team_games: pd.DataFrame | None,
    filters_cls: Any,
    active_filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    ml_info: Mapping[str, Any] | None,
    map_mode_ui: Any,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    meta = _resolve_scope_filters(
        session_state=session_state,
        filters_cls=filters_cls,
        active_filters=active_filters,
        ml_info=ml_info,
        map_mode_ui=map_mode_ui,
    )
    try:
        df_scope = apply_filters_fn(team_games, meta["filters"]).copy()
    except Exception:
        return None, meta
    if df_scope is None or df_scope.empty:
        return None, meta
    map_selector = str(meta.get("map_selector", "") or "").strip()
    if map_selector and "map_number" in df_scope.columns:
        try:
            if map_selector != "All maps" and map_selector.lower() != "avg":
                map_number = int(str(map_selector).replace("Map", "").replace("map", "").strip())
                df_scope = df_scope[df_scope["map_number"] == map_number].copy()
        except Exception:
            pass
    if df_scope is None or df_scope.empty or "team" not in df_scope.columns:
        return None, meta
    return df_scope, meta


def scope_label_from_meta(meta: Mapping[str, Any] | None) -> str:
    payload = dict(meta or {})
    bits: list[str] = []
    for key in ("league", "year", "split", "playoffs", "map_selector"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            bits.append(value)
    if not bits:
        bits.append("Recorte ativo")
    suffix = "manual" if bool(payload.get("manual")) else "auto"
    return " | ".join(bits) + f" ({suffix})"


def build_relevant_championship_options(
    team_games: pd.DataFrame | None,
    *,
    filters_cls: Any,
    active_filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    team_a: str,
    team_b: str,
    selected_league: str = "",
) -> tuple[list[str], str]:
    if team_games is None or team_games.empty or "team" not in team_games.columns or "league" not in team_games.columns:
        return (["Auto"], "Auto")
    base_filters = filters_cls(
        year=getattr(active_filters, "year", None),
        league=None,
        split=getattr(active_filters, "split", None),
        playoffs=getattr(active_filters, "playoffs", None),
    )
    try:
        scoped = apply_filters_fn(team_games, base_filters).copy()
    except Exception:
        scoped = team_games.copy()
    if scoped is None or scoped.empty:
        return (["Auto"], "Auto")
    team_mask = scoped["team"].astype(str).isin([str(team_a), str(team_b)])
    relevant = scoped.loc[team_mask].copy()
    if relevant.empty:
        options = ["Auto"]
        if str(selected_league or "").strip():
            options.append(str(selected_league).strip())
        return (list(dict.fromkeys(options)), str(selected_league).strip() or "Auto")
    if "date" in relevant.columns:
        relevant["date"] = pd.to_datetime(relevant["date"], errors="coerce")
        latest_by_league = (
            relevant.dropna(subset=["date"])
            .groupby("league", dropna=False)["date"]
            .max()
            .sort_values(ascending=False)
        )
        league_list = [str(idx).strip() for idx in latest_by_league.index.tolist() if str(idx).strip()]
    else:
        league_list = [str(x).strip() for x in relevant["league"].dropna().astype(str).unique().tolist() if str(x).strip()]
    options = ["Auto", *league_list]
    selected = str(selected_league).strip()
    if selected and selected not in options:
        options.append(selected)
    if not selected:
        selected = "Auto"
    return (options, selected)
