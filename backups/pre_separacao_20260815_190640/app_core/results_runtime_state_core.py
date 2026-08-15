from __future__ import annotations

from typing import Any, MutableMapping, Sequence


def _normalize_year_opt(raw_value: Any, *, years_list: list[int], default_year: int) -> Any:
    if str(raw_value).strip().lower() == "all":
        return "All"
    try:
        year_value = int(float(raw_value))
    except (TypeError, ValueError):
        return int(default_year)
    if years_list and year_value not in years_list:
        return int(default_year)
    return int(year_value)


def resolve_results_runtime_settings(
    session_state: MutableMapping[str, Any],
    *,
    season_year: int,
    years: Sequence[int] | None,
) -> dict[str, Any]:
    years_list = [int(y) for y in (years or [])]
    default_year = int(season_year)
    if years_list:
        default_year = int(season_year) if int(season_year) in years_list else int(years_list[-1])

    defaults = {
        "totals_mode": "map",
        "ml_mode": "map",
        "map_mode_ui": "avg",
        "league_mode": "auto",
        "year_opt": default_year,
        "split_opt": "All",
        "playoffs_opt": "all",
        "fixed_league": "All",
    }
    for key, value in defaults.items():
        session_state.setdefault(key, value)

    year_opt = _normalize_year_opt(
        session_state.get("year_opt", default_year),
        years_list=years_list,
        default_year=default_year,
    )
    if session_state.get("year_opt", default_year) != year_opt:
        session_state["year_opt"] = year_opt

    weight_mode = str(session_state.get("weight_mode", "Recência (half-life)") or "Recência (half-life)")
    half_life_days = None if weight_mode == "Uniforme" else session_state.get("half_life_days", 180)

    return {
        "year_opt": year_opt,
        "split_opt": str(session_state.get("split_opt", "All") or "All"),
        "playoffs_opt": str(session_state.get("playoffs_opt", "all") or "all"),
        "fixed_league": str(session_state.get("fixed_league", "All") or "All"),
        "totals_mode": str(session_state.get("totals_mode", "map") or "map"),
        "ml_mode": str(session_state.get("ml_mode", "map") or "map"),
        "map_mode_ui": str(session_state.get("map_mode_ui", "avg") or "avg"),
        "league_mode": str(session_state.get("league_mode", "auto") or "auto"),
        "weight_mode": weight_mode,
        "half_life_days": half_life_days,
        "n_sims": int(session_state.get("n_sims", 20000) or 20000),
        "default_year": default_year,
    }


def resolve_hc_kills_team(
    session_state: MutableMapping[str, Any],
    *,
    team_a: Any,
    team_b: Any,
    write_back: bool = True,
) -> tuple[str, list[str]]:
    options = [str(team_a or ""), str(team_b or "")]
    current = str(session_state.get("hc_kills_team") or team_a or "")
    if current not in options:
        current = str(team_a or "")
    if bool(write_back):
        session_state["hc_kills_team"] = current
    return current, options


def get_wf_runtime_settings(session_state: MutableMapping[str, Any]) -> dict[str, float]:
    return {
        "min_train_games": int(session_state.get("wf_min_train_games", 1500) or 1500),
        "k": float(session_state.get("wf_k", 26.0) or 26.0),
        "scale": float(session_state.get("wf_scale", 400.0) or 400.0),
    }
