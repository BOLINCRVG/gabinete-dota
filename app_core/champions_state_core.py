"""State helpers for the Champions tab."""

from __future__ import annotations

from typing import Any, MutableMapping


def ensure_champions_filter_state(session_state: MutableMapping[str, Any]) -> None:
    for key in ("ch_year", "ch_league", "ch_split", "ch_playoffs", "ch_team"):
        if key not in session_state or session_state.get(key) in ("All", "None", None):
            session_state[key] = ""


def reset_champions_league_dependents(session_state: MutableMapping[str, Any]) -> None:
    session_state["ch_split"] = ""
    session_state["ch_team"] = ""


def resolve_blue_side_choice(
    session_state: MutableMapping[str, Any],
    *,
    team_a: str,
    team_b: str,
    ui_key: str = "ch_blue_team_ui",
    state_key: str = "ch_blue_team",
) -> tuple[list[str], int]:
    options = [str(team_a), str(team_b)]
    if len(options) < 2:
        options = [str(team_a or "Time 1"), str(team_b or "Time 2")]
    current = str(session_state.get(state_key, "") or "")
    if current not in options:
        current = options[0]
    ui_current = str(session_state.get(ui_key, current) or "")
    if ui_current not in options:
        session_state[ui_key] = current
        ui_current = current
    try:
        index = options.index(ui_current)
    except Exception:
        index = 0
    return options, int(index)


def sync_blue_side_choice(
    session_state: MutableMapping[str, Any],
    *,
    selected_value: str,
    state_key: str = "ch_blue_team",
) -> None:
    session_state[state_key] = str(selected_value or "")
