"""Session-state helpers for StatsCamps filters and compare state."""

from __future__ import annotations

from typing import Any, MutableMapping

from app_core.tab_state_core import ensure_blank_keys, normalize_select_key, reset_keys


def ensure_statscamps_filter_state(session_state: MutableMapping[str, Any]) -> None:
    ensure_blank_keys(session_state, ["sc_year", "sc_league", "sc_split", "sc_playoffs"])


def reset_statscamps_league_dependents(session_state: MutableMapping[str, Any]) -> None:
    reset_keys(session_state, ["sc_split", "sc_delta_a", "sc_delta_b"])
    session_state["sc_compare_sel"] = []


def normalize_statscamps_select_state(
    session_state: MutableMapping[str, Any],
    *,
    years: list[str],
    splits: list[str],
) -> None:
    normalize_select_key(session_state, "sc_year", ["", *list(years or [])])
    normalize_select_key(session_state, "sc_split", ["", *list(splits or [])])
    normalize_select_key(session_state, "sc_playoffs", ["", "Regular", "Playoffs"])


def get_statscamps_threshold_text(
    session_state: MutableMapping[str, Any],
    *,
    key: str,
    default: str,
) -> str:
    return str(session_state.get(key, default) or default)


def sync_statscamps_delta_targets(
    session_state: MutableMapping[str, Any],
    *,
    selected: list[str],
    default_a: str,
    default_b: str,
) -> tuple[str, str]:
    valid = list(selected or [])
    current_a = str(session_state.get("sc_delta_a", "") or "")
    current_b = str(session_state.get("sc_delta_b", "") or "")
    if current_a not in valid:
        current_a = str(default_a or "")
        session_state["sc_delta_a"] = current_a
    if current_b not in valid:
        current_b = str(default_b or "")
        session_state["sc_delta_b"] = current_b
    return current_a, current_b
