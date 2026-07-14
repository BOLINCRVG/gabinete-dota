"""Session-state helpers for the Agenda/Jogos tab."""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

from app_core.line_store import load_json_store, persist_json_store


_SCHEDULE_LINE_FIELDS = (
    "lines_kills",
    "lines_towers",
    "lines_dragons",
    "lines_barons",
    "lines_inhib",
    "lines_time",
    "teamA_kills_line_text",
    "teamB_kills_line_text",
    "hc_kills_team",
    "hc_kills_text",
    "hc_towers_text",
    "hc_dragons_text",
    "combo_kills_text",
    "combo_time_text",
)

_SCHEDULE_STORE_PATH_KEY = "_schedule_runtime_store_path"
_SCHEDULE_STORE_LOADED_KEY = "_schedule_runtime_store_loaded"


def _build_schedule_runtime_payload(session_state: MutableMapping[str, Any]) -> dict[str, Any]:
    return {
        "games_list": list(session_state.get("games_list") or []),
        "plays_schedule_text": str(session_state.get("plays_schedule_text", "") or ""),
        "plays_pending_games_import": list(session_state.get("plays_pending_games_import") or []),
    }


def _resolve_schedule_store_path(session_state: MutableMapping[str, Any], store_path: str | Path | None = None) -> Path | None:
    if store_path is not None:
        try:
            return Path(store_path)
        except Exception:
            return None
    raw = str(session_state.get(_SCHEDULE_STORE_PATH_KEY, "") or "").strip()
    if not raw:
        return None
    try:
        return Path(raw)
    except Exception:
        return None


def bind_schedule_store_path(session_state: MutableMapping[str, Any], store_path: str | Path) -> str:
    """Register the JSON store path for schedule persistence in session state."""
    text = str(store_path)
    session_state[_SCHEDULE_STORE_PATH_KEY] = text
    return text


def persist_schedule_runtime_state(
    session_state: MutableMapping[str, Any],
    store_path: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize and write current schedule state (games, text, imports) to the JSON store."""
    target = _resolve_schedule_store_path(session_state, store_path)
    payload = _build_schedule_runtime_payload(session_state)
    if target is not None:
        persist_json_store(target, payload)
    return payload


def restore_schedule_runtime_state(
    session_state: MutableMapping[str, Any],
    store_path: str | Path,
) -> dict[str, Any]:
    """Load schedule state from the JSON store into session state (idempotent after first load)."""
    bind_schedule_store_path(session_state, store_path)
    if bool(session_state.get(_SCHEDULE_STORE_LOADED_KEY, False)):
        return _build_schedule_runtime_payload(session_state)
    loaded = load_json_store(Path(store_path))
    session_state["games_list"] = list(loaded.get("games_list") or [])
    session_state["plays_schedule_text"] = str(loaded.get("plays_schedule_text", "") or "")
    session_state["plays_pending_games_import"] = list(loaded.get("plays_pending_games_import") or [])
    session_state[_SCHEDULE_STORE_LOADED_KEY] = True
    return _build_schedule_runtime_payload(session_state)


def ensure_schedule_state(session_state: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    """Initialize games_list if absent and return it."""
    games = session_state.get("games_list")
    if not isinstance(games, list):
        games = []
        session_state["games_list"] = games
    return games


def get_schedule_games(session_state: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    """Return a copy of the current scheduled games list."""
    return list(ensure_schedule_state(session_state))


def append_schedule_games(session_state: MutableMapping[str, Any], games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append new games to the existing list and persist."""
    merged = get_schedule_games(session_state) + list(games or [])
    session_state["games_list"] = merged
    persist_schedule_runtime_state(session_state)
    return merged


def save_schedule_games(session_state: MutableMapping[str, Any], games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace the games list with the new value and persist to disk."""
    normalized = list(games or [])
    session_state["games_list"] = normalized
    persist_schedule_runtime_state(session_state)
    return normalized


def clear_schedule_games(session_state: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    """Clear all scheduled games and persist the empty state."""
    session_state["games_list"] = []
    persist_schedule_runtime_state(session_state)
    return []


def get_schedule_default_league(session_state: MutableMapping[str, Any]) -> str:
    """Return the default league to use when parsing schedule text."""
    return str(session_state.get("schedule_default_league", "") or "")


def get_schedule_text(session_state: MutableMapping[str, Any]) -> str:
    """Return the raw schedule text pasted by the user."""
    return str(session_state.get("plays_schedule_text", "") or "")


def set_schedule_text(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Persist the raw schedule text and trigger a store write."""
    text = str(value or "")
    session_state["plays_schedule_text"] = text
    persist_schedule_runtime_state(session_state)
    return text


def clear_schedule_text(session_state: MutableMapping[str, Any]) -> str:
    """Clear the schedule text input and persist."""
    session_state["plays_schedule_text"] = ""
    persist_schedule_runtime_state(session_state)
    return ""


def get_schedule_line_fields() -> tuple[str, ...]:
    """Return the canonical list of per-game line input field names."""
    return _SCHEDULE_LINE_FIELDS


def init_schedule_line_inputs(
    session_state: MutableMapping[str, Any],
    prefix: str,
    saved_lines: dict[str, Any] | None,
    *,
    fields: tuple[str, ...] = _SCHEDULE_LINE_FIELDS,
) -> dict[str, str]:
    """Initialize per-game line inputs in session state from saved values (if not already set)."""
    out: dict[str, str] = {}
    source = dict(saved_lines or {})
    for field in fields:
        key = f"{prefix}{field}"
        if key not in session_state:
            session_state[key] = str(source.get(field, "") or "")
        out[field] = str(session_state.get(key, "") or "")
    return out


def get_schedule_line_inputs(
    session_state: MutableMapping[str, Any],
    prefix: str,
    *,
    fields: tuple[str, ...] = _SCHEDULE_LINE_FIELDS,
) -> dict[str, str]:
    """Return the current values of all per-game line inputs for the given prefix."""
    return {field: str(session_state.get(f"{prefix}{field}", "") or "") for field in fields}


def resolve_schedule_line_hc_team(
    session_state: MutableMapping[str, Any],
    prefix: str,
    *,
    team_a: Any,
    team_b: Any,
    write_back: bool = True,
) -> tuple[str, list[str]]:
    """Return (current_hc_team, options) for a game's HC kills selector, normalizing invalid values."""
    options = [str(team_a or ""), str(team_b or "")]
    key = f"{prefix}hc_kills_team"
    current = str(session_state.get(key) or team_a or "")
    if current not in options:
        current = str(team_a or "")
    if bool(write_back):
        session_state[key] = current
    return current, options


def set_schedule_pending_load(session_state: MutableMapping[str, Any], payload: dict[str, Any]) -> None:
    """Queue a schedule game payload to be loaded on the next render cycle."""
    session_state["pending_load"] = dict(payload or {})


def get_pending_schedule_import(session_state: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    """Return the list of games pending import from parsed schedule text."""
    pending = session_state.get("plays_pending_games_import")
    if not isinstance(pending, list):
        pending = []
        session_state["plays_pending_games_import"] = pending
    return list(pending)


def save_pending_schedule_import(session_state: MutableMapping[str, Any], games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist a list of raw schedule games awaiting user confirmation."""
    normalized = list(games or [])
    session_state["plays_pending_games_import"] = normalized
    persist_schedule_runtime_state(session_state)
    return normalized


def clear_pending_schedule_import(session_state: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    """Clear the pending import queue after games have been confirmed or discarded."""
    session_state["plays_pending_games_import"] = []
    persist_schedule_runtime_state(session_state)
    return []


def ensure_schedule_alias_state(session_state: MutableMapping[str, Any]) -> dict[str, str]:
    """Initialize the team alias dict if absent and return it."""
    aliases = session_state.get("team_aliases_user")
    if not isinstance(aliases, dict):
        aliases = {}
        session_state["team_aliases_user"] = aliases
    return aliases


def get_schedule_aliases(session_state: MutableMapping[str, Any]) -> dict[str, str]:
    """Return the current user-defined team alias mapping."""
    return dict(ensure_schedule_alias_state(session_state))


def save_schedule_aliases(session_state: MutableMapping[str, Any], aliases: dict[str, Any]) -> dict[str, str]:
    """Persist a new alias mapping, filtering out blank keys/values."""
    normalized = {str(k): str(v) for k, v in dict(aliases or {}).items() if str(k).strip() and str(v).strip()}
    session_state["team_aliases_user"] = normalized
    return dict(normalized)


def has_schedule_unknown_aliases(session_state: MutableMapping[str, Any]) -> bool:
    """Return True if any parsed team names didn't match the CSV and need manual aliasing."""
    return bool(session_state.get("agenda_has_unknown_aliases", False))


def set_schedule_unknown_aliases(session_state: MutableMapping[str, Any], value: Any) -> bool:
    """Set the flag indicating there are unresolved team aliases."""
    flag = bool(value)
    session_state["agenda_has_unknown_aliases"] = flag
    return flag
