from __future__ import annotations

from typing import Any, Dict

from app_core.stats_config import flow_line_keys

# Core (lines_kills...lines_time) deriva do registro; extras (kills por time,
# handicaps, combos) são UI e ficam inline.
LINE_FIELDS_DEFAULTS = {
    **{key: "" for key in flow_line_keys()},
    "teamA_kills_line_text": "",
    "teamB_kills_line_text": "",
    "hc_kills_text": "",
    "hc_kills_team": "",
    "hc_towers_text": "",
    "hc_dragons_text": "",
    "combo_kills_text": "",
    "combo_time_text": "",
}


def load_lines_for_matchup(session_state: Dict[str, Any], store_all: Dict[str, Any], key: str, defaults_key: str = "__defaults__") -> None:
    store = store_all.get(key)
    if not isinstance(store, dict) or (not store):
        store = store_all.get(defaults_key, {})
        if not isinstance(store, dict):
            store = {}
    for k, dv in LINE_FIELDS_DEFAULTS.items():
        session_state[k] = str(store.get(k, dv) or "")


def save_lines_for_matchup(session_state: Dict[str, Any], store_all: Dict[str, Any], key: str) -> None:
    store_all[key] = {k: session_state.get(k, "") for k in LINE_FIELDS_DEFAULTS.keys()}

