"""Session-state helpers for sidebar settings state and actions."""

from __future__ import annotations

from typing import Any, MutableMapping


def get_settings_sidebar_state(session_state: MutableMapping[str, Any]) -> dict[str, Any]:
    return {
        "cfg_profile": str(session_state.get("cfg_profile", "Padrão") or "Padrão"),
        "ui_fast_flow": bool(session_state.get("ui_fast_flow", True)),
    }


def queue_settings_sidebar_action(session_state: MutableMapping[str, Any], key: str, value: Any = True) -> None:
    session_state[str(key)] = value
