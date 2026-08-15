"""Small session-state helpers shared by tab renderers."""

from __future__ import annotations

from typing import Any, Iterable, MutableMapping


def ensure_blank_keys(
    session_state: MutableMapping[str, Any],
    keys: Iterable[str],
    *,
    invalid_values: tuple[Any, ...] = ("All", None, "None"),
) -> None:
    for key in keys:
        if key not in session_state or session_state.get(key) in invalid_values:
            session_state[key] = ""


def reset_keys(session_state: MutableMapping[str, Any], keys: Iterable[str], value: str = "") -> None:
    for key in keys:
        session_state[key] = value


def normalize_select_key(
    session_state: MutableMapping[str, Any],
    key: str,
    allowed_values: Iterable[Any],
    *,
    fallback: str = "",
) -> str:
    allowed = {str(v) for v in allowed_values}
    current = str(session_state.get(key, fallback) or fallback)
    if current not in allowed:
        session_state[key] = fallback
        return fallback
    return current


def select_placeholder(value: str, *, blank_label: str = "Selecione ...") -> str:
    return blank_label if str(value) == "" else str(value)


def blank_space_label(value: str) -> str:
    return " " if str(value) == "" else str(value)
