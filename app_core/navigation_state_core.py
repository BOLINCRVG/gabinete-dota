"""Session-state helpers for top-level navigation and transient UI flags."""

from __future__ import annotations

from typing import Any, MutableMapping

DEFAULT_NAV_MAIN = "Plays"
DEFAULT_UI_PAGE_MODE = "Fluxo R\u00e1pido"
# Live-Bet foi FUNDIDA na página Radar (2026-07-04, pedido do usuário):
# entre-mapas e durante-o-mapa agora vivem lá. Navegações antigas pra
# "Live-Bet" caem no Radar (ver resolve_nav_page).
TOP_NAV_OPTIONS = ("Plays", "SUPER-LIVE", "Radar", "Meta", "Tecnico")


def get_nav_pending(session_state: MutableMapping[str, Any]) -> str:
    """Return the pending navigation target (empty string if none)."""
    return str(session_state.get("_nav_pending") or "")


def set_nav_pending(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Schedule a navigation to value on the next render cycle."""
    text = str(value or "")
    session_state["_nav_pending"] = text
    return text


def consume_nav_pending(session_state: MutableMapping[str, Any]) -> str:
    """Pop and return the pending nav target, clearing it from session state."""
    return str(session_state.pop("_nav_pending", "") or "")


def normalize_nav_main(value: Any, default: str = DEFAULT_NAV_MAIN) -> str:
    """Validate and return a nav_main value; falls back to default if not in TOP_NAV_OPTIONS."""
    text = str(value or default).strip() or default
    if text not in TOP_NAV_OPTIONS:
        return str(default or DEFAULT_NAV_MAIN)
    return text


def get_nav_main(session_state: MutableMapping[str, Any], default: str = DEFAULT_NAV_MAIN) -> str:
    """Return the current top-level navigation section."""
    return str(session_state.get("nav_main", default) or default)


def set_nav_main(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Set the top-level nav section after normalizing and validating the value."""
    text = normalize_nav_main(value, default=DEFAULT_NAV_MAIN)
    session_state["nav_main"] = text
    return text


def get_ui_page_mode(session_state: MutableMapping[str, Any], default: str = DEFAULT_UI_PAGE_MODE) -> str:
    """Return the active UI page mode (e.g. 'Fluxo Rápido', 'Completo')."""
    return str(session_state.get("ui_page_mode", default) or default)


def set_ui_page_mode(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Persist the selected UI page mode to session state."""
    text = str(value or DEFAULT_UI_PAGE_MODE)
    session_state["ui_page_mode"] = text
    return text


def get_flash_saved_profile_msg(session_state: MutableMapping[str, Any]) -> str:
    """Return a one-shot flash message for the profile-saved confirmation banner."""
    return str(session_state.get("_flash_saved_profile_msg") or "")


def set_flash_saved_profile_msg(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Set the flash message to display after a profile is saved."""
    text = str(value or "")
    session_state["_flash_saved_profile_msg"] = text
    return text


def consume_flash_saved_profile_msg(session_state: MutableMapping[str, Any]) -> str:
    """Read and clear the flash message in one operation."""
    text = get_flash_saved_profile_msg(session_state)
    session_state["_flash_saved_profile_msg"] = ""
    return text


def get_dismiss_update_banner(session_state: MutableMapping[str, Any]) -> bool:
    """Return True if the user has dismissed the update-available banner this session."""
    return bool(session_state.get("dismiss_update_banner", False))


def set_dismiss_update_banner(session_state: MutableMapping[str, Any], value: Any) -> bool:
    """Record whether the user has dismissed the update banner."""
    flag = bool(value)
    session_state["dismiss_update_banner"] = flag
    return flag


def resolve_nav_page(session_state: MutableMapping[str, Any]) -> str:
    pending = consume_nav_pending(session_state)
    if pending:
        if pending == "Live-Bet":
            pending = "Radar"  # página fundida no Radar
        if pending in {"Plays", "SUPER-LIVE", "Radar", "Meta", "Tecnico"}:
            set_nav_main(session_state, pending)
        else:
            set_nav_main(session_state, "Plays")
            set_ui_page_mode(session_state, "Completo")

    _cur = str(get_nav_main(session_state, "Plays") or "").strip()
    if _cur == "Calibracoes":
        set_nav_main(session_state, "Tecnico")
    elif _cur == "Live-Bet":
        set_nav_main(session_state, "Radar")
    return normalize_nav_main(get_nav_main(session_state, "Plays"))
