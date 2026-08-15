"""Detect when the visible reading is stale — e.g., user changed teams or
filters but didn't refresh the analysis. Renders a warning banner if the
current context differs from the context that produced the last reading.

Used by panels that show analytics computed from a specific context
(team_a, team_b, league, scope). When the user changes any of those
inputs, the panel can mark its current display as "stale until refreshed".
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, MutableMapping, Optional


def context_signature(context: Mapping[str, Any]) -> str:
    """Compute a stable signature for a context dict (sorted keys, json + sha)."""
    try:
        s = json.dumps(context, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        s = str(sorted(context.items()))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def get_last_signature(session_state: Mapping[str, Any], panel_key: str) -> Optional[str]:
    return str(session_state.get(f"_stale_sig_{panel_key}", "") or "") or None


def mark_panel_refreshed(
    session_state: MutableMapping[str, Any],
    panel_key: str,
    context: Mapping[str, Any],
) -> None:
    """Store the current context signature as 'last refreshed' for this panel."""
    session_state[f"_stale_sig_{panel_key}"] = context_signature(context)


def is_stale(
    session_state: Mapping[str, Any],
    panel_key: str,
    context: Mapping[str, Any],
) -> bool:
    """True if the current context differs from the last refreshed context for this panel.

    Returns False if there's no previous signature yet (first render — not stale).
    """
    last = get_last_signature(session_state, panel_key)
    if not last:
        return False
    return context_signature(context) != last


def render_stale_banner_if_needed(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    panel_key: str,
    context: Mapping[str, Any],
    message: str = "Contexto mudou desde a última leitura — clique em atualizar para recalcular.",
) -> bool:
    """Render a stale banner if the panel context changed. Returns True if rendered."""
    if not is_stale(session_state, panel_key, context):
        return False
    try:
        from app_core.ui_notify_core import notify
        notify(st, "warn", message, detail=f"Painel: {panel_key}")
    except Exception:
        st.warning(f"⚠ {message}")
    return True


__all__ = [
    "context_signature",
    "get_last_signature",
    "is_stale",
    "mark_panel_refreshed",
    "render_stale_banner_if_needed",
]
