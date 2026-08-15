"""Unified notification system for the Streamlit UI.

Replaces the mix of `st.caption / st.markdown / st.info / st.warning / st.error`
calls scattered across modules with a single helper that:

  - Picks the right Streamlit method per level
  - Keeps tone consistent (icons, leading marker)
  - Lets the caller add a tiny `detail` string shown as caption below
  - Centralizes the patterns so changing the look in the future is 1 edit

Usage
-----
    from app_core.ui_notify_core import notify

    notify(st, "info", "Selecione os dois times para continuar")
    notify(st, "success", "Modelos atualizados", detail="3 modelos retreinados em 4m32s")
    notify(st, "warn", "Modelo desatualizado", detail="Último treino há 8 dias")
    notify(st, "error", "Falha ao carregar modelo", detail=f"{type(exc).__name__}: {exc}")

Compared to raw Streamlit calls:
  notify(st, "info", msg)     ≡ st.info("ℹ {msg}")
  notify(st, "success", msg)  ≡ st.success("✓ {msg}")
  notify(st, "warn", msg)     ≡ st.warning("⚠ {msg}")
  notify(st, "error", msg)    ≡ st.error("✕ {msg}")

For a quieter caption-only message use `notify_caption(st, msg)`.
"""
from __future__ import annotations

from typing import Any, Literal, Optional


Level = Literal["info", "success", "warn", "warning", "error"]


_PREFIXES = {
    "info": "ℹ",
    "success": "✓",
    "warn": "⚠",
    "warning": "⚠",
    "error": "✕",
}


def notify(
    st: Any,
    level: Level,
    message: str,
    *,
    detail: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """Render a unified notification banner.

    level   : "info" | "success" | "warn" | "error"
    message : main one-line message
    detail  : optional caption shown below in smaller gray text
    icon    : override the default level icon (e.g., "🎯", "🔄")
    """
    lvl = str(level or "info").lower()
    prefix = icon if icon else _PREFIXES.get(lvl, "ℹ")
    body = f"{prefix} {str(message or '').strip()}".strip()

    method = {
        "info": st.info,
        "success": st.success,
        "warn": st.warning,
        "warning": st.warning,
        "error": st.error,
    }.get(lvl, st.info)
    method(body)
    if detail:
        st.caption(str(detail).strip())


def notify_caption(st: Any, message: str) -> None:
    """Quiet caption-only notification (no colored banner)."""
    st.caption(str(message or "").strip())


def notify_exception(
    st: Any,
    exc: BaseException,
    *,
    context: str = "",
) -> None:
    """Render a unified error banner for a caught exception.

    Replaces `except Exception: pass` patterns — surfaces the failure to
    the user without leaking a full traceback.
    """
    headline = str(context or "Falha ao executar a operação").strip()
    detail = f"{type(exc).__name__}: {str(exc) or '(sem mensagem)'}"
    notify(st, "error", headline, detail=detail)


__all__ = ["notify", "notify_caption", "notify_exception"]
