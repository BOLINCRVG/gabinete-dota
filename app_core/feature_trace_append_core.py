# -*- coding: utf-8 -*-
"""Appends feature trace events and optionally persists them to disk."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from app_core.ml_session_runtime_core import (
    get_feature_trace_state as _get_feature_trace_state,
    set_feature_trace_events as _set_feature_trace_events,
    set_feature_trace_error as _set_feature_trace_error,
)
from app_core.trace_core import append_feature_trace_event as _append_feature_trace_event


def append_feature_trace(
    *,
    feature: str,
    event: str,
    enabled: Optional[bool] = None,
    payload: Optional[Dict[str, Any]] = None,
    dedupe_signature: Optional[str] = None,
    max_items: int = 200,
    persist_files: bool = False,
    st: Any,
    ctx: Mapping[str, Any],
) -> None:
    APP_ROOT = ctx.get("APP_ROOT")
    _persist_combined_trace_file = ctx.get("_persist_combined_trace_file")
    try:
        _feature_state = _get_feature_trace_state(st.session_state)
        rows = _append_feature_trace_event(
            _feature_state.get("events", []),
            feature=str(feature),
            event=str(event),
            enabled=(None if enabled is None else bool(enabled)),
            payload=(payload or {}),
            dedupe_signature=(str(dedupe_signature) if dedupe_signature is not None else None),
            max_items=int(max_items),
        )
        _ok_at = datetime.now().isoformat(timespec="seconds")
        _set_feature_trace_events(st.session_state, rows, ok_at=_ok_at)
        if bool(persist_files):
            try:
                (Path(APP_ROOT) / "feature_trace_last.json").write_text(
                    json.dumps(
                        {
                            "generated_at": datetime.now().isoformat(timespec="seconds"),
                            "events": rows,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except (OSError, TypeError, ValueError):
                _set_feature_trace_error(
                    st.session_state,
                    "Falha ao persistir feature_trace_last.json",
                    error_at=datetime.now().isoformat(timespec="seconds"),
                )
            if callable(_persist_combined_trace_file):
                _persist_combined_trace_file()
    except (OSError, TypeError, ValueError) as exc:
        _set_feature_trace_error(
            st.session_state,
            f"{type(exc).__name__}: {exc}",
            error_at=datetime.now().isoformat(timespec="seconds"),
        )
