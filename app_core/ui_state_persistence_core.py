"""Persist key UI preferences to a small local JSON file.

Streamlit `st.session_state` is per-session — close the browser tab and
lose your inputs. This module saves user-chosen keys (selected teams,
league, last draft, view preferences) to `ui_state.json` in the project
root so the next visit picks up where you left off.

Design
------
- Allow-list of keys that get persisted (no accidental leak of secrets/blobs)
- Atomic write via tempfile + os.replace
- Read errors fall back to empty dict (no crash on first run / corrupted file)
- Bounded size: stored values must be JSON-serializable primitives

Usage
-----
    from app_core.ui_state_persistence_core import (
        load_persisted_state, save_persisted_state, sync_session_with_persistence,
    )

    # At app startup:
    sync_session_with_persistence(st.session_state)

    # After a user action that should be remembered:
    save_persisted_state({"lb_team_a": "Gen.G", "lb_team_b": "T1"})
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


_DEFAULT_PATH = Path("ui_state.json")


# Keys allowed to be persisted (extend conservatively).
ALLOWED_KEYS: tuple[str, ...] = (
    # Live-bet selectors
    "lb_team_a", "lb_team_b",
    "lb_championship", "lb_championship_manual",
    "lb_blue_side_team",
    "lb_draft_compact",
    "lb_players_a", "lb_players_b",
    "lb_champs_a", "lb_champs_b",
    # Top-level
    "page_mode",
    # Last-used filters
    "ml_live_state_model_enabled",
    "ml_live_state_models_dir",
    # Resumo prefs
    "resumo_laplace_pick",
    "resumo_mix_w_model", "resumo_mix_w_laplace", "resumo_mix_w_sc",
)


def _resolve_path(path: str | Path | None = None) -> Path:
    return Path(str(path)) if path is not None else _DEFAULT_PATH


def load_persisted_state(path: str | Path | None = None) -> Dict[str, Any]:
    """Read persisted state from disk. Returns {} on any error / missing file."""
    p = _resolve_path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_serializable(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_serializable(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_serializable(x) for k, x in v.items())
    return False


def save_persisted_state(updates: Mapping[str, Any], *, path: str | Path | None = None) -> bool:
    """Merge `updates` into the persisted state on disk. Returns True on success.

    Only keys in ALLOWED_KEYS are written; others are silently dropped.
    Atomic: writes to temp file then os.replace.
    """
    p = _resolve_path(path)
    current = load_persisted_state(p)
    changed = False
    for key, value in (updates or {}).items():
        if key not in ALLOWED_KEYS:
            continue
        if not _is_serializable(value):
            continue
        if current.get(key) != value:
            current[key] = value
            changed = True
    if not changed:
        return True
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(p.parent or "."), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, p)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return False
        return True
    except OSError:
        return False


def sync_session_with_persistence(
    session_state: MutableMapping[str, Any],
    *,
    path: str | Path | None = None,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """At startup, hydrate session_state with persisted values (only for keys not yet set).

    Returns the dict of persisted values found. Caller can use it for diagnostics.
    """
    persisted = load_persisted_state(path)
    target_keys = set(keys) if keys else set(ALLOWED_KEYS)
    applied = {}
    for k, v in persisted.items():
        if k not in target_keys:
            continue
        if k not in session_state or session_state.get(k) in (None, "", []):
            session_state[k] = v
            applied[k] = v
    return applied


def collect_session_snapshot(
    session_state: Mapping[str, Any],
    *,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a dict of {key: value} for persistable keys currently in session_state."""
    target_keys = set(keys) if keys else set(ALLOWED_KEYS)
    out: Dict[str, Any] = {}
    for k in target_keys:
        if k in session_state:
            v = session_state[k]
            if _is_serializable(v):
                out[k] = v
    return out


def validate_number_input(
    value: Any,
    *,
    label: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> tuple[Optional[float], Optional[str]]:
    """Validate a number input. Returns (clean_value, error_message_or_None)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, f"'{label}' precisa ser um número (recebido: {value!r})"
    import math
    if not math.isfinite(v):
        return None, f"'{label}' não pode ser NaN ou infinito"
    if min_value is not None and v < float(min_value):
        return v, f"'{label}' deve ser ≥ {min_value} (recebido: {v})"
    if max_value is not None and v > float(max_value):
        return v, f"'{label}' deve ser ≤ {max_value} (recebido: {v})"
    return v, None


def validate_text_required(
    value: Any,
    *,
    label: str,
    min_length: int = 1,
) -> tuple[str, Optional[str]]:
    """Validate a required text input. Returns (clean_text, error_message_or_None)."""
    s = str(value or "").strip()
    if len(s) < int(min_length):
        return s, f"'{label}' não pode estar vazio (mínimo {min_length} caractere{'s' if min_length > 1 else ''})"
    return s, None


__all__ = [
    "ALLOWED_KEYS",
    "collect_session_snapshot",
    "load_persisted_state",
    "save_persisted_state",
    "sync_session_with_persistence",
    "validate_number_input",
    "validate_text_required",
]
