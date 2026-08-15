"""Core persistence/actions for app settings."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, MutableMapping

from app_core.ml_session_runtime_core import consume_pending_apply_param_values


def settings_path(app_root: Path, filename: str) -> Path:
    return Path(app_root) / str(filename)


def json_safe(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe(val) for k, val in v.items()}
    return str(v)


def load_settings_from_disk(path: Path, date_keys: Iterable[str] = ("ml_as_of_date",)) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        data = raw.get("settings", raw) if isinstance(raw, dict) else {}
        if not isinstance(data, dict):
            return {}
        for k in date_keys:
            if isinstance(data.get(k), str):
                try:
                    data[k] = date.fromisoformat(str(data[k])[:10])
                except Exception:
                    pass
        return data
    except Exception:
        return {}


def apply_settings_to_session(
    session_state: MutableMapping[str, Any],
    settings: Dict[str, Any],
    *,
    persist_keys: Iterable[str],
    force: bool = False,
) -> None:
    pset = set(str(k) for k in persist_keys)
    for k, v in (settings or {}).items():
        if not isinstance(k, str) or not k:
            continue
        if (k in pset) and (force or (k not in session_state)):
            session_state[k] = v


def save_settings_to_disk(path: Path, session_state: MutableMapping[str, Any], *, persist_keys: Iterable[str]) -> None:
    payload: Dict[str, Any] = {
        "version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {},
    }
    s = payload["settings"]
    for k in persist_keys:
        if k in session_state:
            s[str(k)] = json_safe(session_state.get(k))
    if "cfg_profile" not in s:
        s["cfg_profile"] = "Padrão"

    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        try:
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def process_pending_settings_actions(
    session_state: MutableMapping[str, Any],
    *,
    load_settings_fn: Callable[[], Dict[str, Any]],
    apply_settings_fn: Callable[[Dict[str, Any], bool], None],
    apply_recommended_preset_fn: Callable[[str], None],
    apply_excellent_preset_fn: Callable[[], None],
    apply_twenty_ten_preset_fn: Callable[[], None],
    reset_all_tuning_fn: Callable[[], None],
    param_defaults: Dict[str, Any],
) -> None:
    if bool(session_state.get("_pending_reload_settings", False)):
        try:
            data = load_settings_fn()
            apply_settings_fn(data, True)
        finally:
            session_state["_pending_reload_settings"] = False

    if "_pending_apply_preset" in session_state:
        prof = str(session_state.pop("_pending_apply_preset") or "Padrão")
        apply_recommended_preset_fn(prof)

    vals = consume_pending_apply_param_values(session_state)
    if vals:
        for k, v in vals.items():
            if isinstance(k, str) and k:
                session_state[k] = v
                session_state[f"ctl_{k}"] = v

    if bool(session_state.pop("_pending_apply_excellent", False)):
        apply_excellent_preset_fn()
        for _k, _v in (param_defaults or {}).items():
            session_state[f"ctl_{_k}"] = session_state.get(_k, _v)

    if bool(session_state.pop("_pending_apply_twenty_ten", False)):
        apply_twenty_ten_preset_fn()
        for _k, _v in (param_defaults or {}).items():
            session_state[f"ctl_{_k}"] = session_state.get(_k, _v)

    if bool(session_state.pop("_pending_reset_all_tuning", False)):
        reset_all_tuning_fn()
        for _k, _v in (param_defaults or {}).items():
            session_state[f"ctl_{_k}"] = session_state.get(_k, _v)


def init_settings_once(
    session_state: MutableMapping[str, Any],
    *,
    load_settings_fn: Callable[[], Dict[str, Any]],
    apply_settings_fn: Callable[[Dict[str, Any], bool], None],
    process_pending_fn: Callable[[], None],
) -> None:
    if bool(session_state.get("_app_settings_booted", False)):
        process_pending_fn()
        return

    data = load_settings_fn()
    apply_settings_fn(data, False)
    if "cfg_profile" not in session_state:
        session_state["cfg_profile"] = "Padrão"
    session_state["_app_settings_booted"] = True
    process_pending_fn()
