from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json


def load_json_store(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def persist_json_store(path: Path, store: Dict[str, Any]) -> None:
    try:
        if not isinstance(store, dict):
            return
        payload = json.dumps(store, ensure_ascii=False, indent=2)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return


def ensure_session_store(session_state: Dict[str, Any], key: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    store = session_state.get(key)
    if not isinstance(store, dict):
        store = dict(fallback or {})
        session_state[key] = store
    return store

