from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import safe_float as _f
def _safe_int(v: Any, default: int = 0) -> int:
    fv = _f(v, float("nan"))
    if not math.isfinite(fv):
        return int(default)
    try:
        return int(fv)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _read_json_file_safe(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json_file_safe(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _path_signature(path_like: str | Path) -> Dict[str, Any]:
    p = Path(path_like).expanduser()
    try:
        rp = p.resolve(strict=False)
    except (OSError, RuntimeError):
        rp = p
    out: Dict[str, Any] = {"path": str(rp)}
    try:
        st = rp.stat()
        out["size"] = int(st.st_size)
        out["mtime_ns"] = int(st.st_mtime_ns)
    except OSError:
        out["missing"] = True
    return out


def _build_fit_run_signature(
    *,
    artifact_path: str,
    players_artifact_path: str,
    csv_paths: List[str],
    min_train_games: int,
    speed_profile: str,
    run_label: str | None = None,
    start_stage: str | None = None,
    base_params_path: str | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": "fit_run_signature_v1",
        "artifact": _path_signature(artifact_path),
        "players_artifact": _path_signature(players_artifact_path),
        "csvs": [_path_signature(p) for p in csv_paths],
        "fit_code": _path_signature(Path(__file__).resolve()),
        "min_train_games": int(min_train_games),
        "speed_profile": str(speed_profile or "full").strip().lower(),
        "run_label": str(run_label or "").strip(),
        "start_stage": str(start_stage or "").strip(),
    }
    if str(base_params_path or "").strip():
        payload["base_params"] = _path_signature(str(base_params_path))
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)
    payload["sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return payload
