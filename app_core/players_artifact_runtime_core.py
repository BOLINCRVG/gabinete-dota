# -*- coding: utf-8 -*-
"""Runtime resolution of the players artifact path with fallback."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping

from app_core.flow_runtime_core import resolve_players_artifact_path as _resolve_players_artifact_path


def resolve_runtime_players_artifact_path(
    artifact_path: str,
    *,
    ctx: Mapping[str, Any],
) -> str:
    APP_ROOT = ctx.get("APP_ROOT")
    _load_artifact_cached = ctx.get("_load_artifact_cached")
    _file_sig = ctx.get("_file_sig")
    try:
        return _resolve_players_artifact_path(
            str(artifact_path),
            APP_ROOT,
            load_artifact_fn=lambda path: _load_artifact_cached(_file_sig(str(path))),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        pass
    artifact_raw = str(artifact_path or "").strip()
    if artifact_raw:
        return str(Path(artifact_raw).expanduser().with_name("players_artifact.json"))
    return str(Path(APP_ROOT) / "players_artifact.json")
