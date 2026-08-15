# -*- coding: utf-8 -*-
"""Resolves the achievements layer for a given ML artifact."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from mlcore.achievements import build_team_index

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "msi:top16": 6.0,
    "msi:top8": 12.0,
    "msi:top4": 20.0,
    "msi:runner_up": 32.0,
    "msi:champion": 45.0,
    "worlds:top16": 8.0,
    "worlds:top8": 16.0,
    "worlds:top4": 26.0,
    "worlds:runner_up": 42.0,
    "worlds:champion": 60.0,
    "regional:playoffs": 6.0,
    "regional:top8": 8.0,
    "regional:top4": 12.0,
    "regional:runner_up": 18.0,
    "regional:champion": 24.0,
    "other:champion": 12.0,
    "other:runner_up": 8.0,
    "other:top4": 6.0,
}


def _resolve_relpath(base_file: str, maybe_rel: str) -> str:
    p = Path(maybe_rel)
    if p.is_absolute():
        return str(p)
    return str((Path(base_file).parent / p).resolve())


def ach_layer_for_artifact(
    art: Any,
    artifact_path: str,
    ach_override_path: Optional[str] = None,
    *,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve camada de achievements para o cálculo de força.

    Prioridade:
    1) path vindo do artifact (art.meta['achievements']['path'])
    2) fallback: achievements.json ao lado do ml_artifact.json (mesma pasta)
    """
    _file_sig = ctx.get("_file_sig")
    _load_achievements_cached = ctx.get("_load_achievements_cached")

    ach_meta = dict(art.meta.get("achievements") or {})
    ach_path = str(ach_meta.get("path") or "").strip()
    weights_points = dict(ach_meta.get("weights_points") or {})

    ach_path_resolved = ""
    if ach_override_path:
        cand = str(ach_override_path).strip()
        if cand:
            cand_res = _resolve_relpath(artifact_path, cand) if not os.path.isabs(cand) else cand
            if Path(cand_res).exists():
                ach_path_resolved = cand_res

    if not ach_path_resolved:
        if ach_path:
            ach_path_resolved = _resolve_relpath(artifact_path, ach_path)
        else:
            guess = Path(artifact_path).with_name("achievements.json")
            if guess.exists():
                ach_path_resolved = str(guess)

    if not ach_path_resolved or (not Path(ach_path_resolved).exists()):
        return {"enabled": False, "path": "", "team_idx": {}, "weights_points": {}}

    data = _load_achievements_cached(_file_sig(ach_path_resolved))
    if isinstance(data, dict) and data.get("__load_error__"):
        return {
            "enabled": False,
            "path": ach_path_resolved,
            "team_idx": {},
            "weights_points": {},
            "error": str(data.get("__load_error__")),
        }

    team_idx = build_team_index(data) if data else {}

    if (not weights_points) and isinstance(data, dict):
        cfg = data.get("config") or {}
        fwp = cfg.get("fixed_weights_points") or {}
        if isinstance(fwp, dict) and fwp:
            weights_points = {str(k): float(v) for k, v in fwp.items() if v is not None}

    if not weights_points:
        weights_points = dict(_DEFAULT_WEIGHTS)

    return {
        "enabled": True,
        "path": ach_path_resolved,
        "team_idx": team_idx,
        "weights_points": weights_points,
    }
