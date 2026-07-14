# -*- coding: utf-8 -*-
"""Calibrates a raw probability through an artifact's calibrator."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mlcore.utils import clip_prob, safe_float_or_none as _fnum


def calibrate_p_from_artifact(
    artifact_path: str,
    p_raw: float,
    *,
    ctx: Mapping[str, Any],
) -> float:
    _file_sig = ctx.get("_file_sig")
    _load_artifact_cached = ctx.get("_load_artifact_cached")

    p_fallback = _fnum(p_raw)
    if p_fallback is None:
        return float("nan")
    p_fallback = float(clip_prob(p_fallback))
    try:
        p = Path(artifact_path)
        if not p.exists():
            return p_fallback
        art = _load_artifact_cached(_file_sig(str(p)))
        calibrator = getattr(art, "calibrator", None)
        predict_proba = getattr(calibrator, "predict_proba", None) if calibrator is not None else None
        if not callable(predict_proba):
            return p_fallback
        return float(clip_prob(float(predict_proba(p_fallback))))
    except (AttributeError, OSError, TypeError, ValueError):
        return p_fallback
