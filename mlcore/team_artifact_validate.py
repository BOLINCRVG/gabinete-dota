"""Validation and coercion helpers for team artifact parameter maps."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from mlcore.utils import safe_float as _f
from mlcore.team_artifact_normalize import _norm_upper_key


def _coerce_float_map(d: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        kk = _norm_upper_key(k)
        if not kk:
            continue
        fv = _f(v, float("nan"))
        if math.isfinite(fv):
            out[kk] = float(fv)
    return out


def _param_map_sig(d: Any) -> Tuple[Tuple[str, float], ...]:
    mp = _coerce_float_map(d)
    return tuple(sorted((k, float(v)) for k, v in mp.items()))
