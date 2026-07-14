"""Shared helpers for pipeline step tracing."""

from __future__ import annotations

import math
from typing import Any, Dict, List


def _opt_float(v: Any) -> float | None:
    try:
        x = float(v)
        return float(x) if math.isfinite(x) else None
    except Exception:
        return None


def append_pipeline_step(
    *,
    steps: List[Dict[str, Any]] | Any,
    stage: str,
    before: Any = None,
    after: Any = None,
    applied: bool = False,
    **extras: Any,
) -> None:
    if not isinstance(steps, list):
        return
    row: Dict[str, Any] = {
        "stage": str(stage or ""),
        "before": _opt_float(before),
        "after": _opt_float(after),
        "applied": bool(applied),
    }
    for k, v in (extras or {}).items():
        row[str(k)] = v
    steps.append(row)
