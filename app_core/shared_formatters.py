"""Shared display-formatting helpers for all tab renderers."""
from __future__ import annotations

import math
from typing import Any


def fmt_num(v: Any, decimals: int = 1) -> str:
    try:
        fv = float(v)
        if math.isfinite(fv):
            return f"{fv:.{int(decimals)}f}"
    except Exception:
        pass
    return "-"


def fmt_pct(v: Any, decimals: int = 1) -> str:
    """Format a value already in 0-100 percentage scale."""
    try:
        fv = float(v)
        if math.isfinite(fv):
            return f"{fv:.{int(decimals)}f}%"
    except Exception:
        pass
    return "-"


def fmt_delta_pp(v: Any, decimals: int = 1) -> str:
    """Format a percentage-point delta with explicit sign and 'pp' suffix."""
    try:
        fv = float(v)
        if math.isfinite(fv):
            sign = "+" if fv >= 0 else ""
            return f"{sign}{fv:.{int(decimals)}f}pp"
    except Exception:
        pass
    return "-"


def fmt_status(pct: float) -> str:
    """Classify a percentage as Favorável / Neutro / Desfavorável."""
    if not math.isfinite(pct):
        return "—"
    if pct >= 55.0:
        return "Favorável"
    if pct <= 45.0:
        return "Desfavorável"
    return "Neutro"


def fmt_value(v: Any, decimals: int = 1) -> str:
    """Format a numeric value, rendering whole numbers without decimal places."""
    try:
        fv = float(v)
        if not math.isfinite(fv):
            return "-"
        if abs(fv - round(fv)) < 1e-9:
            return str(round(fv))
        return f"{fv:.{int(decimals)}f}"
    except Exception:
        return "-"
