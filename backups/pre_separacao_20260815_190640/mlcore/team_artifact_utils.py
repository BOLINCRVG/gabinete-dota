"""Pure utility helpers for team artifact processing.

Extracted from state_team_artifact.py: date/int coercions, file-signature
helpers, running average, and league-multiplier lookups.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .league_context import league_macro_key as _league_macro_key_shared
from .utils import safe_float as _f


def _safe_nonneg_int(v: Any, default: int = 0) -> int:
    """Convert v to a non-negative int, returning default on failure or negative values."""
    fv = _f(v, float("nan"))
    if not math.isfinite(fv):
        return int(default)
    try:
        return int(max(0, int(fv)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_iso_date(v: Any) -> Optional[str]:
    """Parse v as a date and return ISO-8601 string (YYYY-MM-DD), or None on failure."""
    d = pd.to_datetime(v, errors="coerce")
    if not isinstance(d, pd.Timestamp) or pd.isna(d):
        return None
    return d.date().isoformat()


def _days_between_iso(a: Optional[str], b: Optional[str]) -> int:
    """Return signed integer day difference (b - a) for ISO date strings; 0 on parse failure."""
    if not a or not b:
        return 0
    da = pd.to_datetime(a, errors="coerce")
    db = pd.to_datetime(b, errors="coerce")
    if not isinstance(da, pd.Timestamp) or not isinstance(db, pd.Timestamp):
        return 0
    if pd.isna(da) or pd.isna(db):
        return 0
    return int((db.date() - da.date()).days)


def _file_sig_info(path_s: str) -> Dict[str, Any]:
    """Return a dict with path, exists, size, and mtime_ns for change-detection."""
    p = Path(str(path_s)).expanduser()
    try:
        rp = str(p.resolve(strict=False))
    except (OSError, RuntimeError):
        rp = str(p)
    out: Dict[str, Any] = {
        "path": rp,
        "exists": False,
        "size": None,
        "mtime_ns": None,
    }
    try:
        st = p.stat()
        out["exists"] = True
        out["size"] = int(st.st_size)
        out["mtime_ns"] = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    except OSError:
        pass
    return out


def _same_source_path_order(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    """Return True if two lists of file-sig dicts have the same paths in the same order."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if not isinstance(x, dict) or not isinstance(y, dict):
            return False
        if str(x.get("path") or "") != str(y.get("path") or ""):
            return False
    return True


def _source_sig_changed(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Return True if two file-sig dicts differ in exists, size, or mtime."""
    return (
        bool(a.get("exists")) != bool(b.get("exists"))
        or int(a.get("size") or -1) != int(b.get("size") or -1)
        or int(a.get("mtime_ns") or -1) != int(b.get("mtime_ns") or -1)
    )


def _running_avg(prev_avg: float, n: int, value: float) -> float:
    """Incremental running mean: given previous average over n samples, add one new value."""
    nn = int(max(0, n))
    if nn <= 0:
        return float(value)
    return float(((float(prev_avg) * float(nn)) + float(value)) / float(nn + 1))


def _league_macro_key(league: str) -> str:
    """Resolve a league to its macro-region key (KR/CN/EMEA/NA/BR/APAC/OTHER)."""
    return str(_league_macro_key_shared(league, unknown="OTHER"))


def _league_k_multiplier(league: str) -> float:
    """Per-region K-factor multiplier for Elo updates (KR/CN are boosted; others dampened)."""
    macro = _league_macro_key(league)
    return {
        "CN": 1.05,
        "KR": 1.05,
        "EMEA": 1.00,
        "NA": 0.98,
        "BR": 0.96,
        "APAC": 0.97,
        "OTHER": 0.92,
    }.get(macro, 0.95)


def _league_patch_vol_multiplier(league: str) -> float:
    """Per-region sigma-inflation multiplier when a new patch is detected."""
    macro = _league_macro_key(league)
    return {
        "CN": 1.12,
        "KR": 1.08,
        "EMEA": 1.00,
        "NA": 0.98,
        "BR": 0.96,
        "APAC": 0.98,
        "OTHER": 0.95,
    }.get(macro, 1.00)
