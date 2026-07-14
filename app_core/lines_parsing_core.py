"""Parsing helpers for bet-line text inputs (quick KV format and single values)."""
from __future__ import annotations

import re
from typing import Dict, Optional

from app_core.market import parse_lines_text, parse_time_lines_text

_ALIASES: Dict[str, str] = {
    "k": "kills", "kill": "kills", "kills": "kills",
    "t": "towers", "tower": "towers", "towers": "towers",
    "drg": "dragons", "drag": "dragons", "dragon": "dragons", "dragons": "dragons", "d": "dragons",
    "bar": "barons", "baron": "barons", "barons": "barons", "b": "barons",
    "inib": "inhibitors", "inibs": "inhibitors", "inhib": "inhibitors",
    "inhibitor": "inhibitors", "inhibitors": "inhibitors", "i": "inhibitors",
    "time": "time", "tempo": "time", "tm": "time",
}


def parse_quick_lines_kv(text: str) -> Dict[str, str]:
    """Parse a single-line KV string of bet lines.

    Example:
        kills=29.5 towers=10.5 drg=4.5 bar=1.5 inib=1.5 time=33:00

    Returns dict with canonical keys: kills, towers, dragons, barons, inhibitors, time.
    Values are kept as strings to feed text_input widgets directly.
    """
    text = (text or "").strip()
    if not text:
        return {}
    text = text.replace(",", " ")
    tokens = [t for t in re.split(r"\s+", text) if t.strip()]
    out: Dict[str, str] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = _ALIASES.get(k.strip().lower())
        if not k:
            continue
        v = v.strip()
        if v:
            out[k] = v
    return out


def parse_single_float(text: str) -> Optional[float]:
    vals = parse_lines_text(text)
    return vals[0] if vals else None


def parse_single_time_min(text: str) -> Optional[float]:
    vals = parse_time_lines_text(text)
    return vals[0] if vals else None


__all__ = ["parse_quick_lines_kv", "parse_single_float", "parse_single_time_min"]
