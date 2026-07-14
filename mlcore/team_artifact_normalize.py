"""Key normalization utilities for team artifact processing."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def _norm_team_key(v: Any) -> str:
    s = str(v or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _patch_key(p: Any) -> str:
    s = str(p or "").strip()
    if not s:
        return ""
    m = re.search(r"(\d+)\.(\d+)", s)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return s


def _norm_upper_key(v: Any) -> str:
    return str(v or "").strip().upper()
