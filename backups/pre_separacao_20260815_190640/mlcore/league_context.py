from __future__ import annotations

import re
from typing import Any

from .tiers import canonical_league


_EXPLICIT_INTERREGIONAL_LABELS = {
    "INTER",
    "INTERREGIONAL",
    "INTER REGIONAL",
    "INTERLEAGUE",
    "INTER LEAGUE",
    "CROSS CONFERENCE",
    "AMERICAS",
    "AMERICAS CUP",
}

_LEAGUE_LOOKUP_COMPAT: dict[str, tuple[str, ...]] = {
    "AL": ("LDL", "LPL CL", "LPL CHALLENGERS", "LPL CHALLENGERS LEAGUE"),
    "LCKC": ("LCK CL", "LCK CHALLENGERS", "LCK CHALLENGERS LEAGUE"),
    "LCS": ("LTA", "LTA N", "LTA NORTH"),
    "CBLOL": ("LTA S", "LTA SOUTH"),
    "AMERICAS": (
        "AMERICAS CUP",
        "CROSS CONFERENCE",
        "CROSS-CONFERENCE",
        "INTERLEAGUE",
        "INTER LEAGUE",
        "INTERREGIONAL",
        "INTER REGIONAL",
    ),
    "WORLDS": ("WLDS", "WORLD CHAMPIONSHIP", "WC", "WCS"),
    "MSI": ("MID SEASON INVITATIONAL", "MIDSEASON INVITATIONAL"),
}


def _norm(v: Any) -> str:
    z = str(v or "").strip().upper()
    if not z:
        return ""
    z = z.replace("-", " ").replace("_", " ")
    z = re.sub(r"\s+", " ", z).strip()
    return z


def is_explicit_interregional_league_label(v: Any) -> bool:
    """Return True if v represents an explicit cross-region event (AMERICAS, Cross Conference, etc.)."""
    z = _norm(v)
    if not z:
        return False
    if z in _EXPLICIT_INTERREGIONAL_LABELS:
        return True
    cz = _norm(canonical_league(z))
    return bool(cz in _EXPLICIT_INTERREGIONAL_LABELS)


def league_lookup_keys(league: Any) -> tuple[str, ...]:
    """Return all lookup keys for a league (canonical + raw + known aliases) for CSV matching."""
    raw = _norm(league)
    canon = _norm(canonical_league(raw))
    out: list[str] = []
    seen: set[str] = set()

    def _push(v: Any) -> None:
        z = _norm(v)
        if not z or z in seen:
            return
        out.append(z)
        seen.add(z)

    _push(canon)
    _push(raw)
    for alias in _LEAGUE_LOOKUP_COMPAT.get(canon, ()):
        _push(alias)
    return tuple(out)


def league_macro_key(league: Any, *, unknown: str = "OTHER") -> str:
    """Return the macro-region key for a league (KR/CN/EMEA/NA/BR/APAC/AMERICAS/OTHER)."""
    z = _norm(league)
    if not z:
        return str(unknown)
    # Prefer canonical key to avoid alias drift (e.g., LTA -> LCS).
    cz = _norm(canonical_league(z))
    if cz:
        z = cz

    if is_explicit_interregional_league_label(z):
        return "AMERICAS"
    if z.startswith("LPL") or z in {"LDL", "LPL CL", "CN", "AL"}:
        return "CN"
    if z.startswith("LCK") or z in {"LCK CL", "LCKC", "KR"}:
        return "KR"
    if z.startswith(("LEC", "EMEA")) or z in {"LFL", "LVP SL", "PRIME LEAGUE", "NLC", "UL", "PRM", "CCWS", "LES", "ROL"}:
        return "EMEA"
    if z.startswith("LCS") or z in {"NACL", "NA"}:
        return "NA"
    if z.startswith("CBLOL") or z == "BR":
        return "BR"
    if z.startswith(("PCS", "VCS", "LJL", "LCO", "LCP")) or z == "APAC":
        return "APAC"
    return str(unknown)
