from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional

from mlcore.tiers import canonical_league

MAJOR_LEAGUES = {
    "LCK",
    "LPL",
    "LEC",
    "LCS",
    "PCS",
    "VCS",
    "CBLOL",
    "LLA",
    "AMERICAS",
    "MSI",
    "WORLDS",
    "WC",
    "WORLD CHAMPIONSHIP",
}

_COMPAT_ALIASES = {
    "CCWS": "PRM",
    "CROSSCONFERENCE": "AMERICAS",
    "LCK CHALLENGER": "LCKC",
    "LCK ACADEMY": "LCKC",
    "LPL CHALLENGER": "AL",
    "LPL ACADEMY": "AL",
    "NORTH AMERICA": "NA",
    "SOUTH KOREA": "KR",
    "CHINA": "CN",
}

_STRONG_LEAGUES = {
    "LCK",
    "LPL",
    "LEC",
    "LCS",
    "CBLOL",
    "AMERICAS",
    "LCP",
    "PCS",
    "VCS",
    "LLA",
    "MSI",
    "WORLDS",
}

_TOP_CONF_LEAGUES = {"LCK", "LPL", "LEC", "LCS"}
_CORE_CONF_LEAGUES = {"CBLOL", "AMERICAS", "LCP", "PCS", "VCS", "LLA", "MSI", "WORLDS"}
_EDGE_TIER_KEYS = ("CHALLENG", "ACADEMY", "CL", "LCKC", "NACL", "LFL", "LVP", "LDL", "AL", "LPL ACADEMY")


def norm_league(value: str) -> str:
    """Strip and return the league string as-is (minimal normalization for UI display)."""
    return str(value or "").strip()


def normalize_ascii_upper(value: str) -> str:
    """Strip accents, remove year tokens, uppercase, and collapse whitespace for league key lookup."""
    txt = str(value or "")
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = txt.upper()
    txt = re.sub(r"\b20\d{2}\b", " ", txt)
    txt = re.sub(r"[^A-Z0-9]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def norm_league_key(value: str) -> str:
    """Resolve a league string to its canonical key, applying alias lookup and ASCII normalization."""
    raw = norm_league(value)
    if not raw:
        return ""
    norm = normalize_ascii_upper(raw)
    if not norm:
        return ""
    compact = norm.replace(" ", "")
    if compact in _COMPAT_ALIASES:
        return str(_COMPAT_ALIASES[compact])
    if norm in _COMPAT_ALIASES:
        return str(_COMPAT_ALIASES[norm])
    return str(canonical_league(norm) or norm)


def canon_league_display_label(raw_league: Any) -> str:
    """Return a canonical display label for a league (remaps LTA North → LCS, etc.)."""
    raw = str(raw_league or "").strip()
    if not raw:
        return ""
    key = norm_league_key(raw)
    if key in {"PRM", "LCS", "CBLOL", "AMERICAS"}:
        return str(key)
    return raw


def parse_league_csv(raw: Any) -> set[str]:
    """Parse a comma/pipe/semicolon-delimited league list into a set of canonical keys."""
    try:
        txt = str(raw or "").strip()
    except Exception:
        txt = ""
    if not txt:
        return set()
    out: set[str] = set()
    for part in re.split(r"[,\n;|]+", txt):
        key = norm_league_key(part)
        if key:
            out.add(key)
    return out


def league_is_excluded(league: str, raw_csv: Any) -> bool:
    """Return True if the league's canonical key appears in the comma-delimited exclusion list."""
    league_key = norm_league_key(league)
    if not league_key:
        return False
    return league_key in parse_league_csv(raw_csv)


def is_marginal_league_for_ml(league: str) -> bool:
    """Return True if the league is not in the 'strong' set (requires marginal-trust guard)."""
    league_key = norm_league_key(league)
    if not league_key:
        return True
    return bool(league_key not in _STRONG_LEAGUES)


def ml_league_conf_band(league: str, tier: str = "") -> str:
    """Return the ML confidence band for a league: 'top' (T1), 'core' (regional), or 'edge'."""
    league_key = norm_league_key(league)
    tier_key = str(tier or "").strip().upper()
    if tier_key in {"T2", "MINOR"}:
        return "edge"
    if league_key in _TOP_CONF_LEAGUES:
        return "top"
    if league_key in _CORE_CONF_LEAGUES:
        return "core"
    return "edge"


def ml_default_marginal_band_profiles() -> Dict[str, Dict[str, float]]:
    """Return the default marginal-trust band coefficient profiles for top/core/edge bands."""
    return {
        "top": {
            "strength_scale": 0.35,
            "prior_scale": 0.70,
            "max_shrink_scale": 0.55,
            "cap_shift": +0.06,
            "soft_shift": +10.0,
            "hard_shift": +8.0,
            "gate_score": 80.0,
            "gate_n_eff": 30.0,
        },
        "core": {
            "strength_scale": 0.75,
            "prior_scale": 0.95,
            "max_shrink_scale": 0.85,
            "cap_shift": +0.02,
            "soft_shift": +3.0,
            "hard_shift": +2.0,
            "gate_score": 84.0,
            "gate_n_eff": 42.0,
        },
        "edge": {
            "strength_scale": 1.15,
            "prior_scale": 1.20,
            "max_shrink_scale": 1.05,
            "cap_shift": -0.04,
            "soft_shift": -6.0,
            "hard_shift": -4.0,
            "gate_score": 88.0,
            "gate_n_eff": 55.0,
        },
    }


def sanitize_ml_marginal_band_coeffs(raw: Any) -> Dict[str, Dict[str, float]]:
    """Validate and coerce raw band-coefficient overrides; drop unrecognized keys and non-finite values."""
    out: Dict[str, Dict[str, float]] = {}
    base = ml_default_marginal_band_profiles()
    if not isinstance(raw, dict):
        return out
    for band in ("top", "core", "edge"):
        rv = raw.get(band)
        if not isinstance(rv, dict):
            continue
        row: Dict[str, float] = {}
        for key in base[band].keys():
            if key not in rv:
                continue
            try:
                fv = float(rv.get(key))
                if fv == fv and fv not in {float("inf"), float("-inf")}:
                    row[str(key)] = float(fv)
            except Exception:
                continue
        if row:
            out[band] = row
    return out


def ml_marginal_band_profile(band: str, band_coeffs: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Return the resolved coefficient profile for a band, merging defaults with optional overrides."""
    band_key = str(band or "").strip().lower()
    if band_key not in {"top", "core", "edge"}:
        band_key = "edge"
    base = ml_default_marginal_band_profiles()
    out = dict(base.get(band_key, base["edge"]))
    if isinstance(band_coeffs, dict):
        ov = sanitize_ml_marginal_band_coeffs(band_coeffs).get(band_key, {})
        if isinstance(ov, dict):
            for key, value in ov.items():
                if key in out:
                    out[key] = float(value)
    return out


def tier_from_league(league: str) -> str:
    """Return T1/T2/MINOR/OTHER tier for a league, used by the pricing and ranking layers."""
    league_key = norm_league_key(league)
    if not league_key:
        return "OTHER"
    if any(token in league_key for token in _EDGE_TIER_KEYS):
        return "T2"
    if league_key in MAJOR_LEAGUES:
        return "T1"
    return "MINOR"
