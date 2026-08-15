from __future__ import annotations
import re

"""Tier/league configuration (stable and easy to tune).

This file is the *main knob* to make ML winner pricing behave sensibly across
regions. In esports, domestic leagues are mostly disconnected; without explicit
baselines, a team that dominates a weak region can look "world-class".

The model uses an additive global strength:

  total(team) = team_rating + tier_base(tier) + league_base(league)

- team_rating: learned from results (residual strength)
- tier_base:  coarse region class (big, stable gaps)
- league_base: optional fine adjustment (small, stable gaps)

You can tune baselines without touching any model code.

Default goal of this config:
- Top GLOBAL should be dominated by LCK/LPL/LEC.
- 2026 naming should remain stable (LCS/CBLOL), while legacy aliases resolve
  into modern league keys.

Important:
- Riot's 2026 ecosystem has five Tier 1 circuits (`LCK`, `LPL`, `LEC`, `LCP`,
  `LCS/CBLOL` umbrella split across North/South).
- This file uses *model tiers*, not official tournament branding. We keep
  `LCS/CBLOL` in a lower model bucket than `LCK/LPL/LEC/LCP`, but still assign
  explicit per-league strength so the code can distinguish strong and weak
  leagues inside the same coarse bucket.
"""

# --- Baselines (Elo-like points) ------------------------------------------------
# Bigger gaps -> underdogs from weak regions get much higher odds.

TIER_BASE_POINTS: dict[str, float] = {
    # Major regions (LCK/LPL/LEC)
    "T1": 250.0,
    # Strong-but-below-major (example: LTA, PCS/VCS if you map them here)
    "T2": 80.0,
    # EMEA Regional Leagues / semi-pro
    "ERL": -50.0,
    # Minor / local leagues
    "MINOR": -250.0,
    # Unknown/other
    "OTHER": 0.0,
}

# Optional extra baseline per league (fine tuning). Keep small compared to tiers.
# NOTE: This is where you can force "top world" to be mostly LCK/LPL/LEC.
LEAGUE_BASE_POINTS: dict[str, float] = {
    # Major-region ordering
    "LCK": 42.0,
    "LPL": 38.0,
    "LEC": 18.0,
    # LCP is a top circuit, but still below KR/CN/EU in model strength.
    "LCP": -55.0,

    # Americas / upper domestic leagues
    "LCS": 10.0,
    "AMERICAS": 8.0,
    "CBLOL": 2.0,
    "PCS": 8.0,
    "VCS": 4.0,
    "LJL": -2.0,

    # Strong ERLs / regional circuits
    "LFL": 20.0,
    "PRM": 16.0,
    "LVP SL": 12.0,
    "NLC": 6.0,
    "EM": 10.0,
    "CCWS": 8.0,
    "PRMP": 4.0,
    "TCL": 6.0,
    "LES": 2.0,
    "ROL": -2.0,
    "EBL": -4.0,
    "LPLOL": -8.0,
    "HM": -2.0,

    # Development / challenger ecosystems
    "AL": 70.0,
    "LCKC": 68.0,
    "NACL": 48.0,
    "LFL2": 36.0,
    "CBLOLA": 30.0,

    # International events as context (usually no need to shift)
    "MSI": 0.0,
    "WORLDS": 0.0,
    "EWC": 0.0,
    "FST": 0.0,
}

# Strength score is a structural prior used mainly for interleague logic.
# It is intentionally smoother than TIER_BASE_POINTS so runtime can distinguish:
# - elite vs major
# - strong ERLs vs weak ERLs
# - top development leagues vs generic local leagues
#
# Values are on a 0..1 scale and reflect a pragmatic strength ordering, using
# Riot's 2026 top-circuit structure plus a project-level inference for domestic,
# ERL and academy ecosystems.
LEAGUE_STRENGTH_SCORES: dict[str, float] = {
    # Elite world circuits
    "LCK": 1.00,
    "LPL": 0.98,

    # Major world circuits
    "LEC": 0.92,
    "LCP": 0.88,
    "MSI": 0.96,
    "WORLDS": 0.97,
    "EWC": 0.95,
    "FST": 0.94,

    # Upper domestic / interregional
    "LCS": 0.83,
    "AMERICAS": 0.82,
    "CBLOL": 0.80,
    "PCS": 0.79,
    "VCS": 0.77,
    "LJL": 0.74,

    # Regional semi-pro / ERL
    "LFL": 0.68,
    "PRM": 0.66,
    "LVP SL": 0.65,
    "NLC": 0.60,
    "EM": 0.60,
    "TCL": 0.60,
    "CCWS": 0.61,
    "LES": 0.57,
    "PRMP": 0.57,
    "EBL": 0.56,
    "HM": 0.55,
    "LPLOL": 0.54,
    "ROL": 0.52,

    # Development / challenger
    "AL": 0.50,
    "LCKC": 0.49,
    "NACL": 0.45,
    "LFL2": 0.46,
    "CBLOLA": 0.44,

    # Local / minor
    "LCO": 0.40,
    "LLA": 0.39,
    "LAS": 0.35,
}

DEFAULT_LEAGUE_STRENGTH_SCORE_BY_TIER: dict[str, float] = {
    "T1": 0.90,
    "T2": 0.78,
    "ERL": 0.56,
    "MINOR": 0.36,
    "OTHER": 0.50,
}

LEAGUE_STRENGTH_BANDS: list[tuple[str, float]] = [
    ("elite", 0.96),
    ("major", 0.88),
    ("upper", 0.78),
    ("regional", 0.58),
    ("development", 0.44),
    ("local", 0.00),
]

# --- League -> tier mapping -----------------------------------------------------
# Adjust this mapping to your preference/knowledge.

LEAGUE_TO_TIER: dict[str, str] = {
    # Major/top tier
    "LCK": "T1",
    "LPL": "T1",
    "LEC": "T1",

    # International events (treat as top-tier context)
    "MSI": "T1",
    "WORLDS": "T1",
    "EWC": "T1",
    "FST": "T1",

    # EMEA Regional Leagues / strong semi-pro
    "LFL": "ERL",
    "PRM": "ERL",
    "CCWS": "ERL",
    "PRMP": "ERL",
    "NLC": "ERL",
    "LVP SL": "ERL",
    "EBL": "ERL",
    "LPLOL": "ERL",
    "HM": "ERL",
    "EM": "ERL",


    # Common pro leagues / legacy labels
    "LCS": "T2",
    "CBLOL": "T2",
    "AMERICAS": "T2",
    "CBLOLA": "T2",
    "LJL": "T2",
    "PCS": "T2",
    "VCS": "T2",
    "LCP": "T1",
    "TCL": "ERL",
    "LES": "ERL",
    "ROL": "ERL",
    "LCO": "MINOR",
    "LLA": "MINOR",
    "LCKC": "MINOR",
    "NACL": "MINOR",
    "AL": "MINOR",
    "LIT": "MINOR",
    "RL": "MINOR",
    "LFL2": "ERL",
    "HLL": "MINOR",
    "HW": "MINOR",
    "LRN": "MINOR",
    "LRS": "MINOR",
    "CD": "MINOR",
    # Explicit audit-backed legacy/local leagues that previously relied on fallback.
    "LAS": "MINOR",
    "PCL": "MINOR",
    "NEXO": "MINOR",
    "HC": "MINOR",
    "ESLOL": "MINOR",
    "ASIA MASTER": "MINOR",
    "UL": "MINOR",
    "GLL": "MINOR",
    "EPL": "MINOR",
    "TSC": "MINOR",
    "KESPA": "MINOR",
    "IC": "MINOR",
    "DCUP": "MINOR",
    "CT": "MINOR",
    "NLC AURORA OPEN": "ERL",
    "CDF": "MINOR",
    "IGNIS": "MINOR",
    "ASI": "MINOR",
    "GLLPA": "MINOR",
    "USP": "MINOR",
    "AC": "MINOR",
    "EBLPA": "MINOR",

    # Everything else defaults to MINOR below
}

# If a league is not found in LEAGUE_TO_TIER, we assign it here.
DEFAULT_TIER_UNKNOWN_LEAGUE = "MINOR"

# --- Ranking order (no numeric baselines) ---------------------------------------
# Used to avoid 'lost' teams from weak regions appearing above major-region teams
# when printing global rankings. This is purely an ordering rule, not points.
TIER_ORDER: list[str] = ["T1", "T2", "ERL", "MINOR", "OTHER"]
TIER_PRIORITY: dict[str, int] = {t: i for i, t in enumerate(TIER_ORDER)}

def tier_priority(tier: str | None) -> int:
    """Return sort priority for a tier (lower = stronger: T1=0, T2=1, ERL=2, MINOR=3)."""
    return int(TIER_PRIORITY.get(str(tier) if tier else "OTHER", TIER_PRIORITY["OTHER"]))


LEAGUE_CANON_ALIASES: dict[str, str] = {
    # International naming variants
    "WORLD CHAMPIONSHIP": "WORLDS",
    "WC": "WORLDS",
    "WCS": "WORLDS",
    "WLDS": "WORLDS",
    "MID SEASON INVITATIONAL": "MSI",
    "MIDSEASON INVITATIONAL": "MSI",
    # Development/challenger aliases
    "LCK CL": "LCKC",
    "LCK CHALLENGERS": "LCKC",
    "LCK CHALLENGERS LEAGUE": "LCKC",
    "LPL CL": "AL",
    "LPL CHALLENGERS": "AL",
    "LPL CHALLENGERS LEAGUE": "AL",
    "LDL": "AL",
    # 2025 legacy Americas names mapped to 2026 canonical leagues.
    "LTA": "LCS",
    "LTA N": "LCS",
    "LTA NORTH": "LCS",
    "LTA S": "CBLOL",
    "LTA SOUTH": "CBLOL",
    # Interregional Americas / cross-conference naming.
    "AMERICAS": "AMERICAS",
    "AMERICAS CUP": "AMERICAS",
    "CROSS CONFERENCE": "AMERICAS",
    "CROSS-CONFERENCE": "AMERICAS",
    "INTERLEAGUE": "AMERICAS",
    "INTER LEAGUE": "AMERICAS",
    "INTERREGIONAL": "AMERICAS",
    "INTER REGIONAL": "AMERICAS",
    # Common ERL aliases
    "PRIME LEAGUE": "PRM",
    "PRIMELEAGUE": "PRM",
}


def canonical_league(league: str | None) -> str:
    """Return a canonical league key used by tier and baseline lookups."""
    if not league:
        return ""
    raw = str(league).strip()
    if not raw:
        return ""

    up = raw.upper()
    norm = re.sub(r"\s+", " ", up).strip()
    compact = re.sub(r"[^A-Z0-9]+", "", up)

    if norm in LEAGUE_CANON_ALIASES:
        return str(LEAGUE_CANON_ALIASES[norm])
    if compact in LEAGUE_CANON_ALIASES:
        return str(LEAGUE_CANON_ALIASES[compact])

    # Keep known keys stable (e.g. "LVP SL"), otherwise return normalized form.
    if norm in LEAGUE_TO_TIER or norm in LEAGUE_BASE_POINTS:
        return norm
    return norm


def _tier_fallback_for_canonical_league(league_key: str) -> str:
    z = str(league_key or "").strip().upper()
    if not z:
        return DEFAULT_TIER_UNKNOWN_LEAGUE
    if z.startswith("LCK"):
        return "T1" if z not in {"LCKC"} else "MINOR"
    if z.startswith("LPL"):
        return "T1"
    if z.startswith("LEC"):
        return "T1"
    if z.startswith("LCP"):
        return "T1"
    if z.startswith(("LCS", "CBLOL", "PCS", "VCS", "LJL")):
        return "T2"
    if z.startswith(("LFL", "PRM", "NLC", "LVP")):
        return "ERL"
    if z in {"MSI", "WORLDS", "EWC", "FST"}:
        return "T1"
    return DEFAULT_TIER_UNKNOWN_LEAGUE



def tier_for_league(league: str | None) -> str:
    """Return the tier label (T1/T2/ERL/MINOR/OTHER) for a given league name."""
    key = canonical_league(league)
    if not key:
        return "OTHER"
    if key in LEAGUE_TO_TIER:
        return str(LEAGUE_TO_TIER[key])
    return str(_tier_fallback_for_canonical_league(key))


def tier_base_points(tier: str | None) -> float:
    """Return the structural base-points added by a team's tier (e.g. T1=250, T2=80)."""
    if not tier:
        return float(TIER_BASE_POINTS.get("OTHER", 0.0))
    return float(TIER_BASE_POINTS.get(str(tier), TIER_BASE_POINTS.get("OTHER", 0.0)))


def league_base_points(league: str | None) -> float:
    """Return per-league bonus points on top of the tier baseline (0 if not listed)."""
    key = canonical_league(league)
    if not key:
        return 0.0
    return float(LEAGUE_BASE_POINTS.get(str(key), 0.0))


def league_strength_score(league: str | None) -> float:
    """Return a [0, 1] strength score for the league used by interleague gap calculations."""
    key = canonical_league(league)
    if not key:
        return float(DEFAULT_LEAGUE_STRENGTH_SCORE_BY_TIER.get("OTHER", 0.50))
    if key in LEAGUE_STRENGTH_SCORES:
        return float(LEAGUE_STRENGTH_SCORES[key])
    tier = str(tier_for_league(key) or "OTHER")
    return float(DEFAULT_LEAGUE_STRENGTH_SCORE_BY_TIER.get(tier, DEFAULT_LEAGUE_STRENGTH_SCORE_BY_TIER["OTHER"]))


def league_strength_band(league: str | None) -> str:
    """Return a qualitative strength label (elite/major/upper/regional/development/local)."""
    score = float(max(0.0, min(1.0, league_strength_score(league))))
    for band, floor in LEAGUE_STRENGTH_BANDS:
        if score >= float(floor):
            return str(band)
    return "local"
def enforce_monotone_caps(tier_caps: dict[str, float], tier_order: list[str] | None = None) -> dict[str, float]:
    """Enforce monotone (non-increasing) caps along tier order.

    This is a safety guard for GLOBAL ranking: caps for weaker tiers must not exceed caps for stronger tiers.
    Example: cap[T2] <= cap[T1], cap[ERL] <= cap[T2], etc.

    Returns a *new* dict (does not mutate input).
    """
    order = tier_order or TIER_ORDER
    out = dict(tier_caps or {})
    prev = None
    for t in order:
        if t not in out:
            continue
        v = float(out[t])
        if prev is None:
            prev = v
            continue
        if v > prev:
            v = prev
            out[t] = v
        prev = v
    return out


# --- International events ----------------------------------------------------
# Some parts of the pipeline need to know if a record belongs to an international
# tournament (teams from multiple regions). This is a *classification* helper only;
# it does not assign numeric strength by itself.

INTERNATIONAL_LEAGUE_KEYS: set[str] = {
    # Riot main international
    "WORLDS",
    "WORLDCHAMPIONSHIP",
    "WCS",
    "MSI",
    "MIDSEASONINVITATIONAL",
    "FST",
    "FIRSTSTAND",
    "FIRSTSTANDTOURNAMENT",
    # Multi-region events / circuits
    "AMERICAS",
    "AMERICASCUP",
    "CROSSCONFERENCE",
    "INTERLEAGUE",
    "INTERREGIONAL",
    "EMEAMASTERS",
    "EUM",
    "EU MASTERS",
    "RIFT RIVALS",
    "IW",
    "INTERNATIONALWILDCARD",
    "WILDCARD",
    "EWC",
    "ESPORTSWORLDCUP",
}


def is_international_league(league: str | None) -> bool:
    """Return True if the league/event is an international tournament.

    Works with either canonical short ids (e.g., 'WORLDS', 'MSI', 'EUM') or
    raw labels (e.g., 'EMEA Masters', 'World Championship').
    """
    if not league:
        return False
    s = str(league).strip()
    if not s:
        return False

    # Normalize to compare robustly
    up = s.upper()
    key = re.sub(r"[^A-Z0-9]+", "", up)

    # Direct key matches (normalized)
    direct = {re.sub(r"[^A-Z0-9]+", "", k.upper()) for k in INTERNATIONAL_LEAGUE_KEYS}
    if key in direct:
        return True

    # Substring heuristics for common raw labels
    if "WORLDS" in up or ("WORLD" in up and "CHAMP" in up):
        return True
    if "MID" in up and "SEASON" in up and "INVIT" in up:
        return True
    if "FIRST" in up and "STAND" in up:
        return True
    if "AMERICAS" in up and "CUP" in up:
        return True
    if "CROSS" in up and "CONFERENCE" in up:
        return True
    if "INTER" in up and "LEAGUE" in up:
        return True
    if "INTER" in up and "REGION" in up:
        return True
    if " MSI " in f" {up} ":
        return True
    if "EMEA" in up and "MASTERS" in up:
        return True
    if "EU" in up and "MASTERS" in up:
        return True
    if "RIFT" in up and "RIVALS" in up:
        return True
    if "ESPORTS" in up and "WORLD" in up and "CUP" in up:
        return True

    return False
