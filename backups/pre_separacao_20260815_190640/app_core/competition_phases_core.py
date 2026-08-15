"""Manual competition-phase annotations.

Some events arrive in the CSV without a usable `split` (e.g. EWC: a single
league bucket mixing qualifiers and the main event, all `split` empty). This
module lets the user annotate phases by (league, optional year, date range)
in a small JSON file; a resolver applies those rules to `team_games`, setting
the `split` (and optionally `playoffs`) for matching rows.

Design: SAFE NO-OP by default. If the file is missing or has no rules, the
dataframe is returned unchanged — so it never affects the pipeline until the
user fills in real phase dates. See `competition_phases.json` (template).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Mapping

import pandas as pd


def load_phase_rules(path: str | Path) -> List[dict]:
    """Load the annotation rules from JSON. Returns [] if missing/invalid/empty.

    Expected shape: {"rules": [{"league","year"?,"date_from","date_to","split","playoffs"?}, ...]}
    """
    try:
        p = Path(path)
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    rules = data.get("rules") if isinstance(data, Mapping) else None
    if not isinstance(rules, list):
        return []
    out: List[dict] = []
    for r in rules:
        if not isinstance(r, Mapping):
            continue
        league = str(r.get("league", "") or "").strip()
        date_from = str(r.get("date_from", "") or "").strip()
        date_to = str(r.get("date_to", "") or "").strip()
        split = str(r.get("split", "") or "").strip()
        if not (league and date_from and date_to and split):
            continue  # incomplete rule → skip
        out.append({
            "league": league,
            "year": r.get("year"),
            "date_from": date_from,
            "date_to": date_to,
            "split": split,
            "playoffs": r.get("playoffs"),
        })
    return out


def apply_phase_overrides(df: pd.DataFrame, rules: List[dict]) -> pd.DataFrame:
    """Override `split` (and optional `playoffs`) for rows matching each rule.

    A row matches a rule when: same league (case-insensitive), same year if the
    rule specifies one, and its date falls within [date_from, date_to]. SAFE
    NO-OP when there are no rules or required columns are missing.
    """
    if df is None or not rules or "league" not in df.columns or "date" not in df.columns:
        return df
    out = df.copy()
    _league = out["league"].astype(str).str.strip().str.upper()
    _date = pd.to_datetime(out["date"], errors="coerce")
    _year = pd.to_numeric(out["year"], errors="coerce") if "year" in out.columns else None
    if "split" not in out.columns:
        out["split"] = ""
    for rule in rules:
        try:
            d_from = pd.to_datetime(rule["date_from"], errors="coerce")
            d_to = pd.to_datetime(rule["date_to"], errors="coerce")
        except (KeyError, ValueError):
            continue
        if pd.isna(d_from) or pd.isna(d_to):
            continue
        mask = (_league == str(rule["league"]).strip().upper()) & _date.between(d_from, d_to)
        if rule.get("year") is not None and _year is not None:
            try:
                mask = mask & (_year == int(rule["year"]))
            except (TypeError, ValueError):
                pass
        if not bool(mask.any()):
            continue
        out.loc[mask, "split"] = str(rule["split"])
        if rule.get("playoffs") is not None and "playoffs" in out.columns:
            out.loc[mask, "playoffs"] = bool(rule["playoffs"])
    return out


def apply_competition_phases(df: pd.DataFrame, *, path: str | Path) -> pd.DataFrame:
    """Convenience: load rules from `path` and apply them. No-op if none."""
    return apply_phase_overrides(df, load_phase_rules(path))


# ---------------------------------------------------------------------------
# EWC regional phases (automatic, no manual file needed)
# ---------------------------------------------------------------------------

INTERNATIONAL_LEAGUES = frozenset({"EWC", "MSI", "WORLDS", "FST", "IEM", "FIRST STAND"})

# Domestic league → region. Fallback: the league code itself (so an unmapped
# league still groups its own qualifier, just labeled by the code).
LEAGUE_REGION = {
    "LPL": "CN", "LDL": "CN",
    "LCK": "KR", "LCKC": "KR",
    "LCS": "NA", "NACL": "NA", "LTA N": "NA", "LTA": "NA",
    "LEC": "EU", "LFL": "EU", "EM": "EU", "NLC": "EU", "PRM": "EU", "HLL": "EU",
    "LCP": "APAC", "PCS": "APAC", "VCS": "APAC", "LJL": "APAC",
    "CBLOL": "BR", "CBLOLA": "BR",
}


def region_of_league(league: Any) -> str:
    lg = str(league or "").strip().upper()
    return LEAGUE_REGION.get(lg, lg)


def apply_ewc_regional_phases(df: pd.DataFrame, *, ewc_league: str = "EWC") -> pd.DataFrame:
    """Label EWC games by regional qualifier vs main event.

    The EWC league bucket is the EWC QUALIFY — regional qualifiers (intra-region
    games). So:
      - intra-region game (both teams same domestic region) → split "Qualify <REGION>"
      - genuinely cross-region game (real main event)        → split "Main"
    Region is derived from each team's most-common domestic (non-international)
    league. A team that only shows up in EWC (no domestic games → unknown region)
    inherits its opponent's region, since the qualifier is intra-region — this
    avoids labeling such a game as a spurious "Main". SAFE NO-OP when there are
    no EWC rows or the required columns are missing.
    """
    if (df is None or "league" not in df.columns
            or "team" not in df.columns or "opponent" not in df.columns):
        return df
    lg_up = df["league"].astype(str).str.strip().str.upper()
    ewc_mask = lg_up == str(ewc_league).strip().upper()
    if not bool(ewc_mask.any()):
        return df
    out = df.copy()
    # Domestic region per team, from non-international games.
    dom = out[~lg_up.isin(INTERNATIONAL_LEAGUES)]
    if not dom.empty and "league" in dom.columns:
        team_dom = dom.groupby("team")["league"].agg(
            lambda s: s.astype(str).value_counts().index[0] if len(s) else ""
        )
    else:
        team_dom = pd.Series(dtype=object)

    def _region(team: Any) -> str:
        return region_of_league(team_dom.get(str(team), ""))

    if "split" not in out.columns:
        out["split"] = ""
    for i in out.index[ewc_mask]:
        ra = _region(out.at[i, "team"])
        rb = _region(out.at[i, "opponent"])
        # Unknown region (team only in EWC) inherits the opponent's — qualifiers
        # are intra-region, so this is the correct fill, not a cross-region game.
        if not ra:
            ra = rb
        if not rb:
            rb = ra
        if ra and ra == rb:
            out.at[i, "split"] = f"Qualify {ra}"
        elif not ra and not rb:
            out.at[i, "split"] = "Qualify"   # região desconhecida, mas é qualify
        else:
            out.at[i, "split"] = "Main"       # cross-região genuíno (main event)
    return out
