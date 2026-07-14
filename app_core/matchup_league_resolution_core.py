"""Helpers to resolve per-team runtime league hints and audit matchup league integrity."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable


def _team_candidates(team_name: str, team_aliases_fn: Callable[[str], list[str]] | None) -> list[str]:
    out: list[str] = []

    def _push(value: Any) -> None:
        text = str(value or "").strip()
        if text and text.casefold() not in {item.casefold() for item in out}:
            out.append(text)

    _push(team_name)
    if callable(team_aliases_fn):
        try:
            for value in list(team_aliases_fn(team_name) or []):
                _push(value)
        except Exception:
            pass
    return out


def _infer_team_from_candidates(
    candidates: list[str],
    *,
    year: Any,
    infer_team_league_fn: Callable[[str, Any], str],
) -> str:
    for candidate in candidates:
        try:
            league = str(infer_team_league_fn(candidate, year) or "").strip()
        except Exception:
            league = ""
        if league:
            return league
    return ""


def _infer_match_from_candidates(
    candidates_a: list[str],
    candidates_b: list[str],
    *,
    year: Any,
    infer_match_league_fn: Callable[[str, str, Any], str],
) -> str:
    for team_a in candidates_a:
        for team_b in candidates_b:
            try:
                league = str(infer_match_league_fn(team_a, team_b, year) or "").strip()
            except Exception:
                league = ""
            if league:
                return league
    return ""


def resolve_runtime_team_leagues(
    *,
    team_a: str,
    team_b: str,
    year: Any,
    league_mode: str,
    fixed_league: Any,
    infer_team_league_fn: Callable[[str, Any], str],
    infer_match_league_fn: Callable[[str, str, Any], str],
    tier_from_league_fn: Callable[[str], str],
    team_aliases_fn: Callable[[str], list[str]] | None = None,
) -> Dict[str, str]:
    mode = str(league_mode or "").strip().lower()
    fixed = str(fixed_league or "").strip()
    team_a_name = str(team_a or "").strip()
    team_b_name = str(team_b or "").strip()
    candidates_a = _team_candidates(team_a_name, team_aliases_fn)
    candidates_b = _team_candidates(team_b_name, team_aliases_fn)

    league_a = ""
    league_b = ""
    source = ""

    if mode == "fixed" and fixed and fixed != "All":
        league_a = fixed
        league_b = fixed
        source = "fixed"
    else:
        league_a = _infer_team_from_candidates(
            candidates_a,
            year=year,
            infer_team_league_fn=infer_team_league_fn,
        )
        league_b = _infer_team_from_candidates(
            candidates_b,
            year=year,
            infer_team_league_fn=infer_team_league_fn,
        )

        if league_a and league_b:
            source = "team_pair"
        else:
            match_league = _infer_match_from_candidates(
                candidates_a,
                candidates_b,
                year=year,
                infer_match_league_fn=infer_match_league_fn,
            )
            if not league_a and match_league:
                league_a = match_league
            if not league_b and match_league:
                league_b = match_league
            if match_league and not source:
                source = "match_fallback"

        if not source:
            source = "team_partial" if (league_a or league_b) else "unresolved"

    tier_a = str(tier_from_league_fn(league_a) or "").strip() if league_a else ""
    tier_b = str(tier_from_league_fn(league_b) or "").strip() if league_b else ""
    same_league = bool(league_a and league_b and str(league_a).strip().upper() == str(league_b).strip().upper())
    shared_league = league_a if same_league else ""

    return {
        "league_a": league_a,
        "league_b": league_b,
        "tier_a": tier_a,
        "tier_b": tier_b,
        "same_league": "1" if same_league else "0",
        "shared_league": shared_league,
        "source": source,
    }


def build_matchup_integrity_row(
    *,
    team_a: str,
    team_b: str,
    league_a: str,
    league_b: str,
    source: str,
    trace_league_a: str = "",
    trace_league_b: str = "",
) -> Dict[str, Any]:
    la = str(league_a or "").strip()
    lb = str(league_b or "").strip()
    tla = str(trace_league_a or "").strip()
    tlb = str(trace_league_b or "").strip()
    inferred_same = bool(la and lb and la.upper() == lb.upper())
    trace_same = bool(tla and tlb and tla.upper() == tlb.upper())
    matchup_kind = "same_league" if inferred_same else "cross_league"

    issues: list[str] = []
    if (tla or tlb) and ((tla.upper(), tlb.upper()) != (la.upper(), lb.upper())):
        issues.append("trace_mismatch")
    if not la or not lb:
        issues.append("missing_team_league")
    if trace_same and not inferred_same:
        issues.append("trace_collapsed_to_same_league")

    status = "ok" if not issues else "review"
    return {
        "team_a": str(team_a or ""),
        "team_b": str(team_b or ""),
        "league_a": la,
        "league_b": lb,
        "matchup_kind": matchup_kind,
        "same_league": inferred_same,
        "source": str(source or ""),
        "trace_league_a": tla,
        "trace_league_b": tlb,
        "trace_same_league": trace_same,
        "status": status,
        "issues": ", ".join(issues),
    }


def seed_default_matchup_audit_cases() -> list[dict[str, Any]]:
    return [
        {"team_a": "Partizan Sangal", "team_b": "Unicorns of Love Sexy Edition", "year": 2026},
        {"team_a": "Gen.G", "team_b": "Bilibili Gaming", "year": 2026},
        {"team_a": "T1", "team_b": "Top Esports", "year": 2026},
        {"team_a": "G2 Esports", "team_b": "paiN Gaming", "year": 2026},
        {"team_a": "LOUD", "team_b": "Team Liquid", "year": 2026},
        {"team_a": "Cloud9", "team_b": "PSG Talon", "year": 2026},
        {"team_a": "Natus Vincere", "team_b": "Myth Esports", "year": 2026},
        {"team_a": "Once Upon A Team", "team_b": "DRX Challengers", "year": 2026},
    ]


def merge_matchup_audit_cases(
    base_cases: Iterable[dict[str, Any]],
    extra_cases: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for row in list(base_cases or []) + list(extra_cases or []):
        team_a = str((row or {}).get("team_a") or (row or {}).get("teamA") or "").strip()
        team_b = str((row or {}).get("team_b") or (row or {}).get("teamB") or "").strip()
        year = str((row or {}).get("year") or "").strip()
        if not team_a or not team_b:
            continue
        key = (team_a.casefold(), team_b.casefold(), year)
        if key in seen:
            continue
        seen.add(key)
        out.append({"team_a": team_a, "team_b": team_b, "year": (row or {}).get("year")})
    return out
