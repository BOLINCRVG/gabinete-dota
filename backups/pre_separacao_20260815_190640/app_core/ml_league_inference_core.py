from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import pandas as pd


def _dedupe_team_candidates(team: str, team_aliases: Iterable[str] | None = None) -> list[str]:
    out: list[str] = []
    for value in [team, *(list(team_aliases or []))]:
        text = str(value or "").strip()
        if text and text.casefold() not in {item.casefold() for item in out}:
            out.append(text)
    return out


def _select_top_league(
    leagues: pd.Series,
    *,
    norm_league_key_fn: Callable[[str], str],
    major_leagues: set[str] | frozenset[str],
) -> str:
    if leagues.empty:
        return ""
    vc = leagues.value_counts()
    vc_df = vc.reset_index()
    vc_df.columns = ["league", "count"]
    vc_df["league_norm"] = vc_df["league"].astype(str).apply(norm_league_key_fn)
    vc_df = vc_df.sort_values(["count", "league_norm"], ascending=[False, True])

    top_league = str(vc_df.iloc[0]["league"])
    top_cnt = int(vc_df.iloc[0]["count"])

    majors = vc_df[vc_df["league_norm"].isin(set(major_leagues))]
    if not majors.empty:
        best_major = majors.iloc[0]
        major_league = str(best_major["league"])
        major_cnt = int(best_major["count"])
        min_major_games = 5
        if (norm_league_key_fn(top_league) in set(major_leagues)) or (
            major_cnt >= min_major_games and major_cnt >= max(2, int(0.5 * top_cnt))
        ):
            return major_league

    return top_league


def infer_team_league_from_csv(
    csv_sig: Tuple[str, int, int],
    team: str,
    year: Optional[int],
    *,
    norm_league_key_fn: Callable[[str], str],
    major_leagues: set[str] | frozenset[str],
    team_aliases: Iterable[str] | None = None,
    fallback_any_year: bool = True,
) -> str:
    csv_path, _size, _mtime = csv_sig
    if not csv_path or not team:
        return ""

    usecols = ["teamname", "league", "year", "participantid"]
    try:
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            return ""
        keep = [c for c in usecols if c in df.columns]
        if not keep:
            return ""
        df = df[keep].copy()

    if "teamname" not in df.columns or "league" not in df.columns:
        return ""

    team_keys = [item.casefold() for item in _dedupe_team_candidates(team, team_aliases)]
    series_team = df["teamname"].astype(str).str.strip().str.casefold()
    if "participantid" in df.columns:
        df = df[df["participantid"].isin([100, 200])]

    def _matching_leagues(frame: pd.DataFrame) -> pd.Series:
        return frame.loc[series_team.loc[frame.index].isin(team_keys), "league"].dropna().astype(str).str.strip()

    df_year = df
    if year is not None and "year" in df.columns:
        df_year = df[df["year"] == int(year)]
    leagues = _matching_leagues(df_year)
    if leagues.empty and bool(fallback_any_year):
        df_any = df
        if "year" in df_any.columns:
            years = pd.to_numeric(df_any.loc[series_team.isin(team_keys), "year"], errors="coerce").dropna()
            if not years.empty:
                latest_year = int(years.max())
                df_any = df_any[pd.to_numeric(df_any["year"], errors="coerce") == latest_year]
        leagues = _matching_leagues(df_any)
    if leagues.empty:
        return ""
    return _select_top_league(
        leagues,
        norm_league_key_fn=norm_league_key_fn,
        major_leagues=major_leagues,
    )


def infer_match_league_from_csv(
    csv_sig: Tuple[str, int, int],
    team_a: str,
    team_b: str,
    year: Optional[int],
    *,
    norm_league_key_fn: Callable[[str], str],
    major_leagues: set[str] | frozenset[str],
    load_team_games_fn: Callable[[Tuple[str, int, int]], pd.DataFrame],
    infer_team_league_fn: Callable[[Tuple[str, int, int], str, Optional[int]], str],
    team_aliases_a: Iterable[str] | None = None,
    team_aliases_b: Iterable[str] | None = None,
    fallback_any_year: bool = True,
) -> str:
    if not team_a or not team_b:
        return ""
    try:
        df = load_team_games_fn(csv_sig)
    except Exception:
        df = None
    if df is None or getattr(df, "empty", True):
        lg_a = infer_team_league_fn(csv_sig, team_a, year)
        lg_b = infer_team_league_fn(csv_sig, team_b, year)
        if lg_a and lg_b and norm_league_key_fn(lg_a) == norm_league_key_fn(lg_b):
            return lg_a
        return lg_a or lg_b or ""

    dd = df
    if year is not None and "year" in dd.columns:
        dd = dd[dd["year"] == int(year)]

    team_keys_a = [item.casefold() for item in _dedupe_team_candidates(team_a, team_aliases_a)]
    team_keys_b = [item.casefold() for item in _dedupe_team_candidates(team_b, team_aliases_b)]
    s_team = dd["team"].astype(str).str.strip().str.casefold()
    s_opp = dd["opponent"].astype(str).str.strip().str.casefold()

    mask = (s_team.isin(team_keys_a) & s_opp.isin(team_keys_b)) | (s_team.isin(team_keys_b) & s_opp.isin(team_keys_a))
    h2h = dd.loc[mask, "league"].dropna().astype(str).str.strip()
    if h2h.empty and bool(fallback_any_year) and "year" in df.columns:
        dd_any = df
        mask_any = (
            dd_any["team"].astype(str).str.strip().str.casefold().isin(team_keys_a)
            & dd_any["opponent"].astype(str).str.strip().str.casefold().isin(team_keys_b)
        ) | (
            dd_any["team"].astype(str).str.strip().str.casefold().isin(team_keys_b)
            & dd_any["opponent"].astype(str).str.strip().str.casefold().isin(team_keys_a)
        )
        years = pd.to_numeric(dd_any.loc[mask_any, "year"], errors="coerce").dropna() if "year" in dd_any.columns else pd.Series(dtype=float)
        if not years.empty:
            latest_year = int(years.max())
            dd_any = dd_any[pd.to_numeric(dd_any["year"], errors="coerce") == latest_year]
            mask_any = (
                dd_any["team"].astype(str).str.strip().str.casefold().isin(team_keys_a)
                & dd_any["opponent"].astype(str).str.strip().str.casefold().isin(team_keys_b)
            ) | (
                dd_any["team"].astype(str).str.strip().str.casefold().isin(team_keys_b)
                & dd_any["opponent"].astype(str).str.strip().str.casefold().isin(team_keys_a)
            )
            h2h = dd_any.loc[mask_any, "league"].dropna().astype(str).str.strip()
    if not h2h.empty:
        return _select_top_league(
            h2h,
            norm_league_key_fn=norm_league_key_fn,
            major_leagues=major_leagues,
        )

    lg_a = infer_team_league_fn(csv_sig, team_a, year)
    lg_b = infer_team_league_fn(csv_sig, team_b, year)
    if lg_a and lg_b and norm_league_key_fn(lg_a) == norm_league_key_fn(lg_b):
        return lg_a
    return lg_a or lg_b or ""
