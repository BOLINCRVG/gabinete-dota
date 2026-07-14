"""Core helpers for players/team Elo auxiliary computations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import re

import numpy as np
import pandas as pd


def elo_expected(ra: float, rb: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / float(scale)))


def discover_player_csv_paths(csv_path: str) -> list[str]:
    """Best-effort discovery of historical player CSVs around current CSV path."""
    out: list[str] = []
    if not csv_path:
        return out
    try:
        p = Path(str(csv_path))
        if p.exists():
            out.append(str(p))
        base_dir = p.parent
        root = base_dir.parent if base_dir.name.lower().startswith("csv_") else base_dir
        hist_dir = root / "CSV_HISTORICO"
        if hist_dir.exists():
            for fp in sorted(hist_dir.glob("*.csv")):
                nm = fp.name
                m = re.search(r"(20\d{2})", nm)
                if m:
                    y = int(m.group(1))
                    if y >= 2024:
                        out.append(str(fp))
                else:
                    out.append(str(fp))
        seen: set[str] = set()
        out2: list[str] = []
        for x in out:
            if x not in seen:
                out2.append(x)
                seen.add(x)
        return out2
    except Exception:
        return out


def load_player_rows_multi(
    *,
    sig_list: Tuple[Tuple[str, int, int], ...],
    load_player_rows_fn: Callable[[Tuple[str, int, int]], pd.DataFrame],
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for sig in sig_list:
        try:
            d = load_player_rows_fn(sig)
            if d is not None and not d.empty:
                dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
    return out


def player_key_from_row(row: Any) -> str:
    try:
        pid = getattr(row, "playerid", "")
        if pid is None:
            pid = ""
        pid = str(pid).strip()
        if pid and pid.lower() not in ("nan", "none"):
            return pid
    except Exception:
        pass
    try:
        pn = getattr(row, "playername", "")
        return str(pn).strip()
    except Exception:
        return ""


def compute_player_elo_global(
    *,
    sig_list: Tuple[Tuple[str, int, int], ...],
    load_player_rows_multi_fn: Callable[[Tuple[Tuple[str, int, int], ...]], pd.DataFrame],
    half_life_days: float,
    k_player: float,
    shrink_m: float,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> Dict[str, Any]:
    """Global player Elo (2024+) with inactivity decay and conservative shrink."""
    df = load_player_rows_multi_fn(sig_list)
    if df is None or df.empty:
        return {
            "rating": {},
            "games": {},
            "last": {},
            "id2name": {},
            "name2id": {},
        }

    if "year" in df.columns:
        df = df[df["year"].fillna(0) >= 2024].copy()

    need = ["gameid", "date", "teamname", "result", "playername"]
    for c in need:
        if c not in df.columns:
            return {
                "rating": {},
                "games": {},
                "last": {},
                "id2name": {},
                "name2id": {},
            }

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["gameid", "date", "teamname", "playername", "result"])
    if df.empty:
        return {
            "rating": {},
            "games": {},
            "last": {},
            "id2name": {},
            "name2id": {},
        }

    if "playerid" in df.columns:
        df["player_key"] = df["playerid"].fillna("").astype(str).str.strip()
        mask = (df["player_key"] == "") | (df["player_key"].str.lower().isin(["nan", "none"]))
        df.loc[mask, "player_key"] = df.loc[mask, "playername"].astype(str).str.strip()
    else:
        df["player_key"] = df["playername"].astype(str).str.strip()

    df["teamname"] = df["teamname"].astype(str).str.strip()
    df["playername"] = df["playername"].astype(str).str.strip()

    id2name: Dict[str, str] = {}
    name2id: Dict[str, str] = {}
    if "playerid" in df.columns:
        tmp = df[["playerid", "playername"]].copy()
        tmp["playerid"] = tmp["playerid"].fillna("").astype(str).str.strip()
        tmp = tmp[tmp["playerid"] != ""]
        if not tmp.empty:
            for pid, g in tmp.groupby("playerid"):
                try:
                    nm = g["playername"].mode().iloc[0]
                except Exception:
                    nm = g["playername"].iloc[0]
                id2name[str(pid)] = str(nm)
            for nm, g in tmp.groupby("playername"):
                try:
                    pid = g["playerid"].mode().iloc[0]
                except Exception:
                    pid = g["playerid"].iloc[0]
                name2id[str(nm)] = str(pid)

    df = df.sort_values("date", ascending=True)
    grouped = df.groupby(["gameid", "teamname"], sort=False)

    games_by_gid: Dict[str, Any] = {}
    for (gid, tname), g in grouped:
        dt = g["date"].max()
        res = float(g["result"].iloc[0])
        players = g["player_key"].dropna().astype(str).str.strip().tolist()
        players = [p for p in players if p and p.lower() not in ("nan", "none")]
        if not players:
            continue
        d = games_by_gid.setdefault(str(gid), {"date": dt, "teams": []})
        d["teams"].append((str(tname), players, 1.0 if res >= 1.0 else 0.0))

    items = [(gid, v["date"], v["teams"]) for gid, v in games_by_gid.items() if len(v["teams"]) >= 2]
    items.sort(key=lambda x: x[1])

    rating: Dict[str, float] = {}
    games_cnt: Dict[str, int] = {}
    last_play: Dict[str, Any] = {}

    def _decay_player(pk: str, now_dt: Any) -> None:
        if pk not in rating:
            rating[pk] = float(base_rating)
            games_cnt[pk] = 0
            last_play[pk] = now_dt
            return
        try:
            prev = last_play.get(pk)
            if prev is None:
                last_play[pk] = now_dt
                return
            dt_days = (now_dt - prev).total_seconds() / 86400.0
            if dt_days <= 0:
                last_play[pk] = now_dt
                return
            factor = 0.5 ** (float(dt_days) / float(half_life_days)) if half_life_days and half_life_days > 0 else 1.0
            rating[pk] = float(base_rating) + (float(rating.get(pk, base_rating)) - float(base_rating)) * float(factor)
            last_play[pk] = now_dt
        except Exception:
            last_play[pk] = now_dt

    for _gid, dt, teams in items:
        teams2 = sorted(teams, key=lambda x: len(x[1]), reverse=True)[:2]
        (_ta, pa, sa), (_tb, pb, _sb) = teams2[0], teams2[1]

        for pk in pa:
            _decay_player(pk, dt)
        for pk in pb:
            _decay_player(pk, dt)

        ra = float(np.mean([rating.get(pk, base_rating) for pk in pa])) if pa else float(base_rating)
        rb = float(np.mean([rating.get(pk, base_rating) for pk in pb])) if pb else float(base_rating)
        ea = elo_expected(ra, rb, scale=scale)

        gpa = float(np.mean([games_cnt.get(pk, 0) for pk in pa])) if pa else 0.0
        gpb = float(np.mean([games_cnt.get(pk, 0) for pk in pb])) if pb else 0.0
        shrink = min(gpa, gpb) / (min(gpa, gpb) + float(shrink_m)) if shrink_m and shrink_m > 0 else 1.0
        k_eff = float(k_player) * float(shrink)

        da = k_eff * (float(sa) - float(ea))
        if pa:
            per = da / float(len(pa))
            for pk in pa:
                rating[pk] = float(rating.get(pk, base_rating)) + per
                games_cnt[pk] = int(games_cnt.get(pk, 0) + 1)
        if pb:
            per = (-da) / float(len(pb))
            for pk in pb:
                rating[pk] = float(rating.get(pk, base_rating)) + per
                games_cnt[pk] = int(games_cnt.get(pk, 0) + 1)

    return {
        "rating": rating,
        "games": games_cnt,
        "last": {k: (v.isoformat() if hasattr(v, "isoformat") else str(v)) for k, v in last_play.items()},
        "id2name": id2name,
        "name2id": name2id,
    }


def compute_team_elo_season(
    *,
    team_games: pd.DataFrame,
    filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    half_life_days: float,
    k_team: float,
    shrink_m: float,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> Dict[str, Any]:
    if team_games is None or team_games.empty:
        return {"rating": {}, "games": {}, "last": {}}

    tg = apply_filters_fn(team_games, filters)
    need = ["gameid", "date", "team", "opponent", "win"]
    for c in need:
        if c not in tg.columns:
            return {"rating": {}, "games": {}, "last": {}}

    dfm = tg.copy()
    dfm["team"] = dfm["team"].astype(str)
    dfm["opponent"] = dfm["opponent"].astype(str)
    dfm = dfm[dfm["team"] != dfm["opponent"]].copy()
    dfm = dfm[dfm["team"] < dfm["opponent"]].copy()
    dfm["date"] = pd.to_datetime(dfm["date"], errors="coerce")
    dfm = dfm.dropna(subset=["date"])
    if dfm.empty:
        return {"rating": {}, "games": {}, "last": {}}

    dfm = dfm.sort_values("date", ascending=True)
    rating: Dict[str, float] = {}
    games_cnt: Dict[str, int] = {}
    last_play: Dict[str, Any] = {}

    def _decay_team(t: str, now_dt: Any) -> None:
        if t not in rating:
            rating[t] = float(base_rating)
            games_cnt[t] = 0
            last_play[t] = now_dt
            return
        prev = last_play.get(t)
        if prev is None:
            last_play[t] = now_dt
            return
        dt_days = (now_dt - prev).total_seconds() / 86400.0
        if dt_days <= 0:
            last_play[t] = now_dt
            return
        factor = 0.5 ** (float(dt_days) / float(half_life_days)) if half_life_days and half_life_days > 0 else 1.0
        rating[t] = float(base_rating) + (float(rating.get(t, base_rating)) - float(base_rating)) * float(factor)
        last_play[t] = now_dt

    for row in dfm.itertuples(index=False):
        dt = row.date
        ta = str(row.team)
        tb = str(row.opponent)
        sa = 1.0 if float(row.win) >= 1.0 else 0.0

        _decay_team(ta, dt)
        _decay_team(tb, dt)

        ra = float(rating.get(ta, base_rating))
        rb = float(rating.get(tb, base_rating))
        ea = elo_expected(ra, rb, scale=scale)

        gpa = float(games_cnt.get(ta, 0))
        gpb = float(games_cnt.get(tb, 0))
        shrink = min(gpa, gpb) / (min(gpa, gpb) + float(shrink_m)) if shrink_m and shrink_m > 0 else 1.0
        k_eff = float(k_team) * float(shrink)

        delta = k_eff * (sa - ea)
        rating[ta] = ra + delta
        rating[tb] = rb - delta
        games_cnt[ta] = int(games_cnt.get(ta, 0) + 1)
        games_cnt[tb] = int(games_cnt.get(tb, 0) + 1)

    return {
        "rating": rating,
        "games": games_cnt,
        "last": {k: (v.isoformat() if hasattr(v, "isoformat") else str(v)) for k, v in last_play.items()},
    }



def elo_rank_table(
    team_games: pd.DataFrame,
    *,
    filters: Any,
    half_life_days: float,
    k_factor: float = 24.0,
    base_rating: float = 1500.0,
    scale: float = 400.0,
    format_pct_fn=None,
    format_num_fn=None,
) -> pd.DataFrame:
    """Ranking por Elo com peso de recência."""
    from mlcore.stats_teams import apply_filters, recency_weights

    if team_games is None or team_games.empty:
        return pd.DataFrame(columns=["team", "league", "games", "winrate", "elo"])

    tg = apply_filters(team_games, filters)

    need = ["gameid", "date", "team", "opponent", "win"]
    for c in need:
        if c not in tg.columns:
            return pd.DataFrame(columns=["team", "league", "games", "winrate", "elo"])

    dfm = tg.copy()
    dfm["team"] = dfm["team"].astype(str)
    dfm["opponent"] = dfm["opponent"].astype(str)
    dfm = dfm[dfm["team"] != dfm["opponent"]]
    dfm = dfm[dfm["team"] < dfm["opponent"]].copy()

    dfm["date"] = pd.to_datetime(dfm["date"], errors="coerce")
    ref_date = dfm["date"].max()
    if pd.isna(ref_date):
        ref_date = pd.Timestamp.utcnow().tz_localize(None)

    dfm = dfm.sort_values("date", ascending=True)
    w = recency_weights(dfm["date"], half_life_days=float(half_life_days), ref_date=ref_date)

    teams = sorted(set(dfm["team"].unique().tolist()) | set(dfm["opponent"].unique().tolist()))
    rating = {t: float(base_rating) for t in teams}
    games: Dict[str, int] = {t: 0 for t in teams}
    wins: Dict[str, float] = {t: 0.0 for t in teams}
    leagues: Dict[str, str] = {}
    if "league" in tg.columns:
        tmp = tg[["team", "league"]].copy()
        tmp["team"] = tmp["team"].astype(str)
        tmp["league"] = tmp["league"].fillna("").astype(str)
        for t, g in tmp.groupby("team"):
            if g.shape[0]:
                leagues[t] = g["league"].mode().iloc[0] if not g["league"].mode().empty else ""

    for i, row in enumerate(dfm.itertuples(index=False)):
        ta = row.team
        tb = row.opponent
        sa = 1.0 if float(row.win) >= 1.0 else 0.0
        ra = rating.get(ta, float(base_rating))
        rb = rating.get(tb, float(base_rating))
        ea = elo_expected(ra, rb, scale=scale)
        k = float(k_factor) * float(w[i])
        delta = k * (sa - ea)
        rating[ta] = ra + delta
        rating[tb] = rb - delta
        games[ta] = games.get(ta, 0) + 1
        games[tb] = games.get(tb, 0) + 1
        wins[ta] = wins.get(ta, 0.0) + sa
        wins[tb] = wins.get(tb, 0.0) + (1.0 - sa)

    rows = []
    for t in teams:
        g = int(games.get(t, 0))
        if g <= 0:
            continue
        wr = float(wins.get(t, 0.0)) / float(g) if g else float("nan")
        rows.append({
            "team": t,
            "league": leagues.get(t, "") if leagues else "",
            "games": g,
            "winrate": wr,
            "elo": float(rating.get(t, base_rating)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["team", "league", "games", "winrate", "elo"])

    out = out.sort_values(["elo", "winrate", "games"], ascending=[False, False, False]).reset_index(drop=True)
    if callable(format_pct_fn):
        out["winrate"] = out["winrate"].apply(format_pct_fn)
    if callable(format_num_fn):
        out["elo"] = out["elo"].apply(lambda x: format_num_fn(x, 0))
    return out


def delta_players_effect(delta_raw: float, *, mode: str = "saturado", cap: float = 1.2, slope: float = 1.6) -> float:
    """Transform raw players delta before applying on logit/diff.

    - linear: effect = delta_raw
    - saturado: effect = cap * tanh(slope * delta_raw / cap)
    """
    import math
    try:
        d = float(delta_raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(d):
        return 0.0
    m = str(mode or "saturado").strip().lower()
    if m == "linear":
        return float(d)
    c = float(cap) if math.isfinite(float(cap)) else 1.2
    s = float(slope) if math.isfinite(float(slope)) else 1.6
    c = max(0.05, abs(c))
    s = max(0.1, abs(s))
    return float(c * math.tanh((s * d) / c))
