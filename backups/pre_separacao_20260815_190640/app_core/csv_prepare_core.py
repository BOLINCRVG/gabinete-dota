from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def csv_header_check_path(csv_path: str) -> Dict[str, Any]:
    """Read only the header row of a CSV and report column names and missing required fields."""
    out: Dict[str, Any] = {"ok": False, "columns": [], "missing_required": []}
    try:
        hdr = pd.read_csv(csv_path, nrows=0, low_memory=False)
        cols = [str(c) for c in list(hdr.columns)]
        out["columns"] = cols
        req = ["gameid", "participantid", "teamname", "date"]
        out["missing_required"] = [c for c in req if c not in cols]
        out["ok"] = True
        out["n_columns"] = len(cols)
    except (OSError, ValueError, pd.errors.EmptyDataError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def csv_diag_path(csv_path: str) -> Dict[str, Any]:
    """Return row counts, unique gameids, and completeness stats for a team-rows CSV."""
    out: Dict[str, Any] = {}
    try:
        df = pd.read_csv(csv_path, usecols=["gameid", "participantid"], low_memory=False)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if "gameid" not in df.columns or "participantid" not in df.columns:
                return {}
            df = df[["gameid", "participantid"]].copy()
        except (OSError, ValueError, pd.errors.EmptyDataError):
            return {}
    out["rows"] = len(df)
    gid = df["gameid"]
    miss = int(gid.isna().sum())
    miss += int((gid.fillna("").astype(str).str.strip() == "").sum())
    out["rows_missing_gameid"] = miss
    out["unique_gameids"] = int(df["gameid"].dropna().nunique())

    pid = pd.to_numeric(df["participantid"], errors="coerce")
    team_mask = pid.isin([100, 200])
    df_team = df.loc[team_mask].copy()
    out["team_rows"] = len(df_team)

    df_team = df_team.dropna(subset=["gameid"])
    if df_team.empty:
        out["games_complete"] = 0
        out["games_incomplete"] = 0
        return out

    grp = df_team.groupby("gameid")["participantid"].agg(
        lambda x: set(pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna().astype(int).tolist())
    )
    complete = int(grp.apply(lambda s: (100 in s) and (200 in s)).sum())
    incomplete = int(grp.apply(lambda s: ((100 in s) ^ (200 in s))).sum())
    out["games_complete"] = complete
    out["games_incomplete"] = incomplete
    return out
