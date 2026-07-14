from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

REQUIRED_COLS = [
    "gameid",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "participantid",
    "teamname",
    "result",
    "gamelength",
]

# Optional team-level columns used by advanced state/ranking layers.
OPTIONAL_COL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "datacompleteness": ("datacompleteness",),
    "teamkills": ("teamkills",),
    "teamdeaths": ("teamdeaths",),
    "dragons": ("dragons",),
    "elementaldrakes": ("elementaldrakes",),
    "barons": ("barons",),
    "heralds": ("heralds",),
    "voidgrubs": ("voidgrubs", "void_grubs"),
    "atakhans": ("atakhans",),
    "towers": ("towers",),
    "turretplates": ("turretplates",),
    "inhibitors": ("inhibitors",),
    "firstblood": ("firstblood",),
    "firstdragon": ("firstdragon",),
    "firstbaron": ("firstbaron",),
    "firstherald": ("firstherald",),
    "firsttower": ("firsttower",),
    "firstmidtower": ("firstmidtower",),
    "firsttothreetowers": ("firsttothreetowers",),
    "golddiffat10": ("golddiffat10",),
    "golddiffat15": ("golddiffat15",),
    "golddiffat20": ("golddiffat20",),
    "xpdiffat10": ("xpdiffat10",),
    "xpdiffat15": ("xpdiffat15",),
    "xpdiffat20": ("xpdiffat20",),
    "csdiffat10": ("csdiffat10",),
    "csdiffat15": ("csdiffat15",),
    "csdiffat20": ("csdiffat20",),
}

FLOAT_STEMS: Tuple[str, ...] = (
    "teamkills",
    "teamdeaths",
    "dragons",
    "elementaldrakes",
    "barons",
    "heralds",
    "voidgrubs",
    "atakhans",
    "towers",
    "turretplates",
    "inhibitors",
    "golddiffat10",
    "golddiffat15",
    "golddiffat20",
    "xpdiffat10",
    "xpdiffat15",
    "xpdiffat20",
    "csdiffat10",
    "csdiffat15",
    "csdiffat20",
)

INT_STEMS: Tuple[str, ...] = (
    "firstblood",
    "firstdragon",
    "firstbaron",
    "firstherald",
    "firsttower",
    "firstmidtower",
    "firsttothreetowers",
)

TEXT_STEMS: Tuple[str, ...] = ("datacompleteness",)


def _resolve_optional_columns(cols: Iterable[str]) -> Tuple[List[str], Dict[str, str]]:
    available = set(cols)
    use_cols: List[str] = []
    rename_map: Dict[str, str] = {}
    for canonical, aliases in OPTIONAL_COL_ALIASES.items():
        chosen = next((a for a in aliases if a in available), None)
        if chosen is None:
            continue
        use_cols.append(chosen)
        if chosen != canonical:
            rename_map[chosen] = canonical
    return use_cols, rename_map


def _coerce_bool(v: Any) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n", ""}:
        return False
    try:
        return bool(int(float(v)))
    except Exception:
        return bool(v)


def _row_float(row: pd.Series, key: str) -> float | None:
    if key in row and pd.notna(row[key]):
        try:
            return float(row[key])
        except Exception:
            return None
    return None


def _row_int(row: pd.Series, key: str) -> int | None:
    if key in row and pd.notna(row[key]):
        try:
            return int(row[key])
        except Exception:
            return None
    return None


def _row_text(row: pd.Series, key: str) -> str | None:
    if key in row and pd.notna(row[key]):
        s = str(row[key]).strip()
        if s:
            return s
    return None


def load_oracle_elixir_games(
    csv_path: str,
    *,
    keep_playoffs: bool | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
) -> pd.DataFrame:
    """Load Oracle's Elixir match data and return a game-level DataFrame.

    Oracle's Elixir CSVs contain rows for each participant; team-level rows are
    participantid 100 (blue) and 200 (red). We collapse those into one row per gameid.

    Output columns:
      - gameid, date, league, year, split, playoffs, patch, game (map # in series)
      - blue, red, blue_win (int 0/1)
      - gamelength (seconds)
    """
    hdr = pd.read_csv(csv_path, low_memory=False, nrows=0)
    cols = list(hdr.columns)
    opt_present, opt_rename = _resolve_optional_columns(cols)
    use_cols = REQUIRED_COLS + opt_present
    df = pd.read_csv(csv_path, low_memory=False, usecols=use_cols)
    if opt_rename:
        df = df.rename(columns=opt_rename)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Keep only team rows
    team_cols = REQUIRED_COLS + [c for c in OPTIONAL_COL_ALIASES if c in df.columns]
    t = df[df["participantid"].isin([100, 200])][team_cols].copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")

    # IMPORTANT: keep_playoffs=True means "include playoffs" (do not filter).
    # keep_playoffs=False excludes playoffs games.
    if keep_playoffs is False:
        playoffs_bool = pd.to_numeric(t["playoffs"], errors="coerce").fillna(0).astype(int).astype(bool)
        t = t[~playoffs_bool]
    if min_date is not None:
        t = t[t["date"] >= pd.to_datetime(min_date)]
    if max_date is not None:
        t = t[t["date"] <= pd.to_datetime(max_date)]

    # Normalize strings
    t["teamname"] = t["teamname"].astype(str).str.strip()
    t["league"] = t["league"].astype(str).str.strip()
    t["split"] = t["split"].astype(str).str.strip()
    t["patch"] = t["patch"].astype(str).str.strip()

    # Build per-game records
    rows = []
    for gid, g in t.groupby("gameid", sort=False):
        blue = g[g["participantid"] == 100]
        red = g[g["participantid"] == 200]
        if len(blue) == 0 or len(red) == 0:
            continue
        b = blue.iloc[0]
        r = red.iloc[0]

        # Winner consistency check (some edge cases may exist; trust blue row)
        blue_win = int(b["result"] == 1)
        row: Dict[str, Any] = {
            "gameid": gid,
            "date": b["date"],
            "league": b["league"],
            "year": int(b["year"]) if pd.notna(b["year"]) else None,
            "split": b["split"],
            "playoffs": _coerce_bool(b["playoffs"]),
            "patch": b["patch"],
            "game": int(b["game"]) if pd.notna(b["game"]) else None,
            "blue": b["teamname"],
            "red": r["teamname"],
            "blue_win": blue_win,
            "gamelength": int(b["gamelength"]) if pd.notna(b["gamelength"]) else None,
        }

        for stem in FLOAT_STEMS:
            row[f"blue_{stem}"] = _row_float(b, stem)
            row[f"red_{stem}"] = _row_float(r, stem)
        for stem in INT_STEMS:
            row[f"blue_{stem}"] = _row_int(b, stem)
            row[f"red_{stem}"] = _row_int(r, stem)
        for stem in TEXT_STEMS:
            row[f"blue_{stem}"] = _row_text(b, stem)
            row[f"red_{stem}"] = _row_text(r, stem)

        rows.append(row)

    out_cols = [
        "gameid",
        "date",
        "league",
        "year",
        "split",
        "playoffs",
        "patch",
        "game",
        "blue",
        "red",
        "blue_win",
        "gamelength",
    ]
    for stem in FLOAT_STEMS:
        out_cols.extend([f"blue_{stem}", f"red_{stem}"])
    for stem in INT_STEMS:
        out_cols.extend([f"blue_{stem}", f"red_{stem}"])
    for stem in TEXT_STEMS:
        out_cols.extend([f"blue_{stem}", f"red_{stem}"])

    # Ensure required output columns exist even when no games were found.
    games = pd.DataFrame(rows, columns=out_cols)
    games = games.dropna(subset=["date", "blue", "red", "blue_win"])
    if not games.empty:
        games = games.sort_values(["date", "gameid"]).reset_index(drop=True)
    return games
