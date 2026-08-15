"""Core helpers for Players lineup and byTournament impact utilities."""

from __future__ import annotations

from typing import Any, Callable
import difflib

import pandas as pd

from mlcore.team_artifact_normalize import _norm_team_key


def parse_pct(x: Any) -> float:
    s = str(x or "").strip().replace("%", "")
    if not s:
        return float("nan")
    try:
        return float(s) / 100.0
    except Exception:
        return float("nan")


def load_players_impact(
    *,
    csv_path: str,
    norm_pos_fn: Callable[[str], str],
    parse_pct_fn: Callable[[Any], float] | None = None,
) -> pd.DataFrame:
    """Load and score byTournament player impact table."""
    parser = parse_pct_fn if callable(parse_pct_fn) else parse_pct
    df = pd.read_csv(csv_path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["team", "pos", "player", "gp", "score"])

    df = df.copy()
    col_player = "Player" if "Player" in df.columns else ("player" if "player" in df.columns else None)
    col_team = "Team" if "Team" in df.columns else ("team" if "team" in df.columns else None)
    col_pos = "Pos" if "Pos" in df.columns else ("pos" if "pos" in df.columns else None)
    if not (col_player and col_team and col_pos):
        return pd.DataFrame(columns=["team", "pos", "player", "gp", "score"])

    df["player"] = df[col_player].astype(str).str.strip()
    df["team"] = df[col_team].astype(str).str.strip()
    df["pos"] = df[col_pos].astype(str).map(norm_pos_fn)

    col_gp = "GP" if "GP" in df.columns else ("gp" if "gp" in df.columns else None)
    if col_gp:
        df["gp"] = pd.to_numeric(df[col_gp], errors="coerce").fillna(0).astype(int)
    else:
        df["gp"] = 0

    feats = {
        "gd10": ("GD10", "gd10"),
        "xpd10": ("XPD10", "xpd10"),
        "csd10": ("CSD10", "csd10"),
        "dmgp": ("DMG%", "DMG_PCT", "dmg%", "dmgp"),
        "dthp": ("DTH%", "DTH_PCT", "dth%", "dthp"),
    }

    def _get_first_col(cols: tuple[str, ...]) -> str | None:
        for c in cols:
            if c in df.columns:
                return c
        return None

    col_gd10 = _get_first_col(feats["gd10"])
    col_xpd10 = _get_first_col(feats["xpd10"])
    col_csd10 = _get_first_col(feats["csd10"])
    col_dmgp = _get_first_col(feats["dmgp"])
    col_dthp = _get_first_col(feats["dthp"])

    df["gd10"] = pd.to_numeric(df[col_gd10], errors="coerce") if col_gd10 else float("nan")
    df["xpd10"] = pd.to_numeric(df[col_xpd10], errors="coerce") if col_xpd10 else float("nan")
    df["csd10"] = pd.to_numeric(df[col_csd10], errors="coerce") if col_csd10 else float("nan")
    df["dmgp"] = df[col_dmgp].apply(parser) if col_dmgp else float("nan")
    df["dthp"] = df[col_dthp].apply(parser) if col_dthp else float("nan")

    rows: list[pd.DataFrame] = []
    for _pos, g in df.groupby("pos"):
        gg = g.copy()
        for c in ["gd10", "xpd10", "csd10", "dmgp", "dthp"]:
            med = gg[c].median(skipna=True)
            gg[c] = gg[c].fillna(med if pd.notna(med) else 0.0)

        def z(col: str) -> pd.Series:
            mu = float(gg[col].mean()) if gg.shape[0] else 0.0
            sd = float(gg[col].std(ddof=0)) if gg.shape[0] else 0.0
            if not (sd > 1e-9):
                return gg[col] * 0.0
            return (gg[col] - mu) / sd

        z_gd = z("gd10")
        z_xp = z("xpd10")
        z_cs = z("csd10")
        z_dmg = z("dmgp")
        z_dth = z("dthp")
        gg["score_raw"] = (1.0 * z_gd) + (0.8 * z_xp) + (0.6 * z_cs) + (0.8 * z_dmg) + (-0.8 * z_dth)
        rows.append(gg)

    out = pd.concat(rows, ignore_index=True) if rows else df.copy()
    out = out[["team", "pos", "player", "gp", "score_raw"]].copy()
    out = out.rename(columns={"score_raw": "score"})
    return out


def apply_shrink_m(df_impact: pd.DataFrame, m: int) -> pd.DataFrame:
    if df_impact is None or df_impact.empty:
        return df_impact
    mm = max(1, int(m) if m is not None else 10)
    out = df_impact.copy()
    gp = pd.to_numeric(out.get("gp"), errors="coerce").fillna(0.0)
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0) * (gp / (gp + float(mm)))
    return out


def best_match(name: str, choices: list, cutoff: float = 0.72) -> str:
    n = str(name or "").strip()
    if not n:
        return ""
    if n in choices:
        return n
    low = {str(c).lower(): str(c) for c in choices}
    if n.lower() in low:
        return low[n.lower()]
    m = difflib.get_close_matches(n, [str(c) for c in choices], n=1, cutoff=cutoff)
    return str(m[0]) if m else ""


def latest_lineup_names(
    *,
    df_in: pd.DataFrame,
    team_name: str,
    norm_pos_fn: Callable[[str], str],
) -> list[str]:
    if df_in is None or df_in.empty or "team" not in df_in.columns:
        return []
    # Match exato primeiro; se não casar (CSV de players grafa o time diferente
    # do CSV de team-games, ex. "Gen.G" vs "GEN"), cai para match normalizado.
    # Sem isso o autofill "não pegava nenhum" quando as grafias divergiam.
    d = df_in[df_in["team"].astype(str) == str(team_name)].copy()
    if d.empty:
        key = _norm_team_key(team_name)
        if key:
            d = df_in[df_in["team"].apply(lambda x: _norm_team_key(x) == key)].copy()
    if d.empty:
        return []
    if "date" in d.columns:
        d = d.sort_values("date")
    d["pos_norm"] = d.get("pos", "").apply(norm_pos_fn)
    out: list[str] = []
    for pos in ["top", "jng", "mid", "bot", "sup"]:
        sub = d[d["pos_norm"] == pos].copy()
        if sub.empty:
            continue
        nm = str(sub.iloc[-1].get("player", "") or "").strip()
        if nm:
            out.append(nm)
    return out


def latest_lineup_for_team(
    *,
    df_player_rows: pd.DataFrame,
    team: str,
    filters: Any,
    league_fixed: str | None = None,
    pos_col: str = "pos",
    player_col: str = "player",
    pos_list: list[str] | None = None,
    norm_pos_fn: Callable[[str], str] | None = None,
    norm_pos_ml_fn: Callable[[str], str] | None = None,
) -> dict:
    if df_player_rows is None or df_player_rows.empty:
        return {}
    df = df_player_rows.copy()
    pos_list = list(pos_list or [])
    if not pos_list:
        return {}
    if "team" not in df.columns or player_col not in df.columns or pos_col not in df.columns:
        return {}

    df = df[df["team"].astype(str) == str(team)].copy()
    if league_fixed and "league" in df.columns:
        df = df[df["league"].astype(str) == str(league_fixed)].copy()
    if filters is not None:
        if getattr(filters, "year", None) is not None and "year" in df.columns:
            df = df[df["year"] == int(filters.year)].copy()
        if getattr(filters, "split", None) and "split" in df.columns:
            sp = str(getattr(filters, "split", "") or "").strip()
            if sp and sp.lower() not in ("all", "todos", "todas", "any"):
                df = df[df["split"].astype(str).str.strip() == sp].copy()
        if getattr(filters, "playoffs", None) is not None and "playoffs" in df.columns:
            target = 1 if bool(getattr(filters, "playoffs", False)) else 0
            po = pd.to_numeric(df["playoffs"], errors="coerce").fillna(0).astype(int)
            df = df[po == int(target)].copy()

    if df.empty:
        return {}

    if pos_col == "pos_ml":
        norm_func = norm_pos_ml_fn if callable(norm_pos_ml_fn) else (lambda x: str(x or ""))
    else:
        norm_func = norm_pos_fn if callable(norm_pos_fn) else (lambda x: str(x or ""))

    df["pos_can"] = df[pos_col].apply(norm_func)
    df = df[df["pos_can"].isin(pos_list)].copy()
    if df.empty:
        return {}

    if "gameid" not in df.columns or "date" not in df.columns:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["gameid", "date"])
    if df.empty:
        return {}

    df = df.sort_values("date", ascending=True)
    last_gid = str(df.iloc[-1]["gameid"])
    g = df[df["gameid"].astype(str) == last_gid].copy()

    out: dict[str, str] = {}
    for pos in pos_list:
        cand = g[g["pos_can"] == pos][player_col].dropna().astype(str).tolist()
        if cand:
            out[pos] = cand[0].strip()
    return out

