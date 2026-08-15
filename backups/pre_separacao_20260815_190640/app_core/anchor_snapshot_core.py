"""Compute lightweight odds-anchor snapshot from a CSV file (Laplace WR ratio)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app_core.matchup_selection_core import pick_team_name

_ANCHORS_SPEC: List[tuple[str, List[str], List[str]]] = [
    # LPL
    ("we_vs_tes", ["team we", "we"], ["top esports", "tes"]),
    ("lng_vs_tt", ["lng esports", "lng"], ["thundertalk gaming", "tt"]),
    ("blg_vs_jdg", ["bilibili gaming", "blg"], ["jd gaming", "jdg"]),
    ("blg_vs_al", ["bilibili gaming", "blg"], ["anyone's legend", "al"]),
    ("jdg_vs_al", ["jd gaming", "jdg"], ["anyone's legend", "al"]),
    ("wbg_vs_ig", ["weibo gaming", "wbg"], ["invictus gaming", "ig"]),
    ("tes_vs_lng", ["top esports", "tes"], ["lng esports", "lng"]),
    ("we_vs_jdg", ["team we", "we"], ["jd gaming", "jdg"]),
    # LCK
    ("geng_vs_t1", ["gen g", "geng"], ["t1"]),
    ("hle_vs_dk", ["hanwha life esports", "hle"], ["dplus kia", "dk"]),
    ("kt_vs_drx", ["kt rolster", "kt"], ["drx"]),
    ("kdf_vs_ns", ["kwangdong freecs", "kdf"], ["nongshim redforce", "ns"]),
    ("fearx_vs_dns", ["bnk fearx", "fearx"], ["dn freecs", "dn soopers", "dns"]),
    ("t1_vs_hle", ["t1"], ["hanwha life esports", "hle"]),
    # LEC / EMEA
    ("g2_vs_fnc", ["g2 esports", "g2"], ["fnatic", "fnc"]),
    ("kc_vs_mkoi", ["karmine corp", "kc"], ["movistar koi", "mkoi", "koi"]),
    ("g2_vs_kc", ["g2 esports", "g2"], ["karmine corp", "kc"]),
    ("fnc_vs_mkoi", ["fnatic", "fnc"], ["movistar koi", "mkoi", "koi"]),
    # Americas / others
    ("tl_vs_c9", ["team liquid", "tl"], ["cloud9", "c9"]),
    ("fly_vs_c9", ["flyquest", "fly"], ["cloud9", "c9"]),
    ("cfo_vs_gz", ["ctbc flying oyster", "ctbc", "cfo"], ["ground zero", "gz", "groundzero"]),
    ("psg_vs_cfo", ["psg talon", "psg"], ["ctbc flying oyster", "ctbc", "cfo"]),
]


def compute_anchor_snapshot(csv_path: str) -> Dict[str, Any]:
    """Lightweight, deterministic odds anchors from CSV (game-level team WR Laplace)."""
    p = Path(str(csv_path or "")).expanduser()
    if not p.exists():
        return {"error": f"csv ausente: {p}"}

    cols = ["gameid", "date", "participantid", "teamname", "result", "league"]
    df = pd.read_csv(str(p), usecols=cols, low_memory=False)
    df = df[pd.to_numeric(df.get("participantid"), errors="coerce").isin([100, 200])].copy()
    if df.empty:
        return {"error": "csv sem linhas de time (participantid 100/200)"}
    df["teamname"] = df["teamname"].astype(str).str.strip()
    df["league"] = df["league"].astype(str).str.strip().str.upper()
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df = df.dropna(subset=["teamname", "result"])

    wr = (
        df.groupby("teamname", as_index=False)
        .agg(
            wins=("result", "sum"),
            games=("result", "count"),
            league=("league", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
        )
    )
    wr["p"] = (wr["wins"] + 1.0) / (wr["games"] + 2.0)
    teams_all = wr["teamname"].astype(str).tolist()

    rows: List[Dict[str, Any]] = []
    for key, pa, pb in _ANCHORS_SPEC:
        ta = pick_team_name(teams_all, pa)
        tb = pick_team_name(teams_all, pb)
        if not ta or not tb:
            rows.append({"key": key, "ok": False, "reason": "team_not_found"})
            continue
        ra = wr[wr["teamname"] == ta].iloc[0]
        rb = wr[wr["teamname"] == tb].iloc[0]
        pa_wr = float(ra["p"])
        pb_wr = float(rb["p"])
        p_a = float(pa_wr / max(1e-9, pa_wr + pb_wr))
        o_a = float(1.0 / max(1e-9, p_a))
        o_b = float(1.0 / max(1e-9, 1.0 - p_a))
        rows.append(
            {
                "key": key,
                "ok": True,
                "team_a": str(ta),
                "team_b": str(tb),
                "league_a": str(ra.get("league") or ""),
                "league_b": str(rb.get("league") or ""),
                "games_a": int(ra.get("games") or 0),
                "games_b": int(rb.get("games") or 0),
                "p_a": p_a,
                "odd_a": o_a,
                "odd_b": o_b,
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "laplace_wr_ratio_v1",
        "anchors": rows,
    }


__all__ = ["compute_anchor_snapshot"]
