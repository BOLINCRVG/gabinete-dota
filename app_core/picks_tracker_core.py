"""Persistent tracker for the user's closed picks.

Lets the user mark a pick as bet, then auto-resolves it later by matching
against the team_games dataframe (we own the source of truth — the CSV).
ROI is computed from settled picks. Open picks wait for the game to land.

JSON schema (one file at ops/picks_tracker.json):

    {"picks": [
        {"id": "<sha1[:10]>",
         "ts": "2026-05-30T18:00:00",       # ISO when user marked it
         "team_a": "Gen.G", "team_b": "T1",
         "league": "LCK",
         "metric_key": "kills",              # canonical, not display
         "side_key": "Over",                 # canonical: Over/Under/A/B
         "line": 27.5,                       # null for ML
         "odd_casa": 1.85,
         "stake_units": 1.0,                 # 0.5/0.75/1/1.25/1.5/2
         "p_hist": 0.62,                     # da FREQUENCIA historica, nao do modelo
         "p_modelo": 0.58,                   # o que o MODELO disse na hora da aposta
         "momento": "pre",                   # pre|live|desconhecido
         "edge_hist_pct": 5.5,
         "status": "open",                   # open|won|lost|void
         "profit_units": null,
         "resolved_at": null,
         "resolved_gameid": null
        }, ...]
    }
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

import pandas as pd


# Mapping from canonical metric_key (used in picks) to OracleElixir column.
METRIC_KEY_TO_COL: dict[str, str] = {
    "kills": "total_kills",
    "towers": "total_towers",
    "dragons": "total_dragons",
    "barons": "total_nashors",
    "inhibitors": "total_inhibitors",
    "time": "game_time_min",
}


@dataclass
class Pick:
    id: str
    ts: str
    team_a: str
    team_b: str
    league: str
    metric_key: str
    side_key: str
    line: Optional[float]
    odd_casa: float
    stake_units: float
    p_hist: Optional[float] = None
    # O QUE O MODELO DISSE, e QUANDO a aposta foi feita.
    #
    # POR QUE ENTRARAM (21/09/2026). Ele observou: "NORMALMENTE, os ml's pres
    # sao vindos do modelo de alguma forma, talvez 80% deles no minimo, os
    # live, nos nao costumamos levar o modelo em conta".
    #
    # Isso invalidou uma coisa que eu tinha afirmado. Eu usei "ML: -2,89% de
    # ROI em 884 apostas" como prova de que o modelo de vencedor e ruim. So
    # que o tracker nao separa pre de live, e as 884 misturam apostas que
    # vieram do modelo com apostas que nunca passaram por ele. O numero nao
    # sustenta a conclusao.
    #
    # E pior: `p_hist` NAO e a probabilidade do modelo -- e a da frequencia
    # historica (odd_to_prob(odd_hist)). Sao coisas diferentes no codigo
    # (odd_model, odd_hist, p_stats) e so a do meio era gravada. Ou seja, o
    # tracker nunca conseguiu responder "o modelo ganha dinheiro?", nem para
    # apostas antigas nem para novas.
    #
    # `momento` nasce "desconhecido" de proposito: as 4.788 apostas ja
    # gravadas nao dizem se foram pre ou live, e chuta-las como "pre" seria
    # inventar o dado que falta. Quem nao sabe, diz que nao sabe.
    p_modelo: Optional[float] = None
    momento: str = "desconhecido"
    edge_hist_pct: Optional[float] = None
    status: str = "open"
    profit_units: Optional[float] = None
    resolved_at: Optional[str] = None
    resolved_gameid: Optional[str] = None
    source: str = "manual"  # "manual" (marcada na UI) | "xlsx" (backfill planilha)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_pick_id(ts: str, team_a: str, team_b: str,
                      metric_key: str, side_key: str, line: Optional[float]) -> str:
    raw = f"{ts}|{team_a}|{team_b}|{metric_key}|{side_key}|{line if line is not None else '-'}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def new_pick(*, team_a: str, team_b: str, league: str,
              metric_key: str, side_key: str, line: Optional[float],
              odd_casa: float, stake_units: float,
              p_hist: Optional[float] = None,
              p_modelo: Optional[float] = None,
              momento: str = "pre",
              edge_hist_pct: Optional[float] = None,
              ts: Optional[str] = None) -> Pick:
    """Construct a fresh Pick with a deterministic id.

    `momento` cai em "pre" porque o app so precifica pre-jogo. Uma aposta
    live registrada aqui tem de dizer isso na chamada.
    """
    ts_iso = ts or datetime.now().isoformat(timespec="seconds")
    pid = _compute_pick_id(ts_iso, team_a, team_b, metric_key, side_key, line)
    return Pick(
        id=pid, ts=ts_iso, team_a=str(team_a), team_b=str(team_b),
        league=str(league or ""), metric_key=str(metric_key),
        side_key=str(side_key), line=(float(line) if line is not None else None),
        odd_casa=float(odd_casa), stake_units=float(stake_units),
        p_hist=(float(p_hist) if p_hist is not None else None),
        p_modelo=(float(p_modelo)
                  if (p_modelo is not None and math.isfinite(float(p_modelo)))
                  else None),
        momento=(str(momento) if momento in ("pre", "live", "desconhecido")
                 else "desconhecido"),
        edge_hist_pct=(float(edge_hist_pct) if edge_hist_pct is not None else None),
    )


def load_picks(path: str) -> list[Pick]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return []
    raw_picks = data.get("picks") or []
    out: list[Pick] = []
    for raw in raw_picks:
        if not isinstance(raw, dict):
            continue
        try:
            out.append(Pick(**{k: v for k, v in raw.items() if k in Pick.__annotations__}))
        except (TypeError, ValueError):
            continue
    return out


def save_picks(path: str, picks: Iterable[Pick]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"picks": [p.to_dict() for p in picks]}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def upsert_pick(picks: list[Pick], pick: Pick) -> list[Pick]:
    """Insert or replace pick by id; returns updated list (does NOT save)."""
    out = [p for p in picks if p.id != pick.id]
    out.append(pick)
    return out


def remove_pick(picks: list[Pick], pick_id: str) -> list[Pick]:
    return [p for p in picks if p.id != pick_id]


# --- Resolution ----------------------------------------------------------

def _find_resolving_game(pick: Pick, team_games: pd.DataFrame) -> Optional[pd.Series]:
    """Find the FIRST game where pick.team_a played pick.team_b on or AFTER
    pick.ts. Returns team_a's row (so we can check win + metric).

    Why FIRST and not LAST: a pick is placed on a specific upcoming match. The
    earliest game at/after the pick timestamp is that match. Using the latest
    would resolve against a rematch weeks later (wrong game). Within a single
    series (BO3/BO5) the maps share a date; we take the earliest map row, which
    is the closest deterministic choice without per-map info on the Pick.

    Returns None when no such game has happened yet.
    """
    if team_games is None or team_games.empty:
        return None
    needed = {"team", "opponent", "date"}
    if not needed.issubset(team_games.columns):
        return None
    sub = team_games[
        (team_games["team"].astype(str) == pick.team_a)
        & (team_games["opponent"].astype(str) == pick.team_b)
    ].copy()
    if sub.empty:
        return None
    sub["_d"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["_d"])
    if sub.empty:
        return None
    pick_ts = pd.to_datetime(pick.ts, errors="coerce")
    if not pd.isna(pick_ts):
        sub = sub[sub["_d"] >= pick_ts]
    if sub.empty:
        return None
    # FIRST game at/after the pick timestamp (closest to when the bet was made).
    return sub.sort_values("_d").iloc[0]


def _settle_pick(pick: Pick, row: pd.Series) -> Pick:
    """Compute new status + profit_units from a resolving row."""
    status = "open"
    if pick.metric_key in ("ml_A", "ml_B", "ml"):
        win_val = pd.to_numeric(row.get("win"), errors="coerce")
        if pd.isna(win_val):
            return pick
        a_won = float(win_val) > 0
        # Pick side: "A" = team_a wins; "B" = team_b wins.
        side_for_a = str(pick.side_key).upper() == "A"
        won = (side_for_a and a_won) or ((not side_for_a) and (not a_won))
        status = "won" if won else "lost"
    elif pick.metric_key in METRIC_KEY_TO_COL:
        col = METRIC_KEY_TO_COL[pick.metric_key]
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(val) or pick.line is None:
            return pick
        v = float(val); ln = float(pick.line)
        if math.isclose(v, ln, abs_tol=1e-9):
            status = "void"
        elif str(pick.side_key).lower() == "over":
            status = "won" if v > ln else "lost"
        else:  # Under
            status = "won" if v < ln else "lost"
    else:
        return pick  # Unknown metric — leave open.

    if status == "open":
        return pick
    if status == "void":
        profit = 0.0
    elif status == "won":
        profit = pick.stake_units * (pick.odd_casa - 1.0)
    else:
        profit = -pick.stake_units
    gameid_raw = row.get("gameid", "")
    gameid_str = "" if pd.isna(gameid_raw) else str(gameid_raw)
    return Pick(
        **{**pick.to_dict(),
           "status": status,
           "profit_units": float(profit),
           "resolved_at": datetime.now().isoformat(timespec="seconds"),
           "resolved_gameid": gameid_str},
    )


def resolve_open_picks(picks: list[Pick], team_games: pd.DataFrame) -> tuple[list[Pick], int]:
    """Settle open picks where the game already happened. Returns (new_list, n_settled)."""
    out: list[Pick] = []
    settled = 0
    for p in picks:
        if p.status != "open":
            out.append(p); continue
        row = _find_resolving_game(p, team_games)
        if row is None:
            out.append(p); continue
        updated = _settle_pick(p, row)
        if updated.status != "open":
            settled += 1
        out.append(updated)
    return out, settled


# --- ROI / aggregates ----------------------------------------------------

def roi_summary(picks: list[Pick]) -> dict[str, Any]:
    """Aggregate ROI across settled picks. Open picks ignored.

    ROI = sum(profit_units) / sum(stake_units of NON-void settled picks).
    Void picks return 0 profit and don't count in the denominator.
    """
    settled = [p for p in picks if p.status in ("won", "lost", "void")]
    n_total = len(settled)
    wins = sum(1 for p in settled if p.status == "won")
    losses = sum(1 for p in settled if p.status == "lost")
    voids = sum(1 for p in settled if p.status == "void")
    profit = sum(float(p.profit_units or 0.0) for p in settled)
    risked = sum(float(p.stake_units) for p in settled if p.status != "void")
    roi_pct = (profit / risked * 100.0) if risked > 0 else None

    by_metric: dict[str, dict[str, float]] = {}
    for p in settled:
        bucket = by_metric.setdefault(p.metric_key, {"profit": 0.0, "risked": 0.0, "n": 0, "wins": 0})
        bucket["profit"] += float(p.profit_units or 0.0)
        bucket["n"] += 1
        if p.status == "won":
            bucket["wins"] += 1
        if p.status != "void":
            bucket["risked"] += float(p.stake_units)
    for _k, v in by_metric.items():
        v["roi_pct"] = (v["profit"] / v["risked"] * 100.0) if v["risked"] > 0 else None
        v["winrate_pct"] = (100.0 * v["wins"] / v["n"]) if v["n"] > 0 else None

    return {
        "n_open": sum(1 for p in picks if p.status == "open"),
        "n_settled": n_total,
        "wins": wins, "losses": losses, "voids": voids,
        "profit_units": profit,
        "risked_units": risked,
        "roi_pct": roi_pct,
        "by_metric": by_metric,
    }


__all__ = [
    "METRIC_KEY_TO_COL",
    "Pick",
    "load_picks",
    "new_pick",
    "remove_pick",
    "resolve_open_picks",
    "roi_summary",
    "save_picks",
    "upsert_pick",
]
