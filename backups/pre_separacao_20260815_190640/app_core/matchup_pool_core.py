# -*- coding: utf-8 -*-
"""Pool do confronto (redesign Campeões, 2026-07-16).

Princípio acordado com o usuário: a leitura global de força de campeão é da
aba Meta; o que SÓ o pool pode dizer é "o que ESTE time joga" — conforto,
concentração (banibilidade) e o cruzamento pool × meta do patch atual
("o meta tirou 2 dos 3 comfort picks deste time").

Pesquisa que fomentou (2026-07-15): PandaSkill/iTero confirmam que força de
time >> draft; Machine LoLing inspirou a leitura de "banibilidade" do pool.
Nada aqui mexe em probabilidade — é leitura descritiva com n visível.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

# Régua (mesma filosofia do Radar B2): conforto exige amostra mínima;
# banibilidade olha a concentração dos top picks.
COMFORT_MIN_GAMES = 5
POOL_TOP_N = 8
_STRONG_TIERS = {"S", "A"}


def build_team_pool(
    players_df: Optional[pd.DataFrame],
    team: str,
    *,
    year: Optional[int] = None,
    top_n: int = POOL_TOP_N,
) -> pd.DataFrame:
    """Pool de um time a partir das rows por-jogador do CSV.

    Retorna DataFrame [champion, games, wins, wr, last_date, player] ordenado
    por games desc. `year` recorta a temporada (None = tudo que veio no df).
    """
    cols = ["champion", "games", "wins", "wr", "last_date", "player"]
    if players_df is None or getattr(players_df, "empty", True):
        return pd.DataFrame(columns=cols)
    df = players_df
    need = {"team", "champion"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame(columns=cols)
    m = df["team"].astype(str).str.strip().str.casefold() == str(team or "").strip().casefold()
    df = df[m]
    if year is not None and "year" in df.columns:
        df = df[pd.to_numeric(df["year"], errors="coerce") == int(year)]
    df = df[df["champion"].astype(str).str.strip() != ""]
    if df.empty:
        return pd.DataFrame(columns=cols)

    res = pd.to_numeric(df.get("result"), errors="coerce") if "result" in df.columns else None
    dts = pd.to_datetime(df.get("date"), errors="coerce") if "date" in df.columns else None
    work = pd.DataFrame({
        "champion": df["champion"].astype(str).str.strip(),
        "win": (res.fillna(0.0) if res is not None else 0.0),
        "dt": (dts if dts is not None else pd.NaT),
        "player": df["playername"].astype(str) if "playername" in df.columns else "",
    })
    g = work.groupby("champion", sort=False)
    out = pd.DataFrame({
        "games": g.size(),
        "wins": g["win"].sum(),
        "last_date": g["dt"].max(),
    })
    # jogador que mais pilotou o campeão no time (dono do conforto)
    try:
        out["player"] = g["player"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
    except Exception:
        out["player"] = ""
    out["wr"] = out["wins"] / out["games"].where(out["games"] > 0)
    out = out.reset_index().sort_values(["games", "wr"], ascending=[False, False])
    return out.head(int(top_n))[cols].reset_index(drop=True)


def cross_pool_with_meta(
    pool: pd.DataFrame,
    meta_map: Optional[Mapping[str, Mapping[str, Any]]],
) -> pd.DataFrame:
    """Anexa tier/presença do patch atual (aba Meta) a cada campeão do pool."""
    out = pool.copy()
    mm = meta_map or {}

    def _tier(ch: str) -> str:
        return str((mm.get(str(ch)) or {}).get("tier") or "")

    def _presence(ch: str):
        v = (mm.get(str(ch)) or {}).get("presence")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    out["tier"] = out["champion"].map(_tier) if not out.empty else ""
    out["presence"] = out["champion"].map(_presence) if not out.empty else None
    return out


def pool_concentration(pool: pd.DataFrame, *, top: int = 3) -> Optional[float]:
    """Fração dos jogos do pool nos `top` campeões mais jogados — proxy de
    banibilidade (pool concentrado = fácil de atacar no ban)."""
    if pool is None or pool.empty or "games" not in pool.columns:
        return None
    tot = float(pool["games"].sum())
    if tot <= 0:
        return None
    return float(pool["games"].head(int(top)).sum()) / tot


def pool_verdict_lines(team: str, pool_meta: pd.DataFrame) -> list[str]:
    """Frases secas (régua do Radar: número junto, sem veredito de aposta)."""
    lines: list[str] = []
    if pool_meta is None or pool_meta.empty:
        return [f"{team}: sem amostra de picks no recorte."]
    comfort = pool_meta[pool_meta["games"] >= COMFORT_MIN_GAMES]
    if not comfort.empty and "tier" in comfort.columns:
        n_comf = int(len(comfort))
        strong = comfort[comfort["tier"].isin(_STRONG_TIERS)]
        n_strong = int(len(strong))
        lines.append(
            f"{team}: {n_strong}/{n_comf} comfort picks (≥{COMFORT_MIN_GAMES} jogos) "
            f"estão S/A no patch atual."
        )
        weak = comfort[~comfort["tier"].isin(_STRONG_TIERS) & (comfort["tier"] != "")]
        if len(weak) >= 2:
            names = ", ".join(weak["champion"].head(3).tolist())
            lines.append(f"{team}: conforto fora do meta — {names}.")
    conc = pool_concentration(pool_meta)
    if conc is not None and conc >= 0.55:
        lines.append(
            f"{team}: pool concentrado — top-3 picks = {conc * 100:.0f}% dos jogos (banível)."
        )
    return lines


__all__ = [
    "COMFORT_MIN_GAMES",
    "POOL_TOP_N",
    "build_team_pool",
    "cross_pool_with_meta",
    "pool_concentration",
    "pool_verdict_lines",
]
