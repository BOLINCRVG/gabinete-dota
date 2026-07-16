# -*- coding: utf-8 -*-
"""Valor individual por jogador — "quem joga hoje vale quanto" (redesign
Players, 2026-07-16).

Lição roubada do PandaSkill (arXiv 2501.10049, validado c/ traders):
1) comparar jogador DENTRO da própria função (suporte não disputa dano com
   mid) → percentil 0-100 por role;
2) separar jogador de time: o índice usa desempenho individual por jogo
   (KLA, dano/min, CS/min, visão/min), NUNCA vitória — um jogador pode ser
   top-percentil num time perdedor.

Leitura descritiva com n visível; não mexe em probabilidade (quem mexe é a
camada de players do ML, já gateada por precisão).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

MIN_GAMES_DEFAULT = 8
_METRICS = ("kla", "dpm", "cspm", "vspm")


def _per_game_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva métricas por jogo das rows de players do OracleElixir."""
    out = pd.DataFrame(index=df.index)
    k = pd.to_numeric(df.get("kills"), errors="coerce")
    d = pd.to_numeric(df.get("deaths"), errors="coerce")
    a = pd.to_numeric(df.get("assists"), errors="coerce")
    glen_min = pd.to_numeric(df.get("gamelength"), errors="coerce") / 60.0
    glen_min = glen_min.where(glen_min > 0)
    out["kla"] = (k.fillna(0) + a.fillna(0)) / d.fillna(0).clip(lower=1)
    out["dpm"] = pd.to_numeric(df.get("damagetochampions"), errors="coerce") / glen_min
    out["cspm"] = pd.to_numeric(df.get("cspm"), errors="coerce")
    out["vspm"] = pd.to_numeric(df.get("visionscore"), errors="coerce") / glen_min
    return out


def build_player_role_scores(
    players_df: Optional[pd.DataFrame],
    *,
    year: Optional[int] = None,
    min_games: int = MIN_GAMES_DEFAULT,
) -> pd.DataFrame:
    """Percentil de desempenho POR FUNÇÃO.

    Retorna DataFrame [player, pos, team, games, percentile, score] — um por
    jogador×função com amostra ≥ min_games no recorte. `team` = último time.
    """
    cols = ["player", "pos", "team", "games", "percentile", "score"]
    if players_df is None or getattr(players_df, "empty", True):
        return pd.DataFrame(columns=cols)
    df = players_df
    need = {"playername", "position"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame(columns=cols)
    if year is not None and "year" in df.columns:
        df = df[pd.to_numeric(df["year"], errors="coerce") == int(year)]
    df = df[(df["playername"].astype(str).str.strip() != "")
            & (df["position"].astype(str).str.strip() != "")]
    if df.empty:
        return pd.DataFrame(columns=cols)

    met = _per_game_metrics(df)
    work = pd.DataFrame({
        "player": df["playername"].astype(str).str.strip(),
        "pos": df["position"].astype(str).str.strip().str.lower(),
        "team": df["team"].astype(str) if "team" in df.columns else "",
        "dt": pd.to_datetime(df.get("date"), errors="coerce"),
    }).join(met)

    g = work.groupby(["player", "pos"], sort=False)
    agg = g.agg(
        games=("kla", "size"),
        kla=("kla", "mean"),
        dpm=("dpm", "mean"),
        cspm=("cspm", "mean"),
        vspm=("vspm", "mean"),
    ).reset_index()
    # último time do jogador (para exibição/matching)
    last_team = work.sort_values("dt").groupby(["player", "pos"])["team"].last()
    agg = agg.merge(last_team.rename("team").reset_index(), on=["player", "pos"], how="left")

    agg = agg[agg["games"] >= int(min_games)]
    if agg.empty:
        return pd.DataFrame(columns=cols)

    # z-score DENTRO da função, média dos z disponíveis → score → percentil
    parts = []
    for pos, sub in agg.groupby("pos"):
        sub = sub.copy()
        zs = []
        for mcol in _METRICS:
            v = pd.to_numeric(sub[mcol], errors="coerce")
            sd = float(v.std(ddof=0) or 0.0)
            zs.append((v - float(v.mean())) / sd if sd > 1e-9 else v * 0.0)
        z = pd.concat(zs, axis=1)
        sub["score"] = z.mean(axis=1)
        sub["percentile"] = sub["score"].rank(pct=True) * 100.0
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return out[cols].sort_values("percentile", ascending=False).reset_index(drop=True)


def lineup_value_view(
    lineup: list[str],
    role_scores: pd.DataFrame,
    *,
    fuzzy_match_one_fn: Any = None,
) -> pd.DataFrame:
    """Cruza um lineup (5 nomes) com a tabela de percentis.

    Retorna [player, pos, percentile, games] na ordem do lineup; jogador sem
    amostra vem com percentile NaN (mostrado como "sem amostra", nunca 0 —
    contrato NaN≠zero do projeto).
    """
    cols = ["player", "pos", "percentile", "games"]
    rows = []
    rs = role_scores if role_scores is not None else pd.DataFrame(columns=cols)
    names = [str(x or "").strip() for x in (lineup or [])]
    known = sorted({str(p) for p in rs["player"]}) if not rs.empty else []
    for name in names:
        if not name:
            continue
        match = name
        if known and name not in known and callable(fuzzy_match_one_fn):
            try:
                m = fuzzy_match_one_fn(name, known)
                match = str(m) if m else name
            except Exception:
                match = name
        sub = rs[rs["player"].astype(str).str.casefold() == match.casefold()] if not rs.empty else rs
        if sub is not None and not sub.empty:
            best = sub.sort_values("games", ascending=False).iloc[0]
            rows.append({"player": name, "pos": str(best["pos"]),
                         "percentile": float(best["percentile"]),
                         "games": int(best["games"])})
        else:
            rows.append({"player": name, "pos": "", "percentile": float("nan"), "games": 0})
    return pd.DataFrame(rows, columns=cols)


__all__ = [
    "MIN_GAMES_DEFAULT",
    "build_player_role_scores",
    "lineup_value_view",
]
