"""REGRA ÚNICA de escopo das janelas (Ano/L15/L10/L5) — fonte da verdade.

Regra do usuário (feedback_cross_league_e_ewc_janelas, 2026-06):
1. "Ano todo" = ano inteiro SEM filtro de liga (todas as ligas/torneios).
2. CROSS-LEAGUE (liga dominante de A ≠ de B): janelas recentes (L15/L10/L5)
   usam o campeonato MAIS RECENTE de cada time (liga do último jogo dele).
3. Mesma liga: janelas recentes seguem o recorte selecionado (df_used).

Todo lugar que mostra janelas (Resumo, Radar, Análise de Linha) deve derivar
os frames DAQUI — divergência entre telas foi bug real (2026-07-06, TSW×LYON
no MSI: Resumo usava LCP/LCS enquanto a Análise usava MSI).
Teste de consistência: mlcore/tests/test_window_scope_consistency.py.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def dominant_league(df: Optional[pd.DataFrame]) -> str:
    """Liga mais comum do frame (identifica a liga 'de casa' do recorte)."""
    if df is None or df.empty or "league" not in df.columns:
        return ""
    vc = df["league"].fillna("").astype(str).str.strip().value_counts()
    return str(vc.index[0]) if not vc.empty else ""


def recent_championship_df(df_year: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Jogos do campeonato MAIS RECENTE do time (liga do último jogo do ano)."""
    if df_year is None or df_year.empty or "league" not in df_year.columns:
        return df_year if df_year is not None else pd.DataFrame()
    d = df_year.sort_values("date")
    lg = str(d["league"].iloc[-1]).strip()
    return d[d["league"].astype(str).str.strip() == lg].copy()


def full_year_team_df(
    team_games: pd.DataFrame,
    *,
    team: str,
    year: Any = None,
    map_mode: str = "avg",
    filters_cls: Any = None,
) -> pd.DataFrame:
    """Ano inteiro do time, TODAS as ligas (regra 1), com o filtro de mapa."""
    from mlcore.lines_engine import filter_team_games
    from mlcore.stats_teams import Filters
    F = filters_cls or Filters
    try:
        yr = int(year) if year is not None else None
    except (TypeError, ValueError):
        yr = None
    f_yr = F(year=yr, league=None, split=None, playoffs=None)
    return filter_team_games(team_games, team=str(team), filters=f_yr,
                             league_mode="all", map_mode=str(map_mode or "avg"))


def resolve_window_scopes(
    *,
    team_games: pd.DataFrame,
    df_a_used: pd.DataFrame,
    df_b_used: pd.DataFrame,
    team_a: str,
    team_b: str,
    year: Any = None,
    map_mode: str = "avg",
    filters_cls: Any = None,
) -> dict[str, Any]:
    """Frames das janelas pela regra. Retorna:
    - df_a_win/df_b_win: janelas RECENTES (recorte, ou campeonato mais recente
      se cross-league);
    - df_a_full/df_b_full: "Ano todo" (ano inteiro sem filtro de liga);
    - cross_league: bool (liga dominante de A ≠ de B).
    Em erro/dado faltando, degrada para o comportamento antigo (df_used)."""
    out = {"df_a_win": df_a_used, "df_b_win": df_b_used,
           "df_a_full": None, "df_b_full": None, "cross_league": False}
    try:
        df_a_full = full_year_team_df(team_games, team=team_a, year=year,
                                      map_mode=map_mode, filters_cls=filters_cls)
        df_b_full = full_year_team_df(team_games, team=team_b, year=year,
                                      map_mode=map_mode, filters_cls=filters_cls)
        out["df_a_full"], out["df_b_full"] = df_a_full, df_b_full
        la, lb = dominant_league(df_a_used), dominant_league(df_b_used)
        cross = bool(la and lb and la != lb)
        out["cross_league"] = cross
        if cross and df_a_full is not None and df_b_full is not None \
                and not df_a_full.empty and not df_b_full.empty:
            out["df_a_win"] = recent_championship_df(df_a_full).sort_values("date")
            out["df_b_win"] = recent_championship_df(df_b_full).sort_values("date")
    except Exception:
        return {"df_a_win": df_a_used, "df_b_win": df_b_used,
                "df_a_full": None, "df_b_full": None, "cross_league": False}
    return out
