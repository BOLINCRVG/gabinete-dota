# -*- coding: utf-8 -*-
"""Consistência de ESCOPO das janelas entre motores (Resumo/Radar/Linha).

Bug real que motivou isto (2026-07-06, TSW×LYON no MSI): o Resumo calculava
janelas e Odd Hist na liga DOMINANTE (doméstica) enquanto a Análise da Linha
seguia a regra do usuário (cross-league → campeonato mais recente; Ano todo →
sem filtro de liga). Regra única agora em app_core/window_scope_core.py —
este teste trava o contrato pra divergência nova quebrar o teste, não o usuário.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app_core.window_scope_core import (  # noqa: E402
    dominant_league,
    recent_championship_df,
    resolve_window_scopes,
)


def _mk_games(team: str, league: str, dates: list[str], kills: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "gameid": [f"{team}_{league}_{i}" for i in range(len(dates))],
        "date": pd.to_datetime(dates),
        "team": team, "opponent": "Rival", "league": league,
        "year": [pd.Timestamp(d).year for d in dates],
        "split": "", "playoffs": 0, "map_number": 1,
        "win": 1, "kills_for": kills, "kills_against": kills,
        "gamelength": 1800.0,
    })


def _world() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time A: 6 jogos LIGA1 + 3 MSI. Time B: 6 jogos LIGA2 + 3 MSI."""
    a = pd.concat([
        _mk_games("TimeA", "LIGA1", ["2026-01-10", "2026-01-17", "2026-02-01",
                                     "2026-02-08", "2026-03-01", "2026-03-08"]),
        _mk_games("TimeA", "MSI", ["2026-07-01", "2026-07-03", "2026-07-05"]),
    ], ignore_index=True)
    b = pd.concat([
        _mk_games("TimeB", "LIGA2", ["2026-01-11", "2026-01-18", "2026-02-02",
                                     "2026-02-09", "2026-03-02", "2026-03-09"]),
        _mk_games("TimeB", "MSI", ["2026-07-02", "2026-07-04", "2026-07-06"]),
    ], ignore_index=True)
    team_games = pd.concat([a, b], ignore_index=True)
    df_a_used = a[a["league"] == "LIGA1"]
    df_b_used = b[b["league"] == "LIGA2"]
    return team_games, df_a_used, df_b_used


class TestWindowScopeConsistency(unittest.TestCase):
    def setUp(self) -> None:
        self.team_games, self.df_a_used, self.df_b_used = _world()

    def test_dominant_league_e_recent_championship(self) -> None:
        self.assertEqual(dominant_league(self.df_a_used), "LIGA1")
        a_year = self.team_games[self.team_games["team"] == "TimeA"]
        recent = recent_championship_df(a_year)
        self.assertEqual(set(recent["league"]), {"MSI"})
        self.assertEqual(len(recent), 3)

    def test_cross_league_usa_campeonato_recente(self) -> None:
        scopes = resolve_window_scopes(
            team_games=self.team_games, df_a_used=self.df_a_used,
            df_b_used=self.df_b_used, team_a="TimeA", team_b="TimeB",
            year=2026, map_mode="avg",
        )
        self.assertTrue(scopes["cross_league"])
        # janelas recentes = SÓ o campeonato mais recente (MSI)
        self.assertEqual(set(scopes["df_a_win"]["league"]), {"MSI"})
        self.assertEqual(set(scopes["df_b_win"]["league"]), {"MSI"})
        # "Ano todo" = ano inteiro, TODAS as ligas
        self.assertEqual(len(scopes["df_a_full"]), 9)
        self.assertEqual(set(scopes["df_a_full"]["league"]), {"LIGA1", "MSI"})

    def test_mesma_liga_mantem_recorte(self) -> None:
        scopes = resolve_window_scopes(
            team_games=self.team_games, df_a_used=self.df_a_used,
            df_b_used=self.df_a_used, team_a="TimeA", team_b="TimeA",
            year=2026, map_mode="avg",
        )
        self.assertFalse(scopes["cross_league"])
        # comportamento antigo preservado: janelas recentes = recorte passado
        self.assertIs(scopes["df_a_win"], self.df_a_used)

    def test_resumo_temporal_profile_segue_a_regra(self) -> None:
        """Ano todo do card = ano inteiro; L5 = campeonato recente — o
        contrato que o bug TSW×LYON violava."""
        from app_core.resumo_score_core import compute_temporal_profile

        scopes = resolve_window_scopes(
            team_games=self.team_games, df_a_used=self.df_a_used,
            df_b_used=self.df_b_used, team_a="TimeA", team_b="TimeB",
            year=2026, map_mode="avg",
        )
        tp = compute_temporal_profile(
            scopes["df_a_win"], metric_key="kills", line=50.5, side_key="Under",
            team_name="TimeA", full_year_df=scopes["df_a_full"],
        )
        by_label = {w.label: w for w in tp.windows}
        self.assertEqual(by_label["Ano todo"].n, 9)   # 6 LIGA1 + 3 MSI
        self.assertEqual(by_label["L5"].n, 3)         # só o MSI

    def test_radar_windows_seguem_a_regra(self) -> None:
        """Radar em cross-league: janela longa = ano; L5 = campeonato recente."""
        from app_core.radar_confronto_core import build_team_windows

        a_year = self.team_games[self.team_games["team"] == "TimeA"]
        wins_all = build_team_windows(a_year, "TimeB")
        recent = build_team_windows(recent_championship_df(a_year), "TimeB")
        self.assertEqual(len(wins_all["liga"]), 9)    # frame do ano completo
        self.assertEqual(len(recent["l5"]), 3)        # MSI tem 3 jogos
        self.assertEqual(set(recent["l5"]["league"]), {"MSI"})


if __name__ == "__main__":
    unittest.main()
