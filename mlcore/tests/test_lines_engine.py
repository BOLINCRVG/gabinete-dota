import math
import unittest

import numpy as np
import pandas as pd

from mlcore.lines_engine import (
    MatchupTotals,
    _norm_cdf,
    _poisson_cdf,
    build_profile,
    fair_odds_from_p,
    filter_team_games,
    matchup_expected_totals,
    normalize_map_mode,
    parse_time_line,
    prob_over_from_total,
    prob_over_series_from_sims,
    simulate_series_total,
)
from mlcore.stats_teams import Filters


class TestLinesEngine(unittest.TestCase):
    def test_matchup_expected_totals_include_elemental_dragons(self) -> None:
        team_games = pd.DataFrame(
            [
                {
                    "team": "A",
                    "opponent": "X",
                    "date": "2026-01-01",
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "map_number": 1,
                    "kills_for": 12,
                    "kills_against": 10,
                    "towers_for": 6,
                    "towers_against": 4,
                    "dragons_for": 4,
                    "dragons_against": 1,
                    "elementaldrakes_for": 4,
                    "elementaldrakes_against": 1,
                    "barons_for": 1,
                    "barons_against": 0,
                    "inhibitors_for": 1,
                    "inhibitors_against": 0,
                    "game_time_min": 31.0,
                },
                {
                    "team": "A",
                    "opponent": "Y",
                    "date": "2026-01-05",
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "map_number": 1,
                    "kills_for": 15,
                    "kills_against": 11,
                    "towers_for": 7,
                    "towers_against": 5,
                    "dragons_for": 3,
                    "dragons_against": 2,
                    "elementaldrakes_for": 2,
                    "elementaldrakes_against": 1,
                    "barons_for": 2,
                    "barons_against": 1,
                    "inhibitors_for": 2,
                    "inhibitors_against": 1,
                    "game_time_min": 32.0,
                },
                {
                    "team": "B",
                    "opponent": "M",
                    "date": "2026-01-02",
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "map_number": 1,
                    "kills_for": 11,
                    "kills_against": 14,
                    "towers_for": 5,
                    "towers_against": 7,
                    "dragons_for": 2,
                    "dragons_against": 3,
                    "elementaldrakes_for": 2,
                    "elementaldrakes_against": 3,
                    "barons_for": 1,
                    "barons_against": 2,
                    "inhibitors_for": 1,
                    "inhibitors_against": 2,
                    "game_time_min": 33.0,
                },
                {
                    "team": "B",
                    "opponent": "N",
                    "date": "2026-01-08",
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "map_number": 1,
                    "kills_for": 10,
                    "kills_against": 13,
                    "towers_for": 4,
                    "towers_against": 6,
                    "dragons_for": 1,
                    "dragons_against": 4,
                    "elementaldrakes_for": 1,
                    "elementaldrakes_against": 2,
                    "barons_for": 0,
                    "barons_against": 1,
                    "inhibitors_for": 0,
                    "inhibitors_against": 1,
                    "game_time_min": 34.0,
                },
            ]
        )

        filters = Filters(year=2026, league="LCK", split="Spring", playoffs=False)
        p_a = build_profile(team_games, team="A", filters=filters, half_life_days=None)
        p_b = build_profile(team_games, team="B", filters=filters, half_life_days=None)
        totals = matchup_expected_totals(p_a, p_b)

        self.assertIn("dragons_elemental", totals)
        elemental = totals["dragons_elemental"]
        self.assertEqual(elemental.dist, "poisson")
        self.assertAlmostEqual(elemental.mean, 4.0, places=9)
        self.assertAlmostEqual(elemental.sd, 2.0, places=9)

        p_over = prob_over_from_total(elemental, 4.5)
        self.assertTrue(math.isfinite(p_over))
        self.assertTrue(0.0 < p_over < 1.0)

    def test_matchup_expected_totals_skip_elemental_dragons_when_missing(self) -> None:
        team_games = pd.DataFrame(
            [
                {
                    "team": "A",
                    "opponent": "X",
                    "date": "2026-01-01",
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "map_number": 1,
                    "kills_for": 12,
                    "kills_against": 10,
                    "towers_for": 6,
                    "towers_against": 4,
                    "dragons_for": 4,
                    "dragons_against": 1,
                    "barons_for": 1,
                    "barons_against": 0,
                    "inhibitors_for": 1,
                    "inhibitors_against": 0,
                    "game_time_min": 31.0,
                },
                {
                    "team": "B",
                    "opponent": "M",
                    "date": "2026-01-02",
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "map_number": 1,
                    "kills_for": 11,
                    "kills_against": 14,
                    "towers_for": 5,
                    "towers_against": 7,
                    "dragons_for": 2,
                    "dragons_against": 3,
                    "barons_for": 1,
                    "barons_against": 2,
                    "inhibitors_for": 1,
                    "inhibitors_against": 2,
                    "game_time_min": 33.0,
                },
            ]
        )

        filters = Filters(year=2026, league="LCK", split="Spring", playoffs=False)
        p_a = build_profile(team_games, team="A", filters=filters, half_life_days=None)
        p_b = build_profile(team_games, team="B", filters=filters, half_life_days=None)
        totals = matchup_expected_totals(p_a, p_b)

        self.assertNotIn("dragons_elemental", totals)


class TestParseTimeLine(unittest.TestCase):
    """Parser da linha de tempo digitada — 35:30 são 35.5 min, não 35.3."""

    def test_formato_mm_ss(self):
        self.assertAlmostEqual(parse_time_line("35:30"), 35.5)
        self.assertAlmostEqual(parse_time_line("32:45"), 32.75)

    def test_decimal_com_virgula_br_e_ponto(self):
        self.assertAlmostEqual(parse_time_line("35,5"), 35.5)
        self.assertAlmostEqual(parse_time_line("35.5"), 35.5)
        self.assertAlmostEqual(parse_time_line("35"), 35.0)

    def test_virgula_dentro_do_mm_ss(self):
        self.assertAlmostEqual(parse_time_line("35:30,0"), 35.5)

    def test_vazio_levanta_erro_claro(self):
        with self.assertRaises(ValueError):
            parse_time_line("   ")


class TestNormalizeMapMode(unittest.TestCase):
    def test_sinonimos_de_media(self):
        for s in (None, "avg", "all", "maps", "overall", " AVG "):
            self.assertEqual(normalize_map_mode(s), "avg")

    def test_formatos_de_mapa(self):
        for s in ("m2", "map2", "2", "M2", "MAP2"):
            self.assertEqual(normalize_map_mode(s), "m2")

    def test_invalido_levanta(self):
        for s in ("m9", "map0", "xyz", "6"):
            with self.assertRaises(ValueError):
                normalize_map_mode(s)


class TestProbOverEDistribuicoes(unittest.TestCase):
    """As CDFs que viram preço de over/under."""

    def test_norm_cdf_contra_valores_conhecidos(self):
        self.assertAlmostEqual(_norm_cdf(0.0, 0.0, 1.0), 0.5, places=9)
        self.assertAlmostEqual(_norm_cdf(1.96, 0.0, 1.0), 0.975, places=3)
        self.assertAlmostEqual(_norm_cdf(-1.96, 0.0, 1.0), 0.025, places=3)

    def test_norm_cdf_sd_zero_vira_degrau(self):
        self.assertEqual(_norm_cdf(4.9, 5.0, 0.0), 0.0)
        self.assertEqual(_norm_cdf(5.1, 5.0, 0.0), 1.0)

    def test_poisson_cdf_contra_soma_manual(self):
        # P(X<=2 | lam=3) = e^-3 (1 + 3 + 4.5)
        esperado = math.exp(-3.0) * (1.0 + 3.0 + 4.5)
        self.assertAlmostEqual(_poisson_cdf(2, 3.0), esperado, places=12)
        self.assertEqual(_poisson_cdf(-1, 3.0), 0.0)

    def test_linha_na_media_da_prob_meio(self):
        tot = MatchupTotals(mean=27.5, sd=4.0, dist="normal")
        self.assertAlmostEqual(prob_over_from_total(tot, 27.5), 0.5, places=9)

    def test_linha_mais_alta_da_prob_menor(self):
        tot = MatchupTotals(mean=27.5, sd=4.0, dist="normal")
        self.assertLess(prob_over_from_total(tot, 29.5),
                        prob_over_from_total(tot, 25.5))

    def test_poisson_over_usa_logica_inteira(self):
        # over 4.5 = P(X>=5); complemento da CDF em 4
        tot = MatchupTotals(mean=4.0, sd=0.0, dist="poisson")
        self.assertAlmostEqual(prob_over_from_total(tot, 4.5),
                               1.0 - _poisson_cdf(4, 4.0), places=12)

    def test_media_invalida_vira_nan(self):
        tot = MatchupTotals(mean=float("nan"), sd=4.0, dist="normal")
        self.assertTrue(math.isnan(prob_over_from_total(tot, 27.5)))

    def test_fair_odds_e_o_inverso_da_prob(self):
        self.assertAlmostEqual(fair_odds_from_p(0.5), 2.0, places=9)
        self.assertAlmostEqual(fair_odds_from_p(0.25), 4.0, places=9)
        self.assertTrue(math.isnan(fair_odds_from_p(float("nan"))))


class TestSimulacaoDeSerie(unittest.TestCase):
    """simulate_series_total precifica o over/under da SÉRIE (BO3/BO5).
    Não tinha nenhum teste (achado 19/07)."""

    _TOT = MatchupTotals(mean=27.0, sd=4.0, dist="normal")

    def test_bo1_tem_exatamente_um_mapa(self):
        sims = simulate_series_total(bo=1, p_blue_map=0.5, total=self._TOT, n_sims=4000)
        # 1 mapa: média ~27, desvio ~4
        self.assertAlmostEqual(float(sims.mean()), 27.0, delta=0.3)
        self.assertAlmostEqual(float(sims.std()), 4.0, delta=0.3)

    def test_bo3_soma_entre_2_e_3_mapas(self):
        sims = simulate_series_total(bo=3, p_blue_map=0.5, total=self._TOT, n_sims=4000)
        # 50/50: P(2-0 ou 0-2)=0.5 → esperança de mapas = 2.5 → total ~67.5
        self.assertAlmostEqual(float(sims.mean()), 27.0 * 2.5, delta=1.5)

    def test_favorito_forte_encurta_a_serie(self):
        equil = simulate_series_total(bo=5, p_blue_map=0.5, total=self._TOT, n_sims=4000)
        massacre = simulate_series_total(bo=5, p_blue_map=0.95, total=self._TOT, n_sims=4000)
        self.assertLess(float(massacre.mean()), float(equil.mean()),
                        "série de massacre acaba em menos mapas = total menor")

    def test_mesma_seed_reproduz_identico(self):
        a = simulate_series_total(bo=3, p_blue_map=0.6, total=self._TOT, n_sims=500, seed=42)
        b = simulate_series_total(bo=3, p_blue_map=0.6, total=self._TOT, n_sims=500, seed=42)
        self.assertTrue((a == b).all())

    def test_bo_invalido_levanta(self):
        with self.assertRaises(ValueError):
            simulate_series_total(bo=2, p_blue_map=0.5, total=self._TOT, n_sims=10)

    def test_prob_over_das_sims_coerente(self):
        sims = simulate_series_total(bo=3, p_blue_map=0.5, total=self._TOT, n_sims=4000)
        p_baixa = prob_over_series_from_sims(sims, 40.0)   # linha bem baixa
        p_alta = prob_over_series_from_sims(sims, 100.0)   # linha bem alta
        self.assertGreater(p_baixa, 0.9)
        self.assertLess(p_alta, 0.1)
        self.assertTrue(math.isnan(prob_over_series_from_sims(np.array([]), 60.0)))


class TestFilterTeamGamesAutoLiga(unittest.TestCase):
    """league_mode=auto tem que preferir a liga MAJOR, não a mais recente
    (regra que evita cair em pré-temporada tipo KeSPA)."""

    @staticmethod
    def _df():
        rows = []
        for i in range(4):  # pré-temporada mais RECENTE, porém não-major
            rows.append({"team": "T1", "date": pd.Timestamp(f"2026-06-{10+i:02d}"),
                         "league": "KeSPA Cup", "map_number": 1, "playoffs": 0, "split": ""})
        for i in range(6):
            rows.append({"team": "T1", "date": pd.Timestamp(f"2026-03-{10+i:02d}"),
                         "league": "LCK", "map_number": 1, "playoffs": 0, "split": ""})
        return pd.DataFrame(rows)

    def test_auto_prefere_major(self):
        out = filter_team_games(self._df(), team="T1",
                                filters=Filters(year=None, league=None, split=None, playoffs=None),
                                league_mode="auto")
        self.assertEqual(set(out["league"]), {"LCK"})

    def test_max_games_pega_os_mais_recentes(self):
        out = filter_team_games(self._df(), team="T1",
                                filters=Filters(year=None, league=None, split=None, playoffs=None),
                                league_mode="all", max_games=3)
        self.assertEqual(len(out), 3)
        self.assertTrue((out["date"] >= pd.Timestamp("2026-06-11")).all())


if __name__ == "__main__":
    unittest.main()
