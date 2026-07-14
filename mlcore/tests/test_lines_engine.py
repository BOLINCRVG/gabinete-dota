import math
import unittest

import pandas as pd

from mlcore.lines_engine import build_profile, matchup_expected_totals, prob_over_from_total
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


if __name__ == "__main__":
    unittest.main()
