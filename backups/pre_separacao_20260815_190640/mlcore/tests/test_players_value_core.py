# -*- coding: utf-8 -*-
"""Testes do valor por jogador (redesign Players, 2026-07-16)."""
import math
import unittest

import pandas as pd

from app_core.players_value_core import build_player_role_scores, lineup_value_view


def _rows():
    """Dois mids (Faker forte, MidFraco fraco) + um top; 10 jogos cada."""
    data = []
    for i in range(10):
        data.append({"playername": "Faker", "position": "mid", "team": "T1",
                     "kills": 6, "deaths": 1, "assists": 8, "gamelength": 1800,
                     "damagetochampions": 24000, "cspm": 9.5, "visionscore": 40,
                     "date": f"2026-06-{i+1:02d}", "year": 2026})
        data.append({"playername": "MidFraco", "position": "mid", "team": "Zeta",
                     "kills": 2, "deaths": 5, "assists": 4, "gamelength": 1800,
                     "damagetochampions": 12000, "cspm": 7.0, "visionscore": 25,
                     "date": f"2026-06-{i+1:02d}", "year": 2026})
        data.append({"playername": "Zeus", "position": "top", "team": "T1",
                     "kills": 3, "deaths": 3, "assists": 5, "gamelength": 1800,
                     "damagetochampions": 15000, "cspm": 8.0, "visionscore": 20,
                     "date": f"2026-06-{i+1:02d}", "year": 2026})
    return pd.DataFrame(data)


class TestRoleScores(unittest.TestCase):
    def test_percentile_within_role(self):
        s = build_player_role_scores(_rows(), year=2026, min_games=5)
        faker = s[s["player"] == "Faker"].iloc[0]
        fraco = s[s["player"] == "MidFraco"].iloc[0]
        self.assertGreater(faker["percentile"], fraco["percentile"])
        self.assertEqual(int(faker["games"]), 10)

    def test_role_isolation_single_player_role_not_penalized(self):
        # Zeus é o único top: percentil dele não compete com os mids
        s = build_player_role_scores(_rows(), year=2026, min_games=5)
        zeus = s[s["player"] == "Zeus"]
        self.assertEqual(len(zeus), 1)

    def test_min_games_cuts_small_sample(self):
        s = build_player_role_scores(_rows(), year=2026, min_games=11)
        self.assertTrue(s.empty)

    def test_empty_never_explodes(self):
        self.assertTrue(build_player_role_scores(None).empty)
        self.assertTrue(build_player_role_scores(pd.DataFrame()).empty)


class TestLineupView(unittest.TestCase):
    def test_lineup_order_and_nan_for_unknown(self):
        s = build_player_role_scores(_rows(), year=2026, min_games=5)
        v = lineup_value_view(["Faker", "Desconhecido"], s)
        self.assertEqual(list(v["player"]), ["Faker", "Desconhecido"])
        self.assertTrue(math.isnan(float(v.iloc[1]["percentile"])))
        self.assertFalse(math.isnan(float(v.iloc[0]["percentile"])))

    def test_fuzzy_match_used_when_provided(self):
        s = build_player_role_scores(_rows(), year=2026, min_games=5)
        v = lineup_value_view(
            ["faker "], s,
            fuzzy_match_one_fn=lambda name, choices: "Faker",
        )
        self.assertFalse(math.isnan(float(v.iloc[0]["percentile"])))


if __name__ == "__main__":
    unittest.main()
