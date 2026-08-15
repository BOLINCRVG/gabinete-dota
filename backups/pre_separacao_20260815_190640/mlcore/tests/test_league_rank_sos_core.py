import unittest

import pandas as pd

from app_core.league_rank_sos_core import (
    _data_completeness_weight,
    _norm_margin,
    apply_league_rank_sos_live,
)


class TestNormMargin(unittest.TestCase):

    def test_zero_value_returns_zero(self):
        self.assertEqual(_norm_margin(0.0, 100.0), 0.0)

    def test_positive_value_normalized(self):
        result = _norm_margin(50.0, 100.0)
        self.assertAlmostEqual(result, 0.5)

    def test_clamped_to_one_max(self):
        result = _norm_margin(200.0, 100.0)
        self.assertEqual(result, 1.0)

    def test_clamped_to_neg_one_min(self):
        result = _norm_margin(-500.0, 100.0)
        self.assertEqual(result, -1.0)

    def test_zero_scale_returns_zero(self):
        self.assertEqual(_norm_margin(50.0, 0.0), 0.0)

    def test_none_value_returns_zero(self):
        self.assertEqual(_norm_margin(None, 100.0), 0.0)

    def test_returns_float(self):
        self.assertIsInstance(_norm_margin(10.0, 100.0), float)


class TestDataCompletenessWeight(unittest.TestCase):

    def test_complete_returns_one(self):
        self.assertEqual(_data_completeness_weight("complete"), 1.0)

    def test_full_returns_one(self):
        self.assertEqual(_data_completeness_weight("full"), 1.0)

    def test_partial_returns_0_82(self):
        self.assertAlmostEqual(_data_completeness_weight("partial"), 0.82)

    def test_incomplete_returns_0_82(self):
        self.assertAlmostEqual(_data_completeness_weight("incomplete"), 0.82)

    def test_unknown_returns_0_92(self):
        self.assertAlmostEqual(_data_completeness_weight("unknown"), 0.92)

    def test_none_returns_0_92(self):
        self.assertAlmostEqual(_data_completeness_weight(None), 0.92)

    def test_returns_float(self):
        self.assertIsInstance(_data_completeness_weight("complete"), float)


class TestApplyLeagueRankSosLive(unittest.TestCase):

    def _rows(self):
        return [
            {"team": "Witchcraft", "opponent": "Partizan Sangal", "league": "EM", "win": 1, "year": 2026},
            {"team": "Partizan Sangal", "opponent": "Witchcraft", "league": "EM", "win": 0, "year": 2026},
            {"team": "Witchcraft", "opponent": "Other Team", "league": "EM", "win": 1, "year": 2026},
            {"team": "Other Team", "opponent": "Witchcraft", "league": "EM", "win": 0, "year": 2026},
            {"team": "Partizan Sangal", "opponent": "Other Team", "league": "EM", "win": 1, "year": 2026},
            {"team": "Other Team", "opponent": "Partizan Sangal", "league": "EM", "win": 0, "year": 2026},
            {"team": "Fourth Team", "opponent": "Other Team", "league": "EM", "win": 1, "year": 2026},
            {"team": "Other Team", "opponent": "Fourth Team", "league": "EM", "win": 0, "year": 2026},
        ]

    def test_alias_resolves_team_a(self):
        out = apply_league_rank_sos_live(
            p_map_in=0.55,
            team_games=pd.DataFrame(self._rows()),
            team_a="Los Ratones",
            team_b="Partizan Sangal",
            team_aliases_a=["Witchcraft"],
            team_aliases_b=[],
            league="EM",
            year_opt=2026,
            bo=3,
        )
        self.assertTrue(bool(out.get("team_a_found", False)))
        self.assertEqual(str(out.get("team_a_lookup")), "Witchcraft")

    def test_returns_dict(self):
        out = apply_league_rank_sos_live(
            p_map_in=0.55,
            team_games=pd.DataFrame(self._rows()),
            team_a="Team X",
            team_b="Team Y",
            league="EM",
            year_opt=2026,
            bo=3,
        )
        self.assertIsInstance(out, dict)

    def test_empty_df_returns_passthrough(self):
        out = apply_league_rank_sos_live(
            p_map_in=0.55,
            team_games=pd.DataFrame(),
            team_a="T1",
            team_b="GEN",
            league="LCK",
            year_opt=2026,
            bo=3,
        )
        self.assertIsInstance(out, dict)


if __name__ == "__main__":
    unittest.main()
