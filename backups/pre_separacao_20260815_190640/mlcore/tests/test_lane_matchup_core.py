from __future__ import annotations

import unittest

from app_core.lane_matchup_core import (
    _all_in_score,
    _confidence_label,
    _fight_setup_score,
    _lane_pressure_score,
    _pick_score,
    _scale_score,
    _side_score,
    _supportive_score,
    _team_from_delta,
    evaluate_lane_matchup,
)


class TestTeamFromDelta(unittest.TestCase):

    def test_positive_delta_above_threshold_returns_team_a(self):
        result = _team_from_delta(1.0, team_a="T1", team_b="GEN")
        self.assertEqual(result, "T1")

    def test_negative_delta_below_threshold_returns_team_b(self):
        result = _team_from_delta(-1.0, team_a="T1", team_b="GEN")
        self.assertEqual(result, "GEN")

    def test_zero_delta_returns_equilibrado(self):
        result = _team_from_delta(0.0, team_a="T1", team_b="GEN")
        self.assertEqual(result, "Equilibrado")

    def test_custom_threshold(self):
        result = _team_from_delta(0.5, team_a="T1", team_b="GEN", threshold=0.4)
        self.assertEqual(result, "T1")

    def test_returns_string(self):
        self.assertIsInstance(_team_from_delta(0.0, team_a="A", team_b="B"), str)


class TestConfidenceLabel(unittest.TestCase):

    def test_high_score_returns_alta(self):
        self.assertEqual(_confidence_label(2.0), "Alta")

    def test_medium_score_returns_media(self):
        self.assertEqual(_confidence_label(1.0), "Media")

    def test_low_score_returns_baixa(self):
        self.assertEqual(_confidence_label(0.5), "Baixa")

    def test_boundary_alta(self):
        self.assertEqual(_confidence_label(1.75), "Alta")

    def test_boundary_media(self):
        self.assertEqual(_confidence_label(0.9), "Media")

    def test_returns_string(self):
        self.assertIsInstance(_confidence_label(1.0), str)


class TestLanePressureScore(unittest.TestCase):

    def test_all_zero_traits_gives_zero(self):
        self.assertAlmostEqual(_lane_pressure_score({}), 0.0)

    def test_lane_priority_contributes(self):
        result = _lane_pressure_score({"lane_priority": 1.0})
        self.assertAlmostEqual(result, 1.0)

    def test_poke_weighted(self):
        result = _lane_pressure_score({"poke": 1.0})
        self.assertAlmostEqual(result, 0.70)

    def test_multiple_traits_sum(self):
        traits = {"lane_priority": 1.0, "poke": 1.0}
        result = _lane_pressure_score(traits)
        self.assertAlmostEqual(result, 1.70, places=5)

    def test_returns_float(self):
        self.assertIsInstance(_lane_pressure_score({"lane_priority": 1.0}), float)


class TestAllInScore(unittest.TestCase):

    def test_empty_traits_gives_zero(self):
        self.assertAlmostEqual(_all_in_score({}), 0.0)

    def test_all_in_trait_contributes_fully(self):
        result = _all_in_score({"all_in": 1.0})
        self.assertAlmostEqual(result, 1.0)

    def test_engage_weighted(self):
        result = _all_in_score({"engage": 1.0})
        self.assertAlmostEqual(result, 0.55)

    def test_multiple_traits(self):
        result = _all_in_score({"all_in": 1.0, "engage": 1.0})
        self.assertAlmostEqual(result, 1.55, places=5)


class TestFightSetupScore(unittest.TestCase):

    def test_empty_traits_gives_zero(self):
        self.assertAlmostEqual(_fight_setup_score({}), 0.0)

    def test_teamfight_contributes_fully(self):
        result = _fight_setup_score({"teamfight": 1.0})
        self.assertAlmostEqual(result, 1.0)

    def test_frontline_weighted(self):
        result = _fight_setup_score({"frontline": 1.0})
        self.assertAlmostEqual(result, 0.55)

    def test_returns_float(self):
        self.assertIsInstance(_fight_setup_score({}), float)


class TestScaleScore(unittest.TestCase):

    def test_empty_gives_zero(self):
        self.assertAlmostEqual(_scale_score({}), 0.0)

    def test_late_contributes_fully(self):
        self.assertAlmostEqual(_scale_score({"late": 1.0}), 1.0)

    def test_range_weighted(self):
        self.assertAlmostEqual(_scale_score({"range": 1.0}), 0.55)


class TestSideScore(unittest.TestCase):

    def test_empty_gives_zero(self):
        self.assertAlmostEqual(_side_score({}), 0.0)

    def test_splitpush_contributes_fully(self):
        self.assertAlmostEqual(_side_score({"splitpush": 1.0}), 1.0)

    def test_mobility_weighted(self):
        self.assertAlmostEqual(_side_score({"mobility": 1.0}), 0.35)


class TestPickScore(unittest.TestCase):

    def test_empty_gives_zero(self):
        self.assertAlmostEqual(_pick_score({}), 0.0)

    def test_pick_trait_contributes_fully(self):
        self.assertAlmostEqual(_pick_score({"pick": 1.0}), 1.0)

    def test_mobility_weighted(self):
        self.assertAlmostEqual(_pick_score({"mobility": 1.0}), 0.45)


class TestSupportiveScore(unittest.TestCase):

    def test_empty_gives_zero(self):
        self.assertAlmostEqual(_supportive_score({}), 0.0)

    def test_peel_contributes_fully(self):
        self.assertAlmostEqual(_supportive_score({"peel": 1.0}), 1.0)

    def test_teamfight_weighted(self):
        self.assertAlmostEqual(_supportive_score({"teamfight": 1.0}), 0.45)


class LaneMatchupCoreTest(unittest.TestCase):
    def test_lane_matchup_detects_early_vs_late_tradeoff(self) -> None:
        registry = {
            "champions": {
                "renekton": {
                    "name": "Renekton",
                    "traits": {
                        "early": 1.8,
                        "mid": 1.1,
                        "late": 0.2,
                        "lane_priority": 1.3,
                        "all_in": 1.0,
                        "teamfight": 0.6,
                        "frontline": 0.4,
                    },
                },
                "ornn": {
                    "name": "Ornn",
                    "traits": {
                        "early": 0.3,
                        "mid": 0.9,
                        "late": 1.8,
                        "lane_priority": 0.1,
                        "all_in": 0.1,
                        "teamfight": 1.4,
                        "frontline": 1.5,
                    },
                },
            },
            "aliases": {},
        }
        out = evaluate_lane_matchup(
            registry=registry,
            role="top",
            champ_a="Renekton",
            champ_b="Ornn",
            team_a="Time A",
            team_b="Time B",
            empirical_edge_pp=4.0,
            h2h_games=5,
            player_games_a=8,
            player_games_b=8,
        )
        self.assertEqual(out["early_team"], "Time A")
        self.assertEqual(out["late_team"], "Time B")
        self.assertIn("Renekton", out["summary"])
        self.assertIn("Ornn", out["summary"])
        self.assertTrue(any(token in out["summary"].casefold() for token in ("lane no early", "all-in", "troca curta")))
        self.assertIn("5v5", out["summary"])
        self.assertIn("Renekton", str(out.get("team_a_plan")))
        self.assertIn("Ornn", str(out.get("team_b_plan")))
        self.assertIn(out["confidence"], {"Media", "Alta"})

    def test_jungle_matchup_uses_setup_language_instead_of_scaling(self) -> None:
        registry = {
            "champions": {
                "pantheon": {
                    "name": "Pantheon",
                    "traits": {
                        "early": 2.0,
                        "mid": 1.2,
                        "late": -1.0,
                        "all_in": 1.2,
                        "pick": 1.3,
                        "dive": 1.0,
                        "mobility": 1.0,
                    },
                },
                "jarvaniv": {
                    "name": "Jarvan IV",
                    "traits": {
                        "early": 0.8,
                        "mid": 1.6,
                        "late": 0.3,
                        "engage": 1.7,
                        "teamfight": 1.8,
                        "frontline": 1.5,
                        "objective_control": 1.1,
                    },
                },
            },
            "aliases": {"jarvaniv": "jarvaniv"},
        }
        out = evaluate_lane_matchup(
            registry=registry,
            role="jng",
            champ_a="Pantheon",
            champ_b="Jarvan IV",
            team_a="Time A",
            team_b="Time B",
            empirical_edge_pp=1.5,
            h2h_games=3,
            player_games_a=6,
            player_games_b=6,
        )
        self.assertEqual(out["early_team"], "Time A")
        self.assertEqual(out["late_team"], "Time B")
        self.assertIn("Pantheon", out["summary"])
        self.assertIn("Jarvan IV", out["summary"])
        self.assertIn("primeiras jogadas", out["summary"])
        self.assertIn("engage", out["summary"].casefold())
        self.assertNotIn("escala melhor", out["summary"])
        self.assertIn("ganks", str(out.get("team_a_plan")).casefold())
        self.assertIn("engage", str(out.get("team_b_plan")).casefold())


if __name__ == "__main__":
    unittest.main()
