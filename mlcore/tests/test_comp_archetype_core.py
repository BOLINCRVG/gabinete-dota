from __future__ import annotations

import unittest

from app_core.comp_archetype_core import (
    _style_score,
    _top_styles,
    _winner_label,
    build_comp_profile,
    compare_comp_profiles,
)


class TestWinnerLabel(unittest.TestCase):

    def test_positive_delta_above_threshold_returns_team_a(self):
        result = _winner_label(1.0, team_a="T1", team_b="GEN")
        self.assertEqual(result, "T1")

    def test_negative_delta_below_threshold_returns_team_b(self):
        result = _winner_label(-1.0, team_a="T1", team_b="GEN")
        self.assertEqual(result, "GEN")

    def test_zero_delta_returns_equilibrado(self):
        result = _winner_label(0.0, team_a="T1", team_b="GEN")
        self.assertEqual(result, "Equilibrado")

    def test_custom_threshold(self):
        result = _winner_label(0.5, team_a="A", team_b="B", threshold=0.4)
        self.assertEqual(result, "A")

    def test_returns_string(self):
        self.assertIsInstance(_winner_label(0.0, team_a="A", team_b="B"), str)


class TestTopStyles(unittest.TestCase):

    def test_returns_top_two_by_default(self):
        scores = {"dive": 1.5, "poke": 1.2, "siege": 0.5}
        result = _top_styles(scores)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "dive")

    def test_limit_one(self):
        scores = {"dive": 1.5, "poke": 1.2}
        result = _top_styles(scores, limit=1)
        self.assertEqual(len(result), 1)

    def test_zero_scores_excluded(self):
        scores = {"dive": 0.0, "poke": 1.0}
        result = _top_styles(scores)
        self.assertNotIn("dive", result)

    def test_returns_list(self):
        self.assertIsInstance(_top_styles({"dive": 1.0}), list)


class TestStyleScore(unittest.TestCase):

    def test_existing_key_returns_float(self):
        scores = {"dive": 1.5}
        result = _style_score(scores, "dive")
        self.assertAlmostEqual(result, 1.5)

    def test_missing_key_returns_zero(self):
        result = _style_score({}, "nonexistent")
        self.assertAlmostEqual(result, 0.0)

    def test_returns_float(self):
        self.assertIsInstance(_style_score({"x": 1.0}, "x"), float)


class CompArchetypeCoreTest(unittest.TestCase):
    def test_comp_compare_separates_early_and_late_strength(self) -> None:
        registry = {
            "champions": {
                "renekton": {"name": "Renekton", "styles": ["dive", "skirmish"], "window": "early", "traits": {"early": 1.7, "mid": 0.9, "late": 0.2, "dive": 1.1, "skirmish": 1.2}},
                "nidalee": {"name": "Nidalee", "styles": ["poke", "skirmish"], "window": "early", "traits": {"early": 1.6, "mid": 0.8, "late": 0.1, "poke": 0.8, "skirmish": 1.1, "objective_control": 0.6}},
                "ahri": {"name": "Ahri", "styles": ["pick", "poke"], "window": "mid", "traits": {"early": 0.9, "mid": 1.2, "late": 0.7, "pick": 1.2, "mobility": 1.1}},
                "caitlyn": {"name": "Caitlyn", "styles": ["siege", "poke"], "window": "mid", "traits": {"early": 1.0, "mid": 1.2, "late": 0.8, "siege": 1.4, "range": 1.4}},
                "karma": {"name": "Karma", "styles": ["poke", "peel"], "window": "mid", "traits": {"early": 1.0, "mid": 1.0, "late": 0.7, "poke": 1.0, "peel": 0.8}},
                "ornn": {"name": "Ornn", "styles": ["frontline", "teamfight"], "window": "late", "traits": {"early": 0.2, "mid": 0.9, "late": 1.7, "frontline": 1.6, "teamfight": 1.2, "objective_control": 0.5}},
                "sejuani": {"name": "Sejuani", "styles": ["engage", "frontline"], "window": "mid", "traits": {"early": 0.5, "mid": 1.1, "late": 1.0, "engage": 1.2, "frontline": 1.2, "objective_control": 0.8}},
                "orianna": {"name": "Orianna", "styles": ["teamfight", "peel"], "window": "late", "traits": {"early": 0.3, "mid": 1.0, "late": 1.5, "teamfight": 1.5, "peel": 0.9}},
                "jinx": {"name": "Jinx", "styles": ["siege", "teamfight"], "window": "late", "traits": {"early": 0.2, "mid": 0.9, "late": 1.8, "siege": 1.3, "teamfight": 1.1, "objective_control": 1.1}},
                "nautilus": {"name": "Nautilus", "styles": ["engage", "frontline"], "window": "mid", "traits": {"early": 0.7, "mid": 1.0, "late": 0.5, "engage": 1.4, "frontline": 0.9}},
            },
            "aliases": {},
        }
        comp_a = build_comp_profile(registry=registry, picks=["Renekton", "Nidalee", "Ahri", "Caitlyn", "Karma"], role_order=["top", "jng", "mid", "bot", "sup"])
        comp_b = build_comp_profile(registry=registry, picks=["Ornn", "Sejuani", "Orianna", "Jinx", "Nautilus"], role_order=["top", "jng", "mid", "bot", "sup"])
        out = compare_comp_profiles(comp_a=comp_a, comp_b=comp_b, team_a="Time A", team_b="Time B")
        self.assertEqual(out["early_team"], "Time A")
        self.assertEqual(out["late_team"], "Time B")
        self.assertIn("Early favorece Time A.", out["reasons"])
        self.assertEqual(str(((out.get("plan_profiles") or {}).get("Time A") or {}).get("identity")), "Skirmish cedo")
        self.assertIn("forcar 2v2/3v3", str(((out.get("plan_profiles") or {}).get("Time A") or {}).get("win_condition")))
        self.assertEqual(str(((out.get("plan_profiles") or {}).get("Time B") or {}).get("identity")), "Front-to-back")
        self.assertIn("proteger o carry", str(((out.get("plan_profiles") or {}).get("Time B") or {}).get("win_condition")))
        self.assertTrue(
            any(
                ("Time B" in reason and "late" in reason.casefold()) or ("Time B" in reason and "jogo alonga" in reason.casefold())
                for reason in out["reasons"]
            )
        )

    def test_comp_compare_flags_low_tower_environment_when_no_real_side_pressure(self) -> None:
        registry = {
            "champions": {
                "rumble": {"name": "Rumble", "styles": ["teamfight"], "window": "mid", "traits": {"early": 1.5, "mid": 1.6, "late": 0.9, "waveclear": 1.1, "teamfight": 1.7, "splitpush": 0.0}},
                "pantheon": {"name": "Pantheon", "styles": ["pick"], "window": "early", "traits": {"early": 2.0, "mid": 1.0, "late": -0.8, "all_in": 1.1, "pick": 1.2, "dive": 1.0, "splitpush": 0.0}},
                "aurora": {"name": "Aurora", "styles": ["pick"], "window": "mid", "traits": {"early": 1.0, "mid": 1.8, "late": 1.0, "pick": 1.8, "mobility": 1.3, "waveclear": 1.0, "splitpush": 0.0}},
                "yunara": {"name": "Yunara", "styles": ["siege"], "window": "late", "traits": {"early": 0.0, "mid": 0.8, "late": 1.8, "siege": 1.7, "range": 1.8, "splitpush": 0.0}},
                "milio": {"name": "Milio", "styles": ["peel"], "window": "late", "traits": {"early": 0.3, "mid": 1.0, "late": 1.3, "peel": 1.4, "teamfight": 1.1, "splitpush": 0.0}},
                "ornn": {"name": "Ornn", "styles": ["frontline"], "window": "late", "traits": {"early": -0.8, "mid": 0.6, "late": 2.1, "frontline": 1.8, "teamfight": 1.6, "splitpush": 0.1}},
                "jarvaniv": {"name": "Jarvan IV", "styles": ["engage"], "window": "mid", "traits": {"early": 0.8, "mid": 1.4, "late": 0.3, "engage": 1.8, "frontline": 1.4, "objective_control": 1.1, "splitpush": 0.0}},
                "leblanc": {"name": "LeBlanc", "styles": ["pick"], "window": "mid", "traits": {"early": 1.1, "mid": 1.8, "late": 0.8, "pick": 1.8, "mobility": 1.2, "waveclear": 0.9, "splitpush": 0.0}},
                "aphelios": {"name": "Aphelios", "styles": ["siege"], "window": "late", "traits": {"early": -0.3, "mid": 1.0, "late": 2.1, "siege": 1.7, "range": 1.8, "teamfight": 1.1, "splitpush": 0.0}},
                "thresh": {"name": "Thresh", "styles": ["engage"], "window": "mid", "traits": {"early": 0.4, "mid": 1.2, "late": 0.8, "engage": 1.5, "peel": 0.9, "teamfight": 1.4, "splitpush": 0.0}},
            },
            "aliases": {"jarvaniv": "jarvaniv"},
        }
        comp_a = build_comp_profile(registry=registry, picks=["Rumble", "Pantheon", "Aurora", "Yunara", "Milio"], role_order=["top", "jng", "mid", "bot", "sup"])
        comp_b = build_comp_profile(registry=registry, picks=["Ornn", "Jarvan IV", "LeBlanc", "Aphelios", "Thresh"], role_order=["top", "jng", "mid", "bot", "sup"])
        out = compare_comp_profiles(comp_a=comp_a, comp_b=comp_b, team_a="Gen.G", team_b="T1")
        self.assertEqual(out["tower_total_call"], "Menos torres")
        self.assertLess(float(out["tower_total_bias"]), 0.0)
        self.assertIn("side/split", out["tower_total_reason"])
        self.assertEqual(out["dragons_total_call"], "Menos dragoes")
        self.assertLess(float(out["dragons_total_bias"]), 0.0)
        self.assertTrue(str(out["dragons_total_reason"]))
        self.assertEqual(out["barons_total_call"], "Menos baroes")
        self.assertLess(float(out["barons_total_bias"]), 0.0)
        self.assertTrue(str(out["barons_total_reason"]))
        self.assertTrue(str(((out.get("plan_profiles") or {}).get("Gen.G") or {}).get("win_condition")))
        self.assertTrue(str(((out.get("plan_profiles") or {}).get("T1") or {}).get("avoid_condition")))


if __name__ == "__main__":
    unittest.main()
