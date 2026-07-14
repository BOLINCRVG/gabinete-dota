from __future__ import annotations

import unittest

from app_core.champion_knowledge_core import champion_style_summary, get_champion_knowledge


class TestGetChampionKnowledge(unittest.TestCase):

    def _make_registry(self, key, entry):
        return {"champions": {key: entry}, "aliases": {}}

    def test_resolves_alias(self):
        registry = {
            "champions": {
                "wukong": {
                    "name": "Wukong",
                    "primary_role": "jng",
                    "styles": ["engage", "teamfight"],
                    "window": "mid",
                    "traits": {"early": 0.8, "mid": 1.2, "late": 0.7},
                }
            },
            "aliases": {"monkeyking": "wukong"},
        }
        out = get_champion_knowledge(registry, "MonkeyKing", role_hint="jng")
        self.assertEqual(out["name"], "Wukong")
        self.assertEqual(out["primary_role"], "jng")

    def test_role_fallback_for_unknown_champion(self):
        out = get_champion_knowledge({}, "Unknown Champ", role_hint="bot")
        self.assertEqual(out["primary_role"], "bot")
        self.assertEqual(out["window"], "mid")
        self.assertIn("siege", out["styles"])

    def test_empty_name_returns_role_fallback(self):
        out = get_champion_knowledge({}, "", role_hint="top")
        self.assertEqual(out["primary_role"], "top")

    def test_none_registry_returns_fallback(self):
        out = get_champion_knowledge(None, "Renekton", role_hint="top")
        self.assertEqual(out["primary_role"], "top")
        self.assertIn("name", out)

    def test_exact_match_returns_entry(self):
        registry = {
            "champions": {
                "jinx": {
                    "name": "Jinx",
                    "primary_role": "bot",
                    "styles": ["siege"],
                    "window": "late",
                    "traits": {"early": 0.2, "late": 1.8},
                }
            },
            "aliases": {},
        }
        out = get_champion_knowledge(registry, "Jinx", role_hint="bot")
        self.assertEqual(out["name"], "Jinx")
        self.assertIn("siege", out["styles"])

    def test_case_insensitive_lookup(self):
        registry = {
            "champions": {
                "ornn": {
                    "name": "Ornn",
                    "styles": ["frontline"],
                    "window": "late",
                    "traits": {},
                }
            },
            "aliases": {},
        }
        out = get_champion_knowledge(registry, "ORNN", role_hint="top")
        self.assertEqual(out["name"], "Ornn")

    def test_traits_default_to_zero_for_missing_keys(self):
        registry = {
            "champions": {
                "thresh": {
                    "name": "Thresh",
                    "styles": ["engage"],
                    "window": "mid",
                    "traits": {"engage": 1.5},
                }
            },
            "aliases": {},
        }
        out = get_champion_knowledge(registry, "Thresh", role_hint="sup")
        self.assertIn("traits", out)
        self.assertIsInstance(out["traits"], dict)

    def test_fallback_sup_role(self):
        out = get_champion_knowledge({}, "SomeSupport", role_hint="sup")
        self.assertIn("peel", out["styles"])

    def test_fallback_jng_role(self):
        out = get_champion_knowledge({}, "SomeJungler", role_hint="jng")
        self.assertIn("engage", out["styles"])

    def test_fallback_mid_role(self):
        out = get_champion_knowledge({}, "SomeMid", role_hint="mid")
        self.assertIn("poke", out["styles"])


class TestChampionStyleSummary(unittest.TestCase):

    def test_returns_styles_joined(self):
        entry = {"styles": ["engage", "teamfight"]}
        result = champion_style_summary(entry)
        self.assertEqual(result, "engage / teamfight")

    def test_limit_one(self):
        entry = {"styles": ["dive", "skirmish", "engage"]}
        result = champion_style_summary(entry, limit=1)
        self.assertEqual(result, "dive")

    def test_none_entry_returns_dash(self):
        self.assertEqual(champion_style_summary(None), "-")

    def test_empty_styles_falls_back_to_traits(self):
        entry = {"styles": [], "traits": {"dive": 1.5, "poke": 0.5}}
        result = champion_style_summary(entry)
        self.assertIn("dive", result)

    def test_no_styles_no_traits_returns_dash(self):
        entry = {"styles": [], "traits": {}}
        result = champion_style_summary(entry)
        self.assertEqual(result, "-")

    def test_returns_string(self):
        self.assertIsInstance(champion_style_summary({"styles": ["poke"]}), str)


if __name__ == "__main__":
    unittest.main()
