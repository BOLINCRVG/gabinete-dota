"""Tests for app_core/live_bet_champ_profile_core.py — champion profile helpers."""
import unittest

from app_core.live_bet_champ_profile_core import _champ_trait, _player_tactical_profile


class TestChampTrait(unittest.TestCase):

    def test_returns_trait_value(self):
        entry = {"traits": {"engage": 1.5, "carry": 0.8}}
        self.assertAlmostEqual(_champ_trait(entry, "engage"), 1.5)

    def test_missing_trait_returns_zero(self):
        entry = {"traits": {"engage": 1.5}}
        self.assertAlmostEqual(_champ_trait(entry, "poke"), 0.0)

    def test_none_entry_returns_zero(self):
        self.assertAlmostEqual(_champ_trait(None, "engage"), 0.0)

    def test_empty_entry_returns_zero(self):
        self.assertAlmostEqual(_champ_trait({}, "engage"), 0.0)

    def test_returns_float(self):
        entry = {"traits": {"engage": 1}}
        self.assertIsInstance(_champ_trait(entry, "engage"), float)


class TestPlayerTacticalProfile(unittest.TestCase):

    def _profile(self, role, champ="Jinx"):
        return _player_tactical_profile(role, champ, registry=None)

    def test_returns_dict_with_required_keys(self):
        profile = self._profile("bot")
        for key in ("primary_label", "risk_label", "carry_score", "engage_score"):
            self.assertIn(key, profile)

    def test_bot_role_boosts_carry(self):
        profile_bot = self._profile("bot")
        profile_sup = self._profile("sup")
        self.assertGreater(profile_bot["carry_score"], profile_sup["carry_score"])

    def test_risk_label_is_string(self):
        profile = self._profile("mid")
        self.assertIn(profile["risk_label"], ("Muito exposto", "Exposto", "Seguro"))

    def test_primary_label_is_string(self):
        profile = self._profile("top")
        self.assertIsInstance(profile["primary_label"], str)

    def test_exposure_score_is_float(self):
        profile = self._profile("adc")
        self.assertIsInstance(profile["exposure_score"], float)

    def test_sup_role_has_setup_boost(self):
        profile = self._profile("sup")
        self.assertGreater(profile["setup_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
