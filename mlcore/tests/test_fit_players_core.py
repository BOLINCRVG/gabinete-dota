"""Tests for mlcore/fit_players_core.py — pure logit-delta and settings helpers."""
import math
import unittest

from mlcore.fit_players_core import _apply_logit_delta, _players_context_settings_from_cfg


class TestApplyLogitDelta(unittest.TestCase):

    def test_zero_delta_returns_unchanged(self):
        result = _apply_logit_delta(0.6, 0.0)
        self.assertAlmostEqual(result, 0.6, places=5)

    def test_positive_delta_increases_prob(self):
        result = _apply_logit_delta(0.5, 1.0)
        self.assertGreater(result, 0.5)

    def test_negative_delta_decreases_prob(self):
        result = _apply_logit_delta(0.5, -1.0)
        self.assertLess(result, 0.5)

    def test_result_clipped_below_1(self):
        result = _apply_logit_delta(0.99, 100.0)
        self.assertLess(result, 1.0)

    def test_result_clipped_above_0(self):
        result = _apply_logit_delta(0.01, -100.0)
        self.assertGreater(result, 0.0)

    def test_returns_float(self):
        result = _apply_logit_delta(0.5, 0.5)
        self.assertIsInstance(result, float)

    def test_nan_delta_returns_p_unchanged(self):
        result = _apply_logit_delta(0.6, float("nan"))
        self.assertAlmostEqual(result, 0.6, places=5)

    def test_inf_delta_returns_p_unchanged(self):
        result = _apply_logit_delta(0.6, float("inf"))
        self.assertAlmostEqual(result, 0.6, places=5)

    def test_tiny_delta_returns_approx_unchanged(self):
        result = _apply_logit_delta(0.5, 1e-13)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_symmetric_positive_negative(self):
        p_up = _apply_logit_delta(0.5, 0.5)
        p_down = _apply_logit_delta(0.5, -0.5)
        self.assertAlmostEqual(p_up + p_down, 1.0, places=5)

    def test_result_is_finite(self):
        result = _apply_logit_delta(0.7, 2.0)
        self.assertTrue(math.isfinite(result))


class TestPlayersContextSettingsFromCfg(unittest.TestCase):

    def test_empty_cfg_returns_defaults(self):
        cfg = _players_context_settings_from_cfg({})
        self.assertTrue(cfg["ml_players_context_enabled"])
        self.assertEqual(cfg["ml_players_context_lookback_games"], 24)
        self.assertAlmostEqual(cfg["ml_players_context_half_life_days"], 45.0)
        self.assertEqual(cfg["ml_players_context_min_games"], 6)
        self.assertAlmostEqual(cfg["ml_players_context_strength"], 0.35)
        self.assertAlmostEqual(cfg["ml_players_context_cap"], 0.22)
        self.assertAlmostEqual(cfg["ml_players_context_min_reliability"], 0.28)
        self.assertAlmostEqual(cfg["ml_players_context_inter_mult"], 0.88)

    def test_none_cfg_treated_as_empty(self):
        cfg = _players_context_settings_from_cfg(None)
        self.assertEqual(cfg["ml_players_context_lookback_games"], 24)

    def test_custom_lookback_games(self):
        cfg = _players_context_settings_from_cfg({"lookback_games": 30})
        self.assertEqual(cfg["ml_players_context_lookback_games"], 30)

    def test_custom_strength(self):
        cfg = _players_context_settings_from_cfg({"strength": 0.5})
        self.assertAlmostEqual(cfg["ml_players_context_strength"], 0.5)

    def test_enabled_false(self):
        cfg = _players_context_settings_from_cfg({"enabled": False})
        self.assertFalse(cfg["ml_players_context_enabled"])

    def test_returns_dict_with_all_keys(self):
        cfg = _players_context_settings_from_cfg({})
        expected_keys = [
            "ml_players_context_enabled",
            "ml_players_context_lookback_games",
            "ml_players_context_half_life_days",
            "ml_players_context_min_games",
            "ml_players_context_strength",
            "ml_players_context_cap",
            "ml_players_context_min_reliability",
            "ml_players_context_inter_mult",
        ]
        for key in expected_keys:
            self.assertIn(key, cfg)

    def test_lookback_games_is_int(self):
        cfg = _players_context_settings_from_cfg({})
        self.assertIsInstance(cfg["ml_players_context_lookback_games"], int)

    def test_half_life_days_is_float(self):
        cfg = _players_context_settings_from_cfg({})
        self.assertIsInstance(cfg["ml_players_context_half_life_days"], float)

    def test_enabled_is_bool(self):
        cfg = _players_context_settings_from_cfg({})
        self.assertIsInstance(cfg["ml_players_context_enabled"], bool)


if __name__ == "__main__":
    unittest.main()
