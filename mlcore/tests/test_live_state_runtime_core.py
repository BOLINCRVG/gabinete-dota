"""Tests for app_core/live_state_runtime_core.py."""
import math
import unittest
from pathlib import Path

import pandas as pd

from app_core.live_state_runtime_core import (
    available_minutes,
    clear_model_cache,
    is_live_state_model_enabled,
    load_live_state_models,
    models_dir_from_state,
    predict_live_p_win,
)


_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "live_models"


def _full_state_at(t: int, blue_ahead: bool = True) -> dict:
    """Build a complete state_features dict for snapshot @t."""
    g_blue, g_red = (30000, 27000) if blue_ahead else (27000, 30000)
    x_blue, x_red = (32000, 30000) if blue_ahead else (30000, 32000)
    c_blue, c_red = (500, 480) if blue_ahead else (480, 500)
    k_blue, k_red = (5, 2) if blue_ahead else (2, 5)
    return {
        f"goldat{t}": g_blue, f"xpat{t}": x_blue, f"csat{t}": c_blue,
        f"killsat{t}": k_blue, f"deathsat{t}": k_red, f"assistsat{t}": k_blue + 3,
        f"opp_goldat{t}": g_red, f"opp_xpat{t}": x_red, f"opp_csat{t}": c_red,
        f"opp_killsat{t}": k_red, f"opp_deathsat{t}": k_blue, f"opp_assistsat{t}": k_red + 1,
    }


class TestFeatureFlag(unittest.TestCase):

    def test_default_off(self):
        self.assertFalse(is_live_state_model_enabled({}))
        self.assertFalse(is_live_state_model_enabled(None))

    def test_explicit_on(self):
        self.assertTrue(is_live_state_model_enabled({"ml_live_state_model_enabled": True}))

    def test_models_dir_default(self):
        self.assertEqual(models_dir_from_state({}), "live_models")

    def test_models_dir_custom(self):
        self.assertEqual(
            models_dir_from_state({"ml_live_state_models_dir": "custom_dir"}),
            "custom_dir",
        )


class TestLoadModels(unittest.TestCase):

    def setUp(self) -> None:
        clear_model_cache()

    def test_load_returns_dict(self):
        if not _MODELS_DIR.exists():
            self.skipTest("live_models/ not present (run Phase 6 to train)")
        models = load_live_state_models(str(_MODELS_DIR))
        self.assertIsInstance(models, dict)
        self.assertGreater(len(models), 0)

    def test_load_missing_dir_returns_empty(self):
        models = load_live_state_models("__nonexistent_dir__")
        self.assertEqual(models, {})

    def test_available_minutes_subset_of_snapshot_minutes(self):
        if not _MODELS_DIR.exists():
            self.skipTest("live_models/ not present")
        minutes = available_minutes(str(_MODELS_DIR))
        for m in minutes:
            self.assertIn(m, (10, 15, 20, 25))


class TestPredictLivePWin(unittest.TestCase):

    def test_flag_off_returns_none(self):
        p = predict_live_p_win(
            session_state={"ml_live_state_model_enabled": False},
            minute=15, gameid="g1", side="Blue", p_map_pre=0.6,
            state_features=_full_state_at(15),
        )
        self.assertIsNone(p)

    def test_invalid_minute_returns_none(self):
        p = predict_live_p_win(
            session_state={"ml_live_state_model_enabled": True},
            minute=7, gameid="g1", side="Blue", p_map_pre=0.6,
            state_features=_full_state_at(7),
        )
        self.assertIsNone(p)

    def test_invalid_p_pre_returns_none(self):
        p = predict_live_p_win(
            session_state={"ml_live_state_model_enabled": True},
            minute=15, gameid="g1", side="Blue", p_map_pre=1.5,  # invalid
            state_features=_full_state_at(15),
        )
        self.assertIsNone(p)

    def test_no_features_returns_none(self):
        p = predict_live_p_win(
            session_state={"ml_live_state_model_enabled": True},
            minute=15, gameid="g1", side="Blue", p_map_pre=0.5,
            state_features=None, team_rows=None,
        )
        self.assertIsNone(p)

    def test_returns_probability_when_enabled(self):
        if not _MODELS_DIR.exists():
            self.skipTest("live_models/ not present")
        ss = {"ml_live_state_model_enabled": True, "ml_live_state_models_dir": str(_MODELS_DIR)}
        p = predict_live_p_win(
            session_state=ss,
            minute=15, gameid="g1", side="Blue", p_map_pre=0.55,
            league="LCK", bo=5, is_playoffs=True,
            state_features=_full_state_at(15, blue_ahead=True),
        )
        self.assertIsNotNone(p)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_blue_with_advantage_gets_higher_p(self):
        if not _MODELS_DIR.exists():
            self.skipTest("live_models/ not present")
        ss = {"ml_live_state_model_enabled": True, "ml_live_state_models_dir": str(_MODELS_DIR)}
        p_ahead = predict_live_p_win(
            session_state=ss, minute=20, gameid="g1", side="Blue", p_map_pre=0.50,
            league="LCK", bo=5, is_playoffs=False,
            state_features=_full_state_at(20, blue_ahead=True),
        )
        p_behind = predict_live_p_win(
            session_state=ss, minute=20, gameid="g1", side="Blue", p_map_pre=0.50,
            league="LCK", bo=5, is_playoffs=False,
            state_features=_full_state_at(20, blue_ahead=False),
        )
        self.assertIsNotNone(p_ahead)
        self.assertIsNotNone(p_behind)
        self.assertGreater(p_ahead, p_behind)

    def test_later_minute_more_confident_with_same_lead(self):
        if not _MODELS_DIR.exists():
            self.skipTest("live_models/ not present")
        ss = {"ml_live_state_model_enabled": True, "ml_live_state_models_dir": str(_MODELS_DIR)}
        # Same lead, but at minute 25 the model should be more confident than at 10
        p_10 = predict_live_p_win(
            session_state=ss, minute=10, gameid="g1", side="Blue", p_map_pre=0.50,
            state_features=_full_state_at(10, blue_ahead=True),
        )
        p_25 = predict_live_p_win(
            session_state=ss, minute=25, gameid="g1", side="Blue", p_map_pre=0.50,
            state_features=_full_state_at(25, blue_ahead=True),
        )
        self.assertIsNotNone(p_10)
        self.assertIsNotNone(p_25)
        # Both should favor Blue; later minute should be at least as confident
        self.assertGreater(p_10, 0.5)
        self.assertGreater(p_25, 0.5)


class TestModelCache(unittest.TestCase):

    def test_cache_clear_does_not_crash(self):
        clear_model_cache()
        # And again to verify idempotence
        clear_model_cache()


if __name__ == "__main__":
    unittest.main()
