"""Tests for app_core/live_markets_runtime_core.py."""
import unittest
from pathlib import Path

from app_core.live_markets_runtime_core import (
    ACTIVE_MARKETS,
    available_markets,
    clear_market_model_cache,
    is_live_market_models_enabled,
    markets_dir_from_state,
    predict_market,
)


_MARKETS_DIR = Path(__file__).resolve().parent.parent.parent / "live_models" / "markets"


class TestActiveMarkets(unittest.TestCase):

    def test_active_markets_does_not_include_barons(self):
        # Barons REJECTED by the gate
        self.assertNotIn("total_barons", ACTIVE_MARKETS)

    def test_active_markets_includes_kills_towers_dragons_time(self):
        for m in ("total_kills", "total_towers", "total_dragons", "gamelength_min"):
            self.assertIn(m, ACTIVE_MARKETS)


class TestFeatureFlag(unittest.TestCase):

    def test_flag_off_returns_disabled(self):
        r = predict_market(
            session_state={"ml_live_state_model_enabled": False},
            market="total_kills", minute=15, line=27.5,
        )
        self.assertIsNone(r["mean"])
        self.assertEqual(r["reason"], "disabled")

    def test_flag_on_changes_enabled(self):
        self.assertFalse(is_live_market_models_enabled({}))
        self.assertTrue(is_live_market_models_enabled({"ml_live_state_model_enabled": True}))


class TestPredictMarket(unittest.TestCase):

    def setUp(self):
        if not _MARKETS_DIR.exists():
            self.skipTest("live_models/markets/ not present (run Phase 7 to train)")
        clear_market_model_cache()

    def _ss(self):
        return {
            "ml_live_state_model_enabled": True,
            "ml_live_state_models_dir": str(_MARKETS_DIR.parent),
        }

    def test_predict_kills_at_15(self):
        r = predict_market(
            session_state=self._ss(),
            market="total_kills", minute=15, line=27.5,
            p_map_pre=0.55, gold_diff=2000.0,
            kills_a=4, kills_b=2,
        )
        self.assertIsNotNone(r["mean"])
        self.assertGreater(r["mean"], 0.0)
        self.assertGreater(r["p_over"], 0.0)
        self.assertLess(r["p_over"], 1.0)

    def test_rejected_market_returns_none(self):
        # total_barons is not in ACTIVE_MARKETS
        r = predict_market(
            session_state=self._ss(),
            market="total_barons", minute=15, line=1.5,
        )
        self.assertIsNone(r["mean"])
        self.assertEqual(r["reason"], "market_not_active")

    def test_invalid_minute(self):
        r = predict_market(
            session_state=self._ss(),
            market="total_kills", minute=50, line=27.5,
        )
        self.assertIsNone(r["mean"])
        self.assertEqual(r["reason"], "minute_out_of_window")

    def test_no_line_no_p_over(self):
        r = predict_market(
            session_state=self._ss(),
            market="total_kills", minute=15,
        )
        # mean is computed even without a line
        self.assertIsNotNone(r["mean"])
        self.assertIsNone(r["p_over"])

    def test_higher_line_lower_p_over(self):
        ss = self._ss()
        kwargs = {
            "market": "total_kills", "minute": 20,
            "p_map_pre": 0.55, "gold_diff": 2500.0,
        }
        p_low = predict_market(session_state=ss, line=10.0, **kwargs)["p_over"]
        p_high = predict_market(session_state=ss, line=50.0, **kwargs)["p_over"]
        self.assertGreater(p_low, p_high)

    def test_supported_markets_load(self):
        avail = available_markets(str(_MARKETS_DIR))
        for m in ACTIVE_MARKETS:
            self.assertIn(m, avail)
            for minute in (10, 15, 20, 25):
                self.assertIn(minute, avail[m])


if __name__ == "__main__":
    unittest.main()
