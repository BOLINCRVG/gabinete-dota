"""Tests for app_core/stats_config.py — STATS_MARKET_CONFIG structure."""
import unittest

from app_core.stats_config import STATS_MARKET_CONFIG


class TestStatsMarketConfig(unittest.TestCase):

    REQUIRED_KEYS = ("metric_key", "combined_col", "threshold_key", "default_thresholds", "pct_prefix")
    EXPECTED_MARKETS = {"Kills", "Torres", "Dragões", "Nashors", "Inibidores", "Tempo"}

    def test_is_dict(self):
        self.assertIsInstance(STATS_MARKET_CONFIG, dict)

    def test_expected_markets_present(self):
        for market in self.EXPECTED_MARKETS:
            self.assertIn(market, STATS_MARKET_CONFIG, f"missing market: {market}")

    def test_each_entry_has_required_keys(self):
        for market, cfg in STATS_MARKET_CONFIG.items():
            for key in self.REQUIRED_KEYS:
                self.assertIn(key, cfg, f"{market} missing key: {key}")

    def test_metric_keys_are_strings(self):
        for market, cfg in STATS_MARKET_CONFIG.items():
            self.assertIsInstance(cfg["metric_key"], str, f"{market}.metric_key not str")

    def test_kills_metric_key(self):
        self.assertEqual(STATS_MARKET_CONFIG["Kills"]["metric_key"], "kills")

    def test_tempo_metric_key(self):
        self.assertEqual(STATS_MARKET_CONFIG["Tempo"]["metric_key"], "time")

    def test_pct_prefix_non_empty(self):
        for market, cfg in STATS_MARKET_CONFIG.items():
            self.assertTrue(len(cfg["pct_prefix"]) > 0, f"{market}.pct_prefix is empty")

    def test_at_least_five_markets(self):
        self.assertGreaterEqual(len(STATS_MARKET_CONFIG), 5)


if __name__ == "__main__":
    unittest.main()
