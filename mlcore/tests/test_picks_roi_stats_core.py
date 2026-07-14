"""Tests for picks_roi_stats_core + the live-ROI overlay in picks_anti_loss_core."""
from __future__ import annotations

import unittest

from app_core.picks_roi_stats_core import compute_roi_maps
from app_core import picks_anti_loss_core as al


class _FakePick:
    def __init__(self, metric_key, side_key, status, profit_units, stake_units, league=""):
        self.metric_key = metric_key
        self.side_key = side_key
        self.status = status
        self.profit_units = profit_units
        self.stake_units = stake_units
        self.league = league


def _make(metric, side, n_won, n_lost, *, odd=2.0, league="LCK", stake=1.0):
    """Build n_won winners + n_lost losers for a market. odd=2.0 → win=+1u, loss=-1u."""
    picks = []
    for _ in range(n_won):
        picks.append(_FakePick(metric, side, "won", stake * (odd - 1.0), stake, league))
    for _ in range(n_lost):
        picks.append(_FakePick(metric, side, "lost", -stake, stake, league))
    return picks


class TestComputeRoiMaps(unittest.TestCase):

    def test_basic_market_roi(self):
        # 60 won + 40 lost @ odd 2.0, 1u each:
        # profit = 60*1 - 40*1 = +20; risked = 100; ROI = +20%
        picks = _make("kills", "Under", 60, 40)
        m = compute_roi_maps(picks, min_n_market=30)
        self.assertIn(("kills", "Under"), m["by_market"])
        self.assertAlmostEqual(m["by_market"][("kills", "Under")], 20.0, places=3)
        self.assertEqual(m["n_by_market"][("kills", "Under")], 100)

    def test_market_below_threshold_dropped(self):
        picks = _make("barons", "Over", 5, 5)  # n=10 < 30
        m = compute_roi_maps(picks, min_n_market=30)
        self.assertNotIn(("barons", "Over"), m["by_market"])

    def test_void_excluded_from_denominator(self):
        picks = _make("kills", "Under", 30, 0)  # 30 wins, +30u, risk 30
        picks += [_FakePick("kills", "Under", "void", 0.0, 1.0, "LCK") for _ in range(5)]
        m = compute_roi_maps(picks, min_n_market=30)
        # ROI still +100% (void adds 0 profit, 0 risk); n counts the voids though
        self.assertAlmostEqual(m["by_market"][("kills", "Under")], 100.0, places=3)
        self.assertEqual(m["n_by_market"][("kills", "Under")], 35)

    def test_ml_grouped_without_side(self):
        picks = [_FakePick("ml", "A", "won", 1.0, 1.0, "LCK") for _ in range(20)]
        picks += [_FakePick("ml", "B", "lost", -1.0, 1.0, "LCK") for _ in range(20)]
        m = compute_roi_maps(picks, min_n_market=30)
        self.assertIn(("ml", ""), m["by_market"])  # A and B merged under ("ml","")
        self.assertEqual(m["n_by_market"][("ml", "")], 40)

    def test_league_roi(self):
        picks = _make("kills", "Under", 60, 40, league="LPL")
        m = compute_roi_maps(picks, min_n_league=100)
        self.assertIn("LPL", m["by_league"])
        self.assertAlmostEqual(m["by_league"]["LPL"], 20.0, places=3)

    def test_league_below_threshold_dropped(self):
        picks = _make("kills", "Under", 30, 20, league="TINY")  # n=50 < 100
        m = compute_roi_maps(picks, min_n_league=100)
        self.assertNotIn("TINY", m["by_league"])

    def test_open_picks_ignored(self):
        picks = _make("kills", "Under", 60, 40)
        picks += [_FakePick("kills", "Under", "open", None, 1.0, "LCK") for _ in range(10)]
        m = compute_roi_maps(picks, min_n_market=30)
        self.assertEqual(m["n_by_market"][("kills", "Under")], 100)  # opens not counted

    def test_empty_returns_empty_maps(self):
        m = compute_roi_maps([])
        self.assertEqual(m["by_market"], {})
        self.assertEqual(m["by_league"], {})
        self.assertEqual(m["n_total"], 0)


class TestLiveOverlay(unittest.TestCase):

    def tearDown(self):
        al.set_live_roi(None, None)  # always clear overlay between tests

    def test_overlay_supersedes_snapshot(self):
        # Frozen snapshot has Over Torres at -52.45. Overlay says -10.
        al.set_live_roi({("towers", "Over"): -10.0}, {})
        self.assertEqual(al.market_historical_roi("towers", "Over"), -10.0)

    def test_fallback_to_snapshot_when_no_overlay(self):
        al.set_live_roi(None, None)
        self.assertEqual(al.market_historical_roi("towers", "Over"), -52.45)

    def test_overlay_new_market_not_in_snapshot(self):
        # barons Under isn't in the frozen snapshot; overlay provides it.
        al.set_live_roi({("barons", "Under"): 14.77}, {})
        self.assertEqual(al.market_historical_roi("barons", "Under"), 14.77)

    def test_demotion_uses_overlay(self):
        # Overlay makes kills Under terrible → must demote.
        al.set_live_roi({("kills", "Under"): -25.0}, {})
        new_cat, warn = al.demote_classification("recommend", metric_key="kills", side_key="Under")
        self.assertEqual(new_cat, "avoid")
        self.assertIsNotNone(warn)

    def test_strength_caption_uses_overlay(self):
        al.set_live_roi({("kills", "Under"): 8.15}, {})
        cap = al.market_strength_caption("kills", "Under")
        self.assertIsNotNone(cap)
        self.assertIn("8.2", cap)  # 8.15 rounds to +8.2% in the display

    def test_league_caption_uses_overlay(self):
        al.set_live_roi({}, {"LEC": -1.92})
        cap = al.league_caption("LEC")
        self.assertIsNotNone(cap)
        self.assertIn("LEC", cap)

    def test_league_caption_none_when_positive(self):
        al.set_live_roi({}, {"LPL": 7.82})
        self.assertIsNone(al.league_caption("LPL"))


if __name__ == "__main__":
    unittest.main()
