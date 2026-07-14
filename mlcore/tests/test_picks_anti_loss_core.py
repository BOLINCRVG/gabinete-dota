"""Tests for app_core.picks_anti_loss_core."""
from __future__ import annotations

import unittest

from app_core.picks_anti_loss_core import (
    BAD_LEAGUE_ROI,
    BAD_MARKET_ROI,
    STRONG_MARKET_ROI,
    demote_classification,
    league_caption,
    market_strength_caption,
)


class TestDemoteClassification(unittest.TestCase):

    def test_over_torres_demoted_to_avoid_from_recommend(self):
        """Over Torres has ROI -52% (severe) — recommend must go to avoid."""
        new, warn = demote_classification("recommend", metric_key="towers", side_key="Over")
        self.assertEqual(new, "avoid")
        self.assertIsNotNone(warn)
        self.assertIn("severo", warn)

    def test_over_torres_demoted_from_watch_to_avoid(self):
        new, _warn = demote_classification("watch", metric_key="towers", side_key="Over")
        self.assertEqual(new, "avoid")

    def test_over_dragons_severe_demotion(self):
        """Over Drag has ROI -20% — at the severe threshold."""
        new, _warn = demote_classification("recommend", metric_key="dragons", side_key="Over")
        self.assertEqual(new, "avoid")

    def test_under_torres_not_demoted(self):
        """Under Torres has +18.81% ROI — must NOT be demoted."""
        new, warn = demote_classification("recommend", metric_key="towers", side_key="Under")
        self.assertEqual(new, "recommend")
        self.assertIsNone(warn)

    def test_under_kills_not_demoted(self):
        """Under Kills carries +8.33% ROI (largest sample). Not in bad set."""
        new, warn = demote_classification("recommend", metric_key="kills", side_key="Under")
        self.assertEqual(new, "recommend")
        self.assertIsNone(warn)

    def test_unknown_market_no_change(self):
        new, warn = demote_classification("recommend", metric_key="exotic", side_key="Over")
        self.assertEqual(new, "recommend")
        self.assertIsNone(warn)

    def test_case_insensitive_metric_key(self):
        new, _ = demote_classification("recommend", metric_key="TOWERS", side_key="Over")
        self.assertEqual(new, "avoid")


class TestLeagueCaption(unittest.TestCase):

    def test_lec_returns_warning(self):
        cap = league_caption("LEC")
        self.assertIsNotNone(cap)
        self.assertIn("LEC", cap)
        self.assertIn("-4", cap)  # -3.99% rounds to -4.0%

    def test_lfl_returns_warning(self):
        cap = league_caption("LFL")
        self.assertIsNotNone(cap)
        self.assertIn("LFL", cap)

    def test_lck_no_warning(self):
        """LCK has +6% ROI — not in bad set."""
        self.assertIsNone(league_caption("LCK"))

    def test_lpl_no_warning(self):
        self.assertIsNone(league_caption("LPL"))

    def test_unknown_league_no_warning(self):
        self.assertIsNone(league_caption("MADE_UP"))

    def test_empty_returns_none(self):
        self.assertIsNone(league_caption(""))
        self.assertIsNone(league_caption(None))

    def test_case_insensitive(self):
        self.assertIsNotNone(league_caption("lec"))


class TestMarketStrengthCaption(unittest.TestCase):

    def test_under_torres_strong(self):
        cap = market_strength_caption("towers", "Under")
        self.assertIsNotNone(cap)
        self.assertIn("+18", cap)

    def test_over_torres_not_strong(self):
        """Over Torres bleeds; doesn't get a positive caption."""
        self.assertIsNone(market_strength_caption("towers", "Over"))

    def test_under_kills_strong(self):
        cap = market_strength_caption("kills", "Under")
        self.assertIsNotNone(cap)
        self.assertIn("+8", cap)

    def test_unknown_returns_none(self):
        self.assertIsNone(market_strength_caption("exotic", "Over"))


class TestDataIntegrity(unittest.TestCase):
    """Sanity: every BAD_MARKET entry has a negative ROI; STRONG has positive."""

    def test_bad_markets_all_negative(self):
        for k, v in BAD_MARKET_ROI.items():
            self.assertLess(v, 0, f"{k} should have negative ROI, got {v}")

    def test_bad_leagues_all_negative(self):
        for k, v in BAD_LEAGUE_ROI.items():
            self.assertLess(v, 0, f"{k} should have negative ROI, got {v}")

    def test_strong_markets_all_positive(self):
        for k, v in STRONG_MARKET_ROI.items():
            self.assertGreater(v, 0, f"{k} should have positive ROI, got {v}")

    def test_no_overlap_bad_strong(self):
        """A (metric, side) cannot be both bad and strong."""
        overlap = set(BAD_MARKET_ROI) & set(STRONG_MARKET_ROI)
        self.assertEqual(overlap, set(), f"Overlap: {overlap}")


if __name__ == "__main__":
    unittest.main()
