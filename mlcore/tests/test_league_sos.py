"""Tests for SOS-adjusted league strength (Bradley-Terry) + model integration."""
from __future__ import annotations

import unittest

from mlcore.league_sos import fit_league_bt, sos_league_base_points
from mlcore.elo import EloModel, EloParams


class TestBradleyTerry(unittest.TestCase):

    def test_recovers_ordering(self):
        # A beats B most of the time, B beats C most of the time → A > B > C.
        pairs = []
        pairs += [("A", "B", 1.0)] * 8 + [("A", "B", 0.0)] * 2
        pairs += [("B", "C", 1.0)] * 8 + [("B", "C", 0.0)] * 2
        pairs += [("A", "C", 1.0)] * 9 + [("A", "C", 0.0)] * 1
        theta = fit_league_bt(pairs)
        self.assertGreater(theta["A"], theta["B"])
        self.assertGreater(theta["B"], theta["C"])

    def test_controls_for_strength_of_schedule(self):
        # WEAK league W only plays the very weak league VW and beats it (high
        # raw win rate). STRONG league S plays the elite E and loses often (low
        # raw win rate). BT must still rank S above W (SOS-adjusted).
        pairs = []
        pairs += [("W", "VW", 1.0)] * 18 + [("W", "VW", 0.0)] * 2   # W: 90% but vs trash
        pairs += [("S", "E", 0.0)] * 12 + [("S", "E", 1.0)] * 8     # S: 40% but vs elite
        pairs += [("E", "VW", 1.0)] * 10                            # link the graph: E >> VW
        theta = fit_league_bt(pairs)
        self.assertGreater(theta["S"], theta["W"],
                           "SOS-adjusted: strong league (tough schedule) must outrank padded weak league")

    def test_empty_returns_empty(self):
        self.assertEqual(fit_league_bt([]), {})

    def test_min_games_filter(self):
        pairs = [("A", "B", 1.0)] * 30 + [("C", "D", 1.0)] * 5  # C,D have few games
        pts = sos_league_base_points(pairs, min_games=20)
        self.assertIn("A", pts); self.assertIn("B", pts)
        self.assertNotIn("C", pts); self.assertNotIn("D", pts)


class TestModelLeagueBasePointsMap(unittest.TestCase):

    def test_map_overrides_and_fallback(self):
        m = EloModel(EloParams())
        # Unknown-to-map league falls back to global static prior (no crash).
        _ = m._league_base_points("LCK")
        m.league_base_points_map = {"LCK": 123.0}
        self.assertEqual(m._league_base_points("LCK"), 123.0)
        # A league not in the map still resolves (global fallback).
        self.assertIsInstance(m._league_base_points("LPL"), float)

    def test_map_serialization_roundtrip(self):
        m = EloModel(EloParams())
        m.league_base_points_map = {"LCK": 50.0, "LPL": 30.0}
        m.ratings = {"T1": 1.0}
        d = m.to_dict()
        self.assertIn("league_base_points_map", d)
        m2 = EloModel.from_dict(d)
        self.assertEqual(m2.league_base_points_map, {"LCK": 50.0, "LPL": 30.0})

    def test_legacy_artifact_without_map(self):
        # Old artifacts have no league_base_points_map → loads as empty (legacy behavior).
        m = EloModel.from_dict({"params": {}, "ratings": {}})
        self.assertEqual(m.league_base_points_map, {})


if __name__ == "__main__":
    unittest.main()
