"""Tests for app_core.picks_tracker_core — persistent picks + resolution + ROI."""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime

import pandas as pd

from app_core.picks_tracker_core import (
    Pick,
    load_picks,
    new_pick,
    resolve_open_picks,
    roi_summary,
    save_picks,
    upsert_pick,
)


def _mk_team_game(team: str, opp: str, date: str, *, win: int = 1,
                   kills: float = 28, towers: float = 11,
                   gameid: str = "g1") -> dict:
    return {
        "team": team, "opponent": opp, "date": date, "gameid": gameid,
        "win": win, "total_kills": kills, "total_towers": towers,
        "total_dragons": 4, "total_nashors": 1, "total_inhibitors": 1,
        "game_time_min": 33.0,
    }


class TestNewPick(unittest.TestCase):

    def test_deterministic_id(self):
        p1 = new_pick(team_a="A", team_b="B", league="LCK",
                      metric_key="kills", side_key="Over", line=27.5,
                      odd_casa=1.85, stake_units=1.0, ts="2026-05-30T18:00:00")
        p2 = new_pick(team_a="A", team_b="B", league="LCK",
                      metric_key="kills", side_key="Over", line=27.5,
                      odd_casa=2.00, stake_units=0.5, ts="2026-05-30T18:00:00")
        self.assertEqual(p1.id, p2.id, "Same matchup/market/line/ts → same id")

    def test_id_changes_with_line(self):
        p1 = new_pick(team_a="A", team_b="B", league="LCK",
                      metric_key="kills", side_key="Over", line=27.5,
                      odd_casa=1.85, stake_units=1.0, ts="2026-05-30T18:00:00")
        p2 = new_pick(team_a="A", team_b="B", league="LCK",
                      metric_key="kills", side_key="Over", line=28.5,
                      odd_casa=1.85, stake_units=1.0, ts="2026-05-30T18:00:00")
        self.assertNotEqual(p1.id, p2.id)


class TestPersistence(unittest.TestCase):

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "picks.json")
            p = new_pick(team_a="A", team_b="B", league="LCK",
                          metric_key="kills", side_key="Over", line=27.5,
                          odd_casa=1.85, stake_units=1.0)
            save_picks(path, [p])
            self.assertEqual(len(load_picks(path)), 1)
            self.assertEqual(load_picks(path)[0].id, p.id)

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_picks("/nonexistent/path.json"), [])

    def test_upsert_replaces_existing(self):
        p1 = new_pick(team_a="A", team_b="B", league="LCK",
                       metric_key="kills", side_key="Over", line=27.5,
                       odd_casa=1.85, stake_units=1.0, ts="2026-05-30T18:00:00")
        p2 = Pick(**{**p1.to_dict(), "status": "won", "profit_units": 0.85})
        out = upsert_pick([p1], p2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].status, "won")


class TestResolveOpenPicks(unittest.TestCase):

    def test_kills_over_wins(self):
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="kills", side_key="Over", line=27.5,
                         odd_casa=1.85, stake_units=1.0,
                         ts="2026-05-25T18:00:00")
        tg = pd.DataFrame([
            _mk_team_game("T1", "Gen.G", "2026-05-26", kills=30),
        ])
        out, n = resolve_open_picks([pick], tg)
        self.assertEqual(n, 1)
        self.assertEqual(out[0].status, "won")
        self.assertAlmostEqual(out[0].profit_units, 0.85, places=4)

    def test_kills_over_loses(self):
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="kills", side_key="Over", line=27.5,
                         odd_casa=2.0, stake_units=1.0,
                         ts="2026-05-25T18:00:00")
        tg = pd.DataFrame([_mk_team_game("T1", "Gen.G", "2026-05-26", kills=25)])
        out, _ = resolve_open_picks([pick], tg)
        self.assertEqual(out[0].status, "lost")
        self.assertAlmostEqual(out[0].profit_units, -1.0)

    def test_under_wins(self):
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="kills", side_key="Under", line=27.5,
                         odd_casa=1.95, stake_units=0.5,
                         ts="2026-05-25T18:00:00")
        tg = pd.DataFrame([_mk_team_game("T1", "Gen.G", "2026-05-26", kills=24)])
        out, _ = resolve_open_picks([pick], tg)
        self.assertEqual(out[0].status, "won")

    def test_exact_line_is_void(self):
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="kills", side_key="Over", line=27.0,
                         odd_casa=1.85, stake_units=1.0,
                         ts="2026-05-25T18:00:00")
        tg = pd.DataFrame([_mk_team_game("T1", "Gen.G", "2026-05-26", kills=27)])
        out, _ = resolve_open_picks([pick], tg)
        self.assertEqual(out[0].status, "void")
        self.assertEqual(out[0].profit_units, 0.0)

    def test_ml_a_wins(self):
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="ml", side_key="A", line=None,
                         odd_casa=1.45, stake_units=1.25,
                         ts="2026-05-25T18:00:00")
        tg = pd.DataFrame([_mk_team_game("T1", "Gen.G", "2026-05-26", win=1)])
        out, _ = resolve_open_picks([pick], tg)
        self.assertEqual(out[0].status, "won")
        self.assertAlmostEqual(out[0].profit_units, 1.25 * 0.45)

    def test_ml_b_wins_when_a_loses(self):
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="ml", side_key="B", line=None,
                         odd_casa=2.85, stake_units=1.0,
                         ts="2026-05-25T18:00:00")
        tg = pd.DataFrame([_mk_team_game("T1", "Gen.G", "2026-05-26", win=0)])
        out, _ = resolve_open_picks([pick], tg)
        self.assertEqual(out[0].status, "won")

    def test_no_game_yet_stays_open(self):
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="kills", side_key="Over", line=27.5,
                         odd_casa=1.85, stake_units=1.0,
                         ts="2026-06-01T18:00:00")
        tg = pd.DataFrame([_mk_team_game("T1", "Gen.G", "2026-05-26", kills=30)])
        out, n = resolve_open_picks([pick], tg)
        self.assertEqual(n, 0)
        self.assertEqual(out[0].status, "open")

    def test_resolves_against_first_game_not_rematch(self):
        """A pick must settle against the FIRST game at/after its ts, not a
        later rematch. Guards the BO/rematch bug."""
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="kills", side_key="Over", line=27.5,
                         odd_casa=1.85, stake_units=1.0,
                         ts="2026-05-01T18:00:00")
        tg = pd.DataFrame([
            _mk_team_game("T1", "Gen.G", "2026-05-02", kills=30, gameid="first"),   # WON Over 27.5
            _mk_team_game("T1", "Gen.G", "2026-06-20", kills=10, gameid="rematch"),  # would LOSE
        ])
        out, n = resolve_open_picks([pick], tg)
        self.assertEqual(n, 1)
        self.assertEqual(out[0].status, "won")  # resolved vs the first game
        self.assertEqual(out[0].resolved_gameid, "first")

    def test_series_takes_earliest_map_row(self):
        """Within a same-day series, resolution is deterministic (earliest row)."""
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="kills", side_key="Under", line=27.5,
                         odd_casa=1.90, stake_units=1.0,
                         ts="2026-05-01T00:00:00")
        tg = pd.DataFrame([
            _mk_team_game("T1", "Gen.G", "2026-05-02 10:00", kills=24, gameid="m1"),
            _mk_team_game("T1", "Gen.G", "2026-05-02 11:00", kills=33, gameid="m2"),
        ])
        out, _ = resolve_open_picks([pick], tg)
        self.assertEqual(out[0].resolved_gameid, "m1")  # earliest
        self.assertEqual(out[0].status, "won")  # 24 < 27.5

    def test_already_settled_stays_settled(self):
        pick = new_pick(team_a="T1", team_b="Gen.G", league="LCK",
                         metric_key="kills", side_key="Over", line=27.5,
                         odd_casa=1.85, stake_units=1.0,
                         ts="2026-05-25T18:00:00")
        pick = Pick(**{**pick.to_dict(), "status": "won", "profit_units": 0.85})
        tg = pd.DataFrame([_mk_team_game("T1", "Gen.G", "2026-05-26", kills=15)])
        out, n = resolve_open_picks([pick], tg)
        self.assertEqual(n, 0)  # already settled, not touched
        self.assertEqual(out[0].status, "won")


class TestRoiSummary(unittest.TestCase):

    def _settled(self, status: str, profit: float, stake: float = 1.0,
                  metric: str = "kills") -> Pick:
        p = new_pick(team_a="A", team_b="B", league="LCK",
                      metric_key=metric, side_key="Over", line=27.5,
                      odd_casa=2.0, stake_units=stake,
                      ts=f"2026-05-{20 + len(metric):02d}T18:00:00")
        return Pick(**{**p.to_dict(), "status": status, "profit_units": profit})

    def test_basic_roi(self):
        # 1u stake, 2.0 odd. Win → +1u; Loss → -1u. 1W 1L → 0 ROI.
        picks = [
            self._settled("won", 1.0),
            self._settled("lost", -1.0, metric="towers"),
        ]
        s = roi_summary(picks)
        self.assertEqual(s["wins"], 1); self.assertEqual(s["losses"], 1)
        self.assertEqual(s["profit_units"], 0.0)
        self.assertEqual(s["roi_pct"], 0.0)

    def test_void_not_in_denominator(self):
        picks = [
            self._settled("won", 1.0),
            self._settled("void", 0.0, metric="towers"),
        ]
        s = roi_summary(picks)
        self.assertEqual(s["voids"], 1)
        # Profit = 1, risked = 1 (void excluded), ROI = 100%
        self.assertEqual(s["roi_pct"], 100.0)

    def test_open_excluded_from_roi(self):
        p_open = new_pick(team_a="A", team_b="B", league="LCK",
                           metric_key="kills", side_key="Over", line=27.5,
                           odd_casa=1.85, stake_units=1.0)
        picks = [p_open, self._settled("won", 0.85)]
        s = roi_summary(picks)
        self.assertEqual(s["n_open"], 1)
        self.assertEqual(s["n_settled"], 1)

    def test_by_metric_breakdown(self):
        picks = [
            self._settled("won", 1.0, metric="kills"),
            self._settled("lost", -1.0, metric="kills"),
            self._settled("won", 0.85, metric="towers"),
        ]
        s = roi_summary(picks)
        self.assertIn("kills", s["by_metric"])
        self.assertEqual(s["by_metric"]["kills"]["wins"], 1)
        self.assertEqual(s["by_metric"]["kills"]["n"], 2)
        self.assertEqual(s["by_metric"]["towers"]["wins"], 1)


if __name__ == "__main__":
    unittest.main()
