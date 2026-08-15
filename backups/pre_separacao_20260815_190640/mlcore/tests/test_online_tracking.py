"""Tests for online prediction tracking + calibration aggregation."""
from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from mlcore.online_tracking import (
    OutcomeRecord,
    PredictionRecord,
    compute_calibration_summary,
    join_predictions_with_outcomes,
    load_outcomes,
    load_predictions,
    record_outcome,
    record_prediction,
    summarize_by_market,
)


class TestRecordRoundtrip(unittest.TestCase):

    def test_write_and_read_back_predictions(self):
        with TemporaryDirectory() as td:
            rec = PredictionRecord(
                gameid="G1", market="p_win", p_predicted=0.62,
                side="Blue", league="LCK", bo=5, is_playoffs=True,
            )
            record_prediction(rec, base_dir=td)
            loaded = load_predictions(td)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["gameid"], "G1")
            self.assertAlmostEqual(loaded[0]["p_predicted"], 0.62)

    def test_write_and_read_back_outcomes(self):
        with TemporaryDirectory() as td:
            rec = OutcomeRecord(gameid="G1", market="p_win", y_true=1.0)
            record_outcome(rec, base_dir=td)
            loaded = load_outcomes(td)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["gameid"], "G1")
            self.assertEqual(loaded[0]["y_true"], 1.0)

    def test_multiple_appends(self):
        with TemporaryDirectory() as td:
            for i in range(5):
                record_prediction(PredictionRecord(
                    gameid=f"G{i}", market="p_win", p_predicted=0.5,
                ), base_dir=td)
            loaded = load_predictions(td)
            self.assertEqual(len(loaded), 5)


class TestJoin(unittest.TestCase):

    def test_inner_join_by_gameid_market(self):
        with TemporaryDirectory() as td:
            record_prediction(PredictionRecord(
                gameid="G1", market="p_win", p_predicted=0.7,
            ), base_dir=td)
            record_outcome(OutcomeRecord(
                gameid="G1", market="p_win", y_true=1.0,
            ), base_dir=td)
            preds = load_predictions(td)
            outs = load_outcomes(td)
            joined = join_predictions_with_outcomes(preds, outs)
            self.assertEqual(len(joined), 1)
            self.assertAlmostEqual(joined[0]["p_predicted"], 0.7)
            self.assertEqual(joined[0]["y_true"], 1.0)

    def test_unmatched_predictions_are_dropped(self):
        with TemporaryDirectory() as td:
            record_prediction(PredictionRecord(
                gameid="G1", market="p_win", p_predicted=0.7,
            ), base_dir=td)
            # No outcome for G1
            joined = join_predictions_with_outcomes(load_predictions(td),
                                                     load_outcomes(td))
            self.assertEqual(len(joined), 0)

    def test_line_used_in_join_key(self):
        with TemporaryDirectory() as td:
            record_prediction(PredictionRecord(
                gameid="G1", market="total_kills", p_predicted=0.65, line=27.5,
            ), base_dir=td)
            record_outcome(OutcomeRecord(
                gameid="G1", market="total_kills", y_true=1.0, line=27.5,
            ), base_dir=td)
            joined = join_predictions_with_outcomes(load_predictions(td),
                                                     load_outcomes(td))
            self.assertEqual(len(joined), 1)

    def test_line_mismatch_skips(self):
        with TemporaryDirectory() as td:
            record_prediction(PredictionRecord(
                gameid="G1", market="total_kills", p_predicted=0.65, line=27.5,
            ), base_dir=td)
            record_outcome(OutcomeRecord(
                gameid="G1", market="total_kills", y_true=1.0, line=30.0,
            ), base_dir=td)
            joined = join_predictions_with_outcomes(load_predictions(td),
                                                     load_outcomes(td))
            self.assertEqual(len(joined), 0)


class TestCalibrationSummary(unittest.TestCase):

    def test_brier_known_value(self):
        # Predict 0.6, outcome 1 → brier = (1-0.6)^2 = 0.16
        joined = [{"market": "p_win", "p_predicted": 0.6, "y_true": 1.0,
                   "line": None, "side": "Blue", "minute": None, "league": "",
                   "gameid": "G", "ts_predicted": "", "ts_resolved": ""}]
        s = compute_calibration_summary(joined)
        self.assertAlmostEqual(s["brier"], 0.16, places=5)

    def test_perfect_calibration_at_boundary(self):
        # Predict 1.0, outcome 1 → brier=0, logloss≈0
        # (1.0 gets clipped to 1-1e-6, so logloss = -log(1-1e-6) ≈ 1e-6)
        joined = [{"market": "p_win", "p_predicted": 1.0, "y_true": 1.0,
                   "line": None, "side": "Blue", "minute": None, "league": "",
                   "gameid": "G", "ts_predicted": "", "ts_resolved": ""}]
        s = compute_calibration_summary(joined)
        self.assertAlmostEqual(s["brier"], 0.0, places=5)
        self.assertLess(s["logloss"], 1e-3)

    def test_empty_input(self):
        s = compute_calibration_summary([])
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["brier"])

    def test_summary_by_market(self):
        joined = [
            {"market": "p_win", "p_predicted": 0.6, "y_true": 1.0, "line": None,
             "side": "Blue", "minute": None, "league": "", "gameid": "G1",
             "ts_predicted": "", "ts_resolved": ""},
            {"market": "p_win", "p_predicted": 0.7, "y_true": 0.0, "line": None,
             "side": "Blue", "minute": None, "league": "", "gameid": "G2",
             "ts_predicted": "", "ts_resolved": ""},
            {"market": "first_blood", "p_predicted": 0.5, "y_true": 1.0, "line": None,
             "side": "Blue", "minute": None, "league": "", "gameid": "G3",
             "ts_predicted": "", "ts_resolved": ""},
        ]
        by_m = summarize_by_market(joined)
        self.assertIn("p_win", by_m)
        self.assertIn("first_blood", by_m)
        self.assertEqual(by_m["p_win"]["n"], 2)
        self.assertEqual(by_m["first_blood"]["n"], 1)

    def test_ece_well_calibrated(self):
        # 100 predictions at 0.7 with 70% of outcomes 1 → low ECE
        joined = []
        for i in range(100):
            y = 1.0 if i < 70 else 0.0
            joined.append({"market": "p_win", "p_predicted": 0.7, "y_true": y,
                           "line": None, "side": "Blue", "minute": None,
                           "league": "", "gameid": f"G{i}",
                           "ts_predicted": "", "ts_resolved": ""})
        s = compute_calibration_summary(joined)
        # ECE should be tiny — predicted 0.7, actual 0.7
        self.assertLess(s["ece"], 0.01)


class TestMonthRolling(unittest.TestCase):

    def test_predictions_path_uses_month(self):
        from mlcore.online_tracking import predictions_path
        with TemporaryDirectory() as td:
            p = predictions_path(td, when="2026-05-15T10:00:00Z")
            self.assertTrue(str(p).endswith("predictions_202605.jsonl"))


if __name__ == "__main__":
    unittest.main()
