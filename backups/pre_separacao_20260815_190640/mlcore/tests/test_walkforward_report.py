import unittest
import warnings
from unittest.mock import patch

import pandas as pd

from mlcore.walkforward_report import _month_key, _normalize_walkforward_input_df, _roi_proxy_fold, run_walkforward


class WalkForwardHelpersTest(unittest.TestCase):
    def test_month_key(self):
        self.assertEqual(_month_key(pd.Timestamp("2026-02-09")), "2026-02")

    def test_roi_proxy_fold_runs(self):
        y = [1, 0, 1, 1, 0]
        p = [0.6, 0.55, 0.7, 0.52, 0.48]
        r = _roi_proxy_fold(y, p)
        self.assertTrue(isinstance(r, float))

    def test_normalize_walkforward_input_df_enforces_required_columns(self):
        df = pd.DataFrame(
            {
                "gameid": ["g1", "", None],
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "blue": ["A", "B", "C"],
                "red": ["X", "Y", ""],
                "blue_win": [1, 0, 1],
                "league": ["LCK", None, "LPL"],
            }
        )
        out = _normalize_walkforward_input_df(df)
        self.assertListEqual(list(out.columns), ["gameid", "date", "blue", "red", "blue_win", "league"])
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.iloc[0]["gameid"]), "g1")

    def test_run_walkforward_ignores_sparse_frames_without_concat_warning(self):
        sparse = pd.DataFrame({"foo": [None]})
        valid = pd.DataFrame(
            {
                "gameid": ["g1", "g2", "g3", "g4"],
                "date": ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"],
                "blue": ["A", "B", "A", "B"],
                "red": ["C", "D", "C", "D"],
                "blue_win": [1, 0, 1, 0],
                "league": ["LCK", "LCK", "LCK", "LCK"],
            }
        )
        with patch("mlcore.walkforward_report.load_oracle_elixir_games", side_effect=[sparse, valid]):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                rep = run_walkforward(["sparse.csv", "valid.csv"], min_train_games=1, holdout_months=1)
        concat_warn = [w for w in rec if "concatenation with empty or all-NA entries" in str(w.message)]
        self.assertFalse(concat_warn, "unexpected pandas concat warning")
        self.assertIn("overall", rep)

    def test_run_walkforward_uses_active_model_when_artifacts_are_provided(self):
        cal_payload = {
            "generated_at": "2026-03-19T00:00:00+00:00",
            "models": {
                "fused_active": {"n": 42, "logloss": 0.51, "brier": 0.18, "ece": 0.03},
            },
            "monthly_fused": [
                {"month": "2026-01", "n": 20, "logloss": 0.50, "brier": 0.17, "ece": 0.02, "mean_p": 0.55, "emp_rate": 0.60},
                {"month": "2026-02", "n": 22, "logloss": 0.52, "brier": 0.19, "ece": 0.04, "mean_p": 0.53, "emp_rate": 0.50},
            ],
            "by_league": {
                "LCK": {
                    "fused_active": {"n": 30, "logloss": 0.49, "brier": 0.16, "ece": 0.02},
                }
            },
        }
        with patch("mlcore.walkforward_report.run_calibration_diagnostics", return_value=cal_payload):
            with patch("pathlib.Path.exists", return_value=True):
                rep = run_walkforward(
                    ["daily.csv"],
                    artifact_path="ml_artifact.json",
                    players_artifact_path="players_artifact.json",
                )
        self.assertEqual(rep.get("mode"), "active_model")
        self.assertEqual(rep.get("source"), "calibration_diagnostics")
        self.assertEqual((rep.get("overall") or {}).get("n_total"), 42)
        self.assertEqual(len(rep.get("folds") or []), 2)
        self.assertIn("LCK", rep.get("by_league") or {})


if __name__ == "__main__":
    unittest.main()
