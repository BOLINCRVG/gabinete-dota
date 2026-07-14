import unittest

from app_core.runtime_tuning_core import (
    compute_confidence_governor_budget_pp,
    compute_series_coherence_max_dev_pp,
)


class TestRuntimeTuningCore(unittest.TestCase):
    def test_confidence_governor_budget_reduces_for_strong_league_with_positive_fit(self):
        out = compute_confidence_governor_budget_pp(
            base_budget_pp=8.0,
            league_hint="LCK",
            enabled=True,
            use_fitted_by_league=True,
            params_fit={
                "best_by_league": {
                    "LCK": {
                        "n_total": 2400,
                        "improvement_logloss_vs_baseline": 0.010,
                    }
                }
            },
            confidence_baseline={
                "by_league": {
                    "LCK": {
                        "quality_band": "strong",
                        "lv_n": 2500,
                        "wf_n": 900,
                    }
                }
            },
        )
        self.assertEqual(str(out.get("band")), "strong")
        self.assertLess(float(out.get("budget_pp")), 8.0)
        self.assertGreaterEqual(float(out.get("budget_pp")), 3.0)

    def test_confidence_governor_budget_increases_for_low_sample_league(self):
        out = compute_confidence_governor_budget_pp(
            base_budget_pp=8.0,
            league_hint="MINOR-X",
            enabled=True,
            use_fitted_by_league=False,
            confidence_baseline={
                "by_league": {
                    "MINOR-X": {
                        "quality_band": "low_sample",
                        "lv_n": 60,
                        "wf_n": 0,
                    }
                }
            },
        )
        self.assertEqual(str(out.get("band")), "low_sample")
        self.assertGreater(float(out.get("budget_pp")), 8.0)

    def test_series_coherence_cap_tighter_for_low_sample(self):
        out = compute_series_coherence_max_dev_pp(
            bo=5,
            base_bo3_pp=16.0,
            base_bo5_pp=20.0,
            league_hint="MINOR-X",
            enabled=True,
            use_fitted_by_league=False,
            confidence_baseline={
                "by_league": {
                    "MINOR-X": {
                        "quality_band": "low_sample",
                        "lv_n": 80,
                        "wf_n": 0,
                    }
                }
            },
        )
        self.assertEqual(int(out.get("bo")), 5)
        self.assertLess(float(out.get("max_dev_pp")), 20.0)
        self.assertGreaterEqual(float(out.get("max_dev_pp")), 10.0)

    def test_series_coherence_cap_relaxes_for_strong_league_with_good_fit(self):
        out = compute_series_coherence_max_dev_pp(
            bo=3,
            base_bo3_pp=16.0,
            base_bo5_pp=20.0,
            league_hint="LPL",
            enabled=True,
            use_fitted_by_league=True,
            params_fit={
                "series_gamma_by_league": {
                    "LPL": {
                        "rows": 400,
                        "bo3": {"n": 280, "improvement_logloss_vs_gamma1": 0.012},
                    }
                }
            },
            confidence_baseline={
                "by_league": {
                    "LPL": {
                        "quality_band": "strong",
                        "lv_n": 3200,
                        "wf_n": 1200,
                    }
                }
            },
        )
        self.assertEqual(int(out.get("bo")), 3)
        self.assertGreater(float(out.get("max_dev_pp")), 16.0)
        self.assertLessEqual(float(out.get("max_dev_pp")), 28.0)


if __name__ == "__main__":
    unittest.main()

