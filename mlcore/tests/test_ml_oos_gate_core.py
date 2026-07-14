"""Tests for app_core/ml_oos_gate_core.py — OOS gate and WF correction helpers."""
import math
import unittest

from app_core.ml_oos_gate_core import (
    evaluate_confidence_baseline_oos_gate,
    wf_correct_p_map,
    wf_gate_multiplier,
)


# ---------------------------------------------------------------------------
# evaluate_confidence_baseline_oos_gate
# ---------------------------------------------------------------------------

class TestEvaluateConfidenceBaselineOosGate(unittest.TestCase):

    def _cand(self, **oos):
        return {"oos_eval": oos}

    def _prev(self, **oos):
        return {"oos_eval": oos}

    def test_disabled_zero_gate_enabled_stays_on(self):
        # oos_gate_enabled=0 does NOT disable due to `or 1` guard in implementation;
        # gate stays enabled and passes with no previous data.
        cand = {"thresholds": {"oos_gate_enabled": 0}, "oos_eval": {}}
        out = evaluate_confidence_baseline_oos_gate(candidate=cand)
        self.assertTrue(out["enabled"])
        self.assertTrue(out["pass"])

    def test_no_previous_passes(self):
        out = evaluate_confidence_baseline_oos_gate(candidate=self._cand(wf_logloss_mean=0.65))
        self.assertTrue(out["pass"])
        self.assertEqual(out["reason"], "no_previous_oos_eval")

    def test_same_metrics_passes(self):
        cand = self._cand(wf_logloss_mean=0.65, wf_brier_mean=0.24)
        prev = self._prev(wf_logloss_mean=0.65, wf_brier_mean=0.24)
        out = evaluate_confidence_baseline_oos_gate(candidate=cand, previous=prev)
        self.assertTrue(out["pass"])

    def test_slightly_worse_within_tol_passes(self):
        # default logloss tol = 0.0015; diff = 0.001 → pass
        cand = self._cand(wf_logloss_mean=0.651, wf_brier_mean=0.24)
        prev = self._prev(wf_logloss_mean=0.650, wf_brier_mean=0.24)
        out = evaluate_confidence_baseline_oos_gate(candidate=cand, previous=prev)
        # diff=0.001 <= tol=0.0015 → pass
        self.assertTrue(out["pass"])

    def test_significantly_worse_fails(self):
        # diff = 0.02 >> tol = 0.0015 → fail
        cand = self._cand(wf_logloss_mean=0.67, wf_brier_mean=0.24)
        prev = self._prev(wf_logloss_mean=0.65, wf_brier_mean=0.24)
        out = evaluate_confidence_baseline_oos_gate(candidate=cand, previous=prev)
        self.assertFalse(out["pass"])
        self.assertEqual(out["reason"], "degradation_detected")

    def test_output_has_required_keys(self):
        out = evaluate_confidence_baseline_oos_gate(candidate={})
        for key in ("enabled", "pass", "reason", "checks"):
            self.assertIn(key, out)

    def test_missing_metrics_skipped_not_failed(self):
        # candidate has metrics but previous doesn't → check is skipped → no active checks → pass
        cand = self._cand(wf_logloss_mean=0.65)
        prev = self._prev()  # no metrics
        out = evaluate_confidence_baseline_oos_gate(candidate=cand, previous=prev)
        # All checks skipped → reason="no_active_checks" or passes
        self.assertTrue(out["pass"])

    def test_checks_list_populated(self):
        cand = self._cand(wf_logloss_mean=0.65, wf_brier_mean=0.24)
        prev = self._prev(wf_logloss_mean=0.65, wf_brier_mean=0.24)
        out = evaluate_confidence_baseline_oos_gate(candidate=cand, previous=prev)
        self.assertIsInstance(out["checks"], list)
        self.assertGreater(len(out["checks"]), 0)

    def test_non_dict_candidate_defaults(self):
        out = evaluate_confidence_baseline_oos_gate(candidate=None)
        self.assertIn("pass", out)

    def test_gain_check_lower_is_worse_fails(self):
        # lv_gain_vs_team_logloss: lower_is_better=False, so positive diff=good, large negative=bad
        # diff = cand - prev = -0.05 → fail (below -tol)
        cand = self._cand(lv_gain_vs_team_logloss=0.00)
        prev = self._prev(lv_gain_vs_team_logloss=0.05)
        out = evaluate_confidence_baseline_oos_gate(candidate=cand, previous=prev)
        self.assertFalse(out["pass"])


# ---------------------------------------------------------------------------
# wf_correct_p_map
# ---------------------------------------------------------------------------

class TestWfCorrectPMap(unittest.TestCase):

    def _report(self, roi=None, league=None, league_n=None, league_drift=None, league_roi=None):
        overall = {}
        if roi is not None:
            overall["roi_proxy_mean"] = roi
        by_league = {}
        if league is not None:
            row = {"n_total": league_n or 0}
            if league_drift is not None:
                row["drift_mean"] = league_drift
            if league_roi is not None:
                row["roi_proxy_mean"] = league_roi
            by_league[league] = row
        return {"overall": overall, "by_league": by_league}

    def test_none_report_returns_p_unchanged(self):
        p_out, info = wf_correct_p_map(None, 0.65)
        self.assertAlmostEqual(p_out, 0.65, places=5)
        self.assertFalse(info["applied"])

    def test_non_dict_report_returns_p_unchanged(self):
        p_out, _info = wf_correct_p_map("invalid", 0.65)
        self.assertAlmostEqual(p_out, 0.65, places=5)

    def test_p_half_stays_half(self):
        report = self._report(roi=0.0)
        p_out, _ = wf_correct_p_map(report, 0.5)
        self.assertAlmostEqual(p_out, 0.5, places=5)

    def test_negative_roi_shrinks_toward_half(self):
        report = self._report(roi=-0.01)
        p_out, _info = wf_correct_p_map(report, 0.70)
        self.assertLess(p_out, 0.70)
        self.assertGreater(p_out, 0.50)

    def test_high_positive_roi_expands_from_half(self):
        report = self._report(roi=0.05)
        p_out, _info = wf_correct_p_map(report, 0.70)
        self.assertGreater(p_out, 0.70)

    def test_info_has_applied_and_shrink(self):
        report = self._report(roi=-0.01)
        _, info = wf_correct_p_map(report, 0.70)
        self.assertTrue(info["applied"])
        self.assertIn("shrink", info)
        self.assertIn("p_in", info)
        self.assertIn("p_out", info)

    def test_result_clamped_to_unit_interval(self):
        report = self._report(roi=0.10)
        p_out, _ = wf_correct_p_map(report, 0.99)
        self.assertLessEqual(p_out, 0.99)
        self.assertGreaterEqual(p_out, 0.01)

    def test_league_high_drift_further_shrinks(self):
        report_nodrift = self._report(roi=0.0)
        report_drift = self._report(roi=0.0, league="LCK", league_n=50, league_drift=0.03)
        p_nodrift, _ = wf_correct_p_map(report_nodrift, 0.70)
        p_drift, _ = wf_correct_p_map(report_drift, 0.70, league_hint="LCK")
        # high drift → shrink → p closer to 0.5
        self.assertLess(p_drift, p_nodrift)

    def test_invalid_p_returned_unchanged(self):
        report = self._report(roi=0.0)
        p_out, info = wf_correct_p_map(report, float("nan"))
        self.assertTrue(math.isnan(p_out) or p_out == float("nan"))
        self.assertFalse(info.get("applied", True))


# ---------------------------------------------------------------------------
# wf_gate_multiplier
# ---------------------------------------------------------------------------

class TestWfGateMultiplier(unittest.TestCase):

    def _report(self, roi=None, league=None, league_n=None, league_drift=None):
        overall = {}
        if roi is not None:
            overall["roi_proxy_mean"] = roi
        by_league = {}
        if league is not None:
            row = {"n_total": league_n or 0}
            if league_drift is not None:
                row["drift_mean"] = league_drift
            by_league[league] = row
        return {"overall": overall, "by_league": by_league}

    def test_none_report_returns_one(self):
        self.assertAlmostEqual(wf_gate_multiplier(None), 1.0)

    def test_non_dict_returns_one(self):
        self.assertAlmostEqual(wf_gate_multiplier("invalid"), 1.0)

    def test_negative_roi_increases_multiplier(self):
        report = self._report(roi=-0.01)
        mult = wf_gate_multiplier(report)
        self.assertGreater(mult, 1.0)

    def test_low_positive_roi_also_increases_multiplier(self):
        # 0 < roi < 0.01 → 1.15
        report = self._report(roi=0.005)
        mult = wf_gate_multiplier(report)
        self.assertGreater(mult, 1.0)

    def test_high_positive_roi_decreases_multiplier(self):
        report = self._report(roi=0.05)
        mult = wf_gate_multiplier(report)
        self.assertLess(mult, 1.0)

    def test_result_clamped_to_range(self):
        for roi in (-1.0, 0.0, 0.05, 1.0):
            mult = wf_gate_multiplier(self._report(roi=roi))
            self.assertGreaterEqual(mult, 0.70)
            self.assertLessEqual(mult, 1.80)

    def test_league_high_drift_with_enough_games(self):
        # n_total >= 80 → league drift applies
        report = self._report(roi=0.0, league="LCK", league_n=100, league_drift=0.03)
        mult = wf_gate_multiplier(report, league_hint="LCK")
        # high drift → widens gate (mult > 1)
        self.assertGreater(mult, 1.0)

    def test_league_negative_drift_shrinks_gate(self):
        # Negative drift reduces the multiplier relative to no-drift baseline
        # (roi=0.0 base factor is 1.30; drift=-0.03 applies *0.95 → 1.235 < 1.30)
        report_nodrift = self._report(roi=0.0)
        report_drift = self._report(roi=0.0, league="LCK", league_n=100, league_drift=-0.03)
        m_base = wf_gate_multiplier(report_nodrift)
        m_drift = wf_gate_multiplier(report_drift, league_hint="LCK")
        self.assertLess(m_drift, m_base)

    def test_league_too_few_games_ignored(self):
        # n_total < 80 → league drift not applied
        without = self._report(roi=0.0)
        with_small = self._report(roi=0.0, league="LCK", league_n=30, league_drift=0.05)
        m1 = wf_gate_multiplier(without)
        m2 = wf_gate_multiplier(with_small, league_hint="LCK")
        self.assertAlmostEqual(m1, m2, places=6)


if __name__ == "__main__":
    unittest.main()
