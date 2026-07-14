"""Tests for app_core/fit_quality_report_core.py — pure dict-in/dict-out functions."""
import math
import unittest

from app_core.fit_quality_report_core import (
    _f,
    _gate_info,
    _summarise_form,
    _summarise_fused_calibration,
    _summarise_microseg,
    _summarise_overall_fit,
    _summarise_players_context,
    _summarise_series_gamma,
    summarise_fit_quality,
)


class TestF(unittest.TestCase):

    def test_valid_float(self):
        self.assertAlmostEqual(_f(3.14), 3.14)

    def test_int_converted(self):
        self.assertEqual(_f(5), 5.0)

    def test_string_float(self):
        self.assertAlmostEqual(_f("2.5"), 2.5)

    def test_none_returns_nan(self):
        self.assertTrue(math.isnan(_f(None)))

    def test_none_with_default(self):
        self.assertEqual(_f(None, 0.0), 0.0)

    def test_inf_returns_default(self):
        self.assertEqual(_f(float("inf"), -1.0), -1.0)

    def test_nan_returns_default(self):
        self.assertEqual(_f(float("nan"), 99.0), 99.0)

    def test_invalid_string_returns_default(self):
        self.assertTrue(math.isnan(_f("abc")))

    def test_returns_float_type(self):
        self.assertIsInstance(_f(1), float)


class TestGateInfo(unittest.TestCase):

    def test_extracts_decision(self):
        cfg = {"gate": {"decision": "promote", "reasons": []}}
        decision, _ = _gate_info(cfg)
        self.assertEqual(decision, "promote")

    def test_extracts_reasons(self):
        cfg = {"gate": {"decision": "shadow_only", "reasons": ["regression_detected"]}}
        _, reasons = _gate_info(cfg)
        self.assertIn("regression_detected", reasons)

    def test_uppercase_decision_normalized(self):
        cfg = {"gate": {"decision": "PROMOTE", "reasons": []}}
        decision, _ = _gate_info(cfg)
        self.assertEqual(decision, "promote")

    def test_none_input_returns_unknown(self):
        decision, reasons = _gate_info(None)
        self.assertEqual(decision, "unknown")
        self.assertEqual(reasons, [])

    def test_empty_dict_returns_unknown(self):
        decision, reasons = _gate_info({})
        self.assertEqual(decision, "unknown")
        self.assertEqual(reasons, [])

    def test_missing_gate_returns_unknown(self):
        decision, _ = _gate_info({"other": "stuff"})
        self.assertEqual(decision, "unknown")


class TestSummariseOverallFit(unittest.TestCase):

    def _make_cfg(self):
        return {
            "baseline": {"logloss": 0.693, "brier": 0.25, "n_total": 1000},
            "best": {"logloss": 0.650, "brier": 0.22, "n_total": 1000, "team_scale": 1.2, "players_scale": 0.8},
        }

    def test_gain_logloss_computed(self):
        result = _summarise_overall_fit(self._make_cfg())
        self.assertAlmostEqual(result["gain_logloss"], 0.043, places=3)

    def test_gain_brier_computed(self):
        result = _summarise_overall_fit(self._make_cfg())
        self.assertAlmostEqual(result["gain_brier"], 0.03, places=3)

    def test_n_total_from_best(self):
        result = _summarise_overall_fit(self._make_cfg())
        self.assertEqual(result["n_total"], 1000)

    def test_team_scale(self):
        result = _summarise_overall_fit(self._make_cfg())
        self.assertAlmostEqual(result["team_scale"], 1.2)

    def test_empty_config_returns_nones(self):
        result = _summarise_overall_fit({})
        self.assertIsNone(result["gain_logloss"])
        self.assertIsNone(result["baseline_logloss"])

    def test_returns_dict(self):
        self.assertIsInstance(_summarise_overall_fit({}), dict)


class TestSummariseForm(unittest.TestCase):

    def test_enabled_true(self):
        cfg = {"best_form": {"enabled": True, "improvement_logloss_vs_no_form": 0.005, "n_total": 800, "settings": {}}}
        result = _summarise_form(cfg)
        self.assertTrue(result["enabled"])

    def test_gain_extracted(self):
        cfg = {"best_form": {"enabled": True, "improvement_logloss_vs_no_form": 0.012, "n_total": 500, "settings": {}}}
        result = _summarise_form(cfg)
        self.assertAlmostEqual(result["gain_logloss_vs_no_form"], 0.012)

    def test_empty_config_disabled(self):
        result = _summarise_form({})
        self.assertFalse(result["enabled"])

    def test_returns_dict(self):
        self.assertIsInstance(_summarise_form({}), dict)


class TestSummarisePlayersContext(unittest.TestCase):

    def test_gate_decision_extracted(self):
        cfg = {"players_context_fit": {"gate": {"decision": "promote", "reasons": []}, "settings": {}, "meta": {}}}
        result = _summarise_players_context(cfg)
        self.assertEqual(result["gate_decision"], "promote")

    def test_empty_config_returns_unknown(self):
        result = _summarise_players_context({})
        self.assertEqual(result["gate_decision"], "unknown")

    def test_returns_dict(self):
        self.assertIsInstance(_summarise_players_context({}), dict)

    def test_required_keys_present(self):
        result = _summarise_players_context({})
        for key in ("gate_decision", "gate_reasons", "n_total", "settings"):
            self.assertIn(key, result)


class TestSummariseFusedCalibration(unittest.TestCase):

    def test_gate_decision_extracted(self):
        cfg = {"fused_post_calibration": {"gate": {"decision": "shadow_only", "reasons": ["unstable"]}}}
        result = _summarise_fused_calibration(cfg)
        self.assertEqual(result["gate_decision"], "shadow_only")

    def test_gate_reasons_extracted(self):
        cfg = {"fused_post_calibration": {"gate": {"decision": "shadow_only", "reasons": ["unstable"]}}}
        result = _summarise_fused_calibration(cfg)
        self.assertIn("unstable", result["gate_reasons"])

    def test_empty_config(self):
        result = _summarise_fused_calibration({})
        self.assertEqual(result["gate_decision"], "unknown")

    def test_returns_dict(self):
        self.assertIsInstance(_summarise_fused_calibration({}), dict)


class TestSummariseMicroseg(unittest.TestCase):

    def test_n_segments_counted(self):
        cfg = {"microseg_calibration": {"table": {"A|B|C|D|E": {}, "*|*|*|*|*": {}}, "pred_rows_n": 500}}
        result = _summarise_microseg(cfg)
        self.assertEqual(result["n_segments"], 2)

    def test_has_global_fallback_detected(self):
        cfg = {"microseg_calibration": {"table": {"*|*|*|*|*": {}}, "pred_rows_n": 100}}
        result = _summarise_microseg(cfg)
        self.assertTrue(result["has_global_fallback"])

    def test_no_global_fallback(self):
        cfg = {"microseg_calibration": {"table": {"A|B|C|D|E": {}}, "pred_rows_n": 100}}
        result = _summarise_microseg(cfg)
        self.assertFalse(result["has_global_fallback"])

    def test_empty_config(self):
        result = _summarise_microseg({})
        self.assertEqual(result["n_segments"], 0)
        self.assertFalse(result["has_global_fallback"])

    def test_returns_dict(self):
        self.assertIsInstance(_summarise_microseg({}), dict)


class TestSummariseSeriesGamma(unittest.TestCase):

    def test_leagues_counted(self):
        cfg = {"series_gamma_by_league": {"LCK": {}, "LPL": {}}, "series_gamma_by_macro": {}, "series_gamma_global": {"gamma": 1.2}}
        result = _summarise_series_gamma(cfg)
        self.assertEqual(result["n_leagues_tuned"], 2)

    def test_global_gamma_extracted(self):
        cfg = {"series_gamma_by_league": {}, "series_gamma_by_macro": {}, "series_gamma_global": {"gamma": 1.5}}
        result = _summarise_series_gamma(cfg)
        self.assertAlmostEqual(result["global_gamma"], 1.5)

    def test_empty_config(self):
        result = _summarise_series_gamma({})
        self.assertEqual(result["n_leagues_tuned"], 0)

    def test_returns_dict(self):
        self.assertIsInstance(_summarise_series_gamma({}), dict)


class TestSummariseFitQuality(unittest.TestCase):

    def test_invalid_input_returns_zero_score(self):
        result = summarise_fit_quality(None)
        self.assertEqual(result["quality_score"], 0.0)
        self.assertIn("invalid_fit_cfg", result["quality_flags"])

    def test_returns_dict_with_required_keys(self):
        result = summarise_fit_quality({})
        for key in ("quality_score", "quality_flags", "generated_at", "n_samples", "components", "overall"):
            self.assertIn(key, result)

    def test_score_range_0_to_100(self):
        result = summarise_fit_quality({})
        self.assertGreaterEqual(result["quality_score"], 0.0)
        self.assertLessEqual(result["quality_score"], 100.0)

    def test_good_config_high_score(self):
        cfg = {
            "baseline": {"logloss": 0.693, "brier": 0.25, "n_total": 1000},
            "best": {"logloss": 0.650, "brier": 0.22, "n_total": 1000, "team_scale": 1.0, "players_scale": 1.0},
            "players_context_fit": {"gate": {"decision": "promote", "reasons": []}, "settings": {}, "meta": {}, "best_on": {"n_total": 1000}},
            "fused_post_calibration": {"gate": {"decision": "promote", "reasons": []}, "meta": {"n_total": 500}},
            "microseg_calibration": {"table": {"*|*|*|*|*": {}, "LCK|A|B|C|D": {}}, "pred_rows_n": 800, "shrink_k": 5.0},
            "series_gamma_global": {"gamma": 1.3},
            "series_gamma_by_league": {"LCK": {}, "LPL": {}},
            "series_gamma_by_macro": {},
        }
        result = summarise_fit_quality(cfg)
        self.assertGreater(result["quality_score"], 70.0)

    def test_low_n_total_deducts_score(self):
        cfg = {
            "best": {"logloss": 0.650, "n_total": 100},
            "baseline": {"logloss": 0.693},
        }
        result = summarise_fit_quality(cfg)
        self.assertTrue(any("low_n_total" in f for f in result["quality_flags"]))

    def test_components_present(self):
        result = summarise_fit_quality({})
        self.assertIn("form", result["components"])
        self.assertIn("players_context", result["components"])
        self.assertIn("microseg", result["components"])

    def test_generated_at_extracted(self):
        result = summarise_fit_quality({"generated_at": "2026-01-01T00:00:00"})
        self.assertEqual(result["generated_at"], "2026-01-01T00:00:00")


if __name__ == "__main__":
    unittest.main()
