"""Tests for app_core/ml_confidence_baseline_core.py."""
import json
import tempfile
import unittest
from pathlib import Path

from app_core.ml_confidence_baseline_core import (
    build_ml_confidence_baseline,
    write_ml_confidence_baseline,
)


def _lv_report(**kwargs):
    base = {
        "models": {
            "fused_active": {
                "n_total": 1500,
                "logloss": 0.63,
                "gain_vs_team_logloss": 0.015,
            },
            "team_only": {"logloss": 0.645},
        },
        "by_league": {},
    }
    base.update(kwargs)
    return base


def _wf_report(**kwargs):
    base = {
        "overall": {
            "n_total": 400,
            "months": 6,
            "logloss_mean": 0.65,
            "drift_mean": 0.005,
            "roi_proxy_mean": 0.02,
        },
        "by_league": {},
        "folds": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Error / invalid inputs
# ---------------------------------------------------------------------------

class TestBuildBaselineErrors(unittest.TestCase):

    def test_none_lv_report_returns_error(self):
        result = build_ml_confidence_baseline(None, None)
        self.assertIn("error", result)

    def test_non_dict_lv_report_returns_error(self):
        result = build_ml_confidence_baseline("not a dict", None)
        self.assertIn("error", result)

    def test_lv_report_with_error_key_returns_error(self):
        result = build_ml_confidence_baseline({"error": "something bad"}, None)
        self.assertIn("error", result)

    def test_none_wf_report_adds_warning(self):
        result = build_ml_confidence_baseline(_lv_report(), None)
        self.assertIn("warnings", result)
        warnings = result["warnings"]
        self.assertTrue(any("wf_report" in w for w in warnings))

    def test_wf_report_with_error_adds_warning(self):
        result = build_ml_confidence_baseline(_lv_report(), {"error": "bad"})
        self.assertIn("warnings", result)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestBuildBaselineStructure(unittest.TestCase):

    def test_required_keys_present(self):
        result = build_ml_confidence_baseline(_lv_report(), _wf_report())
        for key in ("version", "global", "by_league", "thresholds", "oos_eval"):
            self.assertIn(key, result, f"missing key: {key}")

    def test_version_string(self):
        result = build_ml_confidence_baseline(_lv_report(), _wf_report())
        self.assertIn("ml_confidence_baseline_v", result["version"])

    def test_global_has_score_bias(self):
        result = build_ml_confidence_baseline(_lv_report(), _wf_report())
        self.assertIn("score_bias", result["global"])
        self.assertIsInstance(result["global"]["score_bias"], float)

    def test_global_lv_n(self):
        result = build_ml_confidence_baseline(_lv_report(), _wf_report())
        self.assertEqual(result["global"]["lv_n"], 1500)

    def test_global_wf_n(self):
        result = build_ml_confidence_baseline(_lv_report(), _wf_report())
        self.assertEqual(result["global"]["wf_n"], 400)

    def test_thresholds_has_oos_gate_enabled(self):
        result = build_ml_confidence_baseline(_lv_report(), _wf_report())
        self.assertIn("oos_gate_enabled", result["thresholds"])

    def test_source_includes_walkforward(self):
        result = build_ml_confidence_baseline(_lv_report(), _wf_report())
        self.assertIn("walkforward", result["source"])

    def test_source_fallback_without_wf(self):
        result = build_ml_confidence_baseline(_lv_report(), None)
        self.assertIn("fallback", result["source"])


# ---------------------------------------------------------------------------
# by_league population
# ---------------------------------------------------------------------------

class TestBuildBaselineByLeague(unittest.TestCase):

    def _lv_with_league(self):
        return _lv_report(
            by_league={
                "LCK": {
                    "fused_active": {
                        "n_total": 1200,
                        "logloss": 0.62,
                        "gain_vs_team_logloss": 0.025,
                    }
                }
            }
        )

    def _wf_with_league(self):
        return _wf_report(
            by_league={
                "LCK": {
                    "n_total": 300,
                    "months": 5,
                    "logloss_mean": 0.64,
                    "drift_mean": 0.004,
                    "roi_proxy_mean": 0.03,
                }
            }
        )

    def test_by_league_populated(self):
        result = build_ml_confidence_baseline(self._lv_with_league(), self._wf_with_league())
        self.assertIn("by_league", result)
        self.assertTrue(len(result["by_league"]) > 0)

    def test_league_node_has_quality_band(self):
        result = build_ml_confidence_baseline(self._lv_with_league(), self._wf_with_league())
        for node in result["by_league"].values():
            self.assertIn("quality_band", node)

    def test_league_node_quality_band_valid(self):
        result = build_ml_confidence_baseline(self._lv_with_league(), self._wf_with_league())
        valid_bands = {"strong", "good", "ok", "low_sample"}
        for node in result["by_league"].values():
            self.assertIn(node["quality_band"], valid_bands)

    def test_empty_by_league_gives_empty_output(self):
        result = build_ml_confidence_baseline(_lv_report(), _wf_report())
        self.assertIsInstance(result["by_league"], dict)


# ---------------------------------------------------------------------------
# score_bias sign direction
# ---------------------------------------------------------------------------

class TestScoreBiasDirection(unittest.TestCase):

    def test_strong_model_gives_positive_bias(self):
        lv = _lv_report(
            models={
                "fused_active": {
                    "n_total": 2000,
                    "logloss": 0.60,
                    "gain_vs_team_logloss": 0.025,
                },
                "team_only": {"logloss": 0.625},
            }
        )
        wf = _wf_report(
            overall={
                "n_total": 500,
                "months": 8,
                "logloss_mean": 0.63,
                "drift_mean": 0.003,
                "roi_proxy_mean": 0.05,
            }
        )
        result = build_ml_confidence_baseline(lv, wf)
        self.assertGreater(result["global"]["score_bias"], 0.0)

    def test_weak_model_gives_negative_bias(self):
        lv = _lv_report(
            models={
                "fused_active": {
                    "n_total": 200,
                    "logloss": 0.74,
                    "gain_vs_team_logloss": -0.01,
                },
                "team_only": {"logloss": 0.73},
            }
        )
        wf = _wf_report(
            overall={
                "n_total": 90,
                "months": 2,
                "logloss_mean": 0.75,
                "drift_mean": 0.04,
                "roi_proxy_mean": -0.03,
            }
        )
        result = build_ml_confidence_baseline(lv, wf)
        self.assertLess(result["global"]["score_bias"], 0.0)


# ---------------------------------------------------------------------------
# write_ml_confidence_baseline
# ---------------------------------------------------------------------------

class TestWriteBaseline(unittest.TestCase):

    def test_writes_valid_json(self):
        payload = {"version": "test", "data": 42}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"
            write_ml_confidence_baseline(str(path), payload)
            self.assertTrue(path.exists())
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["version"], "test")
            self.assertEqual(loaded["data"], 42)

    def test_roundtrip_baseline(self):
        result = build_ml_confidence_baseline(_lv_report(), _wf_report())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.json"
            write_ml_confidence_baseline(str(path), result)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("version", loaded)


if __name__ == "__main__":
    unittest.main()
