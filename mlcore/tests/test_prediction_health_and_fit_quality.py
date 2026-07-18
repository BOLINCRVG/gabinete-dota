"""Tests for prediction_health_core.py and fit_quality_report_core.py."""

import unittest
from typing import Any, Dict

from app_core.prediction_health_core import (
    check_prediction_health,
    score_prediction_confidence,
)
from app_core.fit_quality_report_core import summarise_fit_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline_out(
    p_map=0.62,
    p_series=0.70,
    bo=3,
    steps=None,
    budget=None,
    coherence_reason="ok",
) -> Dict[str, Any]:
    return {
        "p_map_used": p_map,
        "p_series_used": p_series,
        "trace_pipeline_steps": steps or [],
        "trace_governor_budget_final": budget or {},
        "trace_series_coherence_post_info": {
            "bo": bo,
            "reason": coherence_reason,
        },
    }


def _budget(available=True, exhausted=False, n_limited=0,
            remaining=8.0, max_pp=8.0, used_pp=0.0):
    return {
        "available": available,
        "budget_exhausted": exhausted,
        "n_stages_limited": n_limited,
        "remaining_shrink_pp": remaining,
        "max_total_shrink_pp": max_pp,
        "used_shrink_pp": used_pp,
    }


def _perfect_fit_cfg() -> Dict[str, Any]:
    return {
        "generated_at": "2024-06-01T12:00:00Z",
        "samples_total": 800,
        "min_train_games": 30,
        "speed_profile": "full",
        "baseline": {"logloss": 0.693, "brier": 0.25, "n_total": 800},
        "best": {
            "logloss": 0.685,
            "brier": 0.24,
            "n_total": 800,
            "team_scale": 1.1,
            "players_scale": 0.9,
        },
        "best_form": {
            "enabled": True,
            "improvement_logloss_vs_no_form": 0.002,
            "n_total": 800,
            "settings": {"max_weight": 0.75},
        },
        "players_context_fit": {
            "gate": {"decision": "promote", "reasons": ["pass"]},
            "settings": {},
            "meta": {"cache_entries": 10, "candidates_tested": 5},
            "best_on": {"n_total": 800},
        },
        "fused_post_calibration": {
            "gate": {"decision": "shadow_only", "reasons": ["not_evaluated"]},
            "meta": {"n_total": 0},
            "promoted_model_global": {},
        },
        "microseg_calibration": {
            "table": {
                "*|*|*|*|*": {"bias": 0.0, "n": 100},
                "LCK|Summer|blue|3|regular": {"bias": 0.01, "n": 50},
            },
            "pred_rows_n": 800,
            "shrink_k": 12.0,
            "source": "fitted",
        },
        "series_gamma_global": {"gamma": 1.2},
        "series_gamma_by_league": {"LCK": {"gamma": 1.3}},
        "series_gamma_by_macro": {"KR": {"gamma": 1.2}},
    }


# ===========================================================================
# Tests: check_prediction_health
# ===========================================================================

class TestCheckPredictionHealth(unittest.TestCase):

    def test_healthy_pipeline_returns_healthy_true(self):
        h = check_prediction_health(_pipeline_out())
        self.assertTrue(h["healthy"])
        self.assertEqual(h["n_errors"], 0)

    def test_required_keys_present(self):
        h = check_prediction_health(_pipeline_out())
        for key in ("healthy", "warnings", "errors", "n_warnings", "n_errors",
                    "p_map_used", "p_series_used", "bo"):
            self.assertIn(key, h)

    def test_invalid_input_returns_unhealthy(self):
        h = check_prediction_health(None)
        self.assertFalse(h["healthy"])
        self.assertIn("invalid_pipeline_output", h["errors"])

    def test_p_map_nan_flagged_as_error(self):
        h = check_prediction_health(_pipeline_out(p_map=float("nan")))
        self.assertFalse(h["healthy"])
        self.assertIn("p_map_not_finite", h["errors"])

    def test_p_map_zero_flagged_as_error(self):
        h = check_prediction_health(_pipeline_out(p_map=0.0))
        self.assertFalse(h["healthy"])
        self.assertIn("p_map_out_of_range", h["errors"])

    def test_p_map_one_flagged_as_error(self):
        h = check_prediction_health(_pipeline_out(p_map=1.0))
        self.assertFalse(h["healthy"])
        self.assertIn("p_map_out_of_range", h["errors"])

    def test_p_map_extreme_flagged_as_warning(self):
        # bo=1 skips series check so p_series mismatch doesn't create an error
        h = check_prediction_health(_pipeline_out(p_map=0.03, bo=1))
        self.assertTrue(h["healthy"])  # warning, not error
        self.assertIn("p_map_extreme", h["warnings"])

    def test_p_series_nan_for_bo3_flagged(self):
        h = check_prediction_health(_pipeline_out(p_series=float("nan"), bo=3))
        self.assertFalse(h["healthy"])
        self.assertIn("p_series_not_finite", h["errors"])

    def test_p_series_nan_for_bo1_not_flagged(self):
        h = check_prediction_health(_pipeline_out(p_series=float("nan"), bo=1))
        # BO1 has no series probability check
        self.assertTrue(h["healthy"])

    def test_p_series_flipped_vs_p_map_is_error(self):
        # p_map=0.62 (blue favourite) but p_series=0.45 (blue underdog)
        h = check_prediction_health(_pipeline_out(p_map=0.62, p_series=0.45, bo=3))
        self.assertFalse(h["healthy"])
        self.assertIn("p_series_flipped_vs_p_map", h["errors"])

    def test_p_series_weaker_than_map_for_favourite_is_warning(self):
        # Favourite has p_map=0.62 but p_series=0.60 (series should amplify)
        h = check_prediction_health(_pipeline_out(p_map=0.62, p_series=0.60, bo=3))
        self.assertTrue(h["healthy"])
        self.assertIn("p_series_weaker_than_p_map_for_favourite", h["warnings"])

    def test_governor_budget_exhausted_is_warning(self):
        h = check_prediction_health(_pipeline_out(
            budget=_budget(exhausted=True),
        ))
        self.assertTrue(h["healthy"])
        self.assertIn("governor_budget_exhausted", h["warnings"])

    def test_governor_stages_limited_is_warning(self):
        h = check_prediction_health(_pipeline_out(
            budget=_budget(n_limited=2),
        ))
        self.assertTrue(h["healthy"])
        self.assertIn("governor_limited_2_stage(s)", h["warnings"])

    def test_governor_less_than_10pct_remaining_is_warning(self):
        h = check_prediction_health(_pipeline_out(
            budget=_budget(remaining=0.5, max_pp=8.0),
        ))
        self.assertTrue(h["healthy"])
        self.assertIn("governor_less_than_10pct_remaining", h["warnings"])

    def test_series_coherence_error_prefix_flagged(self):
        h = check_prediction_health(_pipeline_out(coherence_reason="error:flip_detected"))
        self.assertFalse(h["healthy"])
        self.assertTrue(any("series_coherence_" in e for e in h["errors"]))

    def test_series_coherence_ok_not_flagged(self):
        h = check_prediction_health(_pipeline_out(coherence_reason="ok"))
        self.assertNotIn(True, ["series_coherence" in e for e in h["errors"]])

    def test_large_stage_movement_is_warning(self):
        # Use 0.71 - 0.62 = 9pp (avoids float precision issues at exactly 8pp)
        steps = [{"stage": "microseg", "before": 0.62, "after": 0.71}]
        h = check_prediction_health(_pipeline_out(steps=steps), large_move_pp=8.0)
        self.assertTrue(h["healthy"])
        self.assertTrue(any("large_move_microseg" in w for w in h["warnings"]))

    def test_small_stage_movement_not_flagged(self):
        steps = [{"stage": "microseg", "before": 0.62, "after": 0.64}]  # 2pp
        h = check_prediction_health(_pipeline_out(steps=steps), large_move_pp=8.0)
        self.assertFalse(any("large_move" in w for w in h["warnings"]))

    def test_p_map_used_extracted(self):
        h = check_prediction_health(_pipeline_out(p_map=0.62))
        self.assertAlmostEqual(h["p_map_used"], 0.62)

    def test_bo_extracted_from_coherence_info(self):
        h = check_prediction_health(_pipeline_out(bo=5))
        self.assertEqual(h["bo"], 5)

    def test_n_counts_match_lists(self):
        h = check_prediction_health(_pipeline_out())
        self.assertEqual(h["n_warnings"], len(h["warnings"]))
        self.assertEqual(h["n_errors"], len(h["errors"]))


# ===========================================================================
# Tests: score_prediction_confidence
# ===========================================================================

class TestScorePredictionConfidence(unittest.TestCase):

    def test_healthy_clean_pipeline_scores_near_1(self):
        score = score_prediction_confidence(_pipeline_out())
        self.assertGreaterEqual(score, 0.8)
        self.assertLessEqual(score, 1.0)

    def test_unhealthy_pipeline_scores_zero(self):
        out = _pipeline_out(p_map=float("nan"))
        score = score_prediction_confidence(out)
        self.assertEqual(score, 0.0)

    def test_none_input_scores_zero(self):
        self.assertEqual(score_prediction_confidence(None), 0.0)

    def test_score_decreases_with_more_governor_usage(self):
        low_usage = _pipeline_out(budget=_budget(used_pp=1.0, max_pp=8.0))
        high_usage = _pipeline_out(budget=_budget(used_pp=7.0, max_pp=8.0))
        self.assertGreater(
            score_prediction_confidence(low_usage),
            score_prediction_confidence(high_usage),
        )

    def test_score_decreases_with_warnings(self):
        clean = _pipeline_out(p_map=0.62, p_series=0.70)
        with_warning = _pipeline_out(
            p_map=0.62, p_series=0.60,  # p_series_weaker warning
        )
        self.assertGreater(
            score_prediction_confidence(clean),
            score_prediction_confidence(with_warning),
        )

    def test_score_clamped_0_to_1(self):
        # Many warnings should not push score below 0
        steps = [{"stage": f"s{i}", "before": 0.5, "after": 0.59} for i in range(20)]
        out = _pipeline_out(steps=steps, budget=_budget(used_pp=8.0, max_pp=8.0))
        score = score_prediction_confidence(out)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_returns_float(self):
        self.assertIsInstance(score_prediction_confidence(_pipeline_out()), float)


# ===========================================================================
# Tests: summarise_fit_quality
# ===========================================================================

class TestSummariseFitQuality(unittest.TestCase):

    def test_invalid_input_returns_zero_score(self):
        out = summarise_fit_quality(None)
        self.assertEqual(out["quality_score"], 0.0)
        self.assertIn("invalid_fit_cfg", out["quality_flags"])

    def test_required_keys_present(self):
        out = summarise_fit_quality(_perfect_fit_cfg())
        for key in ("quality_score", "quality_flags", "generated_at",
                    "n_samples", "components", "overall"):
            self.assertIn(key, out)

    def test_perfect_config_scores_100(self):
        out = summarise_fit_quality(_perfect_fit_cfg())
        self.assertEqual(out["quality_score"], 100.0)
        self.assertEqual(out["quality_flags"], [])

    def test_low_n_total_deducts_score(self):
        cfg = _perfect_fit_cfg()
        cfg["baseline"]["n_total"] = 100
        cfg["best"]["n_total"] = 100
        out = summarise_fit_quality(cfg)
        self.assertLess(out["quality_score"], 100.0)
        self.assertTrue(any("low_n_total" in f for f in out["quality_flags"]))

    def test_moderate_n_total_small_deduction(self):
        cfg = _perfect_fit_cfg()
        cfg["baseline"]["n_total"] = 400
        cfg["best"]["n_total"] = 400
        out = summarise_fit_quality(cfg)
        self.assertLess(out["quality_score"], 100.0)
        self.assertGreater(out["quality_score"], 90.0)
        self.assertTrue(any("moderate_n_total" in f for f in out["quality_flags"]))

    def test_players_context_shadow_only_deducts(self):
        cfg = _perfect_fit_cfg()
        cfg["players_context_fit"]["gate"] = {
            "decision": "shadow_only", "reasons": ["low_sample"]
        }
        out = summarise_fit_quality(cfg)
        self.assertLess(out["quality_score"], 100.0)
        self.assertTrue(any("players_context_shadow" in f for f in out["quality_flags"]))

    def test_players_context_shadow_regression_bigger_deduction(self):
        cfg_reg = _perfect_fit_cfg()
        cfg_reg["players_context_fit"]["gate"] = {
            "decision": "shadow_only", "reasons": ["oos_regression:-0.001"]
        }
        cfg_low = _perfect_fit_cfg()
        cfg_low["players_context_fit"]["gate"] = {
            "decision": "shadow_only", "reasons": ["low_sample"]
        }
        score_reg = summarise_fit_quality(cfg_reg)["quality_score"]
        score_low = summarise_fit_quality(cfg_low)["quality_score"]
        self.assertLess(score_reg, score_low)

    def test_microseg_no_segments_deducts(self):
        cfg = _perfect_fit_cfg()
        cfg["microseg_calibration"]["table"] = {}
        out = summarise_fit_quality(cfg)
        self.assertLess(out["quality_score"], 100.0)
        self.assertTrue(any("microseg_no_segments" in f for f in out["quality_flags"]))

    def test_microseg_no_global_fallback_small_deduction(self):
        cfg = _perfect_fit_cfg()
        # Remove the wildcard entry but keep specific segments
        cfg["microseg_calibration"]["table"] = {
            "LCK|Summer|blue|3|regular": {"bias": 0.01, "n": 50},
        }
        out = summarise_fit_quality(cfg)
        self.assertLess(out["quality_score"], 100.0)
        self.assertTrue(any("microseg_no_global_fallback" in f for f in out["quality_flags"]))

    def test_overall_gain_very_small_deducts(self):
        cfg = _perfect_fit_cfg()
        # Make gain tiny: baseline ≈ best
        cfg["baseline"]["logloss"] = 0.6930
        cfg["best"]["logloss"] = 0.6929  # gain = 0.0001 < 0.0002
        out = summarise_fit_quality(cfg)
        self.assertLess(out["quality_score"], 100.0)
        self.assertTrue(any("overall_gain_very_small" in f for f in out["quality_flags"]))

    def test_overall_gain_small_minor_deduction(self):
        cfg = _perfect_fit_cfg()
        cfg["baseline"]["logloss"] = 0.6930
        cfg["best"]["logloss"] = 0.6925  # gain = 0.0005, between 0.0002 and 0.001
        out = summarise_fit_quality(cfg)
        self.assertLess(out["quality_score"], 100.0)
        self.assertGreater(out["quality_score"], 90.0)

    def test_components_keys_present(self):
        out = summarise_fit_quality(_perfect_fit_cfg())
        for key in ("form", "players_context", "fused_calibration", "microseg", "series_gamma"):
            self.assertIn(key, out["components"])

    def test_series_gamma_summary(self):
        out = summarise_fit_quality(_perfect_fit_cfg())
        sg = out["components"]["series_gamma"]
        self.assertAlmostEqual(sg["global_gamma"], 1.2)
        self.assertEqual(sg["n_leagues_tuned"], 1)
        self.assertEqual(sg["n_macros_tuned"], 1)

    def test_n_samples_extracted(self):
        out = summarise_fit_quality(_perfect_fit_cfg())
        self.assertEqual(out["n_samples"], 800)

    def test_generated_at_extracted(self):
        out = summarise_fit_quality(_perfect_fit_cfg())
        self.assertEqual(out["generated_at"], "2024-06-01T12:00:00Z")

    def test_score_clamped_0_to_100(self):
        # Very bad config should not go below 0
        cfg = _perfect_fit_cfg()
        cfg["best"]["n_total"] = 1
        cfg["baseline"]["n_total"] = 1
        cfg["players_context_fit"]["gate"] = {
            "decision": "shadow_only", "reasons": ["oos_regression:-0.01"]
        }
        cfg["microseg_calibration"]["table"] = {}
        out = summarise_fit_quality(cfg)
        self.assertGreaterEqual(out["quality_score"], 0.0)
        self.assertLessEqual(out["quality_score"], 100.0)

    def test_fused_calibration_not_evaluated_no_penalty(self):
        # "not_evaluated" should not deduct points (feature disabled, not a failure)
        out = summarise_fit_quality(_perfect_fit_cfg())
        self.assertFalse(any("fused_cal" in f for f in out["quality_flags"]))


if __name__ == "__main__":
    unittest.main()
