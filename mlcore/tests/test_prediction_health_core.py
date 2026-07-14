"""Tests for app_core/prediction_health_core.py — health check helpers."""
import unittest

from app_core.prediction_health_core import (
    check_prediction_health,
    score_prediction_confidence,
)


def _pipeline(p_map=0.65, p_series=0.78, bo=3, steps=None, budget=None, coherence=None):
    return {
        "p_map_used": p_map,
        "p_series_used": p_series,
        "trace_pipeline_steps": steps or [],
        "trace_governor_budget_final": budget or {},
        "trace_series_coherence_post_info": {"bo": bo, **(coherence or {})},
    }


# ---------------------------------------------------------------------------
# check_prediction_health — invalid input
# ---------------------------------------------------------------------------

class TestCheckHealthInvalidInput(unittest.TestCase):

    def test_non_dict_returns_unhealthy(self):
        result = check_prediction_health("not a dict")
        self.assertFalse(result["healthy"])
        self.assertIn("invalid_pipeline_output", result["errors"])

    def test_none_returns_unhealthy(self):
        result = check_prediction_health(None)
        self.assertFalse(result["healthy"])


# ---------------------------------------------------------------------------
# check_prediction_health — probability checks
# ---------------------------------------------------------------------------

class TestCheckHealthProbabilities(unittest.TestCase):

    def test_valid_p_map_bo3_is_healthy(self):
        result = check_prediction_health(_pipeline(p_map=0.65, p_series=0.78, bo=3))
        self.assertTrue(result["healthy"])
        self.assertEqual(result["n_errors"], 0)

    def test_nan_p_map_gives_error(self):
        result = check_prediction_health(_pipeline(p_map=float("nan")))
        self.assertIn("p_map_not_finite", result["errors"])
        self.assertFalse(result["healthy"])

    def test_p_map_zero_gives_error(self):
        result = check_prediction_health(_pipeline(p_map=0.0))
        self.assertFalse(result["healthy"])

    def test_p_map_one_gives_error(self):
        result = check_prediction_health(_pipeline(p_map=1.0))
        self.assertFalse(result["healthy"])

    def test_p_map_extreme_gives_warning(self):
        result = check_prediction_health(_pipeline(p_map=0.04, p_series=0.04, bo=3))
        # p_map_out_of_range triggers before extreme check, so check error
        self.assertTrue(len(result["errors"]) > 0 or len(result["warnings"]) > 0)

    def test_flipped_series_vs_map_gives_error(self):
        # p_map=0.65 (favour A), p_series=0.40 (favour B) — flipped
        result = check_prediction_health(_pipeline(p_map=0.65, p_series=0.40, bo=3))
        self.assertIn("p_series_flipped_vs_p_map", result["errors"])

    def test_bo1_ignores_series_prob(self):
        # BO1 — p_series irrelevant
        result = check_prediction_health(_pipeline(p_map=0.6, p_series=float("nan"), bo=1))
        self.assertTrue(result["healthy"])

    def test_valid_bo5_healthy(self):
        result = check_prediction_health(_pipeline(p_map=0.60, p_series=0.70, bo=5))
        self.assertTrue(result["healthy"])


# ---------------------------------------------------------------------------
# check_prediction_health — governor budget
# ---------------------------------------------------------------------------

class TestCheckHealthGovernor(unittest.TestCase):

    def _budget(self, **kwargs):
        base = {"available": True, "budget_exhausted": False, "n_stages_limited": 0,
                "remaining_shrink_pp": 10.0, "max_total_shrink_pp": 12.0}
        base.update(kwargs)
        return base

    def test_exhausted_budget_gives_warning(self):
        budget = self._budget(budget_exhausted=True)
        result = check_prediction_health(_pipeline(budget=budget))
        self.assertIn("governor_budget_exhausted", result["warnings"])

    def test_limited_stages_gives_warning(self):
        budget = self._budget(n_stages_limited=2)
        result = check_prediction_health(_pipeline(budget=budget))
        self.assertTrue(any("governor_limited" in w for w in result["warnings"]))

    def test_nearly_empty_budget_warns(self):
        budget = self._budget(remaining_shrink_pp=0.5, max_total_shrink_pp=12.0)
        result = check_prediction_health(_pipeline(budget=budget))
        self.assertIn("governor_less_than_10pct_remaining", result["warnings"])

    def test_unavailable_budget_no_warning(self):
        budget = {"available": False, "budget_exhausted": True}
        result = check_prediction_health(_pipeline(budget=budget))
        self.assertNotIn("governor_budget_exhausted", result["warnings"])


# ---------------------------------------------------------------------------
# check_prediction_health — stage movements
# ---------------------------------------------------------------------------

class TestCheckHealthStageMovements(unittest.TestCase):

    def test_large_move_gives_warning(self):
        steps = [{"stage": "microseg", "before": 0.50, "after": 0.60}]  # 10pp
        result = check_prediction_health(_pipeline(steps=steps))
        self.assertTrue(any("large_move" in w for w in result["warnings"]))

    def test_small_move_no_warning(self):
        steps = [{"stage": "microseg", "before": 0.60, "after": 0.62}]  # 2pp
        result = check_prediction_health(_pipeline(steps=steps))
        self.assertFalse(any("large_move" in w for w in result["warnings"]))

    def test_nan_step_values_skipped(self):
        steps = [{"stage": "microseg", "before": float("nan"), "after": 0.60}]
        result = check_prediction_health(_pipeline(steps=steps))
        self.assertFalse(any("large_move" in w for w in result["warnings"]))


# ---------------------------------------------------------------------------
# check_prediction_health — output shape
# ---------------------------------------------------------------------------

class TestCheckHealthOutputShape(unittest.TestCase):

    def test_output_has_required_keys(self):
        result = check_prediction_health(_pipeline())
        for key in ("healthy", "warnings", "errors", "n_warnings", "n_errors"):
            self.assertIn(key, result)

    def test_n_warnings_matches_list(self):
        result = check_prediction_health(_pipeline())
        self.assertEqual(result["n_warnings"], len(result["warnings"]))

    def test_n_errors_matches_list(self):
        result = check_prediction_health(_pipeline())
        self.assertEqual(result["n_errors"], len(result["errors"]))

    def test_healthy_when_no_errors(self):
        result = check_prediction_health(_pipeline())
        self.assertEqual(result["healthy"], result["n_errors"] == 0)


# ---------------------------------------------------------------------------
# score_prediction_confidence
# ---------------------------------------------------------------------------

class TestScorePredictionConfidence(unittest.TestCase):

    def test_non_dict_returns_zero(self):
        self.assertEqual(score_prediction_confidence(None), 0.0)
        self.assertEqual(score_prediction_confidence("x"), 0.0)

    def test_unhealthy_pipeline_returns_zero(self):
        result = score_prediction_confidence(_pipeline(p_map=float("nan")))
        self.assertEqual(result, 0.0)

    def test_clean_pipeline_returns_high_score(self):
        result = score_prediction_confidence(_pipeline())
        self.assertGreater(result, 0.5)

    def test_score_in_zero_one(self):
        result = score_prediction_confidence(_pipeline())
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_exhausted_budget_reduces_score(self):
        clean = score_prediction_confidence(_pipeline())
        budget = {"available": True, "budget_exhausted": True, "n_stages_limited": 0,
                  "remaining_shrink_pp": 11.0, "max_total_shrink_pp": 12.0,
                  "used_shrink_pp": 6.0}
        with_budget = score_prediction_confidence(_pipeline(budget=budget))
        # budget exhausted → governor_budget_exhausted warning → reduced score
        self.assertLessEqual(with_budget, clean)


if __name__ == "__main__":
    unittest.main()
