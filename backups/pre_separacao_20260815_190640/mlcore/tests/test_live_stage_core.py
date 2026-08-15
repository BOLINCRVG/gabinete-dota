import math
import unittest

from app_core.live_stage_core import (
    apply_live_stage_result,
    build_live_stage_error_trace,
    build_live_stage_trace,
)
from mlcore.bo import prob_win_series


class LiveStageCoreTest(unittest.TestCase):
    def test_build_trace_defaults(self):
        tr = build_live_stage_trace(enabled=False, p_map_used=0.55, p_series_used=0.6)
        self.assertFalse(bool(tr.get("enabled")))
        self.assertFalse(bool(tr.get("applied")))
        self.assertEqual(str(tr.get("reason")), "disabled_by_flag")
        self.assertAlmostEqual(float(tr["p_map_before"]), 0.55, places=12)
        self.assertAlmostEqual(float(tr["p_series_before"]), 0.6, places=12)
        self.assertIsNone(tr.get("p_map_after"))
        self.assertIsNone(tr.get("p_series_after"))

    def test_error_trace_contains_error(self):
        err = ValueError("boom")
        tr = build_live_stage_error_trace(
            enabled=True,
            error=err,
            p_map_used=0.51,
            p_series_used=0.52,
        )
        self.assertTrue(str(tr.get("reason", "")).startswith("error:"))
        self.assertIn("boom", str(tr.get("error", "")))
        self.assertAlmostEqual(float(tr["p_map_before"]), 0.51, places=12)

    def test_apply_stage_updates_probs_and_pipeline(self):
        steps = []
        state = {}
        out = {"applied": True, "p_after": 0.62, "method": "x"}
        res = apply_live_stage_result(
            out=out,
            p_map_used=0.55,
            p_series_used=0.57,
            bo=5,
            stage_name="stage_x",
            trace_pipeline_steps=steps,
            prob_win_series_fn=prob_win_series,
            session_state=state,
        )
        p_map = float(res["p_map_used"])
        p_series = float(res["p_series_used"])
        self.assertAlmostEqual(p_map, 0.62, places=12)
        self.assertAlmostEqual(p_series, float(prob_win_series(0.62, 5)), places=12)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["stage"], "stage_x")
        self.assertAlmostEqual(float(state["_p_map_used"]), 0.62, places=12)

    def test_apply_stage_uses_series_from_output_when_available(self):
        steps = []
        out = {"applied": True, "p_after": 0.44, "p_series_after": 0.33, "method": "y"}
        res = apply_live_stage_result(
            out=out,
            p_map_used=0.50,
            p_series_used=0.51,
            bo=3,
            stage_name="stage_y",
            trace_pipeline_steps=steps,
            prob_win_series_fn=prob_win_series,
            out_series_key="p_series_after",
        )
        self.assertAlmostEqual(float(res["p_map_used"]), 0.44, places=12)
        self.assertAlmostEqual(float(res["p_series_used"]), 0.33, places=12)
        self.assertEqual(len(steps), 1)

    def test_apply_stage_noop_when_not_applied(self):
        steps = []
        res = apply_live_stage_result(
            out={"applied": False, "p_after": 0.7},
            p_map_used=0.41,
            p_series_used=0.39,
            bo=3,
            stage_name="stage_z",
            trace_pipeline_steps=steps,
            prob_win_series_fn=prob_win_series,
        )
        self.assertAlmostEqual(float(res["p_map_used"]), 0.41, places=12)
        self.assertAlmostEqual(float(res["p_series_used"]), 0.39, places=12)
        self.assertEqual(len(steps), 0)
        self.assertTrue(math.isnan(float(res["out"].get("p_map_after", float("nan")))))

    def test_apply_stage_honors_confidence_governor(self):
        steps = []
        state = {}

        def _gov_fn(**_kwargs):
            return {
                "enabled": True,
                "limited": True,
                "requested_shrink_pp": 10.0,
                "applied_shrink_pp": 3.0,
                "used_shrink_pp": 3.0,
                "remaining_shrink_pp": 5.0,
                "p_out": 0.67,
            }

        res = apply_live_stage_result(
            out={"applied": True, "p_after": 0.60, "method": "z"},
            p_map_used=0.70,
            p_series_used=0.78,
            bo=3,
            stage_name="stage_gov",
            trace_pipeline_steps=steps,
            prob_win_series_fn=prob_win_series,
            session_state=state,
            apply_confidence_governor_fn=_gov_fn,
            governor_enabled=True,
            governor_budget_pp=8.0,
        )
        self.assertAlmostEqual(float(res["p_map_used"]), 0.67, places=12)
        out = res["out"] if isinstance(res.get("out"), dict) else {}
        gov = out.get("governor") if isinstance(out.get("governor"), dict) else {}
        self.assertTrue(bool(gov.get("enabled", False)))
        self.assertTrue(bool(gov.get("limited", False)))
        self.assertAlmostEqual(float(gov.get("applied_shrink_pp")), 3.0, places=12)
        self.assertEqual(len(steps), 1)
        self.assertTrue(bool(steps[0].get("governor_limited", False)))
        self.assertAlmostEqual(float(steps[0].get("governor_applied_shrink_pp")), 3.0, places=12)


if __name__ == "__main__":
    unittest.main()
