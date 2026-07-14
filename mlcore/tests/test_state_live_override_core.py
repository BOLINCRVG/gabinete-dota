import unittest

from app_core.state_live_override_core import apply_shadow_state_live_override


class TestStateLiveOverrideCore(unittest.TestCase):
    def test_applies_shadow_state_and_builds_pipeline_step(self):
        out = apply_shadow_state_live_override(
            p_map_used=0.60,
            p_series_used=0.65,
            bo=3,
            enabled=True,
            shadow_trace={"applied": True, "p_map_new": 0.58},
            prob_win_series_fn=lambda p, _bo: float(p) + 0.05,
            stage_name="state_alite_live_override",
            source_name="shadow_state_alite",
            promoted_reason="promoted_from_shadow_state_alite",
            not_applied_reason="shadow_state_alite_not_applied",
            session_state={},
            include_qwl_trace=False,
        )
        trace = out.get("trace") if isinstance(out.get("trace"), dict) else {}
        step = out.get("pipeline_step") if isinstance(out.get("pipeline_step"), dict) else {}
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.58, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.63, places=6)
        self.assertTrue(bool(trace.get("applied", False)))
        self.assertEqual(str(step.get("stage", "")), "state_alite_live_override")
        self.assertEqual(str(step.get("source", "")), "shadow_state_alite")

    def test_preserves_qwl_trace_for_areal(self):
        out = apply_shadow_state_live_override(
            p_map_used=0.60,
            p_series_used=0.65,
            bo=3,
            enabled=True,
            shadow_trace={
                "applied": True,
                "p_map_new": 0.57,
                "p_series_new": 0.62,
                "qwl_trace": {"quality": 0.8},
            },
            prob_win_series_fn=lambda p, _bo: float(p),
            stage_name="state_areal_live_override",
            source_name="shadow_state_areal",
            promoted_reason="promoted_from_shadow_state_areal",
            not_applied_reason="shadow_state_areal_not_applied",
            session_state={},
            include_qwl_trace=True,
        )
        trace = out.get("trace") if isinstance(out.get("trace"), dict) else {}
        qwl = trace.get("qwl_trace") if isinstance(trace.get("qwl_trace"), dict) else {}
        self.assertTrue(bool(trace.get("applied", False)))
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.62, places=6)
        self.assertAlmostEqual(float(qwl.get("quality", 0.0)), 0.8, places=6)

    def test_applies_confidence_governor_when_provided(self):
        def _gov_fn(**_kwargs):
            return {
                "enabled": True,
                "limited": True,
                "requested_shrink_pp": 8.0,
                "applied_shrink_pp": 3.0,
                "used_shrink_pp": 3.0,
                "remaining_shrink_pp": 5.0,
                "p_out": 0.57,
            }

        out = apply_shadow_state_live_override(
            p_map_used=0.60,
            p_series_used=0.65,
            bo=3,
            enabled=True,
            shadow_trace={"applied": True, "p_map_new": 0.52, "p_series_new": 0.59},
            prob_win_series_fn=lambda p, _bo: float(p) + 0.01,
            stage_name="state_areal_live_override",
            source_name="shadow_state_areal",
            promoted_reason="promoted_from_shadow_state_areal",
            not_applied_reason="shadow_state_areal_not_applied",
            session_state={},
            include_qwl_trace=False,
            apply_confidence_governor_fn=_gov_fn,
            governor_enabled=True,
            governor_budget_pp=8.0,
        )
        trace = out.get("trace") if isinstance(out.get("trace"), dict) else {}
        gov = trace.get("governor") if isinstance(trace.get("governor"), dict) else {}
        step = out.get("pipeline_step") if isinstance(out.get("pipeline_step"), dict) else {}
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.57, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.58, places=6)
        self.assertTrue(bool(gov.get("limited", False)))
        self.assertTrue(bool(step.get("governor_limited", False)))
        self.assertAlmostEqual(float(step.get("governor_applied_shrink_pp")), 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
