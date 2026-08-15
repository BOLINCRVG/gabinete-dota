import unittest

from app_core.ml_output_models import MlTraceContext
from app_core.ml_trace_payload_core import build_ml_trace_payload


class TestMlTracePayloadCore(unittest.TestCase):
    def test_builds_minimal_payload(self):
        payload = build_ml_trace_payload(
            ctx={
                "season_label": "2026",
                "teamA": "A",
                "teamB": "B",
                "bo": 3,
                "ml_engine": "mlcore_v2",
                "p_map_used": 0.62,
                "p_series_used": 0.70,
                "trace_pipeline_order": [],
                "trace_pipeline_steps": [],
            },
            odd_from_prob_fn=lambda p: 1.0 / p,
        )
        probs = payload.get("probabilities") if isinstance(payload.get("probabilities"), dict) else {}
        odds = payload.get("odds") if isinstance(payload.get("odds"), dict) else {}
        self.assertAlmostEqual(float(probs.get("p_map_used")), 0.62, places=6)
        self.assertAlmostEqual(float(odds.get("teamA_map")), 1.0 / 0.62, places=6)
        self.assertAlmostEqual(float(odds.get("teamB_map")), 1.0 / 0.38, places=6)

    def test_pipeline_order_is_derived_from_steps(self):
        payload = build_ml_trace_payload(
            ctx={
                "season_label": "2026",
                "teamA": "A",
                "teamB": "B",
                "bo": 3,
                "ml_engine": "mlcore_v2",
                "p_map_used": 0.62,
                "p_series_used": 0.70,
                "trace_pipeline_order": ["stale_stage"],
                "trace_pipeline_steps": [
                    {"stage": "uncertainty_layer", "applied": True},
                    {"stage": "engine_lineup", "applied": True},
                    {"stage": "uncertainty_layer", "applied": False},
                    {"stage": "series_gamma", "applied": True},
                ],
            },
            odd_from_prob_fn=lambda p: 1.0 / p,
        )
        pipe = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else {}
        self.assertEqual(list(pipe.get("order", [])), ["uncertainty_layer", "engine_lineup", "series_gamma"])

    def test_accepts_typed_trace_context(self):
        ctx = MlTraceContext.from_mapping(
            {
                "season_label": "2026",
                "teamA": "A",
                "teamB": "B",
                "bo": 3,
                "ml_engine": "mlcore_v2",
                "p_map_used": 0.62,
                "p_series_used": 0.70,
                "trace_pipeline_steps": [{"stage": "series_gamma"}],
            }
        )
        payload = build_ml_trace_payload(
            ctx=ctx,
            odd_from_prob_fn=lambda p: 1.0 / p,
        )
        probs = payload.get("probabilities") if isinstance(payload.get("probabilities"), dict) else {}
        self.assertAlmostEqual(float(probs.get("p_map_used")), 0.62, places=6)
        pipe = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else {}
        self.assertEqual(list(pipe.get("order", [])), ["series_gamma"])

    def test_includes_confidence_governor_summary_and_stage_details(self):
        payload = build_ml_trace_payload(
            ctx={
                "season_label": "2026",
                "teamA": "A",
                "teamB": "B",
                "bo": 3,
                "ml_engine": "mlcore_v2",
                "p_map_used": 0.61,
                "p_series_used": 0.70,
                "ml_confidence_governor_enabled": True,
                "ml_confidence_governor_live_enabled": True,
                "ml_confidence_governor_max_total_shrink_pp": 8.0,
                "_ml_confidence_governor_state": {
                    "max_total_shrink_pp": 8.0,
                    "used_shrink_pp": 5.0,
                    "stages": [
                        {"stage": "uncertainty_layer", "applied_shrink_pp": 2.0, "limited": False},
                        {"stage": "state_alite_live_override", "applied_shrink_pp": 3.0, "limited": True},
                    ],
                },
                "trace_conf_guard_info": {"governor": {"limited": False, "applied_shrink_pp": 0.5}},
                "trace_uncertainty_info": {"governor": {"limited": False, "applied_shrink_pp": 2.0}},
                "trace_patch_guard_info": {"governor": {"limited": False, "applied_shrink_pp": 0.0}},
                "trace_marginal_trust_info": {"governor": {"limited": True, "applied_shrink_pp": 0.0}},
                "trace_pipeline_steps": [
                    {
                        "stage": "uncertainty_layer",
                        "before": 0.62,
                        "after": 0.61,
                        "applied": True,
                        "governor_limited": False,
                        "governor_applied_shrink_pp": 2.0,
                    },
                    {
                        "stage": "state_alite_live_override",
                        "before": 0.61,
                        "after": 0.59,
                        "applied": True,
                        "governor_limited": True,
                        "governor_applied_shrink_pp": 3.0,
                    },
                ],
            },
            odd_from_prob_fn=lambda p: 1.0 / p,
        )
        gov = payload.get("confidence_governor") if isinstance(payload.get("confidence_governor"), dict) else {}
        guards = payload.get("guards") if isinstance(payload.get("guards"), dict) else {}
        unc = payload.get("uncertainty_layer") if isinstance(payload.get("uncertainty_layer"), dict) else {}
        marg = payload.get("marginal_trust_guard") if isinstance(payload.get("marginal_trust_guard"), dict) else {}
        self.assertTrue(bool(gov.get("enabled", False)))
        self.assertTrue(bool(gov.get("live_enabled", False)))
        self.assertAlmostEqual(float(gov.get("used_shrink_pp")), 5.0, places=6)
        self.assertAlmostEqual(float(gov.get("remaining_shrink_pp")), 3.0, places=6)
        self.assertEqual(list(gov.get("limited_stages", [])), ["state_alite_live_override"])
        by_stage = gov.get("applied_shrink_pp_by_stage") if isinstance(gov.get("applied_shrink_pp_by_stage"), dict) else {}
        self.assertAlmostEqual(float(by_stage.get("uncertainty_layer")), 2.0, places=6)
        self.assertAlmostEqual(float(by_stage.get("state_alite_live_override")), 3.0, places=6)
        self.assertTrue(bool(guards.get("confidence_governor_on", False)))
        self.assertTrue(bool(guards.get("confidence_governor_live_on", False)))
        self.assertIn("governor", unc)
        self.assertIn("governor", marg)

    def test_exposes_post_series_coherence_trace(self):
        payload = build_ml_trace_payload(
            ctx={
                "season_label": "2026",
                "teamA": "A",
                "teamB": "B",
                "bo": 5,
                "ml_engine": "mlcore_v2",
                "p_map_used": 0.62,
                "p_series_used": 0.67,
                "ml_series_coherence_post_max_dev_pp_runtime": 20.0,
                "trace_series_coherence_post_info": {
                    "enabled": True,
                    "applied": True,
                    "reason": "clamped_to_economic_coherence",
                    "bo": 5,
                    "p_map": 0.62,
                    "p_series_before": 0.40,
                    "p_series_after": 0.67,
                    "p_target": 0.70,
                    "diff_pp": 27.0,
                    "bounds": {"low": 0.67, "high": 0.73, "max_dev_pp": 20.0},
                },
            },
            odd_from_prob_fn=lambda p: 1.0 / p,
        )
        post = (
            payload.get("series_economic_coherence_post")
            if isinstance(payload.get("series_economic_coherence_post"), dict)
            else {}
        )
        post_raw = (
            payload.get("series_economic_coherence_post_raw")
            if isinstance(payload.get("series_economic_coherence_post_raw"), dict)
            else {}
        )
        guards = payload.get("guards") if isinstance(payload.get("guards"), dict) else {}
        self.assertTrue(bool(post.get("applied", False)))
        self.assertEqual(str(post.get("reason", "")), "clamped_to_economic_coherence")
        self.assertAlmostEqual(float(post.get("p_series_after")), 0.67, places=6)
        self.assertAlmostEqual(float(post.get("max_dev_pp_runtime")), 20.0, places=6)
        self.assertTrue(bool(post_raw.get("applied", False)))
        self.assertTrue(bool(guards.get("series_economic_coherence_post_applied", False)))


if __name__ == "__main__":
    unittest.main()
