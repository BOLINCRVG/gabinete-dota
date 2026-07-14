import json
import unittest

from app_core.ml_live_feature_trace_core import append_ml_live_feature_traces


class TestMlLiveFeatureTraceCore(unittest.TestCase):
    def test_appends_live_feature_traces_when_enabled(self):
        calls = []

        def _append_feature_trace(**kwargs):
            calls.append(dict(kwargs))

        out = append_ml_live_feature_traces(
            append_feature_trace_fn=_append_feature_trace,
            ff_ml_style_b_lite_live=True,
            trace_style_b_lite_live={"applied": True, "p_before": 0.51, "p_after": 0.53},
            ff_ml_league_rank_sos_live=True,
            trace_league_rank_sos_live={"applied": True, "p_before": 0.53, "p_after": 0.54, "p_rank": 0.58},
            ff_ml_roster_continuity_live=True,
            trace_roster_continuity_live={"applied": True, "p_before": 0.54, "p_after": 0.55, "target_p": 0.57},
            ff_ml_hier_calibration_live=True,
            trace_hier_calibration_live={"applied": True, "p_before": 0.55, "p_after": 0.56, "bias_adj": 0.02},
            ff_ml_series_bo_dplite_live=True,
            trace_series_bo_dplite_live={"applied": True, "p_before": 0.56, "p_after": 0.57},
            ff_ml_layer_uncertainty_live=True,
            trace_layer_uncertainty_live={"applied": True, "p_before": 0.57, "p_after": 0.56},
            ff_ml_meta_c_lite_live=True,
            trace_meta_c_lite_live={"applied": True, "p_anchor": 0.55, "p_before": 0.56, "p_after": 0.58},
            season_label="2026",
            teamA="Alpha",
            teamB="Beta",
            bo=3,
            ml_engine="mlcore_v2",
            input_signature_hash="sig123",
        )

        self.assertEqual(len(calls), 7)
        self.assertEqual(
            [c.get("feature") for c in calls],
            [
                "ml_style_matchup_b_lite_live_v1",
                "ml_league_rank_sos_live_v1",
                "ml_roster_continuity_live_v1",
                "ml_hier_calibration_live_v1",
                "ml_series_bo_dplite_live_v1",
                "ml_layer_uncertainty_live_v1",
                "ml_meta_blend_c_lite_live_v1",
            ],
        )
        for call in calls:
            self.assertEqual(str(call.get("event")), "live_apply")
            self.assertEqual(call.get("persist_files"), False)
            payload = call.get("payload") or {}
            self.assertEqual(payload.get("input_signature_sha256"), "sig123")
            self.assertEqual(payload.get("engine"), "mlcore_v2")
            self.assertEqual((payload.get("matchup") or {}).get("bo"), 3)

        dedupe_style = json.loads(str(calls[0].get("dedupe_signature")))
        self.assertEqual(dedupe_style.get("sig"), "sig123")
        self.assertEqual(dedupe_style.get("eng"), "mlcore_v2")
        self.assertAlmostEqual(float(dedupe_style.get("p_before")), 0.51, places=6)
        self.assertAlmostEqual(float(dedupe_style.get("p_after")), 0.53, places=6)
        self.assertTrue(all(bool(v) for v in out.values()))

    def test_continues_when_one_append_raises(self):
        calls = []

        def _append_feature_trace(**kwargs):
            if kwargs.get("feature") == "ml_hier_calibration_live_v1":
                raise RuntimeError("boom")
            calls.append(dict(kwargs))

        out = append_ml_live_feature_traces(
            append_feature_trace_fn=_append_feature_trace,
            ff_ml_style_b_lite_live=True,
            trace_style_b_lite_live={},
            ff_ml_league_rank_sos_live=True,
            trace_league_rank_sos_live={},
            ff_ml_roster_continuity_live=True,
            trace_roster_continuity_live={},
            ff_ml_hier_calibration_live=True,
            trace_hier_calibration_live={},
            ff_ml_series_bo_dplite_live=True,
            trace_series_bo_dplite_live={},
            ff_ml_layer_uncertainty_live=True,
            trace_layer_uncertainty_live={},
            ff_ml_meta_c_lite_live=True,
            trace_meta_c_lite_live={},
            season_label="2026",
            teamA="Alpha",
            teamB="Beta",
            bo=3,
            ml_engine="mlcore_v2",
            input_signature_hash="sig123",
        )

        self.assertEqual(len(calls), 6)
        self.assertFalse(bool(out.get("ml_hier_calibration_live_v1", True)))
        self.assertTrue(bool(out.get("ml_style_matchup_b_lite_live_v1", False)))
        self.assertTrue(bool(out.get("ml_meta_blend_c_lite_live_v1", False)))

    def test_emits_nothing_when_all_flags_disabled(self):
        calls = []
        out = append_ml_live_feature_traces(
            append_feature_trace_fn=lambda **kwargs: calls.append(dict(kwargs)),
            ff_ml_style_b_lite_live=False,
            trace_style_b_lite_live={},
            ff_ml_league_rank_sos_live=False,
            trace_league_rank_sos_live={},
            ff_ml_roster_continuity_live=False,
            trace_roster_continuity_live={},
            ff_ml_hier_calibration_live=False,
            trace_hier_calibration_live={},
            ff_ml_series_bo_dplite_live=False,
            trace_series_bo_dplite_live={},
            ff_ml_layer_uncertainty_live=False,
            trace_layer_uncertainty_live={},
            ff_ml_meta_c_lite_live=False,
            trace_meta_c_lite_live={},
            season_label="2026",
            teamA="Alpha",
            teamB="Beta",
            bo=3,
            ml_engine="mlcore_v2",
            input_signature_hash="sig123",
        )

        self.assertEqual(calls, [])
        self.assertEqual(
            out,
            {
                "ml_style_matchup_b_lite_live_v1": False,
                "ml_league_rank_sos_live_v1": False,
                "ml_roster_continuity_live_v1": False,
                "ml_hier_calibration_live_v1": False,
                "ml_series_bo_dplite_live_v1": False,
                "ml_layer_uncertainty_live_v1": False,
                "ml_meta_blend_c_lite_live_v1": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
