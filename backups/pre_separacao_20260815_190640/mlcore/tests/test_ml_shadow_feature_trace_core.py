import json
import unittest

from app_core.ml_shadow_feature_trace_core import append_ml_shadow_feature_traces


class TestMlShadowFeatureTraceCore(unittest.TestCase):
    def test_appends_three_shadow_events_when_enabled(self):
        calls = []

        def _append_feature_trace(**kwargs):
            calls.append(dict(kwargs))

        out = append_ml_shadow_feature_traces(
            append_feature_trace_fn=_append_feature_trace,
            ff_ml_shadow_vnext=True,
            trace_shadow_ml_vnext={
                "applied": True,
                "reason": "ok",
                "p_map_old": 0.4,
                "p_map_new": 0.42,
                "p_series_old": 0.35,
                "p_series_new": 0.36,
                "diff_pp_map": 2.0,
                "diff_pp_series": 1.0,
                "weights": {"a": 1},
                "inputs": {"b": 2},
            },
            ff_ml_shadow_state_alite=True,
            trace_shadow_state_alite={
                "applied": False,
                "reason": "",
                "p_map_old": 0.4,
                "p_map_new": 0.39,
                "p_series_old": 0.35,
                "p_series_new": 0.34,
                "diff_pp_map": -1.0,
                "diff_pp_series": -1.0,
                "weights": {"c": 3},
                "inputs": {"d": 4},
                "uncertainty": {"n_eff": 10},
            },
            ff_ml_shadow_state_areal=True,
            trace_shadow_state_areal={
                "applied": True,
                "reason": "areal",
                "p_map_old": 0.4,
                "p_map_new": 0.43,
                "p_series_old": 0.35,
                "p_series_new": 0.37,
                "diff_pp_map": 3.0,
                "diff_pp_series": 2.0,
                "weights": {"e": 5},
                "inputs": {"f": 6},
                "state": {"g": 7},
                "qwl_trace": {"h": 8},
            },
            season_label="2026",
            teamA="Alpha",
            teamB="Beta",
            bo=3,
            ml_engine="mlcore_v2",
            input_signature_hash="sig123",
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [c.get("feature") for c in calls],
            [
                "ml_shadow_vnext_map_series_v1",
                "ml_shadow_state_alite_v1",
                "ml_shadow_state_areal_v1",
            ],
        )
        for call in calls:
            self.assertEqual(str(call.get("event")), "shadow_compare")
            self.assertEqual(call.get("persist_files"), False)
            self.assertEqual((call.get("payload") or {}).get("engine"), "mlcore_v2")
            self.assertEqual((call.get("payload") or {}).get("input_signature_sha256"), "sig123")

        dedupe = json.loads(str(calls[0].get("dedupe_signature")))
        self.assertEqual(dedupe.get("sig"), "sig123")
        self.assertEqual(dedupe.get("eng"), "mlcore_v2")
        self.assertAlmostEqual(float(dedupe.get("pm_old")), 0.4, places=6)
        self.assertAlmostEqual(float(dedupe.get("pm_new")), 0.42, places=6)
        self.assertTrue(bool(out.get("ml_shadow_vnext_map_series_v1", False)))
        self.assertTrue(bool(out.get("ml_shadow_state_alite_v1", False)))
        self.assertTrue(bool(out.get("ml_shadow_state_areal_v1", False)))

    def test_keeps_processing_after_one_callback_failure(self):
        calls = []

        def _append_feature_trace(**kwargs):
            if kwargs.get("feature") == "ml_shadow_vnext_map_series_v1":
                raise RuntimeError("boom")
            calls.append(dict(kwargs))

        out = append_ml_shadow_feature_traces(
            append_feature_trace_fn=_append_feature_trace,
            ff_ml_shadow_vnext=True,
            trace_shadow_ml_vnext={"p_map_old": 0.5, "p_map_new": 0.51},
            ff_ml_shadow_state_alite=True,
            trace_shadow_state_alite={"p_map_old": 0.5, "p_map_new": 0.49},
            ff_ml_shadow_state_areal=True,
            trace_shadow_state_areal={"p_map_old": 0.5, "p_map_new": 0.48},
            season_label="2026",
            teamA="Alpha",
            teamB="Beta",
            bo=3,
            ml_engine="mlcore_v2",
            input_signature_hash="sig123",
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [c.get("feature") for c in calls],
            ["ml_shadow_state_alite_v1", "ml_shadow_state_areal_v1"],
        )
        self.assertFalse(bool(out.get("ml_shadow_vnext_map_series_v1", True)))
        self.assertTrue(bool(out.get("ml_shadow_state_alite_v1", False)))
        self.assertTrue(bool(out.get("ml_shadow_state_areal_v1", False)))

    def test_disabled_flags_emit_nothing(self):
        calls = []
        out = append_ml_shadow_feature_traces(
            append_feature_trace_fn=lambda **kwargs: calls.append(dict(kwargs)),
            ff_ml_shadow_vnext=False,
            trace_shadow_ml_vnext={},
            ff_ml_shadow_state_alite=False,
            trace_shadow_state_alite={},
            ff_ml_shadow_state_areal=False,
            trace_shadow_state_areal={},
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
                "ml_shadow_vnext_map_series_v1": False,
                "ml_shadow_state_alite_v1": False,
                "ml_shadow_state_areal_v1": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
