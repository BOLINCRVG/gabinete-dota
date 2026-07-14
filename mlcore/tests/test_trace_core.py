import unittest

from app_core.trace_core import (
    append_feature_trace_event,
    build_combined_trace,
    coerce_feature_flag,
)


class TestTraceCore(unittest.TestCase):
    def test_build_combined_trace_happy_path(self):
        base = {
            "engine": "mlcore_v2",
            "matchup": {"teamA": "A", "teamB": "B"},
            "filters": {"year": "2026"},
        }
        mkt = [{"mercado": "ML", "odd_mix": 1.8}]
        ply = [{"player": "Faker", "odd_mix": 1.9}]
        out = build_combined_trace(
            ml_base_trace=base,
            resumo_markets_trace=mkt,
            players_trace=ply,
            generated_at="2026-02-11T12:00:00",
        )
        self.assertEqual(out["generated_at"], "2026-02-11T12:00:00")
        self.assertEqual(out["engine"], "mlcore_v2")
        self.assertEqual(out["matchup"]["teamA"], "A")
        self.assertEqual(len(out["resumo_markets_trace"]), 1)
        self.assertEqual(len(out["players_trace"]), 1)

    def test_build_combined_trace_sanitizes_types(self):
        out = build_combined_trace(
            ml_base_trace=None,
            resumo_markets_trace=["x", {"ok": 1}],
            players_trace={"a": 1},
        )
        self.assertIsInstance(out["ml_trace"], dict)
        self.assertEqual(len(out["resumo_markets_trace"]), 1)
        self.assertEqual(out["players_trace"], [])

    def test_build_combined_trace_includes_feature_trace(self):
        out = build_combined_trace(
            ml_base_trace={"engine": "mlcore_v2"},
            resumo_markets_trace=[],
            players_trace=[],
            feature_trace=[{"feature": "players_stats_v2", "event": "shadow"}],
        )
        self.assertIn("feature_trace", out)
        self.assertEqual(len(out["feature_trace"]), 1)

    def test_coerce_feature_flag(self):
        self.assertTrue(coerce_feature_flag(True))
        self.assertTrue(coerce_feature_flag("true"))
        self.assertTrue(coerce_feature_flag("Sim"))
        self.assertFalse(coerce_feature_flag("0"))
        self.assertFalse(coerce_feature_flag(None, default=False))
        self.assertTrue(coerce_feature_flag(None, default=True))

    def test_append_feature_trace_event_dedupes_last(self):
        rows = []
        rows = append_feature_trace_event(
            rows,
            feature="players_stats_v2",
            event="shadow_render",
            payload={"rows": 10},
            dedupe_signature="a",
        )
        rows = append_feature_trace_event(
            rows,
            feature="players_stats_v2",
            event="shadow_render",
            payload={"rows": 12},
            dedupe_signature="a",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["payload"]["rows"], 12)


if __name__ == "__main__":
    unittest.main()
