import unittest

from app_core.ml_pre_pipeline_core import run_ml_pre_pipeline
from app_core.ml_trace_payload_core import build_ml_trace_payload


class TestMlOddsE2ETraceCore(unittest.TestCase):
    def test_pre_pipeline_to_trace_odds_snapshot(self):
        def _engine_fn(**_kwargs):
            return {
                "p_map_used": 0.4010,
                "p_series_used": 0.4550,
                "diff_used": 10.0,
                "delta_players": 0.1,
                "delta_players_eff": 0.05,
                "trace_pipeline_steps": [
                    {"stage": "uncertainty_layer", "applied": True},
                    {"stage": "engine_lineup", "applied": True},
                ],
            }

        def _live_fn(**_kwargs):
            return {
                "p_map_used": 0.3765,
                "p_series_used": 0.4421,
                "diff_used": 11.0,
                "delta_players": 0.1,
                "trace_pipeline_steps": [
                    {"stage": "uncertainty_layer", "applied": True},
                    {"stage": "engine_lineup", "applied": True},
                    {"stage": "season_form_blend", "applied": True},
                    {"stage": "series_gamma", "applied": True},
                ],
            }

        out = run_ml_pre_pipeline(
            pre_engine_kwargs={"p_map_used": 0.5},
            pre_live_kwargs={"lineup_mode": "off"},
            run_pre_engine_fn=_engine_fn,
            run_pre_live_fn=_live_fn,
        )
        payload = build_ml_trace_payload(
            ctx={
                "season_label": "2026",
                "teamA": "A",
                "teamB": "B",
                "bo": 3,
                "ml_engine": "mlcore_v2",
                "p_map_used": out.get("p_map_used"),
                "p_series_used": out.get("p_series_used"),
                "trace_pipeline_order": ["stale_should_not_win"],
                "trace_pipeline_steps": out.get("trace_pipeline_steps", []),
            },
            odd_from_prob_fn=lambda p: 1.0 / float(p),
        )

        odds = payload.get("odds") if isinstance(payload.get("odds"), dict) else {}
        pipe = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else {}

        self.assertAlmostEqual(float(odds.get("teamA_map")), 1.0 / 0.3765, places=8)
        self.assertAlmostEqual(float(odds.get("teamB_map")), 1.0 / (1.0 - 0.3765), places=8)
        self.assertEqual(
            list(pipe.get("order", [])),
            ["uncertainty_layer", "engine_lineup", "season_form_blend", "series_gamma"],
        )
        self.assertAlmostEqual(float(pipe.get("final_p_map")), 0.3765, places=8)
        self.assertAlmostEqual(float(pipe.get("final_p_series")), 0.4421, places=8)


if __name__ == "__main__":
    unittest.main()
