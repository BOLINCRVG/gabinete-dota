import unittest

from app_core.ml_output_models import MlPrePipelineOutput
from app_core.ml_pre_pipeline_core import run_ml_pre_pipeline


class TestMlPrePipelineCore(unittest.TestCase):
    def test_merges_engine_and_live_outputs(self):
        captured_live_kwargs = {}

        def _engine_fn(**_kwargs):
            return {
                "p_map_used": 0.61,
                "p_series_used": 0.66,
                "diff_used": 11.0,
                "delta_players": 0.2,
                "delta_players_eff": 0.15,
                "trace_p_map_team": 0.60,
                "trace_p_map_players": 0.62,
                "trace_p_map_fused": 0.61,
                "trace_n_team_eff": 14.0,
                "trace_n_players_eff": 7.0,
                "trace_fusion_info": {"applied": True},
                "trace_uncertainty_info": {"applied": True},
                "_delta_mode": "saturado",
                "_delta_cap": 1.6,
                "_delta_slope": 2.4,
                "lineup_mode": "ML players layer",
                "match_league": "LCK",
                "_elo_info": {"p_cal": 0.61},
                "_p_pipe_prev": 0.61,
                "trace_pipeline_steps": [{"stage": "uncertainty_layer"}],
            }

        def _live_fn(**kwargs):
            captured_live_kwargs.update(kwargs)
            return {
                "p_map_used": 0.63,
                "p_series_used": 0.69,
                "diff_used": 12.0,
                "delta_players": 0.25,
                "trace_form_info": {"applied": True},
                "trace_hybrid_info": {"applied": True},
                "trace_conf_guard_info": {"applied": False},
                "trace_patch_guard_info": {"applied": False},
                "trace_series_gamma_info": {"applied": True},
                "trace_series_coherence_info": {"applied": True},
                "_guard_triggered": False,
                "_guard_dpp": 0.0,
                "_ff_ml_series_bo_dplite_live": True,
                "trace_series_bo_dplite_live": {"applied": True},
                "_lineup_mode_sig": "ML players layer",
                "_lineupA_sig": {"Top": "a"},
                "_lineupB_sig": {"Top": "b"},
                "trace_pipeline_steps": [{"stage": "series_gamma"}],
            }

        out = run_ml_pre_pipeline(
            pre_engine_kwargs={"p_map_used": 0.6},
            pre_live_kwargs={"lineup_mode": "CSV byTournament (manual)"},
            run_pre_engine_fn=_engine_fn,
            run_pre_live_fn=_live_fn,
        )

        self.assertAlmostEqual(float(captured_live_kwargs.get("p_map_used")), 0.61, places=6)
        self.assertAlmostEqual(float(captured_live_kwargs.get("p_series_used")), 0.66, places=6)
        self.assertAlmostEqual(float(captured_live_kwargs.get("trace_n_team_eff")), 14.0, places=6)
        self.assertEqual(str(captured_live_kwargs.get("lineup_mode")), "ML players layer")

        self.assertAlmostEqual(float(out.get("p_map_used")), 0.63, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.69, places=6)
        self.assertAlmostEqual(float(out.get("delta_players_eff")), 0.15, places=6)
        self.assertEqual(str(out.get("_delta_mode")), "saturado")
        self.assertTrue(bool((out.get("trace_form_info") or {}).get("applied", False)))
        self.assertTrue(bool((out.get("trace_series_bo_dplite_live") or {}).get("applied", False)))
        self.assertEqual(str(((out.get("trace_pipeline_steps") or [{}])[0]).get("stage")), "series_gamma")

    def test_accepts_typed_payload_with_to_kwargs(self):
        class _Payload:
            def __init__(self, data):
                self._data = dict(data)

            def to_kwargs(self):
                return dict(self._data)

        def _engine_fn(**_kwargs):
            return {"p_map_used": 0.52, "p_series_used": 0.52, "trace_pipeline_steps": []}

        def _live_fn(**kwargs):
            return {
                "p_map_used": float(kwargs.get("p_map_used", 0.0)),
                "p_series_used": float(kwargs.get("p_series_used", 0.0)),
                "trace_pipeline_steps": [],
            }

        out = run_ml_pre_pipeline(
            pre_engine=_Payload({"p_map_used": 0.5}),
            pre_live=_Payload({"lineup_mode": "off"}),
            run_pre_engine_fn=_engine_fn,
            run_pre_live_fn=_live_fn,
        )
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.52, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.52, places=6)

    def test_can_return_typed_output_model(self):
        out = run_ml_pre_pipeline(
            pre_engine_kwargs={"p_map_used": 0.5},
            pre_live_kwargs={"lineup_mode": "off"},
            run_pre_engine_fn=lambda **_k: {"p_map_used": 0.54, "p_series_used": 0.58},
            run_pre_live_fn=lambda **_k: {"p_map_used": 0.55, "p_series_used": 0.59},
            return_model=True,
        )
        self.assertTrue(isinstance(out, MlPrePipelineOutput))
        self.assertAlmostEqual(float(out.p_map_used), 0.55, places=6)
        self.assertAlmostEqual(float(out.p_series_used), 0.59, places=6)


if __name__ == "__main__":
    unittest.main()
