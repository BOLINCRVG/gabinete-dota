import unittest
from pathlib import Path

from app_core.ml_output_models import MlLiveOutput
from app_core.ml_live_pipeline_core import run_ml_live_layers


class TestMlLivePipelineCore(unittest.TestCase):
    def test_disabled_flags_keep_probabilities(self):
        steps = []
        out = run_ml_live_layers(
            p_map_used=0.61,
            p_series_used=0.67,
            bo=3,
            playoffs_opt=False,
            ml_info={},
            feature_flag_enabled_fn=lambda _k, _d=False: False,
            prob_win_series_fn=lambda p, _bo: float(p),
            session_state={},
            trace_pipeline_steps=steps,
            fit_params_path=Path("params_fitted.__missing__.json"),
        )
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.61, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.67, places=6)
        self.assertFalse(bool(out.get("_ff_ml_style_b_lite_live", True)))
        self.assertFalse(bool(out.get("_ff_ml_layer_uncertainty_live", True)))
        tr_style = out.get("trace_style_b_lite_live") if isinstance(out.get("trace_style_b_lite_live"), dict) else {}
        tr_lu = out.get("trace_layer_uncertainty_live") if isinstance(out.get("trace_layer_uncertainty_live"), dict) else {}
        self.assertFalse(bool(tr_style.get("enabled", True)))
        self.assertFalse(bool(tr_lu.get("enabled", True)))
        self.assertEqual(len(steps), 0)

    def test_can_return_typed_output_model(self):
        out = run_ml_live_layers(
            p_map_used=0.61,
            p_series_used=0.67,
            bo=3,
            playoffs_opt=False,
            ml_info={},
            feature_flag_enabled_fn=lambda _k, _d=False: False,
            prob_win_series_fn=lambda p, _bo: float(p),
            session_state={},
            trace_pipeline_steps=[],
            fit_params_path=Path("params_fitted.__missing__.json"),
            return_model=True,
        )
        self.assertTrue(isinstance(out, MlLiveOutput))
        self.assertAlmostEqual(float(out.p_map_used), 0.61, places=6)
        self.assertAlmostEqual(float(out.p_series_used), 0.67, places=6)

    def test_live_layers_can_apply_confidence_governor(self):
        steps = []

        def _ff(key, _d=False):
            return str(key) == "ff_ml_layer_uncertainty_live"

        def _gov_fn(**_kwargs):
            return {
                "enabled": True,
                "limited": True,
                "requested_shrink_pp": 2.0,
                "applied_shrink_pp": 0.5,
                "used_shrink_pp": 0.5,
                "remaining_shrink_pp": 7.5,
                "p_out": 0.695,
            }

        out = run_ml_live_layers(
            p_map_used=0.70,
            p_series_used=0.78,
            bo=3,
            playoffs_opt=False,
            ml_info={},
            feature_flag_enabled_fn=_ff,
            prob_win_series_fn=lambda p, _bo: float(p),
            session_state={
                "ml_confidence_governor_enabled": True,
                "ml_confidence_governor_live_enabled": True,
                "ml_confidence_governor_max_total_shrink_pp": 8.0,
            },
            trace_pipeline_steps=steps,
            fit_params_path=Path("params_fitted.__missing__.json"),
            apply_confidence_governor_fn=_gov_fn,
        )
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.695, places=6)
        tr_lu = out.get("trace_layer_uncertainty_live") if isinstance(out.get("trace_layer_uncertainty_live"), dict) else {}
        gov = tr_lu.get("governor") if isinstance(tr_lu.get("governor"), dict) else {}
        self.assertTrue(bool(gov.get("limited", False)))
        self.assertAlmostEqual(float(gov.get("applied_shrink_pp")), 0.5, places=6)
        self.assertEqual(len(steps), 1)
        self.assertTrue(bool(steps[0].get("governor_limited", False)))

    def test_rank_sos_does_not_apply_single_league_prior_on_cross_league_matchup(self):
        out = run_ml_live_layers(
            p_map_used=0.60,
            p_series_used=0.68,
            bo=3,
            playoffs_opt=False,
            ml_info={"blue": {"league": "LCK"}, "red": {"league": "LPL"}},
            team_games=None,
            teamA="Gen.G",
            teamB="T1",
            feature_flag_enabled_fn=lambda key, _d=False: str(key) == "ff_ml_league_rank_sos_live",
            prob_win_series_fn=lambda p, _bo: float(p),
            session_state={},
            trace_pipeline_steps=[],
            fit_params_path=Path("params_fitted.__missing__.json"),
        )
        rank = out.get("trace_league_rank_sos_live") if isinstance(out.get("trace_league_rank_sos_live"), dict) else {}
        self.assertFalse(bool(rank.get("applied", False)))
        self.assertEqual(str(rank.get("reason")), "missing_identity")


if __name__ == "__main__":
    unittest.main()
