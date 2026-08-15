import unittest
from pathlib import Path

from app_core.ml_orchestrator_core import run_ml_post_layers


class TestMlOrchestratorCore(unittest.TestCase):
    def test_orchestrates_state_live_marginal_and_pipeline_order(self):
        def _ff(key, default=False):
            if str(key) == "ff_ml_state_areal_live":
                return False
            if str(key) == "ff_ml_state_alite_live":
                return True
            return bool(default)

        def _live_fn(**kwargs):
            steps = kwargs.get("trace_pipeline_steps")
            if isinstance(steps, list):
                steps.append(
                    {
                        "stage": "style_matchup_b_live",
                        "before": 0.58,
                        "after": 0.57,
                        "applied": True,
                    }
                )
            return {
                "p_map_used": 0.57,
                "p_series_used": 0.62,
                "_ff_ml_style_b_lite_live": True,
                "trace_style_b_lite_live": {"applied": True},
                "trace_pipeline_steps": steps if isinstance(steps, list) else [],
            }

        def _marginal_fn(**kwargs):
            self.assertEqual(str(kwargs.get("league_hint", "")), "LCK")
            return {
                "p_map_used": 0.56,
                "p_series_used": 0.61,
                "trace_marginal_trust_info": {"applied": True, "league": "LCK"},
                "pipeline_step": {
                    "stage": "marginal_trust_guard",
                    "before": 0.57,
                    "after": 0.56,
                    "applied": True,
                },
            }

        out = run_ml_post_layers(
            p_map_used=0.60,
            p_series_used=0.64,
            bo=3,
            trace_shadow_state_alite={"applied": True, "p_map_new": 0.58},
            trace_shadow_state_areal={"applied": True, "p_map_new": 0.54},
            feature_flag_enabled_fn=_ff,
            prob_win_series_fn=lambda p, _bo: float(p) + 0.03,
            session_state={},
            trace_pipeline_steps=[],
            live_kwargs={"fit_params_path": Path("params_fitted.__missing__.json")},
            marginal_kwargs={
                "ml_info": {"blue": {"league": "LCK"}},
                "trace_n_team_eff": 12.0,
                "lv_last_report": {},
                "ml_trace_last": {},
                "ml_confidence_fn": lambda *_a, **_k: {"score": 70.0},
                "marginal_adjust_fn": lambda p, *_a, **_k: (float(p), {"applied": False, "league": "LCK"}),
            },
            confidence_baseline_path=Path("baseline.__missing__.json"),
            load_confidence_baseline_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            lg_a_hint="LCK",
            caption_fn=None,
            run_live_layers_fn=_live_fn,
            run_marginal_trust_guard_fn=_marginal_fn,
        )
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.56, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.61, places=6)
        self.assertTrue(bool((out.get("trace_state_alite_live") or {}).get("applied", False)))
        self.assertFalse(bool(out.get("_ff_ml_state_areal_live", True)))
        stages = [str(x) for x in list(out.get("trace_pipeline_order", []))]
        self.assertEqual(stages, ["state_alite_live_override", "style_matchup_b_live", "marginal_trust_guard"])

    def test_governor_is_applied_on_state_override_before_live_layers(self):
        def _ff(key, default=False):
            if str(key) == "ff_ml_state_areal_live":
                return False
            if str(key) == "ff_ml_state_alite_live":
                return True
            return bool(default)

        def _gov_fn(**_kwargs):
            return {
                "enabled": True,
                "limited": True,
                "requested_shrink_pp": 5.0,
                "applied_shrink_pp": 2.0,
                "used_shrink_pp": 2.0,
                "remaining_shrink_pp": 6.0,
                "p_out": 0.59,
            }

        def _live_fn(**kwargs):
            self.assertAlmostEqual(float(kwargs.get("p_map_used")), 0.59, places=6)
            return {
                "p_map_used": float(kwargs.get("p_map_used")),
                "p_series_used": float(kwargs.get("p_series_used")),
                "trace_pipeline_steps": kwargs.get("trace_pipeline_steps", []),
            }

        def _marginal_fn(**kwargs):
            return {
                "p_map_used": float(kwargs.get("p_map_used")),
                "p_series_used": float(kwargs.get("p_series_used")),
                "trace_marginal_trust_info": {"applied": False},
                "pipeline_step": {"stage": "marginal_trust_guard", "applied": False},
            }

        out = run_ml_post_layers(
            p_map_used=0.60,
            p_series_used=0.64,
            bo=3,
            trace_shadow_state_alite={"applied": True, "p_map_new": 0.55},
            trace_shadow_state_areal={},
            feature_flag_enabled_fn=_ff,
            prob_win_series_fn=lambda p, _bo: float(p) + 0.03,
            session_state={
                "ml_confidence_governor_enabled": True,
                "ml_confidence_governor_live_enabled": True,
                "ml_confidence_governor_max_total_shrink_pp": 8.0,
            },
            trace_pipeline_steps=[],
            live_kwargs={"fit_params_path": Path("params_fitted.__missing__.json")},
            marginal_kwargs={
                "ml_info": {},
                "trace_n_team_eff": 12.0,
                "lv_last_report": {},
                "ml_trace_last": {},
                "ml_confidence_fn": lambda *_a, **_k: {"score": 70.0},
                "marginal_adjust_fn": lambda p, *_a, **_k: (float(p), {"applied": False}),
            },
            confidence_baseline_path=Path("baseline.__missing__.json"),
            load_confidence_baseline_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            lg_a_hint="LCK",
            caption_fn=None,
            run_live_layers_fn=_live_fn,
            run_marginal_trust_guard_fn=_marginal_fn,
            apply_confidence_governor_fn=_gov_fn,
        )
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.59, places=6)
        tr = out.get("trace_state_alite_live") if isinstance(out.get("trace_state_alite_live"), dict) else {}
        gov = tr.get("governor") if isinstance(tr.get("governor"), dict) else {}
        self.assertTrue(bool(gov.get("limited", False)))
        steps = list(out.get("trace_pipeline_steps", []))
        self.assertTrue(any(str(s.get("stage")) == "state_alite_live_override" for s in steps))

    def test_applies_post_series_economic_coherence_after_marginal(self):
        def _ff(_key, default=False):
            return bool(default)

        def _live_fn(**kwargs):
            return {
                "p_map_used": float(kwargs.get("p_map_used")),
                "p_series_used": float(kwargs.get("p_series_used")),
                "trace_pipeline_steps": kwargs.get("trace_pipeline_steps", []),
            }

        def _marginal_fn(**_kwargs):
            return {
                "p_map_used": 0.62,
                "p_series_used": 0.40,
                "trace_marginal_trust_info": {"applied": True, "league": "LCK"},
                "pipeline_step": {
                    "stage": "marginal_trust_guard",
                    "before": 0.58,
                    "after": 0.62,
                    "applied": True,
                },
            }

        def _coh_fn(**_kwargs):
            return {
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
            }

        out = run_ml_post_layers(
            p_map_used=0.58,
            p_series_used=0.61,
            bo=5,
            trace_shadow_state_alite={},
            trace_shadow_state_areal={},
            feature_flag_enabled_fn=_ff,
            prob_win_series_fn=lambda p, _bo: float(p),
            session_state={},
            trace_pipeline_steps=[],
            live_kwargs={"fit_params_path": Path("params_fitted.__missing__.json")},
            marginal_kwargs={
                "ml_info": {"blue": {"league": "LCK"}},
                "trace_n_team_eff": 30.0,
                "lv_last_report": {},
                "ml_trace_last": {},
                "ml_confidence_fn": lambda *_a, **_k: {"score": 70.0},
                "marginal_adjust_fn": lambda p, *_a, **_k: (float(p), {"applied": False}),
            },
            confidence_baseline_path=Path("baseline.__missing__.json"),
            load_confidence_baseline_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            lg_a_hint="LCK",
            caption_fn=None,
            run_live_layers_fn=_live_fn,
            run_marginal_trust_guard_fn=_marginal_fn,
            enforce_series_economic_coherence_fn=_coh_fn,
        )
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.62, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.67, places=6)
        tr = (
            out.get("trace_series_coherence_post_info")
            if isinstance(out.get("trace_series_coherence_post_info"), dict)
            else {}
        )
        self.assertTrue(bool(tr.get("applied", False)))
        stages = [str(x.get("stage", "")) for x in list(out.get("trace_pipeline_steps", [])) if isinstance(x, dict)]
        self.assertIn("series_economic_coherence_post", stages)


if __name__ == "__main__":
    unittest.main()
