import unittest

import pandas as pd

from app_core.ml_pre_live_pipeline_core import run_ml_pre_live_pipeline


def _clip_prob(p):
    x = float(p)
    return max(1e-6, min(1.0 - 1e-6, x))


class TestMlPreLivePipelineCore(unittest.TestCase):
    def test_runs_stages_and_builds_trace_outputs(self):
        session = {
            "ml_form_use_fitted": False,
            "ml_hybrid_blend_enabled": True,
            "ml_conf_guard_enabled": True,
            "ml_conf_guard_min_games": 20.0,
            "ml_conf_guard_max_shrink": 0.35,
            "ml_patch_guard_enabled": True,
            "ml_patch_guard_warn_shrink": 0.08,
            "ml_patch_guard_refit_shrink": 0.16,
            "ml_patch_guard_focus_bonus": 0.06,
            "ml_consistency_guard": True,
            "ml_consistency_tol_pp": 99.0,
            "ml_series_gamma_use_fitted": True,
            "_fit_series_bo_dplite_cfg": {},
        }
        out = run_ml_pre_live_pipeline(
            p_map_used=0.60,
            p_series_used=0.65,
            bo=3,
            season_label="2026",
            teamA="A",
            teamB="B",
            ml_engine="mlcore_v2",
            year_opt="2026",
            split_opt="S1",
            playoffs_opt="0",
            league_mode="fixed",
            fixed_league="LCK",
            as_of_date="2026-01-01",
            artifact_path="artifact.json",
            lineup_mode="ML players layer",
            lineupA_sel={"Top": "a"},
            lineupB_sel={"Top": "b"},
            diff_used=10.0,
            delta_players=0.1,
            trace_n_team_eff=12.0,
            trace_n_players_eff=7.0,
            trace_patch_guard_info={},
            trace_style_b_lite_live={"style_volatility": 0.2},
            ml_conf_score_snapshot=70.0,
            lg_a_shadow="",
            lg_b_shadow="",
            csv_path="does_not_exist.csv",
            fit_params_path="does_not_exist_fit.json",
            filters=type("F", (), {"league": "LCK"})(),
            dfA_used=None,
            dfB_used=None,
            session_state=session,
            trace_pipeline_steps=[],
            season_form_blend_fn=lambda **_k: {
                "applied": True,
                "p_blended": 0.62,
                "p_model": 0.60,
                "p_form": 0.64,
                "w_form": 0.2,
                "n_a": 10,
                "n_b": 10,
            },
            current_season_override_fn=lambda **_k: {"applied": False},
            hybrid_momentum_blend_fn=lambda **_k: {
                "applied": True,
                "enabled": True,
                "p_blended": 0.63,
                "p_model": 0.62,
                "p_hybrid": 0.64,
            },
            clip_prob_fn=_clip_prob,
            ml_consistency_signature_fn=lambda **_k: 777,
            feature_flag_enabled_fn=lambda key, default=False: (True if key == "ff_ml_series_bo_dplite_live" else bool(default)),
            apply_series_bo_dplite_live_fn=lambda **_k: {"applied": True, "p_after": 0.71, "method": "series_bo_dplite"},
            enforce_series_economic_coherence_fn=lambda **_k: {"applied": True, "p_series_after": 0.68, "reason": "coherence"},
            series_gamma_for_league_fn=lambda _lg, _bo: (1.10, "global"),
            apply_series_gamma_fn=lambda p, g: float(p) * float(g),
            patch_meta_guard_payload_fn=lambda *_k: {"status": "warn", "changed_leagues": [], "age_days": 3},
            patch_changed_leagues_set_fn=lambda leagues: set(leagues),
            file_sig_fn=lambda _p: ("", 0, 0),
            prob_win_series_fn=lambda p, _bo: float(p) + 0.01,
            fit_form_for_league_fn=lambda _lg: ({}, ""),
            caption_fn=None,
        )

        self.assertTrue(isinstance(out, dict))
        self.assertTrue(bool(out.get("_ff_ml_series_bo_dplite_live", False)))
        self.assertTrue(bool((out.get("trace_series_bo_dplite_live") or {}).get("applied", False)))
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.68, places=6)
        stages = [str((s or {}).get("stage", "")) for s in list(out.get("trace_pipeline_steps", []))]
        self.assertIn("season_form_blend", stages)
        self.assertIn("hybrid_blend", stages)
        self.assertIn("confidence_guard", stages)
        self.assertIn("patch_meta_guard", stages)
        self.assertIn("consistency_guard", stages)
        self.assertIn("series_gamma", stages)

    def test_consistency_guard_uses_previous_snapshot(self):
        session = {
            "ml_form_use_fitted": False,
            "ml_hybrid_blend_enabled": False,
            "ml_conf_guard_enabled": False,
            "ml_patch_guard_enabled": False,
            "ml_consistency_guard": True,
            "ml_consistency_tol_pp": 0.01,
            "ml_series_gamma_use_fitted": False,
            "_ml_consistency_prev": {
                "sig": 123,
                "p_map": 0.55,
                "p_series": 0.60,
                "diff_used": 8.0,
                "delta_players": 0.3,
            },
        }
        out = run_ml_pre_live_pipeline(
            p_map_used=0.70,
            p_series_used=0.75,
            bo=1,
            season_label="2026",
            teamA="A",
            teamB="B",
            ml_engine="mlcore_v2",
            year_opt="2026",
            split_opt="S1",
            playoffs_opt="0",
            league_mode="fixed",
            fixed_league="LCK",
            as_of_date="2026-01-01",
            artifact_path="artifact.json",
            lineup_mode="ML players layer",
            lineupA_sel={},
            lineupB_sel={},
            diff_used=10.0,
            delta_players=0.5,
            trace_n_team_eff=10.0,
            trace_n_players_eff=5.0,
            trace_patch_guard_info={},
            trace_style_b_lite_live={},
            ml_conf_score_snapshot=None,
            lg_a_shadow="",
            lg_b_shadow="",
            csv_path="none.csv",
            fit_params_path="none.json",
            filters=type("F", (), {"league": "LCK"})(),
            dfA_used=None,
            dfB_used=None,
            session_state=session,
            trace_pipeline_steps=[],
            season_form_blend_fn=lambda **_k: {"applied": False},
            current_season_override_fn=lambda **_k: {"applied": False},
            hybrid_momentum_blend_fn=lambda **_k: {"applied": False, "enabled": False},
            clip_prob_fn=_clip_prob,
            ml_consistency_signature_fn=lambda **_k: 123,
            feature_flag_enabled_fn=lambda _k, default=False: bool(default),
            apply_series_bo_dplite_live_fn=lambda **_k: {"applied": False},
            enforce_series_economic_coherence_fn=lambda **_k: {"applied": False},
            series_gamma_for_league_fn=lambda _lg, _bo: (1.0, "global"),
            apply_series_gamma_fn=lambda p, _g: float(p),
            patch_meta_guard_payload_fn=lambda *_k: {"status": "off", "changed_leagues": [], "age_days": 0},
            patch_changed_leagues_set_fn=lambda leagues: set(leagues),
            file_sig_fn=lambda _p: ("", 0, 0),
            prob_win_series_fn=lambda p, _bo: float(p),
            fit_form_for_league_fn=lambda _lg: ({}, ""),
            caption_fn=None,
        )
        self.assertTrue(bool(out.get("_guard_triggered", False)))
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.55, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.60, places=6)
        self.assertAlmostEqual(float(out.get("diff_used")), 8.0, places=6)

    def test_patch_guard_normalizes_accented_status(self):
        session = {
            "ml_form_use_fitted": False,
            "ml_hybrid_blend_enabled": False,
            "ml_conf_guard_enabled": False,
            "ml_patch_guard_enabled": True,
            "ml_patch_guard_warn_shrink": 0.10,
            "ml_patch_guard_refit_shrink": 0.20,
            "ml_patch_guard_focus_bonus": 0.0,
            "ml_consistency_guard": False,
            "ml_series_gamma_use_fitted": False,
        }
        out = run_ml_pre_live_pipeline(
            p_map_used=0.60,
            p_series_used=0.60,
            bo=1,
            season_label="2026",
            teamA="A",
            teamB="B",
            ml_engine="mlcore_v2",
            year_opt="2026",
            split_opt="S1",
            playoffs_opt="0",
            league_mode="fixed",
            fixed_league="LCK",
            as_of_date="2026-01-01",
            artifact_path="artifact.json",
            lineup_mode="off",
            lineupA_sel={},
            lineupB_sel={},
            diff_used=10.0,
            delta_players=0.0,
            trace_n_team_eff=10.0,
            trace_n_players_eff=5.0,
            trace_patch_guard_info={},
            trace_style_b_lite_live={},
            ml_conf_score_snapshot=None,
            lg_a_shadow="",
            lg_b_shadow="",
            csv_path="none.csv",
            fit_params_path="none.json",
            filters=type("F", (), {"league": "LCK"})(),
            dfA_used=None,
            dfB_used=None,
            session_state=session,
            trace_pipeline_steps=[],
            season_form_blend_fn=lambda **_k: {"applied": False},
            current_season_override_fn=lambda **_k: {"applied": False},
            hybrid_momentum_blend_fn=lambda **_k: {"applied": False, "enabled": False},
            clip_prob_fn=_clip_prob,
            ml_consistency_signature_fn=lambda **_k: 123,
            feature_flag_enabled_fn=lambda _k, default=False: bool(default),
            apply_series_bo_dplite_live_fn=lambda **_k: {"applied": False},
            enforce_series_economic_coherence_fn=lambda **_k: {"applied": False},
            series_gamma_for_league_fn=lambda _lg, _bo: (1.0, "global"),
            apply_series_gamma_fn=lambda p, _g: float(p),
            patch_meta_guard_payload_fn=lambda *_k: {"status": "ATEN\u00c7\u00c3O", "changed_leagues": [], "age_days": 0},
            patch_changed_leagues_set_fn=lambda leagues: set(leagues),
            file_sig_fn=lambda _p: ("", 0, 0),
            prob_win_series_fn=lambda p, _bo: float(p),
            fit_form_for_league_fn=lambda _lg: ({}, ""),
            caption_fn=None,
        )
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.59, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.59, places=6)
        self.assertEqual(str((out.get("trace_patch_guard_info") or {}).get("status", "")), "atencao")
        self.assertTrue(bool((out.get("trace_patch_guard_info") or {}).get("applied", False)))

    def test_interregional_pre_live_damps_form_and_hybrid(self):
        session = {
            "ml_form_use_fitted": False,
            "ml_hybrid_blend_enabled": True,
            "ml_conf_guard_enabled": False,
            "ml_patch_guard_enabled": False,
            "ml_consistency_guard": False,
            "ml_series_gamma_use_fitted": False,
            "ml_interregional_form_max_weight_mult": 0.20,
            "ml_interregional_form_knee_mult": 2.0,
            "ml_interregional_disable_form_override": True,
            "ml_interregional_hybrid_max_weight_mult": 0.25,
            "ml_interregional_hybrid_knee_mult": 2.0,
        }
        df_a = pd.DataFrame({"league": ["CBLOL"] * 8, "win": [1] * 8})
        df_b = pd.DataFrame({"league": ["LPL"] * 8, "win": [0] * 8})
        form_calls = []
        override_calls = []
        hybrid_calls = []

        def _season_form_blend_fn(**kwargs):
            form_calls.append(dict(kwargs))
            return {
                "applied": True,
                "p_model": float(kwargs["p_model"]),
                "p_form": 0.85,
                "p_blended": float(kwargs["p_model"]) + (0.30 * float(kwargs["max_weight"])),
                "w_form": float(kwargs["max_weight"]),
                "n_a": 8,
                "n_b": 8,
            }

        def _current_override_fn(**kwargs):
            override_calls.append(dict(kwargs))
            return {"applied": True, "p_out": 0.90}

        def _hybrid_fn(**kwargs):
            hybrid_calls.append(dict(kwargs))
            return {
                "applied": True,
                "enabled": True,
                "p_model": float(kwargs["p_model"]),
                "p_hybrid": 0.80,
                "p_blended": float(kwargs["p_model"]) + (0.20 * float(kwargs["max_weight"])),
            }

        out = run_ml_pre_live_pipeline(
            p_map_used=0.34,
            p_series_used=0.34,
            bo=1,
            season_label="2026",
            teamA="LOUD",
            teamB="JD Gaming",
            ml_engine="mlcore_v2",
            year_opt="2026",
            split_opt="S1",
            playoffs_opt="0",
            league_mode="auto",
            fixed_league="",
            as_of_date="2026-01-01",
            artifact_path="artifact.json",
            lineup_mode="off",
            lineupA_sel={},
            lineupB_sel={},
            diff_used=10.0,
            delta_players=0.0,
            trace_n_team_eff=30.0,
            trace_n_players_eff=10.0,
            trace_patch_guard_info={},
            trace_style_b_lite_live={},
            ml_conf_score_snapshot=None,
            lg_a_shadow="",
            lg_b_shadow="",
            csv_path="none.csv",
            fit_params_path="none.json",
            filters=type("F", (), {"league": None})(),
            dfA_used=df_a,
            dfB_used=df_b,
            session_state=session,
            trace_pipeline_steps=[],
            season_form_blend_fn=_season_form_blend_fn,
            current_season_override_fn=_current_override_fn,
            hybrid_momentum_blend_fn=_hybrid_fn,
            clip_prob_fn=_clip_prob,
            ml_consistency_signature_fn=lambda **_k: 123,
            feature_flag_enabled_fn=lambda _k, default=False: bool(default),
            apply_series_bo_dplite_live_fn=lambda **_k: {"applied": False},
            enforce_series_economic_coherence_fn=lambda **_k: {"applied": False},
            series_gamma_for_league_fn=lambda _lg, _bo: (1.0, "global"),
            apply_series_gamma_fn=lambda p, _g: float(p),
            patch_meta_guard_payload_fn=lambda *_k: {"status": "off", "changed_leagues": [], "age_days": 0},
            patch_changed_leagues_set_fn=lambda leagues: set(leagues),
            file_sig_fn=lambda _p: ("", 0, 0),
            prob_win_series_fn=lambda p, _bo: float(p),
            fit_form_for_league_fn=lambda _lg: ({}, ""),
            caption_fn=None,
        )

        self.assertEqual(len(form_calls), 1)
        self.assertEqual(len(override_calls), 0)
        self.assertEqual(len(hybrid_calls), 1)
        self.assertAlmostEqual(float(form_calls[0]["max_weight"]), 0.0825, places=6)
        self.assertAlmostEqual(float(hybrid_calls[0]["max_weight"]), 0.09625, places=6)
        self.assertTrue(bool(((out.get("trace_form_info") or {}).get("interregional_damp") or {}).get("interregional", False)))
        self.assertEqual(str((((out.get("trace_form_info") or {}).get("override") or {}).get("reason", ""))), "disabled_interregional")
        self.assertLess(float(out.get("p_map_used")), 0.50)

    def test_same_league_keeps_form_override_path_available(self):
        session = {
            "ml_form_use_fitted": False,
            "ml_hybrid_blend_enabled": False,
            "ml_conf_guard_enabled": False,
            "ml_patch_guard_enabled": False,
            "ml_consistency_guard": False,
            "ml_series_gamma_use_fitted": False,
            "ml_interregional_disable_form_override": True,
        }
        df_a = pd.DataFrame({"league": ["LCK"] * 8, "win": [1] * 8})
        df_b = pd.DataFrame({"league": ["LCK"] * 8, "win": [0] * 8})
        override_calls = []

        out = run_ml_pre_live_pipeline(
            p_map_used=0.52,
            p_series_used=0.52,
            bo=1,
            season_label="2026",
            teamA="A",
            teamB="B",
            ml_engine="mlcore_v2",
            year_opt="2026",
            split_opt="S1",
            playoffs_opt="0",
            league_mode="auto",
            fixed_league="",
            as_of_date="2026-01-01",
            artifact_path="artifact.json",
            lineup_mode="off",
            lineupA_sel={},
            lineupB_sel={},
            diff_used=10.0,
            delta_players=0.0,
            trace_n_team_eff=30.0,
            trace_n_players_eff=10.0,
            trace_patch_guard_info={},
            trace_style_b_lite_live={},
            ml_conf_score_snapshot=None,
            lg_a_shadow="",
            lg_b_shadow="",
            csv_path="none.csv",
            fit_params_path="none.json",
            filters=type("F", (), {"league": None})(),
            dfA_used=df_a,
            dfB_used=df_b,
            session_state=session,
            trace_pipeline_steps=[],
            season_form_blend_fn=lambda **kwargs: {
                "applied": True,
                "p_model": float(kwargs["p_model"]),
                "p_form": 0.70,
                "p_blended": 0.60,
                "w_form": float(kwargs["max_weight"]),
                "n_a": 8,
                "n_b": 8,
            },
            current_season_override_fn=lambda **kwargs: override_calls.append(dict(kwargs)) or {"applied": False, "reason": "test"},
            hybrid_momentum_blend_fn=lambda **_k: {"applied": False, "enabled": False},
            clip_prob_fn=_clip_prob,
            ml_consistency_signature_fn=lambda **_k: 123,
            feature_flag_enabled_fn=lambda _k, default=False: bool(default),
            apply_series_bo_dplite_live_fn=lambda **_k: {"applied": False},
            enforce_series_economic_coherence_fn=lambda **_k: {"applied": False},
            series_gamma_for_league_fn=lambda _lg, _bo: (1.0, "global"),
            apply_series_gamma_fn=lambda p, _g: float(p),
            patch_meta_guard_payload_fn=lambda *_k: {"status": "off", "changed_leagues": [], "age_days": 0},
            patch_changed_leagues_set_fn=lambda leagues: set(leagues),
            file_sig_fn=lambda _p: ("", 0, 0),
            prob_win_series_fn=lambda p, _bo: float(p),
            fit_form_for_league_fn=lambda _lg: ({}, ""),
            caption_fn=None,
        )

        self.assertEqual(len(override_calls), 1)
        self.assertFalse(bool(((out.get("trace_form_info") or {}).get("interregional_damp") or {}).get("interregional", False)))

    def test_interregional_confidence_guard_uses_players_reference(self):
        session = {
            "ml_form_use_fitted": False,
            "ml_hybrid_blend_enabled": False,
            "ml_conf_guard_enabled": True,
            "ml_conf_guard_min_games": 12.0,
            "ml_conf_guard_max_shrink": 0.28,
            "ml_interregional_conf_guard_players_eff_weight": 0.12,
            "ml_interregional_conf_guard_players_eff_cap": 18.0,
            "ml_interregional_conf_guard_max_shrink_mult": 0.35,
            "ml_patch_guard_enabled": False,
            "ml_consistency_guard": False,
            "ml_series_gamma_use_fitted": False,
        }
        df_a = pd.DataFrame({"league": ["LPL"] * 8, "win": [1] * 8})
        df_b = pd.DataFrame({"league": ["CBLOL"] * 8, "win": [0] * 8})

        out = run_ml_pre_live_pipeline(
            p_map_used=0.60,
            p_series_used=0.60,
            bo=1,
            season_label="2026",
            teamA="JD Gaming",
            teamB="LOUD",
            ml_engine="mlcore_v2",
            year_opt="2026",
            split_opt="S1",
            playoffs_opt="0",
            league_mode="auto",
            fixed_league="",
            as_of_date="2026-01-01",
            artifact_path="artifact.json",
            lineup_mode="ML players layer",
            lineupA_sel={},
            lineupB_sel={},
            diff_used=10.0,
            delta_players=0.0,
            trace_n_team_eff=4.0,
            trace_n_players_eff=119.0,
            trace_patch_guard_info={},
            trace_style_b_lite_live={},
            ml_conf_score_snapshot=None,
            lg_a_shadow="",
            lg_b_shadow="",
            csv_path="none.csv",
            fit_params_path="none.json",
            filters=type("F", (), {"league": None})(),
            dfA_used=df_a,
            dfB_used=df_b,
            session_state=session,
            trace_pipeline_steps=[],
            season_form_blend_fn=lambda **_k: {"applied": False},
            current_season_override_fn=lambda **_k: {"applied": False},
            hybrid_momentum_blend_fn=lambda **_k: {"applied": False, "enabled": False},
            clip_prob_fn=_clip_prob,
            ml_consistency_signature_fn=lambda **_k: 123,
            feature_flag_enabled_fn=lambda _k, default=False: bool(default),
            apply_series_bo_dplite_live_fn=lambda **_k: {"applied": False},
            enforce_series_economic_coherence_fn=lambda **_k: {"applied": False},
            series_gamma_for_league_fn=lambda _lg, _bo: (1.0, "global"),
            apply_series_gamma_fn=lambda p, _g: float(p),
            patch_meta_guard_payload_fn=lambda *_k: {"status": "off", "changed_leagues": [], "age_days": 0},
            patch_changed_leagues_set_fn=lambda leagues: set(leagues),
            file_sig_fn=lambda _p: ("", 0, 0),
            prob_win_series_fn=lambda p, _bo: float(p),
            fit_form_for_league_fn=lambda _lg: ({}, ""),
            caption_fn=None,
        )

        self.assertGreater(float(out.get("p_map_used")), 0.60)
        conf = out.get("trace_conf_guard_info") or {}
        damp = conf.get("interregional_damp") or {}
        self.assertAlmostEqual(float(conf.get("n_ref", 0.0)), 14.28, places=6)
        self.assertAlmostEqual(float(damp.get("players_ref", 0.0)), 14.28, places=6)
        self.assertAlmostEqual(float(damp.get("team_ref", 0.0)), 4.0, places=6)
        self.assertTrue(bool(damp.get("interregional", False)))
        self.assertFalse(bool(conf.get("applied", False)))
        self.assertAlmostEqual(float(conf.get("p_in", 0.0)), float(out.get("p_map_used")), places=6)
        self.assertAlmostEqual(float(conf.get("p_out", 0.0)), float(out.get("p_map_used")), places=6)


if __name__ == "__main__":
    unittest.main()
