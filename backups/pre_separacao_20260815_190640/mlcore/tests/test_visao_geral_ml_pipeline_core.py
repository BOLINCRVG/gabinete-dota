import tempfile
import unittest
from pathlib import Path

from app_core.visao_geral_ml_pipeline_core import (
    build_ml_trace_context,
    build_pre_engine_input,
    build_pre_live_input,
    build_runtime_input_signature,
    compute_team_games_min_used,
    persist_runtime_analysis_state,
    unpack_post_layer_output,
    unpack_shadow_output,
)


class VisaoGeralMlPipelineCoreTests(unittest.TestCase):
    def test_compute_and_persist_runtime_analysis_state(self):
        state = {}
        team_games_min = persist_runtime_analysis_state(
            session_state=state,
            p_map_used=0.61,
            p_series_used=0.66,
            diff_used=12.5,
            delta_players=3.2,
            delta_players_eff=1.4,
            ml_info={"blue": {"games_played": 20}, "red": {"games_played": 14}},
        )
        self.assertEqual(team_games_min, 14.0)
        self.assertEqual(state["_ml_team_games_min"], 14.0)
        self.assertEqual(state["_p_map_used"], 0.61)
        self.assertEqual(compute_team_games_min_used({"blue": {"games_played": 5}, "red": {"games_played": 9}}), 5.0)

    def test_build_runtime_input_signature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "daily.csv"
            artifact_path = root / "ml_artifact.json"
            params_path = root / "params_fitted.json"
            csv_path.write_text("x", encoding="utf-8")
            artifact_path.write_text("{}", encoding="utf-8")
            params_path.write_text("{}", encoding="utf-8")

            sig = build_runtime_input_signature(
                session_state={"ml_microseg_enabled": True},
                season_label="2026 Spring",
                teamA="A",
                teamB="B",
                bo=3,
                year_opt="2026",
                split_opt="Spring",
                playoffs_opt="false",
                league_mode="auto",
                fixed_league="LCK",
                weight_mode="Recência",
                half_life_days=180,
                ml_engine="mlcore_v2",
                totals_mode="map",
                ml_mode="map",
                map_mode_ui="avg",
                n_sims=20000,
                lineup_mode="manual",
                lineup_sig_a="sig-a",
                lineup_sig_b="sig-b",
                csv_path=str(csv_path),
                artifact_path=str(artifact_path),
                fit_params_path=str(params_path),
                ml_info={"artifact_version": "v1"},
                file_sig_fn=lambda p: (str(p), Path(p).stat().st_size, int(Path(p).stat().st_mtime)),
            )
            self.assertEqual(len(sig["sha256"]), 64)
            self.assertEqual(sig["payload"]["versions"]["artifact_version"], "v1")

    def test_build_pre_pipeline_inputs(self):
        pre_engine = build_pre_engine_input(
            p_map_used=0.55,
            p_series_used=0.60,
            bo=3,
            ml_info={"blue": {"league": "LCK"}},
            lgA="LCK",
            lineup_mode="manual",
            pl_enabled=True,
            pl_players_path="players.csv",
            lineupA_sel={"top": "a"},
            lineupB_sel={"top": "b"},
            as_of_date="2026-03-10",
            ml_engine="mlcore_v2",
            use_lineup=True,
            artifact_path="ml_artifact.json",
            pl_w=1.15,
            playoffs_opt="false",
            teamA="A",
            teamB="B",
            team_aliases_a=["A"],
            team_aliases_b=["B"],
            teamA_key="A",
            teamB_key="B",
            team_games=[],
            csv_path="daily.csv",
            filters={},
            diff_used=10.0,
            delta_players=0.0,
            delta_players_eff=0.0,
            trace_p_map_team=0.55,
            trace_p_map_players=0.56,
            trace_p_map_fused=0.555,
            trace_n_team_eff=100.0,
            trace_n_players_eff=12.0,
            trace_fusion_info={},
            trace_uncertainty_info={},
            league_transfer_features={},
            p_pipe_prev=0.55,
            session_state={},
            trace_pipeline_steps=[],
            callbacks={"league_is_excluded_fn": lambda *a, **k: False},
            format_num_fn=None,
            format_pct_fn=None,
            caption_fn=None,
            warning_fn=None,
            match_league="LCK",
        )
        self.assertEqual(pre_engine.teamA, "A")

        pre_live = build_pre_live_input(
            p_map_used=0.55,
            p_series_used=0.60,
            bo=3,
            season_label="2026 Spring",
            teamA="A",
            teamB="B",
            ml_engine="mlcore_v2",
            year_opt="2026",
            split_opt="Spring",
            playoffs_opt="false",
            league_mode="auto",
            fixed_league="LCK",
            as_of_date="2026-03-10",
            artifact_path="ml_artifact.json",
            lineup_mode="manual",
            lineupA_sel={"top": "a"},
            lineupB_sel={"top": "b"},
            diff_used=10.0,
            delta_players=0.0,
            trace_n_team_eff=100.0,
            trace_n_players_eff=12.0,
            trace_patch_guard_info={},
            trace_style_b_lite_live={},
            ml_conf_score_snapshot=None,
            lg_a_shadow="LCK",
            lg_b_shadow="LPL",
            csv_path="daily.csv",
            fit_params_path="params_fitted.json",
            filters={},
            dfA_used=None,
            dfB_used=None,
            session_state={},
            trace_pipeline_steps=[],
            callbacks={"clip_prob_fn": lambda x: x},
            caption_fn=None,
        )
        self.assertEqual(pre_live.teamB, "B")

    def test_unpack_outputs_and_trace_context(self):
        shadow = unpack_shadow_output(
            {
                "_ff_ml_shadow_vnext": True,
                "trace_shadow_state_areal": {"applied": True},
                "_lg_a_shadow": "LCK",
            }
        )
        self.assertTrue(shadow["_ff_ml_shadow_vnext"])
        self.assertEqual(shadow["_lg_a_shadow"], "LCK")

        post = unpack_post_layer_output(
            {
                "p_map_used": 0.59,
                "p_series_used": 0.63,
                "_ff_ml_state_areal_live": True,
                "trace_state_areal_live": {"applied": True},
                "trace_pipeline_steps": [{"stage": "x"}],
            },
            default_trace_pipeline_steps=[],
        )
        self.assertTrue(post["_ff_ml_state_areal_live"])
        self.assertEqual(post["trace_pipeline_steps"][0]["stage"], "x")

        ctx = build_ml_trace_context(
            session_state={
                "_fit_team_scale": 1.0,
                "_fit_players_scale": 1.0,
                "ml_confidence_governor_enabled": True,
            },
            season_label="2026 Spring",
            teamA="A",
            teamB="B",
            bo=3,
            input_signature_hash="abc",
            input_signature_payload={"x": 1},
            year_opt="2026",
            split_opt="Spring",
            playoffs_opt="false",
            league_mode="auto",
            fixed_league="LCK",
            ml_engine="mlcore_v2",
            p_map_raw=0.57,
            p_map_cal=0.58,
            trace_p_map_team=0.58,
            trace_p_map_players=0.60,
            trace_p_map_fused=0.59,
            p_map_used=0.59,
            p_series_used=0.63,
            use_lineup=True,
            lineup_mode="manual",
            delta_players=2.0,
            delta_players_eff=1.0,
            delta_mode="saturado",
            delta_cap=1.6,
            delta_slope=2.4,
            trace_n_team_eff=100.0,
            trace_n_players_eff=15.0,
            trace_fusion_info={},
            trace_form_info={},
            trace_hybrid_info={},
            trace_conf_guard_info={},
            trace_uncertainty_info={},
            trace_marginal_trust_info={},
            trace_patch_guard_info={},
            trace_series_gamma_info={},
            trace_series_coherence_info={},
            trace_series_coherence_post_info={},
            trace_pipeline_order=["x"],
            trace_pipeline_steps=[{"stage": "x"}],
            guard_triggered=False,
            guard_dpp=0.0,
            wf_info={},
            fused_post_cal_info={},
            shadow_output=shadow,
            post_output=post,
            trace_series_bo_dplite_live={},
            generated_at="2026-03-10T10:00:00",
        )
        ctx_dict = ctx.to_dict()
        self.assertEqual(ctx_dict["teamA"], "A")
        self.assertEqual(ctx_dict["season_label"], "2026 Spring")
        self.assertEqual(ctx_dict["generated_at"], "2026-03-10T10:00:00")


if __name__ == "__main__":
    unittest.main()
