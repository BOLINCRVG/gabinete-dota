import unittest

import pandas as pd

from app_core.ml_pre_engine_core import run_ml_pre_engine_layers


class TestMlPreEngineCore(unittest.TestCase):
    def test_uncertainty_and_engine_stage_are_traced(self):
        out = run_ml_pre_engine_layers(
            p_map_used=0.60,
            p_series_used=0.61,
            bo=3,
            ml_info={"blue": {"league": "LCK"}, "red": {"league": "LCK"}, "diff_points": 10.0},
            lgA="LCK",
            lineup_mode="off",
            pl_enabled=False,
            pl_players_path="players.json",
            lineupA_sel={},
            lineupB_sel={},
            as_of_date="2026-01-01",
            ml_engine="mlcore_v2",
            use_lineup=True,
            artifact_path="artifact.json",
            pl_w=1.0,
            playoffs_opt="0",
            teamA="A",
            teamB="B",
            teamA_key="A",
            teamB_key="B",
            team_games=None,
            csv_path="hist.csv",
            filters=object(),
            diff_used=10.0,
            delta_players=0.0,
            delta_players_eff=0.0,
            trace_p_map_team=0.60,
            trace_p_map_players=float("nan"),
            trace_p_map_fused=0.60,
            trace_n_team_eff=float("nan"),
            trace_n_players_eff=float("nan"),
            trace_fusion_info={},
            trace_uncertainty_info={},
            p_pipe_prev=0.60,
            session_state={
                "ml_uncertainty_enabled": True,
                "ml_uncertainty_strength": 0.55,
                "ml_uncertainty_prior_games": 22.0,
                "ml_uncertainty_max_shrink": 0.65,
                "lineup_delta_mode": "saturado",
                "lineup_delta_cap": 1.6,
                "lineup_delta_slope": 2.4,
            },
            trace_pipeline_steps=[],
            league_is_excluded_fn=lambda _lg, _raw: False,
            apply_ml_uncertainty_adjust_fn=lambda p, *_a, **_k: (
                float(p) + 0.02,
                {"enabled": True, "applied": True, "n_eff": 11.0, "shrink": 0.10, "p_out": float(p) + 0.02},
            ),
            prob_win_series_fn=lambda p, _bo: float(p),
            load_players_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            fit_players_lane_for_league_fn=lambda _lg: ({}, ""),
            fit_players_league_weight_for_league_fn=lambda _lg: (1.0, ""),
            compute_delta_and_confidence_fn=lambda *_a, **_k: {},
            delta_players_effect_fn=lambda d, **_k: float(d),
            transfer_signal_from_details_fn=lambda _d: 0.0,
            lineup_continuity_from_details_fn=lambda _d, **_k: 1.0,
            calibrate_p_from_artifact_fn=lambda _a, p: float(p),
            fuse_probs_by_precision_fn=lambda **_k: {"p_fused": 0.60},
            score_team_fn=lambda *_a, **_k: (0.0, 0, 0, 0.0),
            p_from_diff_points_fn=lambda d, **_k: float(d),
            elo_season_players_p_map_fn=lambda **_k: {},
            format_num_fn=None,
            format_pct_fn=None,
            caption_fn=None,
            warning_fn=None,
            match_league="",
        )
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.62, places=6)
        self.assertTrue(bool((out.get("trace_uncertainty_info") or {}).get("applied", False)))
        stages = [str((s or {}).get("stage", "")) for s in list(out.get("trace_pipeline_steps", []))]
        self.assertIn("uncertainty_layer", stages)
        self.assertIn("engine_lineup", stages)
        self.assertEqual(str(out.get("_delta_mode")), "saturado")

    def test_elo_engine_overrides_probability_and_diff(self):
        out = run_ml_pre_engine_layers(
            p_map_used=0.60,
            p_series_used=0.61,
            bo=3,
            ml_info={"blue": {"league": "LCK"}, "red": {"league": "LCK"}, "diff_points": 10.0},
            lgA="LCK",
            lineup_mode="off",
            pl_enabled=False,
            pl_players_path="players.json",
            lineupA_sel={},
            lineupB_sel={},
            as_of_date="2026-01-01",
            ml_engine="elo_season_players",
            use_lineup=True,
            artifact_path="artifact.json",
            pl_w=1.0,
            playoffs_opt="0",
            teamA="A",
            teamB="B",
            teamA_key="A",
            teamB_key="B",
            team_games=None,
            csv_path="hist.csv",
            filters=object(),
            diff_used=10.0,
            delta_players=0.3,
            delta_players_eff=0.2,
            trace_p_map_team=0.60,
            trace_p_map_players=float("nan"),
            trace_p_map_fused=0.60,
            trace_n_team_eff=float("nan"),
            trace_n_players_eff=float("nan"),
            trace_fusion_info={},
            trace_uncertainty_info={},
            p_pipe_prev=0.60,
            session_state={},
            trace_pipeline_steps=[],
            league_is_excluded_fn=lambda _lg, _raw: False,
            apply_ml_uncertainty_adjust_fn=lambda p, *_a, **_k: (float(p), {"enabled": True, "applied": False}),
            prob_win_series_fn=lambda p, _bo: float(p),
            load_players_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            fit_players_lane_for_league_fn=lambda _lg: ({}, ""),
            fit_players_league_weight_for_league_fn=lambda _lg: (1.0, ""),
            compute_delta_and_confidence_fn=lambda *_a, **_k: {},
            delta_players_effect_fn=lambda d, **_k: float(d),
            transfer_signal_from_details_fn=lambda _d: 0.0,
            lineup_continuity_from_details_fn=lambda _d, **_k: 1.0,
            calibrate_p_from_artifact_fn=lambda _a, p: float(p),
            fuse_probs_by_precision_fn=lambda **_k: {"p_fused": 0.60},
            score_team_fn=lambda *_a, **_k: (0.0, 0, 0, 0.0),
            p_from_diff_points_fn=lambda d, **_k: float(d),
            elo_season_players_p_map_fn=lambda **_k: {"p_cal": 0.71, "diff_points": 15.0},
            format_num_fn=None,
            format_pct_fn=None,
            caption_fn=None,
            warning_fn=None,
            match_league="",
        )
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.71, places=6)
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.71, places=6)
        self.assertAlmostEqual(float(out.get("diff_used")), 15.0, places=6)
        self.assertAlmostEqual(float(out.get("delta_players")), 0.0, places=6)
        self.assertTrue(isinstance(out.get("_elo_info"), dict))

    def test_interregional_uncertainty_is_damped_before_apply(self):
        unc_calls = []

        def _unc_fn(p, *_a, **kwargs):
            unc_calls.append(dict(kwargs))
            return float(p), {"enabled": True, "applied": False, "n_eff": 8.0, "shrink": 0.0, "p_out": float(p)}

        out = run_ml_pre_engine_layers(
            p_map_used=0.60,
            p_series_used=0.61,
            bo=3,
            ml_info={
                "blue": {"league": "LPL", "games_played": 40},
                "red": {"league": "CBLOL", "games_played": 30},
                "diff_points": 10.0,
            },
            lgA="LPL",
            lineup_mode="off",
            pl_enabled=False,
            pl_players_path="players.json",
            lineupA_sel={},
            lineupB_sel={},
            as_of_date="2026-01-01",
            ml_engine="mlcore_v2",
            use_lineup=True,
            artifact_path="artifact.json",
            pl_w=1.0,
            playoffs_opt="0",
            teamA="A",
            teamB="B",
            teamA_key="A",
            teamB_key="B",
            team_games=None,
            csv_path="hist.csv",
            filters=object(),
            diff_used=10.0,
            delta_players=0.0,
            delta_players_eff=0.0,
            trace_p_map_team=0.60,
            trace_p_map_players=float("nan"),
            trace_p_map_fused=0.60,
            trace_n_team_eff=float("nan"),
            trace_n_players_eff=float("nan"),
            trace_fusion_info={},
            trace_uncertainty_info={},
            p_pipe_prev=0.60,
            session_state={
                "ml_uncertainty_enabled": True,
                "ml_uncertainty_strength": 0.55,
                "ml_uncertainty_prior_games": 22.0,
                "ml_uncertainty_max_shrink": 0.65,
                "ml_interregional_uncertainty_strength_mult": 0.30,
                "ml_interregional_uncertainty_prior_mult": 0.60,
                "ml_interregional_uncertainty_cap_mult": 0.40,
                "lineup_delta_mode": "saturado",
                "lineup_delta_cap": 1.6,
                "lineup_delta_slope": 2.4,
            },
            trace_pipeline_steps=[],
            league_is_excluded_fn=lambda _lg, _raw: False,
            apply_ml_uncertainty_adjust_fn=_unc_fn,
            prob_win_series_fn=lambda p, _bo: float(p),
            load_players_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            fit_players_lane_for_league_fn=lambda _lg: ({}, ""),
            fit_players_league_weight_for_league_fn=lambda _lg: (1.0, ""),
            compute_delta_and_confidence_fn=lambda *_a, **_k: {},
            delta_players_effect_fn=lambda d, **_k: float(d),
            transfer_signal_from_details_fn=lambda _d: 0.0,
            lineup_continuity_from_details_fn=lambda _d, **_k: 1.0,
            calibrate_p_from_artifact_fn=lambda _a, p: float(p),
            fuse_probs_by_precision_fn=lambda **_k: {"p_fused": 0.60},
            score_team_fn=lambda *_a, **_k: (0.0, 0, 0, 0.0),
            p_from_diff_points_fn=lambda d, **_k: float(d),
            elo_season_players_p_map_fn=lambda **_k: {},
            format_num_fn=None,
            format_pct_fn=None,
            caption_fn=None,
            warning_fn=None,
            match_league="",
        )

        self.assertEqual(len(unc_calls), 1)
        self.assertAlmostEqual(float(unc_calls[0]["strength"]), 0.165, places=6)
        self.assertAlmostEqual(float(unc_calls[0]["prior_games"]), 13.2, places=6)
        self.assertAlmostEqual(float(unc_calls[0]["max_shrink"]), 0.26, places=6)
        damp = (out.get("trace_uncertainty_info") or {}).get("interregional_damp") or {}
        self.assertTrue(bool(damp.get("interregional", False)))
        self.assertAlmostEqual(float(damp.get("strength_mult", 0.0)), 0.30, places=6)

    def test_manual_players_layer_applies_reliability_to_delta_and_fusion_trace(self):
        def _score_team(team_key, _lineup):
            if str(team_key) == "A":
                return (1.0, 5, 5, 50.0)
            return (0.0, 5, 5, 50.0)

        out = run_ml_pre_engine_layers(
            p_map_used=0.55,
            p_series_used=0.59,
            bo=3,
            ml_info={
                "blue": {"league": "LCK", "games_played": 30},
                "red": {"league": "LPL", "games_played": 28},
                "diff_points": 10.0,
                "scale_points": 160.0,
            },
            lgA="LCK",
            lineup_mode="CSV byTournament (manual)",
            pl_enabled=False,
            pl_players_path="players.json",
            lineupA_sel={},
            lineupB_sel={},
            as_of_date="2026-01-01",
            ml_engine="mlcore_v2",
            use_lineup=True,
            artifact_path="artifact.json",
            pl_w=1.0,
            playoffs_opt="0",
            teamA="A",
            teamB="B",
            teamA_key="A",
            teamB_key="B",
            team_games=None,
            csv_path="hist.csv",
            filters=object(),
            diff_used=10.0,
            delta_players=0.0,
            delta_players_eff=0.0,
            trace_p_map_team=0.55,
            trace_p_map_players=float("nan"),
            trace_p_map_fused=0.55,
            trace_n_team_eff=float("nan"),
            trace_n_players_eff=float("nan"),
            trace_fusion_info={},
            trace_uncertainty_info={},
            p_pipe_prev=0.55,
            session_state={
                "lineup_delta_mode": "saturado",
                "lineup_delta_cap": 1.6,
                "lineup_delta_slope": 2.4,
                "lineup_lambda_points": 100.0,
                "ml_uncertainty_prior_games": 20.0,
                "ml_players_manual_freshness_gp_ref": 10.0,
                "ml_players_reliability_inter_mult": 0.90,
            },
            trace_pipeline_steps=[],
            league_is_excluded_fn=lambda _lg, _raw: False,
            apply_ml_uncertainty_adjust_fn=lambda p, *_a, **_k: (float(p), {"enabled": True, "applied": False}),
            prob_win_series_fn=lambda p, _bo: float(p),
            load_players_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            fit_players_lane_for_league_fn=lambda _lg: ({}, ""),
            fit_players_league_weight_for_league_fn=lambda _lg: (1.0, ""),
            compute_delta_and_confidence_fn=lambda *_a, **_k: {},
            delta_players_effect_fn=lambda d, **_k: float(d),
            transfer_signal_from_details_fn=lambda _d: 0.0,
            lineup_continuity_from_details_fn=lambda _d, **_k: 1.0,
            calibrate_p_from_artifact_fn=lambda _a, p: float(p),
            fuse_probs_by_precision_fn=lambda **_k: {"p_fused": float(_k.get("p_players", 0.5))},
            score_team_fn=_score_team,
            p_from_diff_points_fn=lambda d, **_k: float(min(0.95, max(0.05, 0.5 + (float(d) / 1000.0)))),
            elo_season_players_p_map_fn=lambda **_k: {},
            format_num_fn=None,
            format_pct_fn=None,
            caption_fn=None,
            warning_fn=None,
            match_league="",
        )

        # expected reliability:
        # cov=1.0, freshness=1.0, continuity=1.0 -> base=1.0
        # evidence=10/(10+20)=0.333333 -> rel_pre_inter=0.8
        # interregional multiplier 0.90 -> rel=0.72
        self.assertAlmostEqual(float(out.get("delta_players_eff")), 0.72, places=6)
        self.assertAlmostEqual(float(out.get("diff_used")), 82.0, places=6)
        fus = out.get("trace_fusion_info") or {}
        self.assertAlmostEqual(float(fus.get("players_reliability", 0.0)), 0.72, places=6)
        self.assertTrue(bool(((fus.get("players_reliability_components") or {}).get("interregional", False))))

    def test_manual_players_layer_treats_americas_label_as_interregional(self):
        def _score_team(team_key, _lineup):
            if str(team_key) == "A":
                return (1.0, 5, 5, 50.0)
            return (0.0, 5, 5, 50.0)

        out = run_ml_pre_engine_layers(
            p_map_used=0.55,
            p_series_used=0.59,
            bo=3,
            ml_info={
                "blue": {"league": "Americas", "games_played": 30},
                "red": {"league": "Cross Conference", "games_played": 28},
                "diff_points": 10.0,
                "scale_points": 160.0,
            },
            lgA="LCS",
            lineup_mode="CSV byTournament (manual)",
            pl_enabled=False,
            pl_players_path="players.json",
            lineupA_sel={},
            lineupB_sel={},
            as_of_date="2026-01-01",
            ml_engine="mlcore_v2",
            use_lineup=True,
            artifact_path="artifact.json",
            pl_w=1.0,
            playoffs_opt="0",
            teamA="A",
            teamB="B",
            teamA_key="A",
            teamB_key="B",
            team_games=None,
            csv_path="hist.csv",
            filters=object(),
            diff_used=10.0,
            delta_players=0.0,
            delta_players_eff=0.0,
            trace_p_map_team=0.55,
            trace_p_map_players=float("nan"),
            trace_p_map_fused=0.55,
            trace_n_team_eff=float("nan"),
            trace_n_players_eff=float("nan"),
            trace_fusion_info={},
            trace_uncertainty_info={},
            p_pipe_prev=0.55,
            session_state={
                "lineup_delta_mode": "saturado",
                "lineup_delta_cap": 1.6,
                "lineup_delta_slope": 2.4,
                "lineup_lambda_points": 100.0,
                "ml_uncertainty_prior_games": 20.0,
                "ml_players_manual_freshness_gp_ref": 10.0,
                "ml_players_reliability_inter_mult": 0.90,
            },
            trace_pipeline_steps=[],
            league_is_excluded_fn=lambda _lg, _raw: False,
            apply_ml_uncertainty_adjust_fn=lambda p, *_a, **_k: (float(p), {"enabled": True, "applied": False}),
            prob_win_series_fn=lambda p, _bo: float(p),
            load_players_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            fit_players_lane_for_league_fn=lambda _lg: ({}, ""),
            fit_players_league_weight_for_league_fn=lambda _lg: (1.0, ""),
            compute_delta_and_confidence_fn=lambda *_a, **_k: {},
            delta_players_effect_fn=lambda d, **_k: float(d),
            transfer_signal_from_details_fn=lambda _d: 0.0,
            lineup_continuity_from_details_fn=lambda _d, **_k: 1.0,
            calibrate_p_from_artifact_fn=lambda _a, p: float(p),
            fuse_probs_by_precision_fn=lambda **_k: {"p_fused": float(_k.get("p_players", 0.5))},
            score_team_fn=_score_team,
            p_from_diff_points_fn=lambda d, **_k: float(min(0.95, max(0.05, 0.5 + (float(d) / 1000.0)))),
            elo_season_players_p_map_fn=lambda **_k: {},
            format_num_fn=None,
            format_pct_fn=None,
            caption_fn=None,
            warning_fn=None,
            match_league="AMERICAS",
        )

        self.assertAlmostEqual(float(out.get("delta_players_eff")), 0.72, places=6)
        fus = out.get("trace_fusion_info") or {}
        self.assertTrue(bool(((fus.get("players_reliability_components") or {}).get("interregional", False))))

    def test_manual_players_layer_adds_context_delta_when_enabled(self):
        def _score_team(_team_key, _lineup):
            return (0.0, 5, 5, 30.0)

        rows = []
        base = pd.Timestamp("2026-02-01")
        for i in range(8):
            dt = (base + pd.Timedelta(days=i)).isoformat()
            rows.append(
                {
                    "date": dt,
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "team": "A",
                    "win": 1,
                    "game_time_min": 30.0,
                    "kills_for": 18.0,
                    "kills_against": 9.0,
                    "towers_for": 10.0,
                    "towers_against": 3.0,
                    "dragons_for": 3.0,
                    "dragons_against": 1.0,
                    "barons_for": 1.0,
                    "barons_against": 0.0,
                    "inhibitors_for": 2.0,
                    "inhibitors_against": 0.0,
                    "golddiffat15_for": 1200.0,
                }
            )
            rows.append(
                {
                    "date": dt,
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "team": "B",
                    "win": 0,
                    "game_time_min": 34.0,
                    "kills_for": 9.0,
                    "kills_against": 17.0,
                    "towers_for": 3.0,
                    "towers_against": 9.0,
                    "dragons_for": 1.0,
                    "dragons_against": 3.0,
                    "barons_for": 0.0,
                    "barons_against": 1.0,
                    "inhibitors_for": 0.0,
                    "inhibitors_against": 2.0,
                    "golddiffat15_for": -1000.0,
                }
            )
        tg = pd.DataFrame(rows)

        base_kwargs = dict(
            p_map_used=0.55,
            p_series_used=0.59,
            bo=3,
            ml_info={
                "blue": {"league": "LCK", "games_played": 30},
                "red": {"league": "LCK", "games_played": 28},
                "diff_points": 10.0,
                "scale_points": 160.0,
            },
            lgA="LCK",
            lineup_mode="CSV byTournament (manual)",
            pl_enabled=False,
            pl_players_path="players.json",
            lineupA_sel={},
            lineupB_sel={},
            as_of_date="2026-03-01",
            ml_engine="mlcore_v2",
            use_lineup=True,
            artifact_path="artifact.json",
            pl_w=1.0,
            playoffs_opt="0",
            teamA="A",
            teamB="B",
            teamA_key="A",
            teamB_key="B",
            team_games=tg,
            csv_path="hist.csv",
            filters={"year": 2026, "split": "Spring", "playoffs": False, "league": "LCK"},
            diff_used=10.0,
            delta_players=0.0,
            delta_players_eff=0.0,
            trace_p_map_team=0.55,
            trace_p_map_players=float("nan"),
            trace_p_map_fused=0.55,
            trace_n_team_eff=float("nan"),
            trace_n_players_eff=float("nan"),
            trace_fusion_info={},
            trace_uncertainty_info={},
            p_pipe_prev=0.55,
            trace_pipeline_steps=[],
            league_is_excluded_fn=lambda _lg, _raw: False,
            apply_ml_uncertainty_adjust_fn=lambda p, *_a, **_k: (float(p), {"enabled": True, "applied": False}),
            prob_win_series_fn=lambda p, _bo: float(p),
            load_players_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            fit_players_lane_for_league_fn=lambda _lg: ({}, ""),
            fit_players_league_weight_for_league_fn=lambda _lg: (1.0, ""),
            compute_delta_and_confidence_fn=lambda *_a, **_k: {},
            delta_players_effect_fn=lambda d, **_k: float(d),
            transfer_signal_from_details_fn=lambda _d: 0.0,
            lineup_continuity_from_details_fn=lambda _d, **_k: 1.0,
            calibrate_p_from_artifact_fn=lambda _a, p: float(p),
            fuse_probs_by_precision_fn=lambda **_k: {"p_fused": float(_k.get("p_players", 0.5))},
            score_team_fn=_score_team,
            p_from_diff_points_fn=lambda d, **_k: float(min(0.95, max(0.05, 0.5 + (float(d) / 1000.0)))),
            elo_season_players_p_map_fn=lambda **_k: {},
            format_num_fn=None,
            format_pct_fn=None,
            caption_fn=None,
            warning_fn=None,
            match_league="LCK",
        )

        out_off = run_ml_pre_engine_layers(
            **base_kwargs,
            session_state={
                "lineup_delta_mode": "saturado",
                "lineup_delta_cap": 1.6,
                "lineup_delta_slope": 2.4,
                "lineup_lambda_points": 100.0,
                "ml_uncertainty_prior_games": 20.0,
                "ml_players_manual_freshness_gp_ref": 10.0,
                "ml_players_context_enabled": False,
            },
        )
        out_on = run_ml_pre_engine_layers(
            **base_kwargs,
            session_state={
                "lineup_delta_mode": "saturado",
                "lineup_delta_cap": 1.6,
                "lineup_delta_slope": 2.4,
                "lineup_lambda_points": 100.0,
                "ml_uncertainty_prior_games": 20.0,
                "ml_players_manual_freshness_gp_ref": 10.0,
                "ml_players_context_enabled": True,
                "ml_players_context_lookback_games": 12,
                "ml_players_context_min_games": 6,
                "ml_players_context_strength": 0.60,
                "ml_players_context_cap": 0.30,
            },
        )

        self.assertAlmostEqual(float(out_off.get("delta_players_eff")), 0.0, places=6)
        self.assertGreater(float(out_on.get("delta_players_eff")), 0.01)
        ctx = out_on.get("trace_players_context_info") if isinstance(out_on.get("trace_players_context_info"), dict) else {}
        self.assertTrue(bool(ctx.get("applied", False)))
        self.assertGreater(float(ctx.get("delta", 0.0)), 0.0)

    def test_ml_players_layer_adds_context_delta_when_enabled(self):
        rows = []
        base = pd.Timestamp("2026-02-01")
        for i in range(8):
            dt = (base + pd.Timedelta(days=i)).isoformat()
            rows.append(
                {
                    "date": dt,
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "team": "A",
                    "win": 1,
                    "game_time_min": 30.0,
                    "kills_for": 18.0,
                    "kills_against": 9.0,
                    "towers_for": 10.0,
                    "towers_against": 3.0,
                    "dragons_for": 3.0,
                    "dragons_against": 1.0,
                    "barons_for": 1.0,
                    "barons_against": 0.0,
                    "inhibitors_for": 2.0,
                    "inhibitors_against": 0.0,
                    "golddiffat15_for": 1200.0,
                }
            )
            rows.append(
                {
                    "date": dt,
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "team": "B",
                    "win": 0,
                    "game_time_min": 34.0,
                    "kills_for": 9.0,
                    "kills_against": 17.0,
                    "towers_for": 3.0,
                    "towers_against": 9.0,
                    "dragons_for": 1.0,
                    "dragons_against": 3.0,
                    "barons_for": 0.0,
                    "barons_against": 1.0,
                    "inhibitors_for": 0.0,
                    "inhibitors_against": 2.0,
                    "golddiffat15_for": -1000.0,
                }
            )
        tg = pd.DataFrame(rows)

        base_kwargs = dict(
            p_map_used=0.55,
            p_series_used=0.59,
            bo=3,
            ml_info={
                "blue": {"league": "LCK", "games_played": 30},
                "red": {"league": "LCK", "games_played": 28},
                "p_raw": 0.50,
                "scale_points": 160.0,
            },
            lgA="LCK",
            lineup_mode="ML players layer",
            pl_enabled=True,
            pl_players_path="players.json",
            lineupA_sel={},
            lineupB_sel={},
            as_of_date="2026-03-01",
            ml_engine="mlcore_v2",
            use_lineup=True,
            artifact_path="artifact.json",
            pl_w=1.0,
            playoffs_opt="0",
            teamA="A",
            teamB="B",
            teamA_key="A",
            teamB_key="B",
            team_games=tg,
            csv_path="hist.csv",
            filters={"year": 2026, "split": "Spring", "playoffs": False, "league": "LCK"},
            diff_used=10.0,
            delta_players=0.0,
            delta_players_eff=0.0,
            trace_p_map_team=0.55,
            trace_p_map_players=float("nan"),
            trace_p_map_fused=0.55,
            trace_n_team_eff=float("nan"),
            trace_n_players_eff=float("nan"),
            trace_fusion_info={},
            trace_uncertainty_info={},
            p_pipe_prev=0.55,
            trace_pipeline_steps=[],
            league_is_excluded_fn=lambda _lg, _raw: False,
            apply_ml_uncertainty_adjust_fn=lambda p, *_a, **_k: (float(p), {"enabled": True, "applied": False}),
            prob_win_series_fn=lambda p, _bo: float(p),
            load_players_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            fit_players_lane_for_league_fn=lambda _lg: ({}, ""),
            fit_players_league_weight_for_league_fn=lambda _lg: (1.0, ""),
            compute_delta_and_confidence_fn=lambda *_a, **_k: {
                "delta": 0.0,
                "coverage": 5.0,
                "coverage_total": 5.0,
                "freshness_avg": 1.0,
                "avg_gp_min": 30.0,
                "details": [],
                "league_weight": 1.0,
            },
            delta_players_effect_fn=lambda d, **_k: float(d),
            transfer_signal_from_details_fn=lambda _d: 0.0,
            lineup_continuity_from_details_fn=lambda _d, **_k: 1.0,
            calibrate_p_from_artifact_fn=lambda _a, p: float(p),
            fuse_probs_by_precision_fn=lambda **_k: {
                "p_fused": float(_k.get("p_players", 0.5)),
                "w_team": 0.0,
                "w_players": 1.0,
            },
            score_team_fn=lambda *_a, **_k: (0.0, 0, 0, 0.0),
            p_from_diff_points_fn=lambda d, **_k: float(d),
            elo_season_players_p_map_fn=lambda **_k: {},
            format_num_fn=None,
            format_pct_fn=None,
            caption_fn=None,
            warning_fn=None,
            match_league="LCK",
        )

        out_off = run_ml_pre_engine_layers(
            **base_kwargs,
            session_state={
                "lineup_delta_mode": "saturado",
                "lineup_delta_cap": 1.6,
                "lineup_delta_slope": 2.4,
                "ml_uncertainty_prior_games": 20.0,
                "ml_players_context_enabled": False,
            },
        )
        out_on = run_ml_pre_engine_layers(
            **base_kwargs,
            session_state={
                "lineup_delta_mode": "saturado",
                "lineup_delta_cap": 1.6,
                "lineup_delta_slope": 2.4,
                "ml_uncertainty_prior_games": 20.0,
                "ml_players_context_enabled": True,
                "ml_players_context_lookback_games": 12,
                "ml_players_context_min_games": 6,
                "ml_players_context_strength": 0.60,
                "ml_players_context_cap": 0.30,
            },
        )

        self.assertAlmostEqual(float(out_off.get("delta_players_eff")), 0.0, places=6)
        self.assertGreater(float(out_on.get("delta_players_eff")), 0.01)
        self.assertGreater(float(out_on.get("p_map_used")), float(out_off.get("p_map_used")))
        ctx = out_on.get("trace_players_context_info") if isinstance(out_on.get("trace_players_context_info"), dict) else {}
        self.assertTrue(bool(ctx.get("applied", False)))
        self.assertGreater(float(ctx.get("delta", 0.0)), 0.0)

    def test_ml_players_layer_passes_side_specific_leagues_on_cross_matchup(self):
        captured = {}

        def _compute_delta(*_a, **kwargs):
            captured.update(kwargs)
            return {
                "delta": 0.0,
                "coverage": 5.0,
                "coverage_total": 5.0,
                "freshness_avg": 1.0,
                "avg_gp_min": 30.0,
                "details": [],
                "league_weight": 1.0,
            }

        out = run_ml_pre_engine_layers(
            p_map_used=0.55,
            p_series_used=0.59,
            bo=3,
            ml_info={
                "blue": {"league": "LCK", "games_played": 30},
                "red": {"league": "LPL", "games_played": 28},
                "p_raw": 0.50,
                "scale_points": 160.0,
            },
            lgA="LCK",
            lineup_mode="ML players layer",
            pl_enabled=True,
            pl_players_path="players.json",
            lineupA_sel={},
            lineupB_sel={},
            as_of_date="2026-03-01",
            ml_engine="mlcore_v2",
            use_lineup=True,
            artifact_path="artifact.json",
            pl_w=1.0,
            playoffs_opt="0",
            teamA="A",
            teamB="B",
            teamA_key="A",
            teamB_key="B",
            team_games=None,
            csv_path="hist.csv",
            filters={"year": 2026, "split": "Spring", "playoffs": False, "league": "INTER"},
            diff_used=10.0,
            delta_players=0.0,
            delta_players_eff=0.0,
            trace_p_map_team=0.55,
            trace_p_map_players=float("nan"),
            trace_p_map_fused=0.55,
            trace_n_team_eff=float("nan"),
            trace_n_players_eff=float("nan"),
            trace_fusion_info={},
            trace_uncertainty_info={},
            p_pipe_prev=0.55,
            session_state={
                "lineup_delta_mode": "saturado",
                "lineup_delta_cap": 1.6,
                "lineup_delta_slope": 2.4,
                "ml_uncertainty_prior_games": 20.0,
                "ml_players_context_enabled": False,
            },
            trace_pipeline_steps=[],
            league_is_excluded_fn=lambda _lg, _raw: False,
            apply_ml_uncertainty_adjust_fn=lambda p, *_a, **_k: (float(p), {"enabled": True, "applied": False}),
            prob_win_series_fn=lambda p, _bo: float(p),
            load_players_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            fit_players_lane_for_league_fn=lambda _lg: ({}, ""),
            fit_players_league_weight_for_league_fn=lambda _lg: (1.0, ""),
            compute_delta_and_confidence_fn=_compute_delta,
            delta_players_effect_fn=lambda d, **_k: float(d),
            transfer_signal_from_details_fn=lambda _d: 0.0,
            lineup_continuity_from_details_fn=lambda _d, **_k: 1.0,
            calibrate_p_from_artifact_fn=lambda _a, p: float(p),
            fuse_probs_by_precision_fn=lambda **_k: {
                "p_fused": float(_k.get("p_players", 0.5)),
                "w_team": 0.0,
                "w_players": 1.0,
            },
            score_team_fn=lambda *_a, **_k: (0.0, 0, 0, 0.0),
            p_from_diff_points_fn=lambda d, **_k: float(d),
            elo_season_players_p_map_fn=lambda **_k: {},
            format_num_fn=None,
            format_pct_fn=None,
            caption_fn=None,
            warning_fn=None,
            match_league="LCK",
        )
        self.assertEqual(str(captured.get("match_league", "")), "")
        self.assertEqual(str(captured.get("blue_league", "")), "LCK")
        self.assertEqual(str(captured.get("red_league", "")), "LPL")
        self.assertTrue(isinstance(out, dict))


if __name__ == "__main__":
    unittest.main()
