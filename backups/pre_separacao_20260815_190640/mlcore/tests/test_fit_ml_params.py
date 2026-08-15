import unittest
from pathlib import Path

import pandas as pd

from app_core.fusion_core import fuse_with_coherence_guard
from mlcore.tests._workspace_temp import workspace_tempdir
from mlcore.fit_ml_params import (
    Sample,
    _build_fit_run_signature,
    _path_signature,
    _read_json_file_safe,
    _load_stage_checkpoint,
    _normalize_fit_stage,
    _prime_seed_stage_checkpoints_from_params,
    _save_stage_checkpoint,
    _seed_stage_payload_from_params,
    _write_fit_progress,
    _evaluate_walkforward,
    _fit_form_grid,
    _fit_players_lane_weights,
    _fit_players_league_weights,
    _fit_players_context_tuning,
    _resolve_players_context_budget_profile,
    _blind_backtest_by_league,
    _fuse_probs_by_precision,
)


class FitMlParamsTest(unittest.TestCase):
    def _samples(self):
        # Synthetic monthly stream with strong form signal for team A.
        rows = []
        for i in range(12):
            m = "2026-01" if i < 6 else "2026-02"
            y = 1 if i in (0, 1, 2, 6, 7, 8, 9, 10) else 0
            rows.append(
                Sample(
                    month=m,
                    gameid=f"g_{i}",
                    patch="14.1",
                    league="LPL",
                    split="S1",
                    macro="CN",
                    side="blue",
                    bo_bucket="1",
                    phase="regular",
                    y=y,
                    p_team=0.52,
                    p_players=0.53,
                    n_team=8.0,
                    n_players=7.0,
                    form_n_a=10 + i,
                    form_n_b=10 + i,
                    form_wins_a=8 + (i // 2),
                    form_wins_b=2 + (i // 4),
                    same_league=True,
                    blue_team="TeamA",
                    red_team="TeamB",
                    date_day=f"2026-01-{(i % 27) + 1:02d}" if i < 6 else f"2026-02-{(i % 27) + 1:02d}",
                )
            )
        return rows

    def _team_games(self):
        rows = []
        for i in range(24):
            dt = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
            for team, win, kf, ka, tf, ta, df, da, bf, ba, inf, ina, gd in [
                ("TeamA", 1 if (i % 4 != 0) else 0, 16 + (i % 4), 9 + (i % 3), 8, 4, 3, 1, 1, 0, 2, 1, 1500 + 40 * i),
                ("TeamB", 1 if (i % 4 == 0) else 0, 9 + (i % 3), 16 + (i % 4), 4, 8, 1, 3, 0, 1, 1, 2, -1500 - 40 * i),
            ]:
                rows.append(
                    {
                        "date": dt,
                        "team": team,
                        "league": "LPL",
                        "split": "S1",
                        "playoffs": False,
                        "year": 2026,
                        "win": win,
                        "game_time_min": 30.0 + float(i % 6),
                        "kills_for": float(kf),
                        "kills_against": float(ka),
                        "towers_for": float(tf),
                        "towers_against": float(ta),
                        "dragons_for": float(df),
                        "dragons_against": float(da),
                        "barons_for": float(bf),
                        "barons_against": float(ba),
                        "inhibitors_for": float(inf),
                        "inhibitors_against": float(ina),
                        "golddiffat15_for": float(gd),
                    }
                )
        return pd.DataFrame(rows)

    def test_evaluate_walkforward_with_form_cfg(self):
        samples = self._samples()
        rep = _evaluate_walkforward(
            samples,
            team_scale=1.0,
            players_scale=1.0,
            min_train_games=4,
            form_cfg={
                "enabled": True,
                "max_weight": 0.7,
                "knee_games": 6.0,
                "beta_prior": 2.0,
                "signal_power": 1.0,
                "elite_same_league_parity_strength": 0.2,
            },
        )
        self.assertIn("logloss", rep)
        self.assertIn("brier", rep)
        self.assertGreaterEqual(int(rep.get("n_total") or 0), 1)

    def test_fit_form_grid_returns_expected_keys(self):
        samples = self._samples()
        out = _fit_form_grid(samples, team_scale=1.0, players_scale=1.0, min_train_games=4)
        self.assertIn("baseline_no_form", out)
        self.assertIn("best_form", out)
        self.assertIn("grid_form", out)
        best = out.get("best_form") or {}
        self.assertIn("enabled", best)
        self.assertIn("logloss", best)

    def test_fit_players_lane_weights_shape(self):
        rows = []
        for i in range(80):
            rows.append(
                Sample(
                    month="2026-01" if i < 40 else "2026-02",
                    gameid=f"lane_{i}",
                    patch="14.1",
                    league="LCK" if i % 2 == 0 else "LPL",
                    split="S1",
                    macro="KR" if i % 2 == 0 else "CN",
                    side="blue",
                    bo_bucket="1",
                    phase="regular",
                    y=1 if i % 3 != 0 else 0,
                    p_team=0.52,
                    p_players=0.53,
                    n_team=10.0,
                    n_players=9.0,
                    form_n_a=20,
                    form_n_b=20,
                    form_wins_a=12,
                    form_wins_b=8,
                    lane_top=0.12 if i % 2 == 0 else -0.08,
                    lane_jungle=0.20 if i % 3 == 0 else -0.11,
                    lane_mid=0.30 if i % 4 == 0 else -0.15,
                    lane_bot=0.08 if i % 5 == 0 else -0.04,
                    lane_support=0.05 if i % 7 == 0 else -0.03,
                )
            )
        out = _fit_players_lane_weights(rows, min_n=30)
        self.assertIn("global", out)
        self.assertIn("by_league", out)
        self.assertIn("by_macro", out)
        g = out.get("global") or {}
        for k in ["Top", "Jungle", "Mid", "Bot", "Support"]:
            self.assertIn(k, g)
            self.assertGreaterEqual(float(g[k]), 0.75)
            self.assertLessEqual(float(g[k]), 1.35)

    def test_fit_players_league_weights_shape(self):
        rows = []
        for i in range(120):
            rows.append(
                Sample(
                    month="2026-01" if i < 40 else ("2026-02" if i < 80 else "2026-03"),
                    gameid=f"lg_{i}",
                    patch="14.1",
                    league="LPL" if i % 2 == 0 else "LCK",
                    split="S1",
                    macro="CN" if i % 2 == 0 else "KR",
                    side="blue",
                    bo_bucket="1",
                    phase="regular",
                    y=1 if i % 3 != 0 else 0,
                    p_team=0.52 if i % 2 == 0 else 0.48,
                    p_players=0.56 if i % 2 == 0 else 0.46,
                    n_team=10.0,
                    n_players=9.0,
                    form_n_a=16,
                    form_n_b=16,
                    form_wins_a=10,
                    form_wins_b=8,
                )
            )
        out = _fit_players_league_weights(rows, min_n=20)
        self.assertIn("global", out)
        self.assertIn("by_league", out)
        self.assertIn("by_macro", out)
        self.assertIn("meta", out)

    def test_blind_backtest_by_league_shape(self):
        rows = []
        for i in range(160):
            rows.append(
                Sample(
                    month="2026-01" if i < 50 else ("2026-02" if i < 100 else "2026-03"),
                    gameid=f"bb_{i}",
                    patch="14.1",
                    league="LPL" if i % 2 == 0 else "LCK",
                    split="S1",
                    macro="CN" if i % 2 == 0 else "KR",
                    side="blue",
                    bo_bucket="1",
                    phase="regular",
                    y=1 if i % 3 != 0 else 0,
                    p_team=0.51,
                    p_players=0.54,
                    n_team=12.0,
                    n_players=10.0,
                    form_n_a=20,
                    form_n_b=20,
                    form_wins_a=12,
                    form_wins_b=8,
                )
            )
        out = _blind_backtest_by_league(
            rows,
            team_scale=1.0,
            players_scale=1.0,
            form_cfg={"enabled": True, "max_weight": 0.6, "knee_games": 6.0},
            min_holdout_n=20,
        )
        self.assertIn("rows", out)
        self.assertIn("meta", out)

    def test_fit_players_context_tuning_shape(self):
        samples = self._samples()
        team_games = self._team_games()
        out = _fit_players_context_tuning(
            samples,
            team_games=team_games,
            team_scale=1.0,
            players_scale=1.0,
            min_train_games=4,
            form_cfg={"enabled": True, "max_weight": 0.6, "knee_games": 6.0, "beta_prior": 2.0, "signal_power": 1.0},
            speed_profile="fast",
        )
        self.assertIn("version", out)
        self.assertIn("baseline_off", out)
        self.assertIn("best_on", out)
        self.assertIn("ablation", out)
        self.assertIn("gate", out)
        self.assertIn("settings", out)
        self.assertIn("grid_top", out)
        gate = out.get("gate") or {}
        self.assertIn(str(gate.get("decision") or ""), {"promote", "shadow_only"})

    def test_fit_players_context_tuning_emits_progress(self):
        samples = self._samples()
        team_games = self._team_games()
        events = []

        out = _fit_players_context_tuning(
            samples,
            team_games=team_games,
            team_scale=1.0,
            players_scale=1.0,
            min_train_games=4,
            form_cfg={"enabled": True, "max_weight": 0.6, "knee_games": 6.0, "beta_prior": 2.0, "signal_power": 1.0},
            speed_profile="fast",
            progress_cb=lambda payload: events.append(dict(payload or {})),
        )
        self.assertIn("version", out)
        self.assertTrue(events)
        self.assertTrue(any(str(ev.get("phase") or "") == "candidate_eval" for ev in events))

    def test_fit_players_context_tuning_partial_checkpoint_resume(self):
        samples = self._samples()
        team_games = self._team_games()
        saved = []

        out = _fit_players_context_tuning(
            samples,
            team_games=team_games,
            team_scale=1.0,
            players_scale=1.0,
            min_train_games=4,
            form_cfg={"enabled": True, "max_weight": 0.6, "knee_games": 6.0, "beta_prior": 2.0, "signal_power": 1.0},
            speed_profile="fast",
            checkpoint_save_cb=lambda payload: saved.append(dict(payload or {})),
        )
        self.assertIn("version", out)
        self.assertTrue(saved)
        self.assertIn(str(saved[-1].get("phase") or ""), {"global_done", "complete", "segments_running"})
        out_resume = _fit_players_context_tuning(
            samples,
            team_games=team_games,
            team_scale=1.0,
            players_scale=1.0,
            min_train_games=4,
            form_cfg={"enabled": True, "max_weight": 0.6, "knee_games": 6.0, "beta_prior": 2.0, "signal_power": 1.0},
            speed_profile="fast",
            checkpoint_state=saved[-1],
        )
        self.assertEqual(out_resume.get("version"), out.get("version"))

    def test_players_context_budget_profile_runtime_auto_full(self):
        self.assertEqual(_resolve_players_context_budget_profile("fast", "runtime_auto_fast"), "fast")
        self.assertEqual(_resolve_players_context_budget_profile("full", "runtime_auto_full"), "runtime_full")
        self.assertEqual(_resolve_players_context_budget_profile("full", "manual_quick"), "full")

    def test_fit_players_context_tuning_runtime_full_budget(self):
        samples = self._samples() * 250
        team_games = self._team_games()
        out = _fit_players_context_tuning(
            samples,
            team_games=team_games,
            team_scale=1.0,
            players_scale=1.0,
            min_train_games=4,
            form_cfg={"enabled": True, "max_weight": 0.6, "knee_games": 6.0, "beta_prior": 2.0, "signal_power": 1.0},
            speed_profile="full",
            budget_profile="runtime_full",
        )
        meta = out.get("meta") or {}
        self.assertEqual(str(meta.get("budget_profile") or ""), "runtime_full")
        self.assertTrue(bool(meta.get("segment_fit_restricted")))
        self.assertLessEqual(int(meta.get("candidates_tested") or 0), 8)

    def test_fuse_probs_by_precision_uses_runtime_fusion_semantics(self):
        out_fit = _fuse_probs_by_precision(
            p_team=0.60,
            p_players=0.65,
            n_team=40.0,
            n_players=30.0,
            team_scale=0.25,
            players_scale=3.0,
            coverage=0.90,
            transfer_signal=0.0,
            lineup_continuity=1.0,
            roster_changed=False,
            season_phase="auto",
            players_freshness=1.0,
        )
        out_core = fuse_with_coherence_guard(
            p_team=0.60,
            p_players=0.65,
            n_team=40.0,
            n_players=30.0,
            team_scale=0.25,
            players_scale=3.0,
            coverage=0.90,
            transfer_signal=0.0,
            lineup_continuity=1.0,
            roster_changed=False,
            season_phase="auto",
        )
        self.assertAlmostEqual(float(out_fit), float(out_core.get("p_fused")), places=9)

    def test_fuse_probs_by_precision_applies_players_freshness_to_evidence(self):
        p_stale = _fuse_probs_by_precision(
            p_team=0.55,
            p_players=0.70,
            n_team=20.0,
            n_players=12.0,
            team_scale=1.0,
            players_scale=1.0,
            coverage=0.80,
            transfer_signal=0.0,
            lineup_continuity=1.0,
            roster_changed=False,
            season_phase="auto",
            players_freshness=0.20,
        )
        p_fresh = _fuse_probs_by_precision(
            p_team=0.55,
            p_players=0.70,
            n_team=20.0,
            n_players=12.0,
            team_scale=1.0,
            players_scale=1.0,
            coverage=0.80,
            transfer_signal=0.0,
            lineup_continuity=1.0,
            roster_changed=False,
            season_phase="auto",
            players_freshness=1.0,
        )
        self.assertGreater(float(p_fresh), float(p_stale))

    def test_checkpoint_and_progress_helpers(self):
        with workspace_tempdir("fit_ml_params") as tmp:
            root = Path(tmp)
            artifact = root / "ml_artifact.json"
            players_artifact = root / "players_artifact.json"
            csv_path = root / "sample.csv"
            base_params = root / "params_fitted.base.json"
            artifact.write_text("{}", encoding="utf-8")
            players_artifact.write_text("{}", encoding="utf-8")
            csv_path.write_text("gameid,participantid\n1,1\n", encoding="utf-8")
            base_params.write_text("{}", encoding="utf-8")

            sig = _build_fit_run_signature(
                artifact_path=str(artifact),
                players_artifact_path=str(players_artifact),
                csv_paths=[str(csv_path)],
                min_train_games=300,
                speed_profile="full",
                run_label="fullfit_20260307",
                start_stage="context_models",
                base_params_path=str(base_params),
            )
            self.assertEqual(sig.get("run_label"), "fullfit_20260307")
            self.assertEqual(sig.get("start_stage"), "context_models")
            self.assertEqual(((sig.get("base_params") or {}).get("path")), str(base_params.resolve()))
            ck_dir = root / "checkpoints"
            _save_stage_checkpoint(
                ck_dir,
                "global_scales",
                run_signature=sig,
                payload={"best": {"team_scale": 1.0}},
                summary={"grid_rows": 9},
            )
            loaded = _load_stage_checkpoint(ck_dir, "global_scales", run_signature=sig)
            self.assertEqual(loaded.get("stage"), "global_scales")
            self.assertEqual(((loaded.get("payload") or {}).get("best") or {}).get("team_scale"), 1.0)
            self.assertEqual(((loaded.get("summary") or {}).get("grid_rows")), 9)

            sig_other = _build_fit_run_signature(
                artifact_path=str(artifact),
                players_artifact_path=str(players_artifact),
                csv_paths=[str(csv_path)],
                min_train_games=301,
                speed_profile="full",
                run_label="other",
            )
            self.assertEqual(_load_stage_checkpoint(ck_dir, "global_scales", run_signature=sig_other), {})

            progress_path = root / "progress.json"
            _write_fit_progress(
                progress_path,
                stage="global_scales",
                status="completed",
                run_signature=sig,
                total_stages=9,
                completed_stages=["load_inputs", "player_weights", "global_scales"],
                detail={"best_logloss": 0.61},
                started_at="2026-03-07T09:00:00+00:00",
                elapsed_s=12.5,
            )
            self.assertTrue(progress_path.exists())
            content = progress_path.read_text(encoding="utf-8")
            self.assertIn('"stage": "global_scales"', content)
            self.assertIn('"completed_count": 3', content)
            self.assertIn('"elapsed_s": 12.5', content)
            self.assertIn('"run_label": "fullfit_20260307"', content)
            self.assertIn('"start_stage": "context_models"', content)

    def test_json_and_signature_helpers_handle_invalid_inputs(self):
        with workspace_tempdir("fit_ml_params") as tmp:
            root = Path(tmp)
            broken = root / "broken.json"
            broken.write_text("{invalid", encoding="utf-8")
            self.assertEqual(_read_json_file_safe(broken), {})

            missing_sig = _path_signature(root / "missing.json")
            self.assertEqual(str(missing_sig.get("path")), str((root / "missing.json").resolve()))
            self.assertEqual(missing_sig.get("missing"), True)

    def test_seed_stage_payload_from_params_and_prime_checkpoints(self):
        base_params = {
            "baseline": {"logloss": 0.69, "brier": 0.25, "n_total": 10, "folds": "bad"},
            "best": {"team_scale": 1.1, "players_scale": 0.9, "logloss": 0.63, "brier": 0.22, "n_total": 10, "folds": 2},
            "best_folds": [{"month": "2026-01"}],
            "grid": [{"team_scale": 1.1, "players_scale": 0.9, "logloss": 0.63, "brier": 0.22}],
            "best_form_baseline_no_form": {"logloss": 0.64},
            "best_form": {"enabled": True, "max_weight": 0.5, "settings": {"ml_form_blend_enabled": True}},
            "best_combined": {
                "team_scale": 1.1,
                "players_scale": 0.9,
                "form_enabled": True,
                "form_settings": {"ml_form_blend_enabled": True},
                "logloss": 0.61,
                "brier": 0.21,
                "n_total": 10,
                "folds": 2,
            },
            "microseg_calibration": {"pred_rows_n": 3, "table": {"LPL|S1|blue|1|regular": {"wins": 2}}},
            "fused_post_calibration": {"version": "fused_post_calibration_v1"},
            "best_by_league": {"LPL": {"team_scale": 1.1, "players_scale": 0.9}},
            "best_form_by_league": {"LPL": {"settings": {"ml_form_blend_enabled": True}}},
            "best_form_by_macro": {"CN": {"settings": {"ml_form_blend_enabled": True}}},
            "players_lane_weights": {"global": {"Top": 1.0}},
            "players_league_weights": {"global": {"LPL": 1.0}},
            "series_gamma_global": {"bo3": {"gamma": 1.0}},
            "series_gamma_by_league": {"LPL": {"bo3": {"gamma": 1.02}}},
            "series_gamma_by_macro": {"CN": {"bo3": {"gamma": 1.01}}},
            "series_bo_dplite_tuning": {"global": {"bo3": {"pick_shift": 0.02}}},
            "blind_backtest_by_league": {"rows": [{"league": "LPL"}]},
            "meta_blend_c_model": {"version": "meta_blend_c_model_v3"},
            "style_matchup_b_model": {"version": "style_matchup_b_model_v2"},
            "players_context_fit": {"version": "ml_players_context_fit_v1"},
            "areal_prob_tuning": {"version": "areal_prob_tuning_v3", "global": {"bayes_info_scale": 1.0}},
            "areal_prob_tuning_source": "snapshot",
            "ml_marginal_trust_band_coeffs": {"version": "ml_marginal_trust_band_coeffs_v1", "global": {"alpha": 0.2}},
            "ml_marginal_trust_band_coeffs_source": "snapshot",
        }
        global_seed = _seed_stage_payload_from_params(base_params, "global_scales")
        self.assertEqual(((global_seed.get("baseline") or {}).get("folds")), [])
        seeded = _seed_stage_payload_from_params(base_params, "form_combined")
        self.assertEqual(int(((seeded.get("combined_report") or {}).get("pred_rows_n"))), 3)
        self.assertEqual(len(((seeded.get("combined_report") or {}).get("pred_rows") or [])), 3)
        self.assertEqual(_normalize_fit_stage("context_models"), "context_models")
        with workspace_tempdir("fit_ml_params") as tmp:
            root = Path(tmp)
            artifact = root / "ml_artifact.json"
            players_artifact = root / "players_artifact.json"
            csv_path = root / "sample.csv"
            base_path = root / "params_fitted.base.json"
            artifact.write_text("{}", encoding="utf-8")
            players_artifact.write_text("{}", encoding="utf-8")
            csv_path.write_text("gameid,participantid\n1,1\n", encoding="utf-8")
            base_path.write_text("{}", encoding="utf-8")
            sig = _build_fit_run_signature(
                artifact_path=str(artifact),
                players_artifact_path=str(players_artifact),
                csv_paths=[str(csv_path)],
                min_train_games=300,
                speed_profile="full",
                run_label="fullfit_seed",
                start_stage="context_models",
                base_params_path=str(base_path),
            )
            ck_dir = root / "checkpoints"
            rep = _prime_seed_stage_checkpoints_from_params(
                ck_dir,
                run_signature=sig,
                base_params=base_params,
                start_stage="context_models",
                base_params_path=base_path,
            )
            self.assertEqual(rep.get("missing"), [])
            loaded = _load_stage_checkpoint(ck_dir, "series_models", run_signature=sig)
            self.assertEqual(((loaded.get("summary") or {}).get("seeded_from_base")), True)
            self.assertIn("series_gamma_global", loaded.get("payload") or {})


if __name__ == "__main__":
    unittest.main()
