from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from mlcore.tests._workspace_temp import workspace_tempdir
from app_core.maintenance_runtime_core import (
    build_fit_params_runtime_paths,
    build_fit_params_cmd,
    build_incremental_update_cmd,
    build_retrain_check_cmd,
    build_train_offline_cmd,
    choose_params_primary_csvs,
    params_output_valid_path,
    parse_incremental_update_stdout,
    read_artifact_summary_path,
    resolve_train_offline_reuse_params,
)


class MaintenanceRuntimeCoreTests(unittest.TestCase):
    def test_read_artifact_summary_path_parses_incremental_meta(self) -> None:
        with workspace_tempdir("maintenance_runtime_core") as tmpdir:
            p = Path(tmpdir) / "ml_artifact.json"
            p.write_text(
                json.dumps(
                    {
                        "meta": {
                            "trained_at": "2026-03-07T06:00:00",
                            "data_max_date": "2026-03-06",
                            "n_games_total": 1234,
                            "incremental": {
                                "added_games": 12,
                                "backfill_suspected": True,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            got = read_artifact_summary_path(p)
        self.assertTrue(got["exists"])
        self.assertEqual(got["trained_at"], "2026-03-07T06:00:00")
        self.assertEqual(got["data_max_date"], "2026-03-06")
        self.assertEqual(got["n_games_total"], 1234)
        self.assertEqual(got["added_games_last"], 12)
        self.assertTrue(got["backfill_suspected"])

    def test_parse_incremental_update_stdout_reads_new_games_and_backfill(self) -> None:
        got = parse_incremental_update_stdout(
            "some log\nnew games applied: 17\nbackfill_suspected=true\nother line"
        )
        self.assertEqual(got["new_games"], 17)
        self.assertTrue(got["backfill_suspected"])

    def test_build_incremental_update_cmd(self) -> None:
        cmd = build_incremental_update_cmd(artifact_path="ml_artifact.json", csv_path="daily.csv")
        self.assertEqual(
            cmd,
            [sys.executable, "-m", "mlcore.update_incremental", "--artifact", "ml_artifact.json", "--csv", "daily.csv"],
        )
        # Sem players_path: nao deve incluir --players (retrocompat).
        self.assertNotIn("--players", cmd)

    def test_build_incremental_update_cmd_with_players(self) -> None:
        # Com players_path: os jogos novos vao p/ o arquivo que o app usa,
        # nao p/ um path obsoleto do meta do artifact.
        cmd = build_incremental_update_cmd(
            artifact_path="ml_artifact.json",
            csv_path="daily.csv",
            players_path="players_artifact.json",
        )
        self.assertIn("--players", cmd)
        self.assertEqual(cmd[cmd.index("--players") + 1], "players_artifact.json")

    def test_build_train_offline_cmd(self) -> None:
        cmd = build_train_offline_cmd(
            ["a.csv", "b.csv"],
            out_path="ml_artifact.json",
            holdout_days=180,
            k=26.0,
            scale=400.0,
            calibrate="auto",
            keep_playoffs="yes",
            fit_players=True,
            players_out="players_artifact.json",
        )
        self.assertEqual(
            cmd,
            [
                sys.executable,
                "-m",
                "mlcore.train_offline",
                "--csv",
                "a.csv",
                "--csv",
                "b.csv",
                "--out",
                "ml_artifact.json",
                "--holdout-days",
                "180",
                "--k",
                "26.0",
                "--scale",
                "400.0",
                "--calibrate",
                "auto",
                "--keep-playoffs",
                "yes",
                "--fit-players",
                "--players-out",
                "players_artifact.json",
            ],
        )

    def test_build_fit_params_cmd(self) -> None:
        cmd = build_fit_params_cmd(
            ["x.csv"],
            artifact_path="ml_artifact.json",
            players_artifact_path="players_artifact.json",
            out_path="params_fitted.json",
            speed_profile="fast",
            progress_path="ops/runtime_fit_params/auto/fit_ml_params_progress.json",
            checkpoint_dir="ops/runtime_fit_params/auto/checkpoints",
            run_label="runtime_auto",
            start_stage="context_models",
            base_params_path="params_fitted.base.json",
        )
        self.assertEqual(
            cmd,
            [
                sys.executable,
                "-m",
                "mlcore.fit_ml_params",
                "--artifact",
                "ml_artifact.json",
                "--players-artifact",
                "players_artifact.json",
                "--out",
                "params_fitted.json",
                "--speed-profile",
                "fast",
                "--progress-out",
                "ops/runtime_fit_params/auto/fit_ml_params_progress.json",
                "--checkpoint-dir",
                "ops/runtime_fit_params/auto/checkpoints",
                "--run-label",
                "runtime_auto",
                "--start-stage",
                "context_models",
                "--base-params",
                "params_fitted.base.json",
                "--csv",
                "x.csv",
            ],
        )

    def test_build_fit_params_runtime_paths(self) -> None:
        got = build_fit_params_runtime_paths(app_root="C:/repo/app", scope="manual quick")
        self.assertEqual(got["run_label"], "runtime_manual_quick")
        self.assertTrue(got["run_root"].endswith("ops\\runtime_fit_params\\manual_quick"))
        self.assertTrue(got["progress_path"].endswith("ops\\runtime_fit_params\\manual_quick\\fit_ml_params_progress.json"))
        self.assertTrue(got["checkpoint_dir"].endswith("ops\\runtime_fit_params\\manual_quick\\checkpoints"))

    def test_resolve_train_offline_reuse_params(self) -> None:
        class _Params:
            k = 30.0
            scale_points = 500.0

        class _Model:
            params = _Params()

        class _Cal:
            a = 1.01
            b = 0.01

        class _Art:
            model = _Model()
            calibrator = _Cal()

        got = resolve_train_offline_reuse_params("artifact.json", artifact_loader=lambda _: _Art())
        self.assertEqual(got["k"], 30.0)
        self.assertEqual(got["scale"], 500.0)
        self.assertEqual(got["calibrate"], "no")

    def test_build_retrain_check_cmd(self) -> None:
        cmd = build_retrain_check_cmd(
            artifact_path="ml_artifact.json",
            csv_path="daily.csv",
            keep_playoffs="auto",
            exit_code=True,
        )
        self.assertEqual(
            cmd,
            [
                sys.executable,
                "-m",
                "mlcore.retrain_check",
                "--artifact",
                "ml_artifact.json",
                "--csv",
                "daily.csv",
                "--keep-playoffs",
                "auto",
                "--exit-code",
            ],
        )

    def test_choose_params_primary_csvs_daily_fast(self) -> None:
        with workspace_tempdir("maintenance_runtime_core") as tmpdir:
            daily = Path(tmpdir) / "daily.csv"
            hist = Path(tmpdir) / "hist.csv"
            daily.write_text("x", encoding="utf-8")
            hist.write_text("x", encoding="utf-8")
            got = choose_params_primary_csvs([str(hist), str(daily)], daily_csv=str(daily), fast_mode=True)
        self.assertEqual(len(got["csvs"]), 1)
        self.assertEqual(got["mode"], "daily_only_fast")

    def test_choose_params_primary_csvs_full_mode(self) -> None:
        got = choose_params_primary_csvs(["hist.csv", "daily.csv"], daily_csv="daily.csv", fast_mode=False)
        self.assertEqual(got["csvs"], ["hist.csv", "daily.csv"])
        self.assertEqual(got["mode"], "full_csvs_full")

    def test_params_output_valid_path(self) -> None:
        with workspace_tempdir("maintenance_runtime_core") as tmpdir:
            p = Path(tmpdir) / "params.json"
            p.write_text(json.dumps({"best": {"team_scale": 0.9, "players_scale": 1.1}}), encoding="utf-8")
            self.assertTrue(params_output_valid_path(p))
            p.write_text(json.dumps({"best": {"team_scale": None, "players_scale": 1.1}}), encoding="utf-8")
            self.assertFalse(params_output_valid_path(p))


if __name__ == "__main__":
    unittest.main()
