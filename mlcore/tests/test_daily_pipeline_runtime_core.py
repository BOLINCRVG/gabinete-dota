from __future__ import annotations

import unittest
from pathlib import Path

from mlcore.tests._workspace_temp import workspace_tempdir
from app_core.daily_pipeline_runtime_core import run_daily_pipeline
from app_core.ml_session_runtime_core import get_pre_delivery_report
from app_core.runtime_session_state_core import (
    get_cal_diag_last_report,
    get_daily_pipeline_report,
    get_lv_last_report,
    get_ml_confidence_baseline_last_report,
    get_team_state_artifact_last_report,
    get_wf_last_report,
)


class DailyPipelineRuntimeCoreTests(unittest.TestCase):
    def test_run_daily_pipeline_updates_reports_and_returns_ok(self) -> None:
        state: dict[str, object] = {}
        with workspace_tempdir("daily_pipeline_runtime_core") as tmpdir:
            root = Path(tmpdir)
            daily = root / "daily.csv"
            hist = root / "hist.csv"
            artifact = root / "ml_artifact.json"
            players = root / "players_artifact.json"
            daily.write_text("ok", encoding="utf-8")
            hist.write_text("ok", encoding="utf-8")
            artifact.write_text("{}", encoding="utf-8")
            players.write_text("{}", encoding="utf-8")
            baseline = root / "odds_anchor_baseline.json"
            cache_cleared: list[bool] = []

            rep = run_daily_pipeline(
                state,
                root,
                csv_path_s=str(daily),
                artifact_path_s=str(artifact),
                run_smoke=True,
                fast_mode=True,
                baseline_path=baseline,
                paths_for_ml_fn=lambda _root: (str(daily), [str(hist)], None),
                resolve_players_artifact_path_fn=lambda _artifact_path, _root: str(players),
                compute_anchor_snapshot_fn=lambda _csv: {"ok": True},
                fit_ml_params_run_fn=lambda **kwargs: {"best": {"team_scale": 1.2, "players_scale": 0.8}},
                enrich_params_with_areal_prob_tuning_fn=lambda rep_fit, **kwargs: (rep_fit, "promoted"),
                run_layer_validation_cached_fn=lambda *args: {"n_samples": 12},
                run_calibration_diag_cached_fn=lambda *args: {"models": {"fused": {}}},
                run_walkforward_cached_fn=lambda *args: {"overall": {"folds": 3}},
                load_fitted_params_cached_fn=lambda _sig: {},
                file_sig_fn=lambda _path: ("sig", 1, 1),
                generate_ml_confidence_baseline_fn=lambda **kwargs: {"by_league": {"LCK": {}}},
                generate_team_state_artifact_fn=lambda **kwargs: {"meta": {"teams": 2, "games_used": 10, "update_mode": "full"}},
                run_pre_delivery_checks_fn=lambda *args, **kwargs: {"all_ok": True, "ok_count": 3, "total_count": 3},
                clear_team_state_cache_fn=lambda: cache_cleared.append(True),
            )

            self.assertTrue(rep["ok"])
            self.assertTrue(baseline.exists())
            self.assertTrue((root / "params_fitted.json").exists())
            self.assertEqual(get_lv_last_report(state), {"n_samples": 12})
            self.assertEqual(get_cal_diag_last_report(state), {"models": {"fused": {}}})
            self.assertEqual(get_wf_last_report(state), {"overall": {"folds": 3}})
            self.assertEqual(get_ml_confidence_baseline_last_report(state), {"by_league": {"LCK": {}}})
            self.assertEqual(get_team_state_artifact_last_report(state), {"meta": {"teams": 2, "games_used": 10, "update_mode": "full"}})
            self.assertEqual(get_pre_delivery_report(state), {"all_ok": True, "ok_count": 3, "total_count": 3})
            self.assertEqual(cache_cleared, [True])

            daily_saved = {"ok": rep["ok"], "steps": rep["steps"]}
            from app_core.runtime_session_state_core import set_daily_pipeline_report

            set_daily_pipeline_report(state, daily_saved)
            self.assertEqual(get_daily_pipeline_report(state), daily_saved)


if __name__ == "__main__":
    unittest.main()
