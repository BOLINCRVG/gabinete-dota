from __future__ import annotations

import unittest
from pathlib import Path

from mlcore.tests._workspace_temp import workspace_tempdir
from app_core.pre_delivery_runtime_core import run_pre_delivery_checks


class PreDeliveryRuntimeCoreTests(unittest.TestCase):
    def test_run_pre_delivery_checks_aggregates_rows(self) -> None:
        state = {"ff_ml_state_areal_live": True, "_p_map_used": 0.61}
        with workspace_tempdir("pre_delivery_runtime_core") as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "daily.csv"
            artifact_path = root / "ml_artifact.json"
            csv_path.write_text("ok", encoding="utf-8")
            artifact_path.write_text("{}", encoding="utf-8")
            (root / "params_fitted.json").write_text("{}", encoding="utf-8")
            (root / "ml_confidence_baseline.json").write_text("{}", encoding="utf-8")
            (root / "team_state_artifact.json").write_text("{}", encoding="utf-8")
            (root / "players_artifact.json").write_text("{}", encoding="utf-8")
            (root / "ml_trace_last.json").write_text("{}", encoding="utf-8")
            anchor_path = root / "odds_anchor_baseline.json"
            anchor_path.write_text("{}", encoding="utf-8")

            rep = run_pre_delivery_checks(
                state,
                app_root=root,
                csv_path_s=str(csv_path),
                artifact_path_s=str(artifact_path),
                run_smoke=True,
                assess_confidence_baseline_file_fn=lambda _p: {"ok": True, "detail": "conf"},
                assess_state_artifact_file_fn=lambda _p, optional_when_disabled=False: {"ok": True, "detail": "state"},
                assess_patch_calibration_freshness_fn=lambda **kwargs: {"ok": True, "detail": "patch"},
                assess_combined_trace_file_fn=lambda _p: {"ok": True, "detail": "trace"},
                assess_feature_trace_health_fn=lambda **kwargs: {"ok": True, "detail": "feature"},
                assess_session_trace_fn=lambda **kwargs: {"ok": True, "detail": "session"},
                assess_navigation_integrity_fn=lambda **kwargs: {"ok": True, "detail": "nav"},
                assess_utf8_file_fn=lambda _p: {"ok": True, "detail": "utf8"},
                assess_ranking_coverage_fn=lambda **kwargs: {"ok": True, "detail": "ranking"},
                assess_params_baseline_stability_fn=lambda _p: {"ok": True, "detail": "params"},
                assess_anchor_regression_fn=lambda **kwargs: {"ok": True, "detail": "anchor"},
                run_pre_delivery_smokes_fn=lambda **kwargs: [{"label": "Smoke", "ok": True, "detail": "smoke"}],
                build_patch_snapshot_fn=lambda *args, **kwargs: {},
                recommend_recalibration_fn=lambda **kwargs: {},
                compute_anchor_snapshot_fn=lambda _p: {},
                get_feature_trace_state_fn=lambda _s: {},
                get_ml_trace_snapshot_fn=lambda _s: {},
                pick_tabs_fn=lambda _mode: ["Visão Geral", "Resumo"],
                required_tabs={"Visão Geral"},
                rank_table_fn=lambda *args, **kwargs: [],
                anchor_baseline_path=anchor_path,
                python_executable="python",
            )

            self.assertTrue(rep["all_ok"])
            self.assertGreaterEqual(rep["ok_count"], 1)
            self.assertTrue(any(row["Item"] == "Smoke" for row in rep["rows"]))


    def test_run_pre_delivery_checks_uses_resolved_players_artifact_path(self) -> None:
        state = {"ff_ml_state_areal_live": False, "_p_map_used": 0.61}
        with workspace_tempdir("pre_delivery_runtime_core") as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "custom"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            csv_path = root / "daily.csv"
            artifact_path = artifact_dir / "ml_artifact.json"
            players_path = artifact_dir / "players_artifact.json"
            csv_path.write_text("ok", encoding="utf-8")
            artifact_path.write_text("{}", encoding="utf-8")
            players_path.write_text("{}", encoding="utf-8")
            (root / "params_fitted.json").write_text("{}", encoding="utf-8")
            (root / "ml_confidence_baseline.json").write_text("{}", encoding="utf-8")
            (root / "ml_trace_last.json").write_text("{}", encoding="utf-8")
            anchor_path = root / "odds_anchor_baseline.json"
            anchor_path.write_text("{}", encoding="utf-8")

            rep = run_pre_delivery_checks(
                state,
                app_root=root,
                csv_path_s=str(csv_path),
                artifact_path_s=str(artifact_path),
                run_smoke=False,
                assess_confidence_baseline_file_fn=lambda _p: {"ok": True, "detail": "conf"},
                assess_state_artifact_file_fn=lambda _p, optional_when_disabled=False: {"ok": True, "detail": "state"},
                assess_patch_calibration_freshness_fn=lambda **kwargs: {"ok": True, "detail": "patch"},
                assess_combined_trace_file_fn=lambda _p: {"ok": True, "detail": "trace"},
                assess_feature_trace_health_fn=lambda **kwargs: {"ok": True, "detail": "feature"},
                assess_session_trace_fn=lambda **kwargs: {"ok": True, "detail": "session"},
                assess_navigation_integrity_fn=lambda **kwargs: {"ok": True, "detail": "nav"},
                assess_utf8_file_fn=lambda _p: {"ok": True, "detail": "utf8"},
                assess_ranking_coverage_fn=lambda **kwargs: {"ok": True, "detail": "ranking"},
                assess_params_baseline_stability_fn=lambda _p: {"ok": True, "detail": "params"},
                assess_anchor_regression_fn=lambda **kwargs: {"ok": True, "detail": "anchor"},
                run_pre_delivery_smokes_fn=lambda **kwargs: [],
                build_patch_snapshot_fn=lambda *args, **kwargs: {},
                recommend_recalibration_fn=lambda **kwargs: {},
                compute_anchor_snapshot_fn=lambda _p: {},
                get_feature_trace_state_fn=lambda _s: {},
                get_ml_trace_snapshot_fn=lambda _s: {},
                pick_tabs_fn=lambda _mode: ["VisÃ£o Geral", "Resumo"],
                required_tabs={"VisÃ£o Geral"},
                rank_table_fn=lambda *args, **kwargs: [],
                anchor_baseline_path=anchor_path,
                python_executable="python",
                resolve_players_artifact_path_fn=lambda _artifact_path, _app_root: str(players_path),
            )

            players_row = next(row for row in rep["rows"] if row["Item"] == "players_artifact.json existe")
            self.assertEqual(players_row["Status"], "Aprovado")
            self.assertEqual(players_row["Detalhe"], str(players_path))


if __name__ == "__main__":
    unittest.main()
