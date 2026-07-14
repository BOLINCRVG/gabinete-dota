import json
import unittest
from pathlib import Path

from mlcore.tests._workspace_temp import workspace_tempdir
from app_core.pre_delivery_checks_core import (
    assess_anchor_regression,
    assess_confidence_baseline_file,
    assess_combined_trace_file,
    assess_feature_trace_health,
    assess_navigation_integrity,
    assess_patch_calibration_freshness,
    assess_params_baseline_stability,
    assess_ranking_coverage,
    assess_session_trace,
    assess_state_artifact_file,
    assess_utf8_file,
    extract_p_map_used,
    run_pre_delivery_smokes,
    run_unittest_smoke,
)
import pandas as pd
from unittest.mock import patch


class PreDeliveryChecksCoreTests(unittest.TestCase):
    def test_extract_p_map_used_supports_nested_probabilities(self):
        self.assertEqual(extract_p_map_used({"p_map_used": 0.61}), 0.61)
        self.assertEqual(extract_p_map_used({"probabilities": {"p_map_used": 0.44}}), 0.44)
        self.assertIsNone(extract_p_map_used({"probabilities": {"p_map_used": "nan"}}))

    def test_assess_combined_trace_file(self):
        with workspace_tempdir("pre_delivery_checks_core") as tmpdir:
            trace_path = Path(tmpdir) / "ml_trace_last.json"
            payload = {
                "ml_trace": {"probabilities": {"p_map_used": 0.57}},
                "resumo_markets_trace": [{"mercado": "ML"}],
                "players_trace": [{"player": "mid"}],
                "feature_trace": [],
            }
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            got = assess_combined_trace_file(trace_path)
            self.assertTrue(got["ok"])
            self.assertIn("p_map_used=0.57", got["detail"])

    def test_assess_feature_trace_health_and_session_trace(self):
        with workspace_tempdir("pre_delivery_checks_core") as tmpdir:
            trace_path = Path(tmpdir) / "ml_trace_last.json"
            trace_path.write_text(
                json.dumps({"feature_trace": [{"feature": "shadow"}]}),
                encoding="utf-8",
            )
            feature_health = assess_feature_trace_health(
                trace_path=trace_path,
                feature_state={"last_error": ""},
            )
            self.assertTrue(feature_health["ok"])

            session_trace = assess_session_trace(
                trace_snapshot={"_ml_trace_last": {"p_map_used": 0.52}},
                p_map_used_raw=None,
            )
            self.assertTrue(session_trace["ok"])
            self.assertEqual(session_trace["p_map_used"], 0.52)

    def test_file_checks_and_patch_freshness(self):
        with workspace_tempdir("pre_delivery_checks_core") as tmpdir:
            root = Path(tmpdir)
            conf_path = root / "ml_confidence_baseline.json"
            conf_path.write_text(
                json.dumps({"version": "ml_confidence_baseline_v1", "global": {}, "by_league": {}}),
                encoding="utf-8",
            )
            conf = assess_confidence_baseline_file(conf_path)
            self.assertTrue(conf["ok"])

            state_path = root / "team_state_artifact.json"
            state_path.write_text(
                json.dumps({"version": "team_state_artifact_v1", "teams": {}, "by_league": {}, "meta": {"teams": 2, "games_used": 10}}),
                encoding="utf-8",
            )
            state = assess_state_artifact_file(state_path, optional_when_disabled=False)
            self.assertTrue(state["ok"])

            csv_path = root / "sample.csv"
            csv_path.write_text(
                "gameid,league,patch,date,participantid\n1,LCK,15.1,2026-03-01,100\n1,LCK,15.1,2026-03-01,200\n",
                encoding="utf-8",
            )
            params_path = root / "params_fitted.json"
            params_path.write_text(
                json.dumps({"generated_at": "2026-03-02T10:00:00", "patch_snapshot": {}}),
                encoding="utf-8",
            )
            patch = assess_patch_calibration_freshness(
                csv_path=csv_path,
                params_path=params_path,
                build_patch_snapshot_fn=lambda df: {"games": int(df.shape[0])},
                recommend_recalibration_fn=lambda **kwargs: {"status": "ok", "changed_leagues": [], "age_days": 1},
            )
            self.assertTrue(patch["ok"])
            self.assertIn("status=ok", patch["detail"])

    def test_navigation_utf8_ranking_and_params_checks(self):
        with workspace_tempdir("pre_delivery_checks_core") as tmpdir:
            root = Path(tmpdir)
            utf8_path = root / "sample.py"
            utf8_path.write_text("print('ok')\n", encoding="utf-8")
            utf8_check = assess_utf8_file(utf8_path)
            self.assertTrue(utf8_check["ok"])

            nav_check = assess_navigation_integrity(
                tabs_full=["Visão Geral", "Resumo", "Players"],
                required_tabs={"Visão Geral", "Resumo"},
            )
            self.assertTrue(nav_check["ok"])

            ranking_df = pd.DataFrame({"league": ["LCK", "LPL", "LEC", "LCS", "CBLOL"]})
            ranking_check = assess_ranking_coverage(ranking_df=ranking_df)
            self.assertTrue(ranking_check["ok"])

            params_path = root / "params_fitted.json"
            params_path.write_text(
                json.dumps(
                    {
                        "best": {"team_scale": 1.2, "players_scale": 0.9, "logloss": 0.48},
                        "baseline": {"logloss": 0.49},
                    }
                ),
                encoding="utf-8",
            )
            params_check = assess_params_baseline_stability(params_path)
            self.assertTrue(params_check["ok"])

    def test_assess_anchor_regression(self):
        with workspace_tempdir("pre_delivery_checks_core") as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "sample.csv"
            csv_path.write_text("gameid,league,patch,date,participantid\n", encoding="utf-8")
            baseline_path = root / "odds_anchor_baseline.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "anchors": [
                            {"key": "a", "ok": True, "p_a": 0.51},
                            {"key": "b", "ok": True, "p_a": 0.62},
                            {"key": "c", "ok": True, "p_a": 0.47},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            anchor_check = assess_anchor_regression(
                csv_path=csv_path,
                baseline_path=baseline_path,
                compute_anchor_snapshot_fn=lambda _csv: {
                    "anchors": [
                        {"key": "a", "ok": True, "p_a": 0.52},
                        {"key": "b", "ok": True, "p_a": 0.60},
                        {"key": "c", "ok": True, "p_a": 0.49},
                    ]
                },
            )
            self.assertTrue(anchor_check["ok"])
            self.assertIn("anchors_usadas=3/3", anchor_check["detail"])

    def test_run_unittest_smoke_filters_script_context_noise(self):
        class _CP:
            returncode = 0
            stdout = "ok line\nmissing ScriptRunContext\n"
            stderr = ""

        with patch("app_core.pre_delivery_checks_core.subprocess.run", return_value=_CP()):
            result = run_unittest_smoke(
                test_targets=["mlcore.tests.test_smoke"],
                cwd=".",
                csv_path=None,
                python_executable="python",
            )
        self.assertTrue(result["ok"])
        self.assertIn("ok line", result["detail"])
        self.assertNotIn("ScriptRunContext", result["detail"])

    def test_run_pre_delivery_smokes_returns_all_labels(self):
        with patch(
            "app_core.pre_delivery_checks_core.run_unittest_smoke",
            side_effect=[
                {"ok": True, "detail": "a"},
                {"ok": True, "detail": "b"},
                {"ok": False, "detail": "c"},
            ],
        ):
            rows = run_pre_delivery_smokes(cwd=".", csv_path=None, python_executable="python")
        self.assertEqual([row["label"] for row in rows], ["Smoke test", "Players/Campeões smoke", "Series smoke"])
        self.assertEqual([bool(row["ok"]) for row in rows], [True, True, False])


if __name__ == "__main__":
    unittest.main()
