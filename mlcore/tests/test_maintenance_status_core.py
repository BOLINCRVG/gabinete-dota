import json
import os
import unittest
from datetime import datetime
from pathlib import Path

from mlcore.tests._workspace_temp import workspace_tempdir
from app_core.maintenance_status_core import collect_maintenance_status, safe_mtime


class TestMaintenanceStatusCore(unittest.TestCase):
    def test_safe_mtime(self):
        with workspace_tempdir("maintenance_status_core") as td:
            p = Path(td) / "x.txt"
            p.write_text("ok", encoding="utf-8")
            self.assertIsNotNone(safe_mtime(p))
            self.assertIsNone(safe_mtime(Path(td) / "missing.txt"))

    def test_missing_artifacts_and_params(self):
        with workspace_tempdir("maintenance_status_core") as td:
            root = Path(td)
            csv = root / "daily.csv"
            csv.write_text("gameid,participantid\n", encoding="utf-8")
            out = collect_maintenance_status(
                str(csv),
                str(root / "ml_artifact.json"),
                app_root=root,
                wf_report_path=root / "wf_last_report.json",
                file_sig_fn=lambda p: (str(p), 1, 1),
                patch_meta_guard_payload_fn=lambda a, b: {},
                now=datetime(2026, 3, 7, 8, 0, 0),
            )
            self.assertEqual(out["severity"], "critical")
            self.assertTrue(out["need_retrain_model"])
            self.assertTrue(out["need_retrain_params"])
            self.assertTrue(any("Artifact ausente" in x for x in out["alerts"]))

    def test_walkforward_disk_fallback_and_patch_refit(self):
        with workspace_tempdir("maintenance_status_core") as td:
            root = Path(td)
            csv = root / "daily.csv"
            art = root / "ml_artifact.json"
            params = root / "params_fitted.json"
            players = root / "players_artifact.json"
            wf = root / "wf_last_report.json"
            for p in [csv, art, params, players, wf]:
                p.write_text("{}", encoding="utf-8")
            old_ts = datetime(2026, 2, 20, 12, 0, 0).timestamp()
            os.utime(wf, (old_ts, old_ts))
            os.utime(art, (old_ts, old_ts))
            os.utime(params, (old_ts, old_ts))
            out = collect_maintenance_status(
                str(csv),
                str(art),
                app_root=root,
                wf_report_path=wf,
                wf_last_report_at=None,
                file_sig_fn=lambda p: (str(p), 1, 1),
                patch_meta_guard_payload_fn=lambda a, b: {"status": "refit", "changed_leagues": ["LCK", "LPL"], "age_days": 3.5},
                now=datetime(2026, 3, 7, 8, 0, 0),
            )
            self.assertTrue(out["need_walkforward"])
            self.assertTrue(out["need_patch_recalibration"])
            self.assertTrue(out["need_retrain_params"])
            self.assertTrue(out["resolved_wf_last_report_at"])
            self.assertTrue(any("REFIT" in x for x in out["alerts"]))

    def test_consistency_and_pre_delivery_alerts(self):
        with workspace_tempdir("maintenance_status_core") as td:
            root = Path(td)
            csv = root / "daily.csv"
            art = root / "ml_artifact.json"
            params = root / "params_fitted.json"
            players = root / "players_artifact.json"
            for p in [csv, art, params, players]:
                p.write_text("{}", encoding="utf-8")
            now_dt = datetime(2026, 3, 7, 8, 0, 0)
            ts = now_dt.timestamp()
            for p in [art, params, players, csv]:
                os.utime(p, (ts, ts))
            out = collect_maintenance_status(
                str(csv),
                str(art),
                app_root=root,
                wf_report_path=root / "missing_wf_last_report.json",
                wf_last_report_at=now_dt.isoformat(timespec="seconds"),
                ml_trace_last={"consistency_guard_drift_pp": 0.15},
                consistency_tol_pp=0.05,
                pre_delivery_report={"all_ok": False, "rows": [{"Item": "ABC", "Status": "Reprovado"}]},
                file_sig_fn=lambda p: (str(p), 1, 1),
                patch_meta_guard_payload_fn=lambda a, b: {"status": "ok"},
                now=now_dt,
            )
            self.assertTrue(any("Drift de consist" in x for x in out["alerts"]))
            self.assertTrue(any("Pr" in x and "entrega" in x for x in out["alerts"]))

    def test_params_fit_running_flag(self):
        with workspace_tempdir("maintenance_status_core") as td:
            root = Path(td)
            csv = root / "daily.csv"
            art = root / "ml_artifact.json"
            params = root / "params_fitted.json"
            players = root / "players_artifact.json"
            for p in [csv, art, params, players]:
                p.write_text("{}", encoding="utf-8")
            now_dt = datetime(2026, 3, 11, 17, 0, 0)
            prm_ts = datetime(2026, 3, 10, 12, 0, 0).timestamp()
            art_ts = datetime(2026, 3, 11, 16, 50, 0).timestamp()
            os.utime(params, (prm_ts, prm_ts))
            os.utime(art, (art_ts, art_ts))
            ops_dir = root / "ops" / "runtime_fit_params" / "auto_full"
            ops_dir.mkdir(parents=True, exist_ok=True)
            progress = ops_dir / "fit_ml_params_progress.json"
            progress.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "stage": "context_models",
                        "updated_at": now_dt.isoformat(),
                        "process": {"pid": os.getpid()},
                    }
                ),
                encoding="utf-8",
            )
            out = collect_maintenance_status(
                str(csv),
                str(art),
                app_root=root,
                wf_report_path=root / "wf_last_report.json",
                wf_last_report_at=now_dt.isoformat(timespec="seconds"),
                file_sig_fn=lambda p: (str(p), 1, 1),
                patch_meta_guard_payload_fn=lambda a, b: {"status": "ok"},
                now=now_dt,
            )
            self.assertTrue(out["need_retrain_params"])
            self.assertTrue(out["params_fit_running"])
            self.assertEqual(out["params_fit_runtime"].get("scope"), "auto_full")
            self.assertTrue(any("Retreino de parâmetros em execução" in x for x in out["alerts"]))


    def test_stale_players_artifact_warns_but_does_not_flag_model_retrain(self):
        """Regressão 2026-07-14: players_artifact mais velho que o CSV ligava
        need_retrain_model — flag insaciável (o retreino roda --no-fit-players
        e não toca no players_artifact), toda troca de CSV pagava retreino
        completo de ~16min em loop. Drift de players é aviso, não gatilho."""
        with workspace_tempdir("maintenance_status_core") as td:
            root = Path(td)
            csv = root / "daily.csv"
            art = root / "ml_artifact.json"
            params = root / "params_fitted.json"
            players = root / "players_artifact.json"
            for p in [csv, art, params, players]:
                p.write_text("{}", encoding="utf-8")
            now_dt = datetime(2026, 7, 14, 15, 0, 0)
            # modelo/params recém-treinados e CSV atual; só players ficou velho
            fresh = now_dt.timestamp()
            stale = datetime(2026, 7, 1, 12, 0, 0).timestamp()
            for p in [csv, art, params]:
                os.utime(p, (fresh, fresh))
            os.utime(players, (stale, stale))
            out = collect_maintenance_status(
                str(csv),
                str(art),
                app_root=root,
                wf_report_path=root / "wf_last_report.json",
                wf_last_report_at=now_dt.isoformat(timespec="seconds"),
                file_sig_fn=lambda p: (str(p), 1, 1),
                patch_meta_guard_payload_fn=lambda a, b: {"status": "ok"},
                now=now_dt,
            )
            self.assertTrue(
                any("players_artifact" in x for x in out["alerts"]),
                out["alerts"],
            )
            self.assertFalse(out["need_retrain_model"], out["alerts"])

    def test_players_artifact_resolution_uses_custom_path(self):
        with workspace_tempdir("maintenance_status_core") as td:
            root = Path(td)
            csv = root / "daily.csv"
            artifact_dir = root / "custom"
            art = artifact_dir / "ml_artifact.json"
            players = artifact_dir / "players_artifact.json"
            params = root / "params_fitted.json"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for p in [csv, art, players, params]:
                p.write_text("{}", encoding="utf-8")
            now_dt = datetime(2026, 3, 7, 8, 0, 0)
            ts = now_dt.timestamp()
            for p in [csv, art, players, params]:
                os.utime(p, (ts, ts))
            out = collect_maintenance_status(
                str(csv),
                str(art),
                app_root=root,
                wf_report_path=root / "wf_last_report.json",
                wf_last_report_at=now_dt.isoformat(timespec="seconds"),
                file_sig_fn=lambda p: (str(p), 1, 1),
                patch_meta_guard_payload_fn=lambda a, b: {"status": "ok"},
                resolve_players_artifact_path_fn=lambda _artifact_path, _app_root: str(players),
                now=now_dt,
            )
            self.assertEqual(Path(out["players_artifact_path"]), players)
            self.assertFalse(any("Players artifact ausente" in x for x in out["alerts"]))


if __name__ == "__main__":
    unittest.main()
