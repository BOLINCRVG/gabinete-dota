from __future__ import annotations

import json
import unittest
from datetime import datetime
from pathlib import Path

from mlcore.tests._workspace_temp import workspace_tempdir
from app_core.bootstrap_maintenance_state_core import (
    acquire_bootstrap_maint_lock,
    bootstrap_maint_state_finish,
    bootstrap_maint_state_start,
    bootstrap_maint_state_update,
    clear_bootstrap_auto_maint_run_state,
    cleanup_stale_bootstrap_maint_lock,
    get_bootstrap_runtime_state,
    load_bootstrap_maint_report_for_fp,
    read_bootstrap_maint_state,
    read_json_file_safe,
    release_bootstrap_maint_lock,
    set_bootstrap_auto_maint_last_report,
    set_bootstrap_active_run_id,
    set_bootstrap_auto_maint_run_fp,
    set_bootstrap_last_report,
    set_bootstrap_lock_poll_n,
    set_bootstrap_status_before,
    write_json_file_atomic,
)


class BootstrapMaintenanceStateCoreTests(unittest.TestCase):
    def test_json_helpers_roundtrip(self) -> None:
        with workspace_tempdir("bootstrap_maintenance_state_core") as tmpdir:
            p = Path(tmpdir) / "x.json"
            self.assertTrue(write_json_file_atomic(p, {"a": 1}))
            self.assertEqual(read_json_file_safe(p), {"a": 1})

    def test_state_lifecycle_roundtrip(self) -> None:
        with workspace_tempdir("bootstrap_maintenance_state_core") as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            rid = bootstrap_maint_state_start(
                state_path,
                fp_key="abc",
                owner_token="owner",
                total_steps=4,
                pid_value=111,
                now_dt=datetime(2026, 3, 7, 10, 0, 0),
                now_ts=10.0,
                run_id="run_1",
            )
            self.assertEqual(rid, "run_1")
            bootstrap_maint_state_update(
                state_path,
                run_id=rid,
                fp_key="abc",
                owner_token="owner",
                event={"event": "running", "step": "model", "idx": 1, "total": 4, "status": "rodando"},
                pid_value=111,
                now_dt=datetime(2026, 3, 7, 10, 1, 0),
                now_ts=70.0,
            )
            bootstrap_maint_state_finish(
                state_path,
                run_id=rid,
                fp_key="abc",
                owner_token="owner",
                ok=True,
                report={"duration_s": 12.5, "policy_profile": "fast", "policy_reason": "test"},
                pid_value=111,
                now_dt=datetime(2026, 3, 7, 10, 2, 0),
                now_ts=130.0,
            )
            got = read_bootstrap_maint_state(state_path)
        self.assertEqual(got["run_id"], "run_1")
        self.assertEqual(got["status"], "done")
        self.assertEqual(got["result"]["policy_profile"], "fast")
        self.assertTrue(got["result"]["ok"])

    def test_acquire_and_release_lock(self) -> None:
        with workspace_tempdir("bootstrap_maintenance_state_core") as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            lock_path = Path(tmpdir) / "lock.json"
            ok, msg = acquire_bootstrap_maint_lock(
                lock_path=lock_path,
                state_path=state_path,
                fp_key="abc",
                owner_token="owner",
                stale_s=180,
                ttl_s=1800,
                now_ts=100.0,
                pid_value=999999,
            )
            self.assertTrue(ok, msg)
            release_bootstrap_maint_lock(
                lock_path=lock_path,
                state_path=state_path,
                fp_key="abc",
                owner_token="owner",
            )
            self.assertFalse(lock_path.exists())

    def test_cleanup_stale_lock_without_live_state(self) -> None:
        with workspace_tempdir("bootstrap_maintenance_state_core") as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            lock_path = Path(tmpdir) / "lock.json"
            lock_path.write_text(
                json.dumps({"fp_key": "abc", "pid": 999999, "ts": 1.0, "owner_token": "owner"}),
                encoding="utf-8",
            )
            ok, msg = cleanup_stale_bootstrap_maint_lock(
                lock_path=lock_path,
                state_path=state_path,
                fp_key="abc",
                stale_s=30,
                now_ts=100.0,
            )
            self.assertTrue(ok, msg)
            self.assertIn("cleared:", msg)
            self.assertFalse(lock_path.exists())

    def test_load_bootstrap_maint_report_for_fp(self) -> None:
        with workspace_tempdir("bootstrap_maintenance_state_core") as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(
                json.dumps({"csv_fingerprint": {"a": 1}, "ok": True}),
                encoding="utf-8",
            )
            got = load_bootstrap_maint_report_for_fp(report_path, json.dumps({"a": 1}, ensure_ascii=False, sort_keys=True))
            self.assertIsInstance(got, dict)
            self.assertTrue(got["ok"])

    def test_runtime_state_helpers(self) -> None:
        session_state = {}
        self.assertEqual(set_bootstrap_lock_poll_n(session_state, 3), 3)
        self.assertEqual(set_bootstrap_auto_maint_run_fp(session_state, "fp-1"), "fp-1")
        self.assertEqual(set_bootstrap_active_run_id(session_state, "run-1"), "run-1")
        self.assertEqual(set_bootstrap_status_before(session_state, {"severity": "warning"}), {"severity": "warning"})
        self.assertEqual(set_bootstrap_last_report(session_state, {"ok": True}), {"ok": True})
        self.assertEqual(set_bootstrap_auto_maint_last_report(session_state, {"ok": True, "run": 1}), {"ok": True, "run": 1})
        snapshot = get_bootstrap_runtime_state(session_state)
        self.assertEqual(snapshot["lock_poll_n"], 3)
        self.assertEqual(snapshot["auto_maint_ran_fp"], "fp-1")
        self.assertEqual(snapshot["active_run_id"], "run-1")
        self.assertEqual(snapshot["status_before"], {"severity": "warning"})
        self.assertEqual(snapshot["last_report"], {"ok": True})
        self.assertEqual(snapshot["auto_maint_last_report"], {"ok": True, "run": 1})
        clear_bootstrap_auto_maint_run_state(session_state)
        snapshot2 = get_bootstrap_runtime_state(session_state)
        self.assertEqual(snapshot2["auto_maint_ran_fp"], "")
        self.assertEqual(snapshot2["active_run_id"], "")


if __name__ == "__main__":
    unittest.main()
