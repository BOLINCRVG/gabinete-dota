from __future__ import annotations

import json
import unittest
from datetime import datetime
from pathlib import Path

from mlcore.tests._workspace_temp import workspace_tempdir
from app_core.bootstrap_maintenance_state_core import (
    _lock_busy_reason,
    _stale_clear_decision,
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


class LockBusyReasonTests(unittest.TestCase):
    """Decisão PURA de 'lock ocupado' — extraída de acquire (19/07). É a
    semântica de concorrência que causou o incidente do fit sobrescrito;
    agora cada ramo tem teste próprio."""

    _BASE = dict(
        fp_key="fpX", lock_ts=1000.0, lock_fp="fpX", lock_pid="123",
        lock_age=10.0, lock_pid_alive=True, same_owner=False,
        state_fp="fpX", state_status="running", hb_age=2.0,
        ttl=1800.0, same_fp_busy_s=180.0, hb_busy_s=180.0, other_fp_grace_s=45.0,
    )

    def _r(self, **over):
        kw = dict(self._BASE); kw.update(over)
        return _lock_busy_reason(**kw)

    def test_sem_lock_nao_ocupado(self):
        self.assertEqual(self._r(lock_ts=0.0, lock_fp=""), "")

    def test_mesmo_dono_nunca_ocupado(self):
        self.assertEqual(self._r(same_owner=True), "")

    def test_outro_fp_dentro_do_grace_ocupado(self):
        self.assertIn("lock_busy(fp=fpY", self._r(lock_fp="fpY", lock_age=10.0))

    def test_outro_fp_passou_grace_mas_dentro_do_ttl_ocupado(self):
        # passou grace (45s) mas < ttl e pid vivo -> busy pelo ramo do ttl
        self.assertIn("lock_busy(fp=fpY", self._r(lock_fp="fpY", lock_age=100.0))

    def test_outro_fp_passou_ttl_livre(self):
        self.assertEqual(self._r(lock_fp="fpY", lock_age=2000.0), "")

    def test_mesmo_fp_heartbeat_fresco_ocupado(self):
        self.assertIn("lock_busy_same_fp", self._r(hb_age=2.0))

    def test_mesmo_fp_heartbeat_velho_livre(self):
        # running mas hb velho => stale => não ocupa (deixa readquirir)
        self.assertEqual(self._r(hb_age=500.0, lock_age=500.0), "")

    def test_mesmo_fp_sem_state_dentro_da_janela_ocupado(self):
        self.assertIn("lock_busy_same_fp", self._r(state_status="", lock_age=10.0))

    def test_pid_morto_nao_ocupa(self):
        self.assertEqual(self._r(lock_pid_alive=False, state_status=""), "")


class StaleClearDecisionTests(unittest.TestCase):
    """Decisão PURA de 'pode limpar lock obsoleto' — extraída de cleanup.
    Ordem de prioridade dos motivos preservada."""

    def test_pid_morto_limpa(self):
        ok, why = _stale_clear_decision(pid_alive=False, has_live_state=False,
                                        lock_age=30.0, hb_age=0.0, stale_limit=180.0)
        self.assertTrue(ok)
        self.assertIn("pid_dead", why)

    def test_pid_morto_recem_criado_nao_limpa(self):
        # lock_age < 5s: proteção contra corrida de criação
        ok, why = _stale_clear_decision(pid_alive=False, has_live_state=False,
                                        lock_age=2.0, hb_age=0.0, stale_limit=180.0)
        self.assertFalse(ok)

    def test_sem_state_vivo_e_velho_limpa(self):
        ok, why = _stale_clear_decision(pid_alive=True, has_live_state=False,
                                        lock_age=300.0, hb_age=0.0, stale_limit=180.0)
        self.assertTrue(ok)
        self.assertIn("no_live_state", why)

    def test_running_sem_heartbeat_limpa(self):
        ok, why = _stale_clear_decision(pid_alive=True, has_live_state=True,
                                        lock_age=200.0, hb_age=400.0, stale_limit=180.0)
        self.assertTrue(ok)
        self.assertIn("running_state_no_heartbeat", why)

    def test_running_com_heartbeat_fresco_nao_limpa(self):
        ok, why = _stale_clear_decision(pid_alive=True, has_live_state=True,
                                        lock_age=200.0, hb_age=5.0, stale_limit=180.0)
        self.assertFalse(ok)
        self.assertEqual(why, "not_stale_enough")

    def test_prioridade_pid_morto_vence(self):
        # pid morto E sem state: o motivo reportado é o de pid morto (1º)
        ok, why = _stale_clear_decision(pid_alive=False, has_live_state=False,
                                        lock_age=300.0, hb_age=400.0, stale_limit=180.0)
        self.assertTrue(ok)
        self.assertIn("pid_dead", why)


if __name__ == "__main__":
    unittest.main()
