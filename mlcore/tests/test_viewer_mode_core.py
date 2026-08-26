# -*- coding: utf-8 -*-
"""Testes do modo visualização + botão 'Atualizar agora' (2026-07-16)."""
import os
import unittest
from unittest import mock

from app_core.viewer_mode_core import (
    dispatch_data_workflow,
    is_viewer_mode,
)


class _FakeResp:
    def __init__(self, code):
        self.status_code = code


class TestViewerFlag(unittest.TestCase):
    def test_env_flag_controls_mode(self):
        with mock.patch.dict(os.environ, {"GABINETE_VIEWER": "1"}):
            self.assertTrue(is_viewer_mode())
        # "0" e nao "" (26/08/2026): vazio significa "nao setado" e cai no
        # secrets.toml, entao este teste dependia de o dev NAO ter o secret —
        # no repo do Dota ele existe e o teste falhava sem nada estar errado.
        # "0" testa o contrato de verdade: a env manda nos dois sentidos.
        with mock.patch.dict(os.environ, {"GABINETE_VIEWER": "0"}):
            self.assertFalse(is_viewer_mode())

    def test_truthy_variants(self):
        for v in ("true", "YES", "on"):
            with mock.patch.dict(os.environ, {"GABINETE_VIEWER": v}):
                self.assertTrue(is_viewer_mode(), v)


class TestDispatchWorkflow(unittest.TestCase):
    def test_204_is_success_and_targets_right_url(self):
        seen = {}

        def fake_post(url, **kw):
            seen["url"] = url
            seen["auth"] = kw.get("headers", {}).get("Authorization", "")
            return _FakeResp(204)

        ok, msg = dispatch_data_workflow(token="tok123", repo="X/y", post_fn=fake_post)
        self.assertTrue(ok, msg)
        self.assertIn("/repos/X/y/actions/workflows/dados_diarios.yml/dispatches", seen["url"])
        self.assertEqual(seen["auth"], "Bearer tok123")

    def test_non_204_is_explained_failure(self):
        ok, msg = dispatch_data_workflow(
            token="t", repo="X/y", post_fn=lambda *a, **k: _FakeResp(401)
        )
        self.assertFalse(ok)
        self.assertIn("401", msg)

    def test_network_error_never_raises(self):
        def boom(*a, **k):
            raise ConnectionError("offline")

        ok, msg = dispatch_data_workflow(token="t", repo="X/y", post_fn=boom)
        self.assertFalse(ok)
        self.assertIn("ConnectionError", msg)


class _FakeStWatch:
    """Duble mínimo do st p/ o botão 🔄: grava o que foi renderizado."""

    def __init__(self):
        self.session_state = {}
        self.captions = []
        self.infos = []
        self.successes = []
        self.warnings = []
        self.reran = False

    def caption(self, text, **kw):
        self.captions.append(str(text))

    def info(self, text, **kw):
        self.infos.append(str(text))

    def success(self, text, **kw):
        self.successes.append(str(text))

    def warning(self, text, **kw):
        self.warnings.append(str(text))

    def error(self, text, **kw):
        pass

    def button(self, *a, **kw):
        return False

    def rerun(self):
        self.reran = True


class TestUpdateButtonCooldownKeepsWatch(unittest.TestCase):
    """BUG 18/07 (report dele): no cooldown pós-clique, o return precoce
    matava a tela de acompanhamento no re-render seguinte. O cooldown agora
    TAMBÉM renderiza o watch (que expira sozinho em 12 min)."""

    def test_cooldown_still_renders_watch(self):
        import time as _t

        from app_core import viewer_mode_core as vm

        fake = _FakeStWatch()
        now = _t.time()
        fake.session_state["_upd_now_last_ts"] = now  # cooldown ativo
        fake.session_state["_button_watch_t0"] = now  # acompanhamento ativo

        prog = {"upload_commit": False, "csv_commit": False,
                "data_run": "rodando", "retrain": "aguardando",
                "model_commit": False}
        with mock.patch.object(vm, "_gh_dispatch_config", return_value=("tok", "X/y")), \
                mock.patch.object(vm, "check_publish_progress", return_value=prog), \
                mock.patch("time.sleep"):
            vm.render_update_now_button(fake)

        self.assertTrue(any("aguarda" in c for c in fake.captions), fake.captions)
        self.assertTrue(
            any("Acompanhamento da atualiza" in i for i in fake.infos),
            f"watch sumiu no cooldown: infos={fake.infos} warnings={fake.warnings}",
        )

    def test_cooldown_without_watch_only_caption(self):
        """Sem watch ativo (t0 zerado/expirado), o cooldown mostra só o caption."""
        import time as _t

        from app_core import viewer_mode_core as vm

        fake = _FakeStWatch()
        fake.session_state["_upd_now_last_ts"] = _t.time()

        with mock.patch.object(vm, "_gh_dispatch_config", return_value=("tok", "X/y")):
            vm.render_update_now_button(fake)

        self.assertTrue(any("aguarda" in c for c in fake.captions))
        self.assertEqual(fake.infos, [])


class TestCsvUpload(unittest.TestCase):
    def _csv_bytes(self, n_games, dmax):
        rows = ["gameid,date"]
        for i in range(n_games):
            rows.append(f"g{i},{dmax}")
        return "\n".join(rows).encode()

    def test_valid_upload_against_missing_current(self):
        from app_core.viewer_mode_core import validate_csv_upload

        ok, msg, info = validate_csv_upload(self._csv_bytes(10, "2026-07-16"), "Z:/nao/existe.csv")
        self.assertTrue(ok, msg)
        self.assertEqual(info["games"], 10)

    def test_regression_rejected(self):
        import tempfile, os
        from app_core.viewer_mode_core import validate_csv_upload

        with tempfile.TemporaryDirectory() as td:
            cur = os.path.join(td, "cur.csv")
            with open(cur, "wb") as f:
                f.write(self._csv_bytes(20, "2026-07-16"))
            ok, msg, _ = validate_csv_upload(self._csv_bytes(5, "2026-07-10"), cur)
            self.assertFalse(ok)
            self.assertIn("REGRESS", msg)

    def test_same_content_rejected_as_no_news(self):
        import tempfile, os
        from app_core.viewer_mode_core import validate_csv_upload

        with tempfile.TemporaryDirectory() as td:
            cur = os.path.join(td, "cur.csv")
            with open(cur, "wb") as f:
                f.write(self._csv_bytes(10, "2026-07-16"))
            ok, msg, _ = validate_csv_upload(self._csv_bytes(10, "2026-07-16"), cur)
            self.assertFalse(ok)
            self.assertIn("Sem novidade", msg)

    def test_garbage_rejected(self):
        from app_core.viewer_mode_core import validate_csv_upload

        ok, msg, _ = validate_csv_upload(b"\x00\x01isto nao e csv", "Z:/x.csv")
        self.assertFalse(ok)

    def test_commit_via_real_git_end_to_end(self):
        """Regressão 2026-07-16: NENHUMA API do GitHub aceita o CSV de 47MB
        ('input too large... create the blob in a local clone'). O upload
        agora é git de verdade — validado aqui contra um repositório local."""
        import subprocess
        import tempfile
        from pathlib import Path as P

        from app_core.viewer_mode_core import commit_csv_to_repo

        def _git(*args, cwd=None):
            cp = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
            self.assertEqual(cp.returncode, 0, cp.stderr)
            return cp

        with tempfile.TemporaryDirectory() as td:
            bare = P(td) / "remote.git"
            seed = P(td) / "seed"
            _git("init", "--bare", "-b", "master", str(bare))
            _git("clone", str(bare), str(seed))
            (seed / "CSV_DIARIO").mkdir()
            (seed / "CSV_DIARIO" / "x.csv").write_bytes(self._csv_bytes(3, "2026-07-10"))
            _git("-C", str(seed), "config", "user.email", "t@t")
            _git("-C", str(seed), "config", "user.name", "t")
            _git("-C", str(seed), "add", "-A")
            _git("-C", str(seed), "commit", "-m", "seed")
            _git("-C", str(seed), "push", "origin", "master")

            new_bytes = self._csv_bytes(9, "2026-07-16")
            ok, msg = commit_csv_to_repo(
                new_bytes, token="irrelevante", repo="x/y",
                repo_path="CSV_DIARIO/x.csv", message="upload teste",
                clone_url=bare.as_uri(),
            )
            self.assertTrue(ok, msg)
            shown = subprocess.run(
                ["git", "--git-dir", str(bare), "show", "master:CSV_DIARIO/x.csv"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertEqual(shown.stdout.encode(), new_bytes.replace(b"\r\n", b"\n"))
            log = subprocess.run(
                ["git", "--git-dir", str(bare), "log", "-1", "--format=%s"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            self.assertIn("upload teste", log.stdout)

    def test_commit_failure_is_explained(self):
        from app_core.viewer_mode_core import commit_csv_to_repo

        class _CP:
            returncode = 128
            stderr = "fatal: could not read from remote repository"
            stdout = ""

        ok, msg = commit_csv_to_repo(
            b"d", token="t", repo="U/r", repo_path="CSV_DIARIO/x.csv",
            message="m", run_fn=lambda *a, **k: _CP(),
        )
        self.assertFalse(ok)
        self.assertIn("clone falhou", msg)
        self.assertIn("remote repository", msg)


class TestPublishProgress(unittest.TestCase):
    """Tela de acompanhamento pós-upload (pedido dele 16/07)."""

    class _R:
        def __init__(self, code, payload):
            self.status_code = code
            self._p = payload

        def json(self):
            return self._p

    def _get(self, commits, run):
        def fake_get(url, **kw):
            if "/commits?" in url:
                return self._R(200, commits)
            return self._R(200, {"workflow_runs": [run] if run else []})
        return fake_get

    def test_full_cascade_detected(self):
        from app_core.viewer_mode_core import check_publish_progress

        commits = [
            {"commit": {"message": "Robô: modelo atualizado (incremental pós-CSV novo)"}},
            {"commit": {"message": "Upload manual pelo site: x.csv"}},
        ]
        run = {"status": "completed", "conclusion": "success",
               "created_at": "2026-07-16T18:18:25Z"}
        p = check_publish_progress(token="t", repo="U/r",
                                   since_iso="2026-07-16T18:00:00Z",
                                   get_fn=self._get(commits, run))
        self.assertTrue(p["upload_commit"])
        self.assertEqual(p["retrain"], "ok")
        self.assertTrue(p["model_commit"])

    def test_retrain_running_and_old_run_ignored(self):
        from app_core.viewer_mode_core import check_publish_progress

        run_old = {"status": "completed", "conclusion": "success",
                   "created_at": "2026-07-16T10:00:00Z"}
        p = check_publish_progress(token="t", repo="U/r",
                                   since_iso="2026-07-16T18:00:00Z",
                                   get_fn=self._get([], run_old))
        self.assertEqual(p["retrain"], "aguardando")

        run_now = {"status": "in_progress", "conclusion": None,
                   "created_at": "2026-07-16T18:05:00Z"}
        p2 = check_publish_progress(token="t", repo="U/r",
                                    since_iso="2026-07-16T18:00:00Z",
                                    get_fn=self._get([], run_now))
        self.assertEqual(p2["retrain"], "rodando")

    def test_network_failure_never_raises(self):
        from app_core.viewer_mode_core import check_publish_progress

        def boom(*a, **k):
            raise ConnectionError

        p = check_publish_progress(token="t", repo="U/r",
                                   since_iso="x", get_fn=boom)
        self.assertEqual(p["retrain"], "aguardando")

    def test_button_mode_detects_robot_without_csv(self):
        """Bug dele 17/07: clicou 🔄, robô rodou (cota bloqueou) e a tela
        ficou muda. Agora: data_run=ok + csv_commit=False = mensagem honesta."""
        from app_core.viewer_mode_core import check_publish_progress

        run_ok = {"status": "completed", "conclusion": "success",
                  "created_at": "2026-07-17T10:05:00Z"}

        def fake_get(url, **kw):
            if "/commits?" in url:
                return self._R(200, [])  # nenhum commit novo
            return self._R(200, {"workflow_runs": [run_ok]})

        p = check_publish_progress(token="t", repo="U/r",
                                   since_iso="2026-07-17T10:00:00Z",
                                   get_fn=fake_get, include_data_run=True)
        self.assertEqual(p["data_run"], "ok")
        self.assertFalse(p["csv_commit"])

    def test_button_mode_full_cascade(self):
        from app_core.viewer_mode_core import check_publish_progress

        commits = [
            {"commit": {"message": "Robô: modelo atualizado (incremental pós-CSV novo)"}},
            {"commit": {"message": "Robô: CSV diário do Oracle's Elixir atualizado"}},
        ]
        run_ok = {"status": "completed", "conclusion": "success",
                  "created_at": "2026-07-17T10:05:00Z"}

        def fake_get(url, **kw):
            if "/commits?" in url:
                return self._R(200, commits)
            return self._R(200, {"workflow_runs": [run_ok]})

        p = check_publish_progress(token="t", repo="U/r",
                                   since_iso="2026-07-17T10:00:00Z",
                                   get_fn=fake_get, include_data_run=True)
        self.assertTrue(p["csv_commit"])
        self.assertTrue(p["model_commit"])
        self.assertEqual(p["data_run"], "ok")
        self.assertEqual(p["retrain"], "ok")


if __name__ == "__main__":
    unittest.main()
