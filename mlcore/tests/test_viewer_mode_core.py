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
        with mock.patch.dict(os.environ, {"GABINETE_VIEWER": ""}):
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


if __name__ == "__main__":
    unittest.main()
