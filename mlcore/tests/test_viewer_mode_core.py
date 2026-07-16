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


if __name__ == "__main__":
    unittest.main()
