"""Tests: detecção do full fit (Qualidade profunda) + card de status.

Antes o card era hardcoded ('full fit não rodado'). Agora detecta de verdade:
rodado? (params_fitted speed_profile==full) e desatualizado? (artefato mais novo
que o fit). Estes testes travam os 3 estados.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app_core.maintenance_status_core import detect_deep_fit
from app_core.maintenance_summary_core import _short_date, build_maintenance_quality_status

_T = 1_900_000_000  # epoch fixo de referência


class TestDetectDeepFit(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _params(self, profile: str, epoch: float) -> Path:
        iso = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
        p = self.d / "params_fitted.json"
        p.write_text(f'{{"speed_profile": "{profile}", "generated_at": "{iso}"}}', encoding="utf-8")
        return p

    def _art(self, mtime: float) -> Path:
        a = self.d / "ml_artifact.json"
        a.write_text("{}", encoding="utf-8")
        os.utime(a, (mtime, mtime))
        return a

    def test_no_params(self):
        out = detect_deep_fit(self.d / "nope.json", self.d / "art.json")
        self.assertFalse(out["done"])
        self.assertFalse(out["stale"])

    def test_not_full_profile(self):
        out = detect_deep_fit(self._params("fast", _T), self._art(_T - 3600))
        self.assertFalse(out["done"])

    def test_full_aligned(self):
        out = detect_deep_fit(self._params("full", _T), self._art(_T - 3600))  # artefato + velho
        self.assertTrue(out["done"])
        self.assertFalse(out["stale"])

    def test_full_stale(self):
        out = detect_deep_fit(self._params("full", _T), self._art(_T + 7200))  # artefato + novo
        self.assertTrue(out["done"])
        self.assertTrue(out["stale"])


class TestDeepCard(unittest.TestCase):

    def test_pending_when_never_run(self):
        r = build_maintenance_quality_status({"deep_fit_done": False})["deep"]
        self.assertEqual(r["state"], "pending")
        self.assertIn("não foi rodado", r["text"])

    def test_ready_when_aligned(self):
        r = build_maintenance_quality_status(
            {"deep_fit_done": True, "deep_fit_stale": False, "deep_fit_at": "2026-06-04T11:40:09+00:00"}
        )["deep"]
        self.assertEqual(r["state"], "ready")
        self.assertIn("04/06/2026", r["text"])

    def test_warn_when_stale(self):
        r = build_maintenance_quality_status(
            {"deep_fit_done": True, "deep_fit_stale": True, "deep_fit_at": "2026-06-04T11:40:09+00:00"}
        )["deep"]
        self.assertEqual(r["state"], "ready_warn")
        self.assertIn("vale rodar", r["text"])

    def test_short_date(self):
        self.assertEqual(_short_date("2026-06-04T11:40:09+00:00"), "04/06/2026")
        self.assertEqual(_short_date(""), "")
        self.assertEqual(_short_date("lixo"), "")


if __name__ == "__main__":
    unittest.main()
