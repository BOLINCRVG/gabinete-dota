# -*- coding: utf-8 -*-
"""Blindagem do fit PROFUNDO contra o fit rápido da manutenção (19/07).

Incidente real: o fit de ~11h promovido em 18/07 (players_context com 17
ligas) foi sobrescrito às 09:33 de 19/07 por um fit rápido disparado pela
manutenção automática ao abrir o app — a segmentação por liga foi a zero.
A guarda faz backup e avisa antes de deixar isso acontecer.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _params(n_ligas: int) -> dict:
    return {
        "players_context_fit": {
            "settings_by_league": {f"L{i}": {"strength": 0.5} for i in range(n_ligas)},
            "baseline_off": {"logloss": 0.60}, "best_on": {"logloss": 0.59},
        }
    }


class TestDeepParamsGuard(unittest.TestCase):
    """A guarda vive dentro de run_auto_maintenance_now (closure). Testamos a
    REGRA que ela implementa, sobre a mesma estrutura de arquivo real."""

    def _deep_info(self, path: Path) -> dict:
        try:
            d = json.loads(path.read_text(encoding="utf-8-sig"))
            ligas = (d.get("players_context_fit") or {}).get("settings_by_league") or {}
            return {"deep": bool(ligas), "n_ligas": len(ligas)}
        except (OSError, ValueError, TypeError):
            return {"deep": False, "n_ligas": 0}

    def test_detecta_fit_profundo(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "params_fitted.json"
            p.write_text(json.dumps(_params(17)), encoding="utf-8")
            info = self._deep_info(p)
            self.assertTrue(info["deep"])
            self.assertEqual(info["n_ligas"], 17)

    def test_fit_raso_nao_e_profundo(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "params_fitted.json"
            p.write_text(json.dumps(_params(0)), encoding="utf-8")
            self.assertFalse(self._deep_info(p)["deep"])

    def test_arquivo_invalido_nao_quebra(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "params_fitted.json"
            p.write_text("{lixo", encoding="utf-8")
            self.assertFalse(self._deep_info(p)["deep"])  # não levanta
            self.assertFalse(self._deep_info(Path(td) / "nao_existe.json")["deep"])

    def test_guarda_existe_no_runner(self):
        """A guarda tem que estar CHAMADA no caminho do fit (não só definida)."""
        src = (Path(__file__).resolve().parents[2] / "app_core"
               / "csv_maintenance_runner_core.py").read_text(encoding="utf-8", errors="replace")
        self.assertIn("_guard_deep_params_before_fit", src)
        self.assertIn("_guard_msg = _guard_deep_params_before_fit(speed_profile)", src)
        self.assertIn("deep_before_auto_fast_", src)  # nome do backup


if __name__ == "__main__":
    unittest.main()
