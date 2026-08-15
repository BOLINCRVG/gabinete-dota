# -*- coding: utf-8 -*-
"""Smoke das páginas de navegação de topo (Radar e Meta) — fatia 3 do
fatiamento do plays_app (2026-07-18). O E2E principal cobre só a página
Plays; estas páginas renderizam por _nav_pending e encerram com st.stop().
Roda em subprocesso (mesma razão do test_e2e_smoke: isolamento).
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_DRIVER = r"""
import sys
page = sys.argv[1]
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("plays_app.py", default_timeout=900)
at.run()
entrar = [b for b in at.button if b.label == "Entrar no app"]
if entrar:
    entrar[0].click(); at.run()
at.session_state["_nav_pending"] = page
at.run()
if at.exception:
    print("RESULT|EXC|" + str(getattr(at.exception[0], "message", at.exception[0]))[:300])
else:
    print("RESULT|OK|" + page)
"""


def _run_page(page: str) -> tuple[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["GABINETE_E2E"] = "1"
    cp = subprocess.run(
        [sys.executable, "-c", _DRIVER, page],
        capture_output=True, text=True, timeout=1200,
        cwd=str(ROOT), encoding="utf-8", errors="replace", env=env,
    )
    for line in (cp.stdout or "").splitlines():
        if line.startswith("RESULT|"):
            _, status, payload = line.split("|", 2)
            return status, payload
    return "EXC", f"driver sem RESULT (rc={cp.returncode}): {(cp.stderr or '')[-300:]}"


class TestNavPagesSmoke(unittest.TestCase):
    def test_radar_page_renders(self) -> None:
        status, payload = _run_page("Radar")
        self.assertEqual(status, "OK", f"página Radar quebrou: {payload}")

    def test_meta_page_renders(self) -> None:
        status, payload = _run_page("Meta")
        self.assertEqual(status, "OK", f"página Meta quebrou: {payload}")

    def test_tecnico_page_renders(self) -> None:
        # Fecha a última página do top-nav sem cobertura E2E (força-tarefa 23/07):
        # endurece a rede pro refactor da espinha globals->ctx.
        status, payload = _run_page("Tecnico")
        self.assertEqual(status, "OK", f"página Tecnico quebrou: {payload}")

    def test_live_bet_nav_is_merged_into_radar(self) -> None:
        """Live-Bet foi fundida no Radar (2026-07): navegação antiga cai lá.
        Trava a semântica que torna o branch Live-Bet do plays_app inerte."""
        from app_core.navigation_state_core import resolve_nav_page, set_nav_pending
        state: dict = {}
        set_nav_pending(state, "Live-Bet")
        self.assertEqual(resolve_nav_page(state), "Radar")


if __name__ == "__main__":
    unittest.main()
