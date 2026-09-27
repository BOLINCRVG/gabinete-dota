# -*- coding: utf-8 -*-
"""E2E smoke permanente (Regra de engenharia 4, 2026-07-14): boot → entrar →
selecionar times → Analisar Confronto, p/ o app do LoL E o clone Dota.

Cada app roda em SUBPROCESSO PRÓPRIO — obrigatório: o clone Dota importa
dota_overrides, que MUTA módulos compartilhados (labels/registro) in-place;
rodar in-process contaminava os demais testes da suíte (visto 2026-07-14:
15 falhas fantasma de labels/registro).

Guardas: NO-GO/manutenção = SKIP (estado de dados, não regressão); nunca
clica em nada que dispare retreino; Dota só roda se a pasta existir ao lado.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOTA_APP = ROOT.parent / "GabineteDOTA" / "gabinete_dota_app.py"
# A entrada do app tem nome diferente em cada repo (plays_app.py no LoL,
# gabinete_dota_app.py no Dota). Resolver aqui mantem ESTE arquivo igual
# nos dois — com o caminho fixo, o fluxo E2E nem comecava no Dota.
from app_core.streamlit_apptest_core import resolve_app_file  # noqa: E402

APP = resolve_app_file(ROOT / "plays_app.py")

_DRIVER = r"""
import sys
app_path, hint_a, hint_b = sys.argv[1], sys.argv[2], sys.argv[3]
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(app_path, default_timeout=900)
at.run()
entrar = [b for b in at.button if b.label == "Entrar no app"]
if entrar:
    entrar[0].click(); at.run()
if at.exception:
    print("RESULT|EXC|" + str(getattr(at.exception[0], "message", at.exception[0]))[:300])
    raise SystemExit(0)
t1 = [s for s in at.selectbox if s.label == "Time 1"]
t2 = [s for s in at.selectbox if s.label == "Time 2"]
if not (t1 and t2):
    print("RESULT|SKIP|sem selecao de times (manutencao/NO-GO)")
    raise SystemExit(0)
opts = [o for o in t1[0].options if o]
a = next((o for o in opts if hint_a.lower() in str(o).lower()), opts[0])
b = next((o for o in opts if hint_b.lower() in str(o).lower() and o != a), opts[1])
t1[0].select(a); t2[0].select(b)
at.run()
analisar = [x for x in at.button if x.label == "Analisar Confronto"]
if not analisar:
    print("RESULT|SKIP|botao Analisar ausente")
    raise SystemExit(0)
analisar[0].click(); at.run()
if at.exception:
    print("RESULT|EXC|" + str(getattr(at.exception[0], "message", at.exception[0]))[:300])
    raise SystemExit(0)

# A blindagem do despacho impede que uma aba quebrada derrube o app — e por
# isso ela NAO aparece mais em `at.exception`. Sem esta checagem, a rede de
# seguranca ficaria cega justamente para o defeito que derrubou o site em
# 13/09/2026. `.get` com padrao: o clone Dota pode nao ter a chave.
# `at.session_state` nao e dicionario: `.get` levanta AttributeError e
# `["chave"]` ausente levanta KeyError. Descoberto ao dirigir o app.
try:
    _falhas = list(at.session_state["_tab_falhas"] or [])
except (KeyError, AttributeError):
    _falhas = []
if _falhas:
    print("RESULT|EXC|aba quebrada -> " + "; ".join(
        "%s: %s" % (n, m) for n, m in _falhas)[:300])
    raise SystemExit(0)
tabs = [str(t.label) for t in at.tabs]
print("RESULT|OK|" + ";".join(tabs))
"""


def _run_flow(app_path: Path, hint_a: str, hint_b: str,
              extra_env: dict | None = None) -> tuple[str, str]:
    """Roda o fluxo num subprocesso isolado. Retorna (status, payload)."""
    # PYTHONIOENCODING: sem isso o driver imprime cp1252 no Windows e
    # "Visão Geral" chega como "Vis�o Geral" (mesma família do bug do
    # run_cmd, 2026-07-14).
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env or {})
    cp = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(app_path), hint_a, hint_b],
        capture_output=True, text=True, timeout=1200,
        cwd=str(app_path.parent), encoding="utf-8", errors="replace", env=env,
    )
    for line in (cp.stdout or "").splitlines():
        if line.startswith("RESULT|"):
            _, status, payload = line.split("|", 2)
            return status, payload
    return "EXC", f"driver sem RESULT (rc={cp.returncode}): {(cp.stderr or '')[-300:]}"


class TestE2ESmokeLoL(unittest.TestCase):
    def test_boot_select_analyze(self):
        # GABINETE_E2E=1: suíte SÓ-LEITURA (fase 2 da campanha) — o gate não
        # dispara manutenção real (já mutou ml_artifact no meio da suíte e
        # matou um retreino dele em 15-16/07); subprocessos ficam bloqueados.
        def _sig(name: str):
            p = ROOT / name
            return (p.stat().st_size, p.stat().st_mtime) if p.exists() else None

        # user_schedule_store: agenda DELE — teste não reescreve dado de
        # usuário (ml_trace_last fica de fora: é sink de diagnóstico,
        # escrito por design a cada análise).
        before = {n: _sig(n) for n in ("ml_artifact.json", "params_fitted.json",
                                       "players_artifact.json",
                                       "user_schedule_store.json")}
        status, payload = _run_flow(APP, "T1", "Gen",
                                    extra_env={"GABINETE_E2E": "1"})
        if status == "SKIP":
            self.skipTest(payload)
        self.assertEqual(status, "OK", f"fluxo LoL quebrou: {payload}")
        tabs = payload.split(";")
        self.assertGreaterEqual(len(tabs), 7, f"poucas tabs: {tabs}")
        self.assertIn("Visão Geral", tabs)
        after = {n: _sig(n) for n in before}
        self.assertEqual(before, after,
                         "E2E MUTOU artifacts do modelo — a suíte tem que ser só-leitura")


class TestE2ESmokeLoLViewer(unittest.TestCase):
    def test_viewer_mode_boots_straight_to_app(self):
        """Modo visualização (nuvem): sem gate de manutenção, boot direto,
        abas renderizam. É o modo do Streamlit Community Cloud."""
        status, payload = _run_flow(
            APP, "T1", "Gen",
            extra_env={"GABINETE_VIEWER": "1"},
        )
        if status == "SKIP":
            self.skipTest(payload)
        self.assertEqual(status, "OK", f"fluxo viewer quebrou: {payload}")
        tabs = payload.split(";")
        self.assertGreaterEqual(len(tabs), 7, f"poucas tabs: {tabs}")
        self.assertIn("Visão Geral", tabs)


class TestE2ESmokeDota(unittest.TestCase):
    def test_boot_select_analyze(self):
        if not DOTA_APP.exists():
            self.skipTest("GabineteDOTA não está ao lado (máquina sem o app Dota)")
        status, payload = _run_flow(DOTA_APP, "Spirit", "Falcons",
                                    extra_env={"GABINETE_E2E": "1"})
        if status == "EXC":
            # a esteira P3 pode regenerar os CSVs do Dota bem no meio do boot
            # (corrida vista 16/07): UMA repetição separa transitório de bug.
            status, payload = _run_flow(DOTA_APP, "Spirit", "Falcons",
                                        extra_env={"GABINETE_E2E": "1"})
        if status == "SKIP":
            self.skipTest(payload)
        self.assertEqual(status, "OK", f"fluxo Dota quebrou: {payload}")
        tabs = payload.split(";")
        self.assertGreaterEqual(len(tabs), 8, f"poucas tabs: {tabs}")
        self.assertIn("Megacreeps", tabs, "aba Megacreeps sumiu do Dota")
        self.assertNotIn("Dragões", tabs, "Dragões voltou ao Dota (override falhou)")


if __name__ == "__main__":
    unittest.main()
