# -*- coding: utf-8 -*-
"""Roda a SUITE INTEIRA e grava o resultado em evidence/suite_esteira.json.

POR QUE (22/10/2026). A suite so rodava a mao. Na esteira rodavam 3 modulos-
guarda -- e a nota de Codigo do LoL travava em 3,0 porque a suite falhava no
robo (PyYAML fora do requirements) sem ninguem ver. E nao ha como ler o log do
GitHub Actions daqui: por isso o resultado vira ARQUIVO versionado, que as
reguas e qualquer pessoa leem pelo git. Suite que falha calada e o defeito que
a regua das 28 existe para pegar ("pipeline com teste de integracao").

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score" (Google,
2017), secao "Computing an ML Test Score" -- a regua das 28 deste repo:
"half a point is awarded for executing the test manually [...] A full point is
awarded if there is a system in place to run that test automatically on a
repeated basis."
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Arquivo IDENTICO nos dois repositorios (LoL e Dota): a raiz vem do git.

Uso: python tools/suite_na_esteira.py      (sai com codigo 1 se a suite falhar)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _raiz() -> Path:
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parent), timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return Path(__file__).resolve().parent.parent


def ler_resultado(saida: str) -> dict:
    """Le o resumo do unittest. Puro (testavel)."""
    m = re.search(r"Ran (\d+) tests? in ([\d.]+)s", saida)
    ran = int(m.group(1)) if m else 0
    segundos = float(m.group(2)) if m else 0.0
    final = re.findall(r"^(OK|FAILED)\b(.*)$", saida, re.M)
    status, extra = (final[-1] if final else ("SEM RESULTADO", ""))
    def _n(chave):
        mm = re.search(chave + r"=(\d+)", extra)
        return int(mm.group(1)) if mm else 0
    return {"testes": ran, "segundos": segundos, "status": status,
            "falhas": _n("failures"), "erros": _n("errors"), "pulados": _n("skipped"),
            "ok": status == "OK" and ran > 0}


def main() -> int:
    raiz = _raiz()
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", GABINETE_VIEWER="0")
    cp = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "mlcore/tests", "-p", "test_*.py"],
                        capture_output=True, text=True, cwd=str(raiz), env=env,
                        encoding="utf-8", errors="replace")
    saida = (cp.stdout or "") + (cp.stderr or "")
    res = ler_resultado(saida)
    res["gerado_em"] = datetime.now().isoformat(timespec="seconds")
    res["duracao_total_s"] = round(time.time() - t0, 1)
    res["primeiras_falhas"] = re.findall(r"^(?:FAIL|ERROR): (.+)$", saida, re.M)[:20]
    destino = raiz / "evidence" / "suite_esteira.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(saida[-3000:])
    print(f"[suite_na_esteira] {res['status']} | {res['testes']} testes | "
          f"{res['falhas']} falhas, {res['erros']} erros | salvo em evidence/suite_esteira.json")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
