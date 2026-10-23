# -*- coding: utf-8 -*-
"""TREINO REPRODUZIVEL (23/10/2026): treinar duas vezes no mesmo CSV da o mesmo modelo?

Item "treino reproduzivel" da regua das 28, que estava em 0 declarado ("nao ha
semente nem teste que rode duas vezes e compare"). Medido em 23/10/2026 no CSV
real do LoL: duas rodadas de mlcore.train_offline deram artefatos identicos
byte a byte, salvo o carimbo trained_at. O treino nao sorteia nada; este teste
garante que continue assim (um shuffle ou set() iterado mudaria o modelo de
uma rodada para outra sem ninguem ver).

Roda na suite, que o robo roda todo dia (~80 s). Grava tudo em pasta
temporaria: nao toca nos arquivos do modelo. Arquivo identico nos dois
repositorios (pula onde o treino ou o CSV nao existem).

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
infraestrutura "Training is reproducible" ("train two models on the same data,
and observe any differences").
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CARIMBOS = {"trained_at", "updated_at", "generated_at", "created_at", "saved_at"}


def _sem_carimbo(obj):
    if isinstance(obj, dict):
        return {k: _sem_carimbo(v) for k, v in obj.items() if k not in CARIMBOS}
    if isinstance(obj, list):
        return [_sem_carimbo(v) for v in obj]
    return obj


def _diferencas(a, b, caminho="$", saida=None):
    """Caminhos onde a e b diferem (no maximo 10), para o erro dizer ONDE."""
    saida = [] if saida is None else saida
    if len(saida) >= 10:
        return saida
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            _diferencas(a.get(k), b.get(k), f"{caminho}.{k}", saida)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diferencas(x, y, f"{caminho}[{i}]", saida)
    elif a != b:
        saida.append(f"{caminho}: {str(a)[:60]} != {str(b)[:60]}")
    return saida


def _csv():
    achados = sorted((RAIZ / "CSV_DIARIO").glob("*.csv"))
    return achados[-1] if achados else None


@unittest.skipUnless((RAIZ / "mlcore" / "train_offline.py").exists() and _csv(),
                     "sem treino ou sem CSV neste repositorio")
class TestTreinoReproduzivel(unittest.TestCase):

    def _treinar(self, pasta: Path) -> dict:
        saidas = {"ml": pasta / "ml.json", "players": pasta / "players.json",
                  "pscore": pasta / "pscore.json"}
        cmd = [sys.executable, "-m", "mlcore.train_offline", "--csv", str(_csv()),
               "--out", str(saidas["ml"]), "--holdout-days", "30"]
        ajuda = subprocess.run([sys.executable, "-m", "mlcore.train_offline", "--help"],
                               cwd=str(RAIZ), capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=300).stdout
        if "--players-out" in ajuda:
            cmd += ["--players-out", str(saidas["players"]), "--pscore-out", str(saidas["pscore"])]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run(cmd, cwd=str(RAIZ), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800, env=env)
        self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
        # o artefato anota onde gravou os irmaos (players_artifact: <pasta>/...);
        # a pasta temporaria muda de rodada para rodada, o modelo nao
        def _ler(p: Path):
            obj = json.loads(p.read_text(encoding="utf-8"))
            texto = json.dumps(obj).replace(json.dumps(str(pasta))[1:-1], "<PASTA>")
            return _sem_carimbo(json.loads(texto))
        return {k: _ler(p) for k, p in saidas.items() if p.exists()}

    def test_duas_rodadas_dao_o_mesmo_modelo(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            m1, m2 = self._treinar(Path(a)), self._treinar(Path(b))
        self.assertIn("ml", m1)
        self.assertEqual(sorted(m1), sorted(m2))
        for k in m1:
            self.assertEqual(_diferencas(m1[k], m2[k]), [], f"{k}: duas rodadas deram modelos diferentes")


if __name__ == "__main__":
    unittest.main()
