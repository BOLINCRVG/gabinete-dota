# -*- coding: utf-8 -*-
"""O retreino ao vivo nao pode dizer "ok" quando nem comecou.

Achado 16/08/2026: o Dota estava com ZERO modelos de estado ao vivo — a aba mais
validada do projeto (no LoL vale +9.87 a +29.28pp) simplesmente nao existia la.
A causa: `launch_retrain_background` devolvia ok=True assim que o Popen
INICIAVA. E o Popen sucede mesmo com o script inexistente, porque quem existe e
o python.exe; quem falha e o Python depois, ao abrir o arquivo.

Resultado: o app tentava todo dia, "dava certo", gravava o carimbo de tentativa,
e a unica pista era uma linha num retrain.log que ninguem abre:
    can't open file '...GabineteDOTA\tools\retrain_live_models_all.py'

Quem olhasse o last_auto_retrain.txt veria a data de HOJE e concluiria que estava
tudo funcionando. Mesma anatomia do resto do dia: o carimbo e escrito antes de
verificar o resultado.

╔══ FONTES / CONSERTO DE 04/09/2026 (divida D7) ══════════════════════════════╗
║ Este arquivo ERA a causa de a suite retreinar os modelos de PRODUCAO.       ║
║                                                                             ║
║ Ele chamava `launch_retrain_background(project_root=".")` — a raiz real e o ║
║ `live_models/` real. Dois problemas:                                        ║
║                                                                             ║
║ 1. EFEITO COLATERAL. Cada rodada da suite disparava o retreino de verdade e ║
║    sobrescrevia os 12 arquivos dos 4 minutos. Em 04/09 o retreino acidental ║
║    produziu modelo PIOR (ganho 0,0366 no minuto 10 contra os 0,0645 de      ║
║    producao). Enquanto o repo morou em caminho com acento isso ficou        ║
║    escondido, porque o acento e que impedia a escrita.                      ║
║                                                                             ║
║ 2. O TESTE NEM RODAVA. A chamada estava ANTES da checagem de existencia:    ║
║    lancava o retreino e SO ENTAO via que o script existe neste repo e dava  ║
║    skipTest. Ou seja: pagava o efeito colateral e nao verificava nada.      ║
║                                                                             ║
║ Conserto: raiz TEMPORARIA. O guarda de pre-voo passa a ser exercitado de    ║
║ verdade (o script nao existe na pasta temporaria, entao a funcao devolve    ║
║ ok=False sem lancar processo nenhum), e a suite deixa de tocar em producao. ║
║                                                                             ║
║ Principio: teste hermetico — so o conteudo do proprio teste decide se ele   ║
║ passa; sem dependencia do sistema de arquivos real e SEM efeito colateral,  ║
║ para poder rodar em qualquer lugar, a qualquer hora, com o mesmo resultado. ║
║ Injecao de dependencia (aqui, `project_root`) e o meio de trocar o que e do ║
║ ambiente por algo controlado pelo teste.                                    ║
║ https://testrigor.com/blog/hermetic-testing/                                ║
║ https://fuchsia.dev/fuchsia-src/contribute/testing/scope                    ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app_core.live_models_status_core import build_retrain_command, launch_retrain_background


class TestPreVoo(unittest.TestCase):
    """NUNCA passar project_root real aqui: a funcao LANCA o retreino."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = self._tmp.name          # raiz vazia: sem script de retreino

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_script_ausente_devolve_erro_e_nao_ok(self) -> None:
        """O caso real do Dota de 16/08: a raiz nao tem o script."""
        r = launch_retrain_background(project_root=self.raiz,
                                      save_models_dir="live_models")
        script = Path(build_retrain_command()[1])
        self.assertFalse(r.get("ok"), "dizer ok sem o script foi o que escondeu o bug")
        self.assertIn("nao existe", str(r.get("error") or ""))
        self.assertIn(script.name, str(r.get("error") or ""))

    def test_nao_lanca_processo_quando_o_script_falta(self) -> None:
        """O guarda tem que barrar ANTES do Popen — senao nao e pre-voo."""
        r = launch_retrain_background(project_root=self.raiz,
                                      save_models_dir="live_models")
        self.assertIsNone(r.get("pid"), "barrou depois de lancar: nao e pre-voo")
        self.assertFalse((Path(self.raiz) / "live_models" / "retrain.pid").exists(),
                         "escreveu pid: chegou a lancar")

    def test_o_comando_continua_no_retorno_para_diagnostico(self) -> None:
        r = launch_retrain_background(project_root=self.raiz,
                                      save_models_dir="live_models")
        self.assertTrue(r.get("command"), "sem o comando nao da para diagnosticar")

    def test_raiz_inexistente_nao_estoura(self) -> None:
        r = launch_retrain_background(project_root="/nao/existe",
                                      save_models_dir="live_models")
        self.assertIn("ok", r)

    def test_o_comando_aponta_para_a_fase6_deste_repo(self) -> None:
        """Puro, sem efeito colateral: so monta o comando e confere o alvo."""
        cmd = build_retrain_command()
        self.assertGreater(len(cmd), 1)
        self.assertTrue(str(cmd[1]).endswith(".py"), f"alvo estranho: {cmd[1]}")


if __name__ == "__main__":
    unittest.main()
