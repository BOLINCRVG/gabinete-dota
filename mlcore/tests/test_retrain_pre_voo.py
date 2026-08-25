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
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app_core.live_models_status_core import build_retrain_command, launch_retrain_background


class TestPreVoo(unittest.TestCase):
    def test_script_ausente_devolve_erro_e_nao_ok(self) -> None:
        """O caso real do Dota: o clone nao tem tools/."""
        r = launch_retrain_background(project_root=".", save_models_dir="live_models")
        script = Path(build_retrain_command()[1])
        if (Path(".") / script).exists():
            self.skipTest("este repositorio TEM o script — nada a verificar aqui")
        self.assertFalse(r.get("ok"), "dizer ok sem o script foi o que escondeu o bug")
        self.assertIn("nao existe", str(r.get("error") or ""))
        self.assertIn(script.name, str(r.get("error") or ""))

    def test_o_comando_continua_no_retorno_para_diagnostico(self) -> None:
        r = launch_retrain_background(project_root=".", save_models_dir="live_models")
        self.assertTrue(r.get("command"), "sem o comando nao da para diagnosticar")

    def test_raiz_inexistente_nao_estoura(self) -> None:
        r = launch_retrain_background(project_root="/nao/existe", save_models_dir="live_models")
        self.assertIn("ok", r)


if __name__ == "__main__":
    unittest.main()
