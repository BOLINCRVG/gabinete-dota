# -*- coding: utf-8 -*-
"""O robo de retreino ao vivo do Dota (22/10/2026).

Os modelos ao vivo do Dota ficaram parados de 26/08 a 22/10: o codigo dizia que
"quem treina e o robo do Actions", mas esse robo nao existia no Dota. Aqui: o
robo existe, roda o MESMO comando que o app (build_retrain_command) e so
commita com os 4 minutos carregando.
"""
from __future__ import annotations

import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
YML = RAIZ / ".github" / "workflows" / "retreino_ao_vivo_dota.yml"


@unittest.skipUnless(YML.exists(), "robo de retreino ao vivo so existe no Dota")
class TestRoboRetreinoAoVivoDota(unittest.TestCase):

    def test_mesmo_comando_do_app(self):
        from app_core.live_models_status_core import build_retrain_command
        cmd = build_retrain_command(save_models_dir="live_models")
        self.assertEqual(cmd[1], "run_phase6_live_dota.py")
        yml = YML.read_text(encoding="utf-8")
        self.assertIn("python run_phase6_live_dota.py --save-models-dir live_models", yml)

    def test_guarda_antes_do_commit(self):
        yml = YML.read_text(encoding="utf-8")
        self.assertLess(yml.index("load_live_state_models"), yml.index("git add live_models/"))
        self.assertIn("[10, 15, 20, 25]", yml)


if __name__ == "__main__":
    unittest.main()
