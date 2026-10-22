# -*- coding: utf-8 -*-
"""O placar "modelo x Pinnacle" (25/09/2026).

Ele perguntou a nota do modelo "em relacao as notas da pinnacle". As reguas
gravam o resumo em evidence/regua_pinnacle_*.json, o robo semanal as refaz e
a aba Oportunidades mostra.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    from tools.regua_pinnacle_comum import resumo
except ImportError as _sem_ferramenta:  # ferramenta so do LoL: no Dota o modulo e pulado
    import unittest as _u
    raise _u.SkipTest("tools.regua_pinnacle_comum nao existe neste repositorio: " + str(_sem_ferramenta))
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol

RAIZ = Path(__file__).resolve().parents[2]


def _caso(partida, p_modelo, p_casa, y):
    return {"partida": partida, "p_modelo": p_modelo, "p_casa": p_casa, "y": y}


class TestAConta(unittest.TestCase):

    def test_logloss_e_nota_na_escala_do_projeto(self):
        r = resumo([_caso(1, 0.7, 0.5, 1.0), _caso(2, 0.7, 0.5, 1.0)], "p_modelo", minimo=60)
        self.assertAlmostEqual(r["logloss_modelo"], -math.log(0.7))
        self.assertAlmostEqual(r["logloss_casa"], math.log(2))
        self.assertAlmostEqual(r["nota_casa"], 5.0, places=2)   # moeda = 5,0 (o projeto usa 0,6931)
        self.assertGreater(r["nota_modelo"], r["nota_casa"])
        self.assertEqual(r["bateu"], 2)

    def test_a_incerteza_conta_PARTIDA_e_nao_aposta(self):
        # 10 apostas da mesma partida = 1 partida
        r = resumo([_caso(1, 0.6, 0.5, 1.0)] * 10, "p_modelo", minimo=60)
        self.assertEqual(r["partidas"], 1)
        self.assertIsNone(r["margem_erro"])

    def test_abaixo_do_minimo_nunca_e_conclusivo(self):
        casos = [_caso(i, 0.9, 0.5, 1.0) for i in range(30)]
        self.assertFalse(resumo(casos, "p_modelo", minimo=60)["conclusivo"])

    def test_sem_a_chave_nao_inventa(self):
        self.assertIsNone(resumo([_caso(1, 0.6, 0.5, 1.0)], "p_vitoria", minimo=60))


class TestATelaLe(unittest.TestCase):

    def test_le_o_placar_e_diz_quando_nao_da_para_concluir(self):
        from app_core.oportunidades_tab_core import reguas_pinnacle
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d) / "evidence"
            ev.mkdir()
            (ev / "regua_pinnacle_total_kills.json").write_text(json.dumps({
                "gerado_em": "2026-09-25T17:00:00", "mercado": "Total de kills",
                "so_kills": {"partidas": 26, "minimo": 60, "nota_modelo": 6.2,
                             "nota_casa": 5.2, "diferenca": -0.03, "bateu": 19,
                             "conclusivo": False}}), encoding="utf-8")
            linhas = reguas_pinnacle(Path(d))
        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0]["conclusao"], "ainda nao da para concluir")
        self.assertEqual(linhas[0]["quem acertou mais ate agora"], "modelo")
        self.assertEqual(linhas[0]["partidas"], "26 de 60")

    def test_sem_arquivo_nao_mostra_nada(self):
        from app_core.oportunidades_tab_core import reguas_pinnacle
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(reguas_pinnacle(Path(d)), [])


class TestORobo(unittest.TestCase):

    @so_no_lol("o .gitignore do LoL (o placar das reguas e so do LoL)")
    def test_o_robo_semanal_roda_as_reguas_e_guarda_o_placar(self):
        yml = (RAIZ / ".github" / "workflows" / "reguas_pinnacle.yml").read_text(encoding="utf-8")
        self.assertIn("python tools/abates_contra_a_casa.py", yml)
        self.assertIn("python tools/handicap_contra_a_casa.py", yml)
        self.assertIn("evidence/regua_pinnacle_total_kills.json", yml)

    @so_no_lol("o .gitignore do LoL (o placar das reguas e so do LoL)")
    def test_o_placar_e_versionado(self):
        gi = (RAIZ / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("!evidence/regua_pinnacle_*.json", gi)

    @so_no_lol("o .gitignore do LoL (o placar das reguas e so do LoL)")
    def test_a_regua_do_handicap_mede_a_conta_da_TELA(self):
        fonte = (RAIZ / "tools" / "handicap_contra_a_casa.py").read_text(encoding="utf-8")
        self.assertIn("from mlcore.abates_por_time import prob_cobre_handicap", fonte)
        self.assertNotIn("calibrador(", fonte)


if __name__ == "__main__":
    unittest.main()
