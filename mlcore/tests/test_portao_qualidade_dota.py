# -*- coding: utf-8 -*-
"""tools/portao_qualidade_dota.py (22/10/2026): a trava de qualidade antes de publicar."""
from __future__ import annotations

import unittest
from pathlib import Path

try:
    from tools.portao_qualidade_dota import FAIL_DELTA, MOEDA, decidir
except ImportError as _sem_ferramenta:  # ferramenta so do Dota: no LoL o modulo e pulado
    import unittest as _u
    raise _u.SkipTest("tools.portao_qualidade_dota nao existe neste repositorio: " + str(_sem_ferramenta))

RAIZ = Path(__file__).resolve().parents[2]
ANTIGO = {"n": 5288, "logloss": 0.5735}


class TestDecisao(unittest.TestCase):

    def test_aprova_estavel(self):
        self.assertTrue(decidir({"n": 5300, "logloss": 0.5740}, ANTIGO)[0])

    def test_reprova_sem_medida(self):
        self.assertFalse(decidir(None, ANTIGO)[0])

    def test_reprova_pior_que_moeda(self):
        self.assertFalse(decidir({"n": 5300, "logloss": MOEDA + 0.001}, None)[0])

    def test_reprova_queda_brusca(self):
        self.assertFalse(decidir({"n": 5300, "logloss": 0.5735 + FAIL_DELTA + 0.001}, ANTIGO)[0])

    def test_reprova_csv_truncado(self):
        self.assertFalse(decidir({"n": 3000, "logloss": 0.57}, ANTIGO)[0])

    def test_reprova_nan(self):
        self.assertFalse(decidir({"n": 5300, "logloss": float("nan")}, ANTIGO)[0])

    def test_liga_nova_mais_dificil_nao_conta_como_piora(self):
        # 16/11/2026: entraram ligas novas (mais dificeis); nos torneios comuns o modelo empatou
        antigo_t = [{"torneio": "A", "n": 3000, "logloss": 0.57}, {"torneio": "B", "n": 2000, "logloss": 0.59}]
        novo_t = antigo_t[:1] + [{"torneio": "B", "n": 2010, "logloss": 0.595},
                                 {"torneio": "Nova", "n": 3500, "logloss": 0.64}]
        ok, motivo = decidir({"n": 8510, "logloss": 0.60}, {"n": 5000, "logloss": 0.578}, novo_t, antigo_t)
        self.assertTrue(ok, motivo)
        self.assertIn("mesmos torneios", motivo)

    def test_piora_nos_mesmos_torneios_reprova(self):
        antigo_t = [{"torneio": "A", "n": 3000, "logloss": 0.57}]
        novo_t = [{"torneio": "A", "n": 3000, "logloss": 0.60}]
        self.assertFalse(decidir({"n": 3000, "logloss": 0.60}, {"n": 3000, "logloss": 0.57}, novo_t, antigo_t)[0])

    def test_mesmos_torneios_com_menos_jogos_reprova(self):
        antigo_t = [{"torneio": "A", "n": 3000, "logloss": 0.57}]
        novo_t = [{"torneio": "A", "n": 2000, "logloss": 0.57}, {"torneio": "Nova", "n": 4000, "logloss": 0.6}]
        self.assertFalse(decidir({"n": 6000, "logloss": 0.59}, {"n": 3000, "logloss": 0.57}, novo_t, antigo_t)[0])


class TestEsteira(unittest.TestCase):

    def test_reprovar_mantem_o_modelo_anterior(self):
        yml = (RAIZ / ".github" / "workflows" / "dados_diarios_dota.yml").read_text(encoding="utf-8")
        self.assertIn("python tools/portao_qualidade_dota.py", yml)
        self.assertIn("git checkout HEAD -- ml_artifact.json", yml)
        self.assertLess(yml.index("python dota_scorecard.py"), yml.index("python tools/portao_qualidade_dota.py"))
        self.assertLess(yml.index("python tools/portao_qualidade_dota.py"), yml.index("git add data.csv"))


if __name__ == "__main__":
    unittest.main()
