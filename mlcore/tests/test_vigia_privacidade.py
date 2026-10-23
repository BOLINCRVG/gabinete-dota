# -*- coding: utf-8 -*-
"""O vigia de privacidade (23/10/2026): roda na suite, entao o robo vigia todo dia.
Arquivo identico nos dois repositorios."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.vigia_privacidade import PROIBIDOS, campos_de, varrer

RAIZ = Path(__file__).resolve().parents[2]


class TestVigiaPegaOProblema(unittest.TestCase):

    def test_acha_campo_proibido_aninhado_no_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text(json.dumps({"times": [{"roster": [{"account_id": 1, "personaname": "z"}]}]}),
                         encoding="utf-8")
            self.assertIn("personaname", campos_de(p) & PROIBIDOS)

    def test_acha_coluna_proibida_no_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("gameid,playername,country\n1,a,BR\n", encoding="utf-8")
            self.assertEqual(campos_de(p) & PROIBIDOS, {"country"})

    def test_identificador_profissional_e_permitido(self):
        for campo in ("playername", "playerid", "account_id", "teamname"):
            self.assertNotIn(campo, PROIBIDOS)


class TestOsDadosDesteRepositorio(unittest.TestCase):

    def test_nenhum_dado_pessoal_proibido_nos_dados(self):
        res = varrer(RAIZ)
        self.assertGreater(res["lidos"], 0, "nenhum arquivo de dados encontrado")
        self.assertEqual(res["achados"], {})


if __name__ == "__main__":
    unittest.main()
