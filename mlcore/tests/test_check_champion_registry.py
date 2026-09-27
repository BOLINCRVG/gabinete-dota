# -*- coding: utf-8 -*-
"""Vigia do registro de campeoes: a comparacao tem que ser pelo NOME, nao pelo id.

Ele foi montar um draft em 16/08/2026 e o Locke nao estava la — o registro estava
dez patches atras e nada avisava. O vigia nasceu disso.

Na primeira execucao ele acusou 3 campeoes faltando que NAO faltavam: o Data
Dragon chaveia por um id interno e o registro pelo nome normalizado. Comparar
pelo id daria alarme falso toda semana — e alarme falso semanal e pior que nao
ter alarme, porque ensina a ignorar.
"""
from __future__ import annotations

import unittest

from app_core.champion_knowledge_core import _norm_key
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol


class TestChaveDeCampeao(unittest.TestCase):
    def test_os_tres_casos_que_deram_alarme_falso(self) -> None:
        """id do Riot != nome exibido. A chave do registro segue o NOME."""
        for id_riot, nome, esperado in (
            ("MonkeyKing", "Wukong", "wukong"),
            ("Nunu", "Nunu & Willump", "nunuwillump"),
            ("Renata", "Renata Glasc", "renataglasc"),
        ):
            self.assertEqual(_norm_key(nome), esperado)
            self.assertNotEqual(_norm_key(id_riot), esperado,
                                f"{id_riot}: se batesse, o alarme falso nao teria existido")

    def test_normalizacao_tira_pontuacao_espaco_e_caixa(self) -> None:
        self.assertEqual(_norm_key("Dr. Mundo"), "drmundo")
        self.assertEqual(_norm_key("Kai'Sa"), "kaisa")
        self.assertEqual(_norm_key("LeBlanc"), "leblanc")
        self.assertEqual(_norm_key("  Locke  "), "locke")

    def test_caso_simples_id_igual_nome(self) -> None:
        for n in ("Locke", "Zaahen", "Aurora", "Ambessa"):
            self.assertEqual(_norm_key(n), n.lower())

    def test_entradas_degeneradas(self) -> None:
        self.assertEqual(_norm_key(None), "")
        self.assertEqual(_norm_key(""), "")
        self.assertEqual(_norm_key("   "), "")


class TestRegistroLocal(unittest.TestCase):
    @so_no_lol("campeoes do LoL")
    def test_locke_esta_no_registro(self) -> None:
        """O campeao que ele sentiu falta — trava para nao sumir num rebuild."""
        from app_core.champion_knowledge_core import load_champion_knowledge_registry
        reg = load_champion_knowledge_registry()
        champs = reg.get("champions") or {}
        self.assertIn("locke", champs)
        self.assertEqual(str(champs["locke"].get("primary_role")), "mid")


if __name__ == "__main__":
    unittest.main()
