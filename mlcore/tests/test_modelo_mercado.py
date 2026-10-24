# -*- coding: utf-8 -*-
"""Aba Modelo x Mercado reorganizada (23/10/2026). Identico nos 2 repos."""
from __future__ import annotations

import unittest

from app_core.modelo_mercado_core import (
    aposta, linhas_com_valor, nivel, painel_do_dia, tabela_de_valor)


def _l(jogo, v, **k):
    base = {"jogo": jogo, "casa": "Liquid", "fora": "CTBC Flying Oyster", "liga": "Worlds",
            "vantagem_pct": v, "odd_casa": 1.84, "odd_modelo": 1.74,
            "nome_pt": "Mapa 1 - Total de Kills", "lado": "under", "pontos": 27.5}
    base.update(k)
    return base


COMPLETO = [_l(1, 6.0), _l(1, 8.1, lado="home", tipo="moneyline"), _l(1, None),
            _l(2, 0.5), _l(2, 45.0, suspeito=True), _l(3, 2.0)]


class TestValor(unittest.TestCase):

    def test_ranking_do_dia_ignora_sem_modelo_e_abaixo_do_minimo(self):
        r = linhas_com_valor(COMPLETO, incluir_suspeitas=False)
        self.assertEqual([x["vantagem_pct"] for x in r], [8.1, 6.0, 2.0])

    def test_suspeita_aparece_mas_no_fim(self):
        r = linhas_com_valor(COMPLETO, jogo=2)
        self.assertEqual([x["vantagem_pct"] for x in r], [45.0])
        r = linhas_com_valor(COMPLETO)
        self.assertEqual(r[-1]["vantagem_pct"], 45.0)

    def test_mesma_aposta_de_vencedor_vira_um_cartao_so(self):
        bo1 = [_l(9, 16.6, tipo="moneyline", lado="home", odd_casa=2.59, nome_pt="Vencedor - mapa 1", pontos=None),
               _l(9, 16.3, tipo="moneyline", lado="home", odd_casa=2.59, nome_pt="Vencedor - serie", pontos=None)]
        self.assertEqual(len(linhas_com_valor(bo1)), 1)

    def test_linhas_alternativas_do_mesmo_handicap_viram_uma(self):
        alt = [_l(9, v, tipo="spread", lado="home", pontos=p, nome_pt="Handicap - Kills - mapa 1")
               for v, p in ((19.6, 3.5), (19.1, 6.5), (17.2, 2.5))]
        r = linhas_com_valor(alt)
        self.assertEqual([x["pontos"] for x in r], [3.5])
        self.assertEqual(len(linhas_com_valor(alt, um_por_mercado=False)), 3)

    def test_painel(self):
        p = painel_do_dia(COMPLETO)
        self.assertEqual((p["jogos"], p["linhas_com_valor"], p["jogos_com_valor"]), (3, 3, 2))

    def test_nivel(self):
        self.assertEqual([nivel(x) for x in (0.5, 2, 5, 9, 30)],
                         ["", "baixo", "medio", "alto", "desconfie"])


class TestTabela(unittest.TestCase):

    def test_aposta_legivel(self):
        self.assertEqual(aposta(_l(1, 1)), "abaixo de 27,5")
        self.assertEqual(aposta(_l(1, 1, lado="home", tipo="spread", pontos=-8.5)), "Liquid -8,5")

    def test_colunas_com_e_sem_jogo(self):
        com = tabela_de_valor([_l(1, 6.0)], com_jogo=True)[0]
        sem = tabela_de_valor([_l(1, 6.0)], com_jogo=False)[0]
        self.assertIn("Jogo", com)
        self.assertNotIn("Jogo", sem)
        self.assertEqual(sem["Vantagem"], "+6,0%")
        self.assertEqual(sem["Nivel"], "medio")


if __name__ == "__main__":
    unittest.main()
