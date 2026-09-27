# -*- coding: utf-8 -*-
"""Testes do indice NOME -> id do jogador.

POR QUE ELE EXISTE
As notas sao chaveadas por `playerid` (`oe:player:...`), porque id sobrevive
a troca de nick. O montador de draft do RADAR usa `latest_roster`, que
devolve `playername` — sem indice, nenhuma nota resolve ali e a secao de
escalacao do Radar nao mostra nada.

O CAMINHO DA ODD NUNCA DEPENDEU DISTO: a escalacao dele vem de `lineup_ml_*`,
que guarda a CHAVE do jogador (o editor mostra o nome e grava a chave).
Conferido no app dirigido em 14/09/2026 com o artefato de producao:
applied=True, cobertura 5/5.

ERRO MEU, REGISTRADO PARA NINGUEM REFAZER O CAMINHO: em 13/09/2026 testei o
caminho servido com a escalacao do RADAR (nomes), vi cobertura 0/5 e anunciei
que a camada estava inerte em producao havia dias. Nao estava. Peguei a lista
errada e conclui grande demais a partir de um teste so.
"""
from __future__ import annotations

import unittest

import pandas as pd

from mlcore.players_pscore_core import (
    delta_da_escalacao,
    indice_de_apelidos,
    resolver_chave,
)

NOTAS = {
    "oe:player:aaa|||Top": {"n": 1.0, "gp": 40},
    "oe:player:bbb|||Mid": {"n": -0.5, "gp": 40},
    "oe:player:ccc|||Top": {"n": 0.2, "gp": 40},
}
APELIDOS = {
    "gakgos|||Top": "oe:player:aaa",
    "quad|||Mid": "oe:player:bbb",
}


def _linhas(dados) -> pd.DataFrame:
    """dados = [(playername, pos_can, player_key)]"""
    return pd.DataFrame(
        [{"playername": n, "pos_can": r, "player_key": k} for n, r, k in dados])


class TestOCasoQueQuebrou(unittest.TestCase):

    def test_escalacao_por_NOME_agora_encontra_a_nota(self):
        d, cob = delta_da_escalacao(
            NOTAS, blue_lineup={"Top": "Gakgos"}, red_lineup={"Mid": "Quad"},
            apelidos=APELIDOS)
        self.assertEqual(cob, 1)
        self.assertAlmostEqual(d, 1.0 - (-0.5))

    def test_SEM_o_indice_o_nome_nao_resolve(self):
        """E o caso do Radar antes do indice — nao o do caminho da odd."""
        d, cob = delta_da_escalacao(
            NOTAS, blue_lineup={"Top": "Gakgos"}, red_lineup={"Mid": "Quad"})
        self.assertEqual(cob, 0)
        self.assertEqual(d, 0.0)

    def test_quem_ja_manda_o_ID_continua_funcionando(self):
        d, cob = delta_da_escalacao(
            NOTAS, blue_lineup={"Top": "oe:player:aaa"},
            red_lineup={"Mid": "oe:player:bbb"}, apelidos=APELIDOS)
        self.assertEqual(cob, 1)
        self.assertAlmostEqual(d, 1.5)

    def test_o_id_vem_ANTES_do_nome(self):
        """Quem manda id nao paga por ambiguidade de nome de terceiro."""
        notas = dict(NOTAS)
        notas["Gakgos|||Top"] = {"n": 99.0, "gp": 1}
        self.assertEqual(
            resolver_chave(notas, APELIDOS, "Gakgos", "Top"), "Gakgos|||Top")


class TestAmbiguidadeNaoVIraChute(unittest.TestCase):
    """PRINCIPIO IV: dois jogadores com o mesmo nick nao viram um so."""

    def test_nome_repetido_na_mesma_rota_fica_de_fora(self):
        idx = indice_de_apelidos(_linhas([
            ("Ace", "Bot", "oe:player:111"),
            ("Ace", "Bot", "oe:player:222"),
        ]))
        self.assertNotIn("ace|||Bot", idx)

    def test_ambiguo_derruba_a_cobertura_em_vez_de_somar_errado(self):
        d, cob = delta_da_escalacao(
            NOTAS, blue_lineup={"Top": "Ace"}, red_lineup={"Mid": "Quad"},
            apelidos={"quad|||Mid": "oe:player:bbb"})
        self.assertEqual(cob, 0, "sem os dois lados a conta nao vale")

    def test_mesmo_nome_em_ROTAS_diferentes_nao_e_ambiguidade(self):
        idx = indice_de_apelidos(_linhas([
            ("Ace", "Bot", "oe:player:111"),
            ("Ace", "Top", "oe:player:222"),
        ]))
        self.assertEqual(idx["ace|||Bot"], "oe:player:111")
        self.assertEqual(idx["ace|||Top"], "oe:player:222")

    def test_troca_de_nick_aponta_para_o_mesmo_id(self):
        """Os dois nicks tem de achar a MESMA nota — e o motivo de usar id."""
        idx = indice_de_apelidos(_linhas([
            ("Faker", "Mid", "oe:player:xyz"),
            ("Hide on bush", "Mid", "oe:player:xyz"),
        ]))
        self.assertEqual(idx["faker|||Mid"], "oe:player:xyz")
        self.assertEqual(idx["hide on bush|||Mid"], "oe:player:xyz")


class TestTolerancia(unittest.TestCase):

    def test_caixa_diferente_ainda_acha(self):
        self.assertEqual(
            resolver_chave(NOTAS, APELIDOS, "GAKGOS", "Top"), "oe:player:aaa|||Top")

    def test_espaco_sobrando_ainda_acha(self):
        self.assertEqual(
            resolver_chave(NOTAS, APELIDOS, "  Gakgos ", "Top"), "oe:player:aaa|||Top")

    def test_nome_desconhecido_devolve_vazio_e_nao_estoura(self):
        self.assertEqual(resolver_chave(NOTAS, APELIDOS, "NinguemAqui", "Top"), "")

    def test_apelido_que_aponta_para_nota_inexistente_nao_conta(self):
        """Indice desatualizado nao pode virar nota fantasma."""
        self.assertEqual(
            resolver_chave(NOTAS, {"zzz|||Top": "oe:player:sumiu"}, "zzz", "Top"), "")


class TestOndeOIndiceNasce(unittest.TestCase):

    def test_indice_vazio_quando_faltam_colunas(self):
        self.assertEqual(indice_de_apelidos(pd.DataFrame([{"playername": "x"}])), {})

    def test_indice_vazio_nao_estoura(self):
        self.assertEqual(indice_de_apelidos(pd.DataFrame()), {})
        self.assertEqual(indice_de_apelidos(None), {})

    def test_linha_incompleta_e_ignorada(self):
        idx = indice_de_apelidos(_linhas([
            ("", "Top", "oe:player:111"),
            ("Bom", "", "oe:player:222"),
            ("Bom", "Mid", ""),
            ("Certo", "Mid", "oe:player:333"),
        ]))
        self.assertEqual(idx, {"certo|||Mid": "oe:player:333"})

    def test_o_pacote_construido_carrega_o_indice(self):
        """Se `construir` parar de gravar, a camada volta a ficar muda."""
        import inspect

        from mlcore import players_pscore_core as m
        self.assertIn('ajuste["apelidos"]', inspect.getsource(m.construir))


class TestOCaminhoServidoUsaOIndice(unittest.TestCase):

    def test_o_estagio_servido_repassa_apelidos(self):
        import inspect

        from app_core import ml_pre_live_pipeline_core as m
        self.assertIn("apelidos=pac.get(\"apelidos\")",
                      inspect.getsource(m._aplicar_pscore))


if __name__ == "__main__":
    unittest.main()
