# -*- coding: utf-8 -*-
"""Testes do painel de nota de jogador do Radar.

Pedido do dono em 13/09/2026: ver no Radar a conta nova de jogadores. O Radar
ja mostrava QUEM joga cada rota, e o veredito ja inclui a camada — mas so o
resultado, sem a razao. Faltava ver de onde vem a vantagem de escalacao.

O RISCO PRINCIPAL, e o motivo de metade destes testes: um painel que soma por
conta propria pode mostrar um numero diferente do que a odd usou. O Radar ja
custou caro com isso uma vez (discordava do pre, 24/07/2026). Aqui o total sai
da mesma funcao do caminho servido, e ha teste travando isso.
"""
from __future__ import annotations

import unittest

from app_core.radar_jogadores_core import (
    efeito_na_probabilidade,
    montar_painel,
)

NOTAS = {
    "id_a1|||Top": {"n": 0.40, "gp": 50},
    "id_a2|||Jungle": {"n": 0.06, "gp": 50},
    "id_a3|||Mid": {"n": 0.70, "gp": 50},
    "id_a4|||Bot": {"n": 0.54, "gp": 50},
    "id_a5|||Support": {"n": 0.28, "gp": 50},
    "id_b1|||Top": {"n": 0.09, "gp": 50},
    "id_b2|||Jungle": {"n": 0.38, "gp": 50},
    "id_b3|||Mid": {"n": -0.03, "gp": 50},
    "id_b4|||Bot": {"n": 0.06, "gp": 50},
    "id_b5|||Support": {"n": -0.04, "gp": 50},
}
APELIDOS = {
    "gakgos|||Top": "id_a1", "gryffinn|||Jungle": "id_a2", "quad|||Mid": "id_a3",
    "massu|||Bot": "id_a4", "cryogen|||Support": "id_a5",
    "impact|||Top": "id_b1", "hambak|||Jungle": "id_b2", "darkwings|||Mid": "id_b3",
    "rahel|||Bot": "id_b4", "huhi|||Support": "id_b5",
}
PAC = {"notas": NOTAS, "apelidos": APELIDOS, "w": 0.0686}

A = {"top": "Gakgos", "jng": "Gryffinn", "mid": "Quad", "bot": "Massu", "sup": "Cryogen"}
B = {"top": "Impact", "jng": "HamBak", "mid": "DARKWINGS", "bot": "Rahel", "sup": "huhi"}


class TestOPainel(unittest.TestCase):

    def test_uma_linha_por_rota_na_ordem_do_draft(self):
        p = montar_painel(PAC, roster_a=A, roster_b=B)
        self.assertEqual([l["rotulo"] for l in p["linhas"]],
                         ["TOP", "JNG", "MID", "ADC", "SUP"])

    def test_mostra_a_nota_dos_dois_lados_pelo_NOME(self):
        """O Radar so tem o nome; as notas sao por id. Era o que faltava."""
        l = montar_painel(PAC, roster_a=A, roster_b=B)["linhas"][0]
        self.assertAlmostEqual(l["nota_a"], 0.40)
        self.assertAlmostEqual(l["nota_b"], 0.09)

    def test_o_delta_por_rota_mostra_quem_ganha_onde(self):
        p = montar_painel(PAC, roster_a=A, roster_b=B)
        por_rota = {l["rotulo"]: l["delta"] for l in p["linhas"]}
        self.assertGreater(por_rota["MID"], 0, "Quad ganha do DARKWINGS")
        self.assertLess(por_rota["JNG"], 0, "HamBak ganha do Gryffinn")

    def test_cobertura_cheia_entra_na_conta(self):
        p = montar_painel(PAC, roster_a=A, roster_b=B)
        self.assertEqual(p["cobertura"], 5)
        self.assertTrue(p["entra_na_conta"])


class TestNaoInventaUmaContaParalela(unittest.TestCase):
    """O numero do painel TEM de ser o que a odd usou."""

    def test_o_total_bate_com_a_funcao_do_caminho_servido(self):
        from mlcore.players_pscore_core import delta_da_escalacao
        from mlcore.players_layer import norm_pos
        esperado, _ = delta_da_escalacao(
            NOTAS,
            blue_lineup={norm_pos(k): v for k, v in A.items()},
            red_lineup={norm_pos(k): v for k, v in B.items()},
            apelidos=APELIDOS)
        p = montar_painel(PAC, roster_a=A, roster_b=B)
        self.assertAlmostEqual(p["delta_total"], esperado, places=12)

    def test_o_deslocamento_e_peso_vezes_delta(self):
        p = montar_painel(PAC, roster_a=A, roster_b=B)
        self.assertAlmostEqual(p["deslocamento"], 0.0686 * p["delta_total"], places=12)

    def test_o_painel_chama_a_funcao_de_producao(self):
        """Se alguem trocar por uma soma local, o painel pode divergir da odd."""
        import inspect

        from app_core import radar_jogadores_core as m
        src = inspect.getsource(m.montar_painel)
        self.assertIn("delta_da_escalacao", src)


class TestQuandoNaoDaParaMostrar(unittest.TestCase):
    """PRINCIPIO IV: ausencia aparece como ausencia, nunca como zero."""

    def test_jogador_sem_nota_vira_traco_e_nao_zero(self):
        a = dict(A); a["mid"] = "NinguemConhece"
        l = [x for x in montar_painel(PAC, roster_a=a, roster_b=B)["linhas"]
             if x["rotulo"] == "MID"][0]
        self.assertIsNone(l["nota_a"])
        self.assertIsNone(l["delta"], "sem os dois lados nao ha delta")

    def test_escalacao_incompleta_NAO_entra_na_conta(self):
        a = dict(A); a["sup"] = ""
        p = montar_painel(PAC, roster_a=a, roster_b=B)
        self.assertFalse(p["entra_na_conta"])
        self.assertEqual(p["deslocamento"], 0.0)
        self.assertIn("incompleta", p["motivo"])

    def test_diz_o_MOTIVO_de_nao_entrar(self):
        a = dict(A); a["sup"] = ""
        p = montar_painel(PAC, roster_a=a, roster_b=B)
        self.assertIn("4/5", p["motivo"])

    def test_peso_zero_nao_desloca_nada(self):
        pac = dict(PAC); pac["w"] = 0.0
        p = montar_painel(pac, roster_a=A, roster_b=B)
        self.assertFalse(p["entra_na_conta"])
        self.assertEqual(p["deslocamento"], 0.0)

    def test_sem_arquivo_de_notas_o_painel_nao_aparece(self):
        for ruim in (None, {}, {"notas": None}, {"notas": []}):
            p = montar_painel(ruim, roster_a=A, roster_b=B)
            self.assertFalse(p["disponivel"])
            self.assertTrue(p["motivo"])

    def test_sem_indice_NENHUM_o_painel_se_esconde(self):
        """Sem indice NADA resolveria: o painel sairia com traco nas dez
        celulas. Tabela vazia nao informa — so parece defeito."""
        for pac in ({"notas": NOTAS, "w": 0.0686},
                    {"notas": NOTAS, "apelidos": {}, "w": 0.0686}):
            p = montar_painel(pac, roster_a=A, roster_b=B)
            self.assertFalse(p["disponivel"])
            self.assertIn("indice", p["motivo"])


class TestAPonteAteOProximoTreino(unittest.TestCase):
    """O indice so entra no arquivo de notas no treino COMPLETO, mensal.

    Sem uma ponte, esta tela ficaria semanas escondida esperando uma data. O
    Radar ja carrega as linhas de jogador para montar o draft, entao o indice
    sai de graca dali — sem tocar no modelo e sem mexer em nenhuma odd.
    """

    def test_o_indice_de_fora_faz_o_painel_aparecer(self):
        pac = {"notas": NOTAS, "w": 0.0686}          # arquivo servido hoje
        p = montar_painel(pac, roster_a=A, roster_b=B, apelidos_extra=APELIDOS)
        self.assertTrue(p["disponivel"])
        self.assertEqual(p["cobertura"], 5)

    def test_o_indice_DO_ARQUIVO_tem_preferencia(self):
        """Ele nasceu junto com as notas; nao pode perder para um de fora."""
        proprio = dict(APELIDOS)
        # Aponta para OUTRO jogador, mas da MESMA rota — apelido que cruza de
        # rota nao resolve (e certo: a nota e por jogador E rota).
        proprio["quad|||Mid"] = "id_b3"
        pac = {"notas": NOTAS, "apelidos": proprio, "w": 0.0686}
        l = [x for x in montar_painel(pac, roster_a=A, roster_b=B,
                                      apelidos_extra=APELIDOS)["linhas"]
             if x["rotulo"] == "MID"][0]
        self.assertAlmostEqual(l["nota_a"], NOTAS["id_b3|||Mid"]["n"],
                               msg="o indice do arquivo deveria ter vencido")

    def test_apelido_que_cruza_de_rota_NAO_resolve(self):
        """A nota e por jogador E rota; um mid nao herda a nota de um top."""
        from mlcore.players_pscore_core import resolver_chave
        self.assertEqual(
            resolver_chave(NOTAS, {"x|||Mid": "id_a1"}, "x", "Mid"), "")

    def test_monta_o_indice_a_partir_das_linhas_do_app(self):
        import pandas as pd
        from app_core.radar_jogadores_core import indice_das_linhas
        linhas = pd.DataFrame([
            {"playername": "Gakgos", "playerid": "id_a1", "position": "top"},
            {"playername": "Quad", "playerid": "id_a3", "position": "mid"},
        ])
        idx = indice_das_linhas(linhas)
        self.assertEqual(idx["gakgos|||Top"], "id_a1")
        self.assertEqual(idx["quad|||Mid"], "id_a3")

    def test_linhas_sem_as_colunas_nao_estouram(self):
        import pandas as pd
        from app_core.radar_jogadores_core import indice_das_linhas
        self.assertEqual(indice_das_linhas(None), {})
        self.assertEqual(indice_das_linhas(pd.DataFrame()), {})
        self.assertEqual(indice_das_linhas(pd.DataFrame([{"playername": "x"}])), {})

    def test_ambiguidade_continua_ficando_de_fora_na_ponte(self):
        """A ponte nao pode ser mais frouxa que o indice do treino."""
        import pandas as pd
        from app_core.radar_jogadores_core import indice_das_linhas
        idx = indice_das_linhas(pd.DataFrame([
            {"playername": "Ace", "playerid": "id_1", "position": "bot"},
            {"playername": "Ace", "playerid": "id_2", "position": "bot"},
        ]))
        self.assertNotIn("ace|||Bot", idx)


class TestOTamanhoDaVantagem(unittest.TestCase):
    """Numero no logito nao diz nada a quem aposta; em probabilidade, diz."""

    def test_desfaz_o_deslocamento_para_mostrar_o_sem_jogadores(self):
        p_com = 0.576
        sem = efeito_na_probabilidade(p_com, 0.1055)
        self.assertLess(sem, p_com)
        self.assertAlmostEqual(sem, 0.550, places=3)

    def test_deslocamento_zero_nao_muda_nada(self):
        self.assertAlmostEqual(efeito_na_probabilidade(0.6, 0.0), 0.6, places=12)

    def test_ida_e_volta_fecha(self):
        import math as _m
        p0, d = 0.42, -0.37
        z = _m.log(p0 / (1 - p0)) + d
        p_com = 1 / (1 + _m.exp(-z))
        self.assertAlmostEqual(efeito_na_probabilidade(p_com, d), p0, places=12)

    def test_entrada_invalida_devolve_None_e_nao_um_numero(self):
        for p, d in ((None, 0.1), (0.0, 0.1), (1.0, 0.1), ("x", 0.1),
                     (0.5, float("nan"))):
            self.assertIsNone(efeito_na_probabilidade(p, d), "p=%r d=%r" % (p, d))


if __name__ == "__main__":
    unittest.main()
