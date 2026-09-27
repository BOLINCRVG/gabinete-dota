# -*- coding: utf-8 -*-
"""Testes da busca por vantagem nas linhas coletadas.

O QUE ELE PEDIU (15/09/2026): "so onde ha vantagem", ordenado pela vantagem.
A parte facil e achar a vantagem. A parte que decide se a tela presta e o que
acontece com o que NAO da para precificar -- porque uma tela que mostra 4
apostas e esconde que 200 linhas ficaram de fora parece completa e nao e.
"""
from __future__ import annotations

import unittest

from app_core.oportunidades_core import (
    JOGOS_MINIMOS,
    VANTAGEM_SUSPEITA_PCT,
    avaliar_linha,
    metrica_da_linha,
    montar_oportunidades,
    odd_justa,
    vantagem_pct,
)


class _Total:
    """Fica no lugar do MatchupTotals do motor."""

    def __init__(self, mean, sd=5.0):
        self.mean = mean
        self.sd = sd
        self.dist = "normal"


def _prob_over_falso(total, linha):
    """Metade acima, metade abaixo da media -- previsivel de proposito."""
    if linha < total.mean:
        return 0.60
    if linha > total.mean:
        return 0.40
    return 0.50


def _linha(**k):
    base = {"jogo": 1, "tipo": "total", "periodo": 1, "lado": "over",
            "pontos": 30.5, "odd": 2.00, "unidade": "Kills",
            "mercado": "Total - Kills - mapa 1", "limite": 500,
            "margem_pct": 5.0, "fecha_em": ""}
    base.update(k)
    return base


def _odds(linhas, jogos=None):
    return {
        "lido_em": "2026-09-15T20:21:00-03:00",
        "jogos": jogos or [{"id": 1, "liga": "LPL", "casa": "Top Esports",
                            "fora": "Invictus", "bo": 5, "inicio_utc": ""}],
        "linhas": linhas,
    }


class TestQualLinhaOMotorSABEPrecificar(unittest.TestCase):

    def test_total_de_kills_do_mapa_1_e_precificavel(self):
        self.assertEqual(metrica_da_linha(_linha()), ("kills", "m1"))

    def test_o_mapa_vem_do_periodo(self):
        self.assertEqual(metrica_da_linha(_linha(periodo=3)), ("kills", "m3"))

    def test_handicap_de_kills_NAO_e(self):
        """Handicap pede a distribuicao da DIFERENCA, que o motor nao tem.

        Fingir que da seria o pior desfecho: um numero errado com cara de
        numero certo, em cima de dinheiro dele.
        """
        self.assertIsNone(metrica_da_linha(_linha(tipo="spread", lado="home")))

    def test_quem_faz_mais_kills_NAO_e(self):
        self.assertIsNone(metrica_da_linha(_linha(tipo="moneyline", lado="home", pontos=None)))

    def test_total_da_SERIE_nao_e_total_de_mapa(self):
        """Periodo 0 e a serie inteira; o motor fala de mapa."""
        self.assertIsNone(metrica_da_linha(_linha(periodo=0, unidade="Maps", pontos=2.5)))

    def test_mercado_sem_unidade_NAO_e(self):
        self.assertIsNone(metrica_da_linha(_linha(unidade="")))

    def test_linha_sem_numero_NAO_e(self):
        self.assertIsNone(metrica_da_linha(_linha(pontos=None)))

    def test_torre_e_barao_tambem_entram_quando_abrirem(self):
        """Torre e tempo ainda nao abriram na casa; quando abrirem, passam."""
        self.assertEqual(metrica_da_linha(_linha(unidade="Towers")), ("towers", "m1"))
        self.assertEqual(metrica_da_linha(_linha(unidade="Barons")), ("barons", "m1"))


class TestAContaDaVantagem(unittest.TestCase):
    """vantagem = p * odd - 1. E so isso, e tem de ser exatamente isso."""

    def test_odd_paga_o_justo_da_vantagem_zero(self):
        self.assertAlmostEqual(vantagem_pct(0.50, 2.00), 0.0, places=9)

    def test_odd_acima_do_justo_da_vantagem_positiva(self):
        self.assertAlmostEqual(vantagem_pct(0.50, 2.20), 10.0, places=9)

    def test_odd_abaixo_do_justo_da_vantagem_negativa(self):
        self.assertAlmostEqual(vantagem_pct(0.50, 1.80), -10.0, places=9)

    def test_a_odd_justa_e_o_inverso_da_probabilidade(self):
        self.assertAlmostEqual(odd_justa(0.25), 4.0, places=9)

    def test_sem_probabilidade_NAO_vira_zero(self):
        """PRINCIPIO IV. Vantagem zero e uma afirmacao: "medi, e nao ha".

        Escrever zero onde nao houve medicao transformaria "nao sei" em
        "sei que nao vale" -- e ele decidiria dinheiro em cima disso.
        """
        self.assertIsNone(vantagem_pct(None, 2.00))
        self.assertIsNone(vantagem_pct(float("nan"), 2.00))
        self.assertIsNone(vantagem_pct(0.50, None))

    def test_probabilidade_impossivel_nao_passa(self):
        self.assertIsNone(vantagem_pct(0.0, 2.00))
        self.assertIsNone(vantagem_pct(1.0, 2.00))
        self.assertIsNone(vantagem_pct(-0.3, 2.00))

    def test_odd_impossivel_nao_passa(self):
        self.assertIsNone(vantagem_pct(0.5, 1.0))
        self.assertIsNone(vantagem_pct(0.5, 0.5))


class TestOVeredictoDeUmaLinha(unittest.TestCase):

    def test_over_usa_a_probabilidade_de_passar(self):
        a = avaliar_linha(_linha(pontos=25.5, odd=2.00),
                          {"kills": _Total(30.0)}, prob_over=_prob_over_falso)
        self.assertAlmostEqual(a["probabilidade"], 0.60, places=9)
        self.assertAlmostEqual(a["vantagem_pct"], 20.0, places=9)

    def test_under_usa_o_COMPLEMENTO(self):
        """O erro classico: usar a mesma probabilidade dos dois lados.

        Daria vantagem nos DOIS lados da mesma linha ao mesmo tempo, que e
        impossivel, e a tela recomendaria apostar contra si mesma.
        """
        a = avaliar_linha(_linha(pontos=25.5, lado="under", odd=2.00),
                          {"kills": _Total(30.0)}, prob_over=_prob_over_falso)
        self.assertAlmostEqual(a["probabilidade"], 0.40, places=9)
        self.assertAlmostEqual(a["vantagem_pct"], -20.0, places=9)

    def test_os_dois_lados_nunca_tem_vantagem_juntos_com_odd_justa(self):
        tot = {"kills": _Total(30.0)}
        o = avaliar_linha(_linha(pontos=25.5, lado="over", odd=1 / 0.60),
                          tot, prob_over=_prob_over_falso)
        u = avaliar_linha(_linha(pontos=25.5, lado="under", odd=1 / 0.40),
                          tot, prob_over=_prob_over_falso)
        self.assertAlmostEqual(o["vantagem_pct"], 0.0, places=6)
        self.assertAlmostEqual(u["vantagem_pct"], 0.0, places=6)

    def test_motor_sem_numero_devolve_NADA(self):
        def _nan(total, linha):
            return float("nan")
        self.assertIsNone(avaliar_linha(_linha(), {"kills": _Total(30.0)}, prob_over=_nan))

    def test_motor_que_ESTOURA_nao_derruba_a_tela(self):
        def _explode(total, linha):
            raise RuntimeError("dados de menos")
        self.assertIsNone(avaliar_linha(_linha(), {"kills": _Total(30.0)}, prob_over=_explode))

    def test_metrica_ausente_nos_totais_devolve_NADA(self):
        self.assertIsNone(avaliar_linha(_linha(unidade="Towers"),
                                        {"kills": _Total(30.0)}, prob_over=_prob_over_falso))

    def test_guarda_o_limite_da_casa(self):
        """O limite e a casa dizendo em qual preco ela acredita.

        Vantagem de 8% num mercado de R$ 100 vale menos que 3% num de
        R$ 2.000 -- e sem o limite na tela ele nao teria como saber disso.
        """
        a = avaliar_linha(_linha(limite=2000), {"kills": _Total(30.0)},
                          prob_over=_prob_over_falso)
        self.assertEqual(a["limite"], 2000)


class TestAListaQueEleVaiOLHAR(unittest.TestCase):

    def _totais(self, casa, fora, mapa):
        return {"kills": _Total(30.0)}

    def test_so_o_que_tem_vantagem_entra(self):
        r = montar_oportunidades(
            _odds([_linha(pontos=25.5, odd=2.00),        # +20%
                   _linha(pontos=35.5, odd=2.00)]),      # abaixo da media -> -20%
            self._totais, prob_over=_prob_over_falso)
        self.assertEqual(len(r["achados"]), 1)
        self.assertAlmostEqual(r["achados"][0]["vantagem_pct"], 20.0, places=6)

    def test_a_ordem_e_da_maior_vantagem_para_a_menor(self):
        r = montar_oportunidades(
            _odds([_linha(pontos=25.5, odd=1.80), _linha(pontos=25.5, odd=2.00)]),
            self._totais, prob_over=_prob_over_falso, minimo_pct=0.0)
        vs = [a["vantagem_pct"] for a in r["achados"]]
        self.assertEqual(vs, sorted(vs, reverse=True))

    def test_vantagem_GRANDE_DEMAIS_vai_marcada_e_para_o_FIM(self):
        """A Pinnacle abre com ~5% de margem: ela nao deixa 40% na mesa.

        Uma conta assim quase sempre e o modelo errando aquele confronto. No
        topo da lista ela empurraria justamente para a pior aposta.
        """
        r = montar_oportunidades(
            _odds([_linha(pontos=25.5, odd=4.00),      # +140%: absurdo
                   _linha(pontos=25.5, odd=1.80)]),    # +8%: plausivel
            self._totais, prob_over=_prob_over_falso, minimo_pct=0.0)
        self.assertGreater(r["achados"][0]["vantagem_pct"], 0)
        self.assertFalse(r["achados"][0]["suspeito"])
        self.assertTrue(r["achados"][-1]["suspeito"])
        self.assertGreater(r["achados"][-1]["vantagem_pct"], VANTAGEM_SUSPEITA_PCT)

    def test_o_que_NAO_foi_precificado_e_CONTADO_e_NOMEADO(self):
        """Uma tela com 4 apostas e 200 linhas caladas parece completa.

        Foi exatamente esse tipo de silencio que segurou kills por horas: eu
        rodava, via 274 linhas, e nenhuma era de kills -- sem um unico sinal
        de que algo tinha sido jogado fora.
        """
        r = montar_oportunidades(
            _odds([_linha(pontos=25.5, odd=2.00),
                   _linha(tipo="spread", lado="home", pontos=-9.5,
                          mercado="Handicap - Kills - mapa 1"),
                   _linha(tipo="team_total", pontos=12.5,
                          mercado="Total do time - Kills - mapa 1"),
                   _linha(tipo="moneyline", lado="home", pontos=None, unidade="",
                          mercado="Vencedor - serie")]),
            self._totais, prob_over=_prob_over_falso)
        self.assertEqual(len(r["achados"]), 1)
        self.assertEqual(sum(g["linhas"] for g in r["nao_precificados"]), 3)
        for g in r["nao_precificados"]:
            self.assertTrue(g["mercado"].strip(), "mercado sem nome nao localiza nada")
            self.assertTrue(g["motivo"].strip(), "motivo em branco nao explica nada")

    def test_o_de_fora_e_agrupado_pelo_nome_QUE_A_CASA_USA(self):
        """Ele confere na tela da Pinnacle; o nome tem de ser o de la.

        Agrupar por um nome nosso ("diferenca de kills") obrigaria a traduzir
        de cabeca para achar a mesma aposta no site.
        """
        r = montar_oportunidades(
            _odds([_linha(tipo="spread", lado="home", pontos=-9.5,
                          mercado="Handicap - Kills - mapa 1"),
                   _linha(tipo="spread", lado="away", pontos=9.5,
                          mercado="Handicap - Kills - mapa 1"),
                   _linha(tipo="moneyline", lado="Fuego 3, 9z 0", pontos=None,
                          unidade="Maps", periodo=0, mercado="Correct Score")]),
            self._totais, prob_over=_prob_over_falso)
        por_nome = {g["mercado"]: g for g in r["nao_precificados"]}
        self.assertEqual(por_nome["Handicap - Kills - mapa 1"]["linhas"], 2)
        self.assertEqual(por_nome["Correct Score"]["linhas"], 1)

    def test_o_de_fora_vem_do_maior_para_o_menor(self):
        r = montar_oportunidades(
            _odds([_linha(tipo="team_total", pontos=12.5, mercado="A"),
                   _linha(tipo="spread", lado="home", pontos=-9.5, mercado="B"),
                   _linha(tipo="spread", lado="away", pontos=9.5, mercado="B")]),
            self._totais, prob_over=_prob_over_falso)
        self.assertEqual([g["mercado"] for g in r["nao_precificados"]], ["B", "A"])

    def test_cada_motivo_diz_o_que_FALTA_e_nao_so_que_faltou(self):
        from app_core.oportunidades_core import motivo_de_nao_precificar
        handicap = motivo_de_nao_precificar(
            _linha(tipo="spread", lado="home", pontos=-9.5))
        por_time = motivo_de_nao_precificar(_linha(tipo="team_total", pontos=12.5))
        serie = motivo_de_nao_precificar(
            _linha(tipo="moneyline", lado="home", pontos=None, unidade="", periodo=0))
        # Desde 24/09/2026 o modelo CALCULA handicap de kills e total por time
        # (mlcore/abates_por_time.py). O que falta, quando falta, e o
        # confronto -- ou, no total por time, saber de qual time e a linha.
        self.assertIn("nao conseguiu este confronto", handicap)
        self.assertIn("de qual time", por_time)
        self.assertIn("vencedor", serie)
        self.assertNotEqual(handicap, por_time)
        self.assertNotEqual(handicap, serie)

    def test_placar_exato_NAO_e_chamado_de_handicap(self):
        """Erro meu na primeira versao: "Correct Score" saiu rotulado como
        "diferenca de maps", que nao quer dizer nada para quem le."""
        from app_core.oportunidades_core import motivo_de_nao_precificar
        m = motivo_de_nao_precificar(
            _linha(tipo="moneyline", lado="Fuego 3, 9z 0", pontos=None,
                   unidade="Maps", periodo=0, mercado="Correct Score"))
        self.assertNotIn("diferenca", m)
        self.assertIn("vencedor", m)

    def test_confronto_sem_modelo_e_NOMEADO_com_os_times(self):
        """"Nao consegui" sem dizer de quem nao ajuda ninguem."""
        r = montar_oportunidades(_odds([_linha()]),
                                 lambda casa, fora, mapa: None,
                                 prob_over=_prob_over_falso)
        self.assertEqual(r["achados"], [])
        self.assertEqual(len(r["sem_modelo"]), 1)
        self.assertEqual(r["sem_modelo"][0]["casa"], "Top Esports")
        self.assertEqual(r["sem_modelo"][0]["fora"], "Invictus")

    def test_confronto_sem_modelo_nao_e_tentado_de_novo(self):
        """Sem isto, um confronto ruim custaria uma tentativa por linha."""
        chamadas = []

        def _falha(casa, fora, mapa):
            chamadas.append((casa, fora, mapa))
            return None

        montar_oportunidades(_odds([_linha(), _linha(pontos=31.5), _linha(pontos=32.5)]),
                             _falha, prob_over=_prob_over_falso)
        self.assertEqual(len(chamadas), 1)

    def test_o_motor_e_chamado_UMA_vez_por_confronto_e_mapa(self):
        chamadas = []

        def _conta(casa, fora, mapa):
            chamadas.append((casa, fora, mapa))
            return {"kills": _Total(30.0)}

        montar_oportunidades(
            _odds([_linha(pontos=25.5), _linha(pontos=26.5), _linha(periodo=2, pontos=25.5)]),
            _conta, prob_over=_prob_over_falso)
        self.assertEqual(sorted(chamadas), [("Top Esports", "Invictus", "m1"),
                                            ("Top Esports", "Invictus", "m2")])

    def test_motor_que_ESTOURA_vira_sem_modelo_e_nao_derruba(self):
        def _explode(casa, fora, mapa):
            raise RuntimeError("time fora do CSV")
        r = montar_oportunidades(_odds([_linha()]), _explode, prob_over=_prob_over_falso)
        self.assertEqual(r["achados"], [])
        self.assertEqual(len(r["sem_modelo"]), 1)

    def test_linha_de_jogo_que_nao_esta_na_lista_e_ignorada(self):
        r = montar_oportunidades(_odds([_linha(jogo=999)]), self._totais,
                                 prob_over=_prob_over_falso)
        self.assertEqual(r["achados"], [])

    def test_leva_a_hora_da_leitura_junto(self):
        """Odd sem carimbo de quando foi vista nao serve para decidir nada."""
        r = montar_oportunidades(_odds([_linha()]), self._totais, prob_over=_prob_over_falso)
        self.assertEqual(r["lido_em"], "2026-09-15T20:21:00-03:00")

    def test_arquivo_vazio_nao_estoura(self):
        r = montar_oportunidades({"jogos": [], "linhas": []}, self._totais,
                                 prob_over=_prob_over_falso)
        self.assertEqual(r["achados"], [])
        self.assertEqual(r["avaliadas"], 0)

    def test_lixo_no_meio_das_linhas_nao_derruba_o_resto(self):
        r = montar_oportunidades(_odds(["lixo", None, _linha(pontos=25.5, odd=2.00)]),
                                 self._totais, prob_over=_prob_over_falso)
        self.assertEqual(len(r["achados"]), 1)


class TestDadosFinosAparecem(unittest.TestCase):
    """Vantagem calculada sobre 3 jogos nao e vantagem: e ruido com enfeite."""

    def _totais(self, casa, fora, mapa):
        return {"kills": _Total(30.0)}

    def test_poucos_jogos_ficam_MARCADOS(self):
        r = montar_oportunidades(_odds([_linha(pontos=25.5, odd=2.00)]), self._totais,
                                 prob_over=_prob_over_falso,
                                 jogos_por_time=lambda t: 3)
        self.assertTrue(r["achados"][0]["dados_finos"])
        self.assertEqual(r["achados"][0]["jogos_no_historico"], 3)

    def test_historico_cheio_nao_fica_marcado(self):
        r = montar_oportunidades(_odds([_linha(pontos=25.5, odd=2.00)]), self._totais,
                                 prob_over=_prob_over_falso,
                                 jogos_por_time=lambda t: 40)
        self.assertFalse(r["achados"][0]["dados_finos"])

    def test_vale_o_time_com_MENOS_jogos_dos_dois(self):
        """Um time bem conhecido nao conserta o desconhecido do outro lado."""
        n = {"Top Esports": 40, "Invictus": 2}
        r = montar_oportunidades(_odds([_linha(pontos=25.5, odd=2.00)]), self._totais,
                                 prob_over=_prob_over_falso,
                                 jogos_por_time=lambda t: n[t])
        self.assertEqual(r["achados"][0]["jogos_no_historico"], 2)
        self.assertTrue(r["achados"][0]["dados_finos"])

    def test_contagem_que_estoura_nao_derruba_e_nao_mente(self):
        def _explode(t):
            raise RuntimeError("sem CSV")
        r = montar_oportunidades(_odds([_linha(pontos=25.5, odd=2.00)]), self._totais,
                                 prob_over=_prob_over_falso, jogos_por_time=_explode)
        self.assertEqual(r["achados"][0]["jogos_no_historico"], -1)
        self.assertFalse(r["achados"][0]["dados_finos"])


class TestNaoSeReimplementaOMotor(unittest.TestCase):
    """PRINCIPIO V.

    Se este modulo tivesse a sua propria conta de probabilidade, o mesmo
    confronto daria um numero aqui e outro na aba de analise -- e nenhum dos
    dois mereceria confianca.
    """

    def test_a_conta_de_probabilidade_vem_de_FORA(self):
        import inspect

        from app_core import oportunidades_core as m
        src = inspect.getsource(m)
        self.assertNotIn("def prob_over_from_total", src)
        self.assertNotIn("_norm_cdf", src)
        self.assertNotIn("_poisson_cdf", src)

    def test_o_motor_de_producao_encaixa_na_assinatura_esperada(self):
        """Contrato com o motor de verdade, nao com o dublê dos testes."""
        from mlcore.lines_engine import MatchupTotals, prob_over_from_total
        t = MatchupTotals(mean=30.0, sd=5.0, dist="normal")
        a = avaliar_linha(_linha(pontos=25.5, odd=2.00), {"kills": t},
                          prob_over=prob_over_from_total)
        self.assertIsNotNone(a)
        self.assertGreater(a["probabilidade"], 0.5)
        self.assertLess(a["probabilidade"], 1.0)
        self.assertAlmostEqual(a["odd_justa"], 1.0 / a["probabilidade"], places=9)


class TestSemRede(unittest.TestCase):
    """O app NUNCA vai a rede: quem coleta e o robo.

    Uma visita dele ao site viraria uma ida a casa de apostas, e a hospedagem
    ja provou que nao e confiavel para isso (13/09/2026).
    """

    def test_o_modulo_nao_importa_requests(self):
        import inspect

        from app_core import oportunidades_core as m
        src = inspect.getsource(m)
        self.assertNotIn("import requests", src)
        self.assertNotIn("urlopen", src)


if __name__ == "__main__":
    unittest.main()


class TestOMercadoDeVencedor(unittest.TestCase):
    """ELE ESTRANHOU a aba mostrando zero: "nao faz mt sentido".

    Tinha razao: o numero JA EXISTE. O card do confronto mostra, por exemplo,
    MD5 1,66 x 2,51 (60,1% x 39,9%) -- exatamente a odd justa que faltava.

    O QUE NAO SE FAZ AQUI: recalcular essa probabilidade. O projeto ja teve
    TRES implementacoes do pre-jogo com contagens diferentes, e existe um
    modulo inteiro escrito por causa disso. O numero e RECEBIDO do fluxo do
    confronto -- o mesmo que ele ve na tela.
    """

    def _ml(self, **k):
        base = {"jogo": 1, "tipo": "moneyline", "periodo": 0, "lado": "home",
                "pontos": None, "odd": 2.84, "unidade": "",
                "mercado": "Vencedor - serie"}
        base.update(k)
        return base

    def test_a_serie_usa_p_series_e_o_mapa_usa_p_map(self):
        from app_core.oportunidades_core import prob_do_vencedor
        self.assertAlmostEqual(
            prob_do_vencedor(self._ml(periodo=0), p_map=0.53, p_series=0.56,
                             casa_e_time_a=True), 0.56, places=9)
        self.assertAlmostEqual(
            prob_do_vencedor(self._ml(periodo=1), p_map=0.53, p_series=0.56,
                             casa_e_time_a=True), 0.53, places=9)

    def test_o_lado_de_FORA_e_o_complemento(self):
        from app_core.oportunidades_core import prob_do_vencedor
        self.assertAlmostEqual(
            prob_do_vencedor(self._ml(lado="away"), p_map=0.53, p_series=0.56,
                             casa_e_time_a=True), 0.44, places=9)

    def test_CONFRONTO_INVERTIDO_troca_o_lado(self):
        """O PIOR ERRO POSSIVEL seria este passar batido.

        A casa pode listar "A x B" e a tela ter "B x A". Sem inverter, o
        painel recomendaria apostar no TIME ERRADO com toda a confianca.
        """
        from app_core.oportunidades_core import prob_do_vencedor
        self.assertAlmostEqual(
            prob_do_vencedor(self._ml(), p_map=0.53, p_series=0.56,
                             casa_e_time_a=False), 0.44, places=9)

    def test_quem_faz_mais_KILLS_nao_e_vencedor_da_partida(self):
        """"Vencedor - Kills" tambem e moneyline com lados casa/fora, mas
        pergunta outra coisa. Usar p_series ali seria precificar uma aposta
        de kills com a probabilidade de ganhar a serie."""
        from app_core.oportunidades_core import prob_do_vencedor
        self.assertIsNone(prob_do_vencedor(self._ml(unidade="Kills", periodo=1),
                                           p_map=0.53, p_series=0.56, casa_e_time_a=True))

    def test_placar_exato_nao_entra_por_aqui(self):
        from app_core.oportunidades_core import prob_do_vencedor
        self.assertIsNone(prob_do_vencedor(self._ml(lado="Fuego 3, 9z 0", unidade="Maps"),
                                           p_map=0.53, p_series=0.56, casa_e_time_a=True))

    def test_sem_numero_do_modelo_NAO_inventa(self):
        """PRINCIPIO IV: p_series NaN nao vira 0,5."""
        from app_core.oportunidades_core import prob_do_vencedor
        self.assertIsNone(prob_do_vencedor(self._ml(), p_map=0.53,
                                           p_series=float("nan"), casa_e_time_a=True))
        self.assertIsNone(prob_do_vencedor(self._ml(), p_map=0.53,
                                           p_series=None, casa_e_time_a=True))


class TestEOMESMOConfronto(unittest.TestCase):
    """Comparar a odd de um jogo com o modelo de OUTRO seria desastre."""

    def _jogo(self, casa="Natus Vincere", fora="Movistar KOI", bo=5):
        return {"id": 1, "casa": casa, "fora": fora, "bo": bo}

    def _id(self, n):
        return n

    def test_mesma_ordem_da_verdadeiro(self):
        from app_core.oportunidades_core import mesmo_confronto
        self.assertIs(mesmo_confronto(self._jogo(), "Natus Vincere", "Movistar KOI",
                                      self._id), True)

    def test_ordem_INVERTIDA_da_falso_e_nao_None(self):
        """False quer dizer "e o mesmo jogo, mas de cabeca para baixo" --
        informacao diferente de "nao e este jogo"."""
        from app_core.oportunidades_core import mesmo_confronto
        self.assertIs(mesmo_confronto(self._jogo(), "Movistar KOI", "Natus Vincere",
                                      self._id), False)

    def test_outro_confronto_da_None(self):
        from app_core.oportunidades_core import mesmo_confronto
        self.assertIsNone(mesmo_confronto(self._jogo(), "T1", "Gen.G", self._id))

    def test_um_time_so_batendo_NAO_basta(self):
        from app_core.oportunidades_core import mesmo_confronto
        self.assertIsNone(mesmo_confronto(self._jogo(), "Natus Vincere", "T1", self._id))

    def test_time_que_nao_resolve_da_None(self):
        from app_core.oportunidades_core import mesmo_confronto
        self.assertIsNone(mesmo_confronto(self._jogo(), "Natus Vincere", "Movistar KOI",
                                          lambda n: ""))

    def test_resolvedor_que_estoura_nao_derruba(self):
        from app_core.oportunidades_core import mesmo_confronto

        def _explode(n):
            raise RuntimeError("sem CSV")
        self.assertIsNone(mesmo_confronto(self._jogo(), "a", "b", _explode))


class TestOBODaCasaTemQueBaterComOModelo(unittest.TestCase):
    """UM BO5 PRECIFICADO COMO BO3 muda a probabilidade da serie inteira."""

    def test_bo_igual_passa(self):
        from app_core.oportunidades_core import bo_bate
        self.assertTrue(bo_bate({"bo": 5}, 5))

    def test_bo_diferente_NAO_passa(self):
        from app_core.oportunidades_core import bo_bate
        self.assertFalse(bo_bate({"bo": 5}, 3))

    def test_bo_ausente_ou_ilegivel_NAO_passa(self):
        """PRINCIPIO IV: sem saber o formato, nao se afirma que bate."""
        from app_core.oportunidades_core import bo_bate
        self.assertFalse(bo_bate({}, 5))
        self.assertFalse(bo_bate({"bo": "cinco"}, 5))
        self.assertFalse(bo_bate({"bo": 5}, None))


class TestOVencedorEntraNaLista(unittest.TestCase):

    def _totais(self, casa, fora, mapa):
        return None      # sem totais: so o vencedor deve aparecer

    def _linha_ml(self):
        return {"jogo": 1, "tipo": "moneyline", "periodo": 0, "lado": "home",
                "pontos": None, "odd": 2.84, "unidade": "",
                "mercado": "Vencedor - serie"}

    def test_a_linha_de_vencedor_vira_achado(self):
        r = montar_oportunidades(
            _odds([self._linha_ml()]), self._totais, prob_over=_prob_over_falso,
            minimo_pct=0.0, prob_de_vencedor=lambda jogo, linha: 0.56)
        self.assertEqual(len(r["achados"]), 1)
        self.assertEqual(r["achados"][0]["metrica"], "vencedor")
        self.assertAlmostEqual(r["achados"][0]["vantagem_pct"], 59.04, places=1)

    def test_quando_o_modelo_nao_sabe_a_linha_e_CONTADA_de_fora(self):
        r = montar_oportunidades(
            _odds([self._linha_ml()]), self._totais, prob_over=_prob_over_falso,
            prob_de_vencedor=lambda jogo, linha: None)
        self.assertEqual(r["achados"], [])
        self.assertEqual(sum(g["linhas"] for g in r["nao_precificados"]), 1)

    def test_precificador_que_ESTOURA_nao_derruba_a_tela(self):
        def _explode(jogo, linha):
            raise RuntimeError("sem numero")
        r = montar_oportunidades(
            _odds([self._linha_ml()]), self._totais, prob_over=_prob_over_falso,
            prob_de_vencedor=_explode)
        self.assertEqual(r["achados"], [])
        self.assertEqual(sum(g["linhas"] for g in r["nao_precificados"]), 1)

    def test_vantagem_gigante_no_vencedor_tambem_vai_MARCADA(self):
        """Foi o que apareceu no primeiro uso real: +59,65% num mercado com
        5,36% de margem da casa. A conta estava certa; a leitura e que nao
        pode ser "dinheiro de graca"."""
        r = montar_oportunidades(
            _odds([self._linha_ml()]), self._totais, prob_over=_prob_over_falso,
            minimo_pct=0.0, prob_de_vencedor=lambda jogo, linha: 0.56)
        self.assertTrue(r["achados"][0]["suspeito"])
