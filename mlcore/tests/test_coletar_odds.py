# -*- coding: utf-8 -*-
"""Testes do coletor de odds da Pinnacle.

PARA QUE SERVE
A Pinnacle trabalha com margem baixa e limite alto e aceita dinheiro afiado em
vez de barrar a conta — o preco dela e o mais proximo de probabilidade
verdadeira que o mercado oferece. Guardado COM HORA, ele vira a regua externa
que este projeto nao tem: medir se o modelo bate o fechamento (CLV) e a unica
medicao que responde se existe lucro.

DE ONDE VEM, SEM DISFARCE
E a API interna do site DELES, aberta e sem credencial — a porta dos fundos,
nao a oficial (que fechou, e que para esports exige autorizacao especial).
Ele decidiu usa-la sabendo disso em 15/09/2026. Por isso os testes de
EDUCACAO abaixo: 403/429 tem de parar na hora, sem insistir nem contornar.

O QUE ESTES TESTES PROTEGEM, em ordem:
  1. o historico guarda SO o que mudou — senao viram ~9 MB/mes no repositorio,
     e o que o CLV precisa e o movimento, nao o retrato repetido;
  2. mudanca de formato da API vira ERRO ALTO, nunca lista vazia calada;
  3. preco ilegivel vira ausencia, nunca 1,0 ou 2,0 disfarcado de odd.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.coletar_odds import (
    _ultimas_odds,
    chave_da_linha,
    margem,
    normalizar,
    para_decimal,
)


def _matchup(**k):
    base = {
        "id": 1, "parentId": None, "isLive": False, "bestOfX": 5,
        "startTime": "2026-09-18T20:00:00Z",
        "league": {"name": "League of Legends - LCS"},
        "participants": [
            {"alignment": "home", "name": "Shopify Rebellion"},
            {"alignment": "away", "name": "Sentinels"},
        ],
    }
    base.update(k)
    return base


def _mercado(**k):
    base = {
        "matchupId": 1, "type": "moneyline", "period": 0, "isAlternate": False,
        "cutoffAt": "2026-09-18T20:00:00+00:00",
        "limits": [{"amount": 2000, "type": "maxRiskStake"}],
        "prices": [{"designation": "home", "price": 190},
                   {"designation": "away", "price": -243}],
    }
    base.update(k)
    return base


class TestOddAmericanaViraDecimal(unittest.TestCase):

    def test_positiva(self):
        self.assertAlmostEqual(para_decimal(190), 2.90, places=4)

    def test_negativa(self):
        self.assertAlmostEqual(para_decimal(-243), 1.4115, places=4)

    def test_favorito_pesado(self):
        self.assertAlmostEqual(para_decimal(-853), 1.1172, places=4)

    def test_preco_ilegivel_vira_AUSENCIA_e_nao_odd(self):
        """PRINCIPIO IV: 1,0 ou 2,0 chutado viraria aposta de graca na tela."""
        for ruim in (None, "", "abc", 0):
            self.assertIsNone(para_decimal(ruim), ruim)


class TestAMargemDenunciaAConfianca(unittest.TestCase):
    """Medido em 15/09/2026: LPL 4,8% · LCS 5,3% · WSCI 9,4%.

    Margem alta e a casa dizendo "nao sei direito, entao me protejo".
    """

    def test_calcula_a_margem_das_duas_pernas(self):
        m = margem([2.90, 1.4115])
        self.assertAlmostEqual(m, 5.34, delta=0.1)

    def test_margem_menor_em_mercado_mais_liquido(self):
        self.assertLess(margem([2.150, 1.714]), margem([1.166, 4.220]))

    def test_perna_faltando_NAO_vira_margem_inventada(self):
        self.assertIsNone(margem([2.90]))
        self.assertIsNone(margem([]))

    def test_odd_invalida_nao_vira_margem(self):
        self.assertIsNone(margem([2.90, None]))
        self.assertIsNone(margem([2.90, 1.0]))


class TestOQueEntraEOQueFicaDeFora(unittest.TestCase):

    def test_le_a_serie_com_times_e_formato(self):
        d = normalizar([_matchup()], [_mercado()])
        self.assertEqual(len(d["jogos"]), 1)
        j = d["jogos"][0]
        self.assertEqual(j["casa"], "Shopify Rebellion")
        self.assertEqual(j["fora"], "Sentinels")
        self.assertEqual(j["bo"], 5)
        self.assertEqual(j["liga"], "LCS")

    def test_ao_vivo_fica_de_fora(self):
        """O escopo deste projeto e PRE-JOGO."""
        d = normalizar([_matchup(isLive=True)], [_mercado()])
        self.assertEqual(d["jogos"], [])
        self.assertEqual(d["descartes"]["ao_vivo"], 1)

    def test_confronto_filho_SEM_O_PAI_fica_de_fora(self):
        """Aposta de uma partida que nao esta na lista.

        MUDOU EM 15/09/2026: filho COM o pai presente passou a ser adotado --
        e assim que kills e placar exato chegam. O descarte sobrou so para o
        orfao, que e o caso em que nao da nem para dizer de que jogo a aposta
        e. PRINCIPIO IV.
        """
        d = normalizar([_matchup(parentId=99)], [_mercado()])
        self.assertEqual(d["jogos"], [])
        self.assertEqual(d["descartes"]["filho"], 1)

    def test_outro_jogo_fica_de_fora(self):
        d = normalizar([_matchup(league={"name": "Counter-Strike - BLAST"})], [_mercado()])
        self.assertEqual(d["jogos"], [])
        self.assertEqual(d["descartes"]["outro_jogo"], 1)

    def test_serie_sem_mercado_e_contada(self):
        d = normalizar([_matchup()], [])
        self.assertEqual(d["jogos"], [])
        self.assertEqual(d["descartes"]["sem_mercado"], 1)

    def test_guarda_limite_e_hora_de_fechamento(self):
        """O limite e a casa dizendo em qual preco ela acredita."""
        l = normalizar([_matchup()], [_mercado()])["linhas"][0]
        self.assertEqual(l["limite"], 2000)
        self.assertTrue(l["fecha_em"])

    def test_handicap_e_total_tambem_entram(self):
        mks = [
            _mercado(type="spread", prices=[
                {"designation": "home", "points": 1.5, "price": -114},
                {"designation": "away", "points": -1.5, "price": -106}]),
            _mercado(type="total", prices=[
                {"designation": "over", "points": 4.5, "price": 172},
                {"designation": "under", "points": 4.5, "price": -247}]),
        ]
        tipos = {l["tipo"] for l in normalizar([_matchup()], mks)["linhas"]}
        self.assertEqual(tipos, {"spread", "total"})

    def test_a_linha_do_handicap_e_guardada(self):
        mk = _mercado(type="spread", prices=[
            {"designation": "home", "points": 1.5, "price": -114},
            {"designation": "away", "points": -1.5, "price": -106}])
        pts = sorted(l["pontos"] for l in normalizar([_matchup()], [mk])["linhas"])
        self.assertEqual(pts, [-1.5, 1.5])


class TestSeAAPIMudarELOUD(unittest.TestCase):
    """Lista vazia calada seria o pior desfecho: "sem odds hoje" com toda a
    confianca do mundo, num dia cheio de jogo."""

    def test_matchups_que_nao_e_lista_estoura(self):
        with self.assertRaises(ValueError):
            normalizar({"data": []}, [])

    def test_markets_que_nao_e_lista_estoura(self):
        with self.assertRaises(ValueError):
            normalizar([], {"data": []})

    def test_a_mensagem_aponta_a_causa(self):
        try:
            normalizar("nada", [])
        except ValueError as e:
            self.assertIn("API mudou", str(e))

    def test_lixo_no_meio_da_lista_nao_derruba_o_resto(self):
        d = normalizar([None, "texto", 42, _matchup()], [_mercado()])
        self.assertEqual(len(d["jogos"]), 1)


class TestOHistoricoGuardaSoOMovimento(unittest.TestCase):
    """Retrato repetido daria ~9 MB/mes. O CLV precisa do MOVIMENTO."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.hist = Path(self.tmp.name) / "hist.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _grava(self, registros):
        with open(self.hist, "a", encoding="utf-8") as fh:
            for r in registros:
                fh.write(json.dumps(r) + "\n")

    def test_le_o_ultimo_preco_de_cada_aposta(self):
        self._grava([{"chave": "a", "odd": 2.0}, {"chave": "a", "odd": 2.2},
                     {"chave": "b", "odd": 1.5}])
        u = _ultimas_odds(self.hist)
        self.assertEqual(u["a"], 2.2, "tem de valer o ULTIMO, nao o primeiro")
        self.assertEqual(u["b"], 1.5)

    def test_historico_ausente_nao_estoura(self):
        self.assertEqual(_ultimas_odds(self.hist), {})

    def test_linha_truncada_nao_derruba_o_resto(self):
        with open(self.hist, "w", encoding="utf-8") as fh:
            fh.write('{"chave": "a", "odd": 2.0}\n')
            fh.write('{"chave": "b", "odd\n')          # cortada no meio
            fh.write('{"chave": "c", "odd": 1.8}\n')
        u = _ultimas_odds(self.hist)
        self.assertEqual(u.get("a"), 2.0)
        self.assertEqual(u.get("c"), 1.8)

    def test_a_chave_separa_apostas_diferentes(self):
        base = {"jogo": 1, "tipo": "spread", "periodo": 0, "lado": "home"}
        k1 = chave_da_linha(dict(base, pontos=1.5))
        k2 = chave_da_linha(dict(base, pontos=2.5))
        self.assertNotEqual(k1, k2, "linhas diferentes do handicap sao apostas diferentes")

    def test_a_chave_separa_mapa_de_serie(self):
        base = {"jogo": 1, "tipo": "moneyline", "lado": "home", "pontos": None}
        self.assertNotEqual(chave_da_linha(dict(base, periodo=0)),
                            chave_da_linha(dict(base, periodo=1)))

    def test_a_chave_separa_MERCADOS_com_o_mesmo_lado(self):
        """Sem o nome do mercado na chave, "Yes" atropela "Yes".

        Num mesmo jogo existem "Will A Win At Least One Map? / Yes" e "Will B
        Win At Least One Map? / Yes": mesmo jogo, mesmo tipo, mesmo periodo,
        mesmo lado, sem pontos. Com a chave antiga as duas eram A MESMA
        aposta, e o historico de precos misturava as duas -- que e o dado
        usado para medir se o modelo bate o fechamento.
        """
        base = {"jogo": 1, "tipo": "moneyline", "periodo": 0,
                "lado": "Yes", "pontos": None}
        k1 = chave_da_linha(dict(base, mercado="Will A Win At Least One Map?"))
        k2 = chave_da_linha(dict(base, mercado="Will B Win At Least One Map?"))
        self.assertNotEqual(k1, k2)

    def test_a_chave_separa_os_dois_lados(self):
        base = {"jogo": 1, "tipo": "moneyline", "periodo": 0, "pontos": None}
        self.assertNotEqual(chave_da_linha(dict(base, lado="home")),
                            chave_da_linha(dict(base, lado="away")))


class TestDerivadosKillsTorresTempo(unittest.TestCase):
    """ERRO MEU, corrigido em 15/09/2026 depois de ELE me corrigir.

    Eu afirmei que a Pinnacle "so tem mercado de mapas". Ela TEM kills, torres
    e tempo — eu e que olhei so o nivel do esporte inteiro, que traz os
    mercados principais. Os derivados vivem por confronto, em
    `matchups/{id}/related` + `matchups/{id}/markets/straight`.

    Pior: o codigo DESCARTAVA tudo que tivesse `parentId`, chamando de "filho".
    Entre os descartados estavam justamente os derivados.
    """

    def test_derivado_entra_PENDURADO_no_jogo_pai(self):
        """Um jogo, nao dois.

        "(Map 1) Total Kills" nao e outra partida: e outra aposta da MESMA
        partida. Se virasse um jogo separado, a tela mostraria "Shopify x
        Sentinels" duas vezes e ele nao saberia qual olhar.
        """
        pai = _matchup(id=1)
        deriv = _matchup(id=2, parentId=1)
        deriv["_paiId"] = 1          # marca de derivado buscado de proposito
        d = normalizar([pai], [_mercado(matchupId=1)],
                       matchups_extra=[deriv], mercados_extra=[_mercado(matchupId=2)])
        self.assertEqual(d["descartes"]["filho"], 0)
        self.assertEqual(len(d["jogos"]), 1)
        self.assertEqual({l["jogo"] for l in d["linhas"]}, {1},
                         "a linha do derivado tem de apontar para o jogo pai")

    def test_filho_da_lista_GERAL_e_adotado_DE_GRACA(self):
        """MUDEI DE IDEIA em 15/09/2026, e por medicao.

        Eu escrevia aqui que "placar exato nao interessa". Mas ele JA VEM na
        resposta que pedimos de qualquer jeito: descartar nao economiza nada,
        so joga aposta fora. Medido no mesmo dia: 54 linhas de Placar Exato em
        9 series, custo ZERO em requisicao.
        """
        d = normalizar([_matchup(id=1), _matchup(id=2, parentId=1)],
                       [_mercado(matchupId=1), _mercado(matchupId=2)])
        self.assertEqual(d["descartes"]["filho"], 0)
        self.assertEqual(len(d["jogos"]), 1, "continua sendo UM jogo")
        self.assertEqual(len([l for l in d["linhas"] if l["jogo"] == 1]), 4)

    def test_filho_ORFAO_continua_descartado(self):
        """Sem o pai na resposta nao ha casa nem fora.

        PRINCIPIO IV: uma aposta sem saber de que jogo e nao vira palpite.
        """
        d = normalizar([_matchup(id=2, parentId=999)], [_mercado(matchupId=2)])
        self.assertEqual(d["descartes"]["filho"], 1)
        self.assertEqual(d["jogos"], [])

    def test_lado_NEUTRO_sobrevive_e_ganha_nome(self):
        """O DEFEITO QUE SEGURAVA KILLS, TORRE E TEMPO (15/09/2026).

        Num mercado derivado a casa nao marca "home"/"away": marca todos como
        "neutral" e diz o lado pelo `participantId` -- "Yes", "No", "Odd".
        A checagem antiga exigia casa E fora, e derrubava o derivado num
        `continue` MUDO, sem nem contar. Resultado: eu rodava, via 274 linhas,
        e NENHUMA era de kills -- sem um unico sinal de que algo fora jogado
        fora.
        """
        pai = _matchup(id=1)
        esp = _matchup(id=2, parentId=1, units="Kills")
        esp["special"] = {"category": "Teams", "description": "(Map 1) Total Kills"}
        esp["participants"] = [
            {"id": 900, "alignment": "neutral", "name": "Over 25.5"},
            {"id": 901, "alignment": "neutral", "name": "Under 25.5"},
        ]
        mk = _mercado(matchupId=2, type="total", period=1, prices=[
            {"designation": None, "participantId": 900, "price": -110, "points": 25.5},
            {"designation": None, "participantId": 901, "price": -110, "points": 25.5},
        ])
        d = normalizar([pai, esp], [_mercado(matchupId=1), mk])
        self.assertEqual(d["descartes"]["sem_lados"], 0)
        kills = [l for l in d["linhas"] if l["mercado"] == "(Map 1) Total Kills"]
        self.assertEqual(len(kills), 2, "kills foi descartado de novo")
        self.assertEqual({l["lado"] for l in kills}, {"Over 25.5", "Under 25.5"})
        self.assertEqual({l["pontos"] for l in kills}, {25.5})

    def test_o_proprio_jogo_voltando_na_varredura_NAO_duplica(self):
        """`related` devolve o confronto junto com os derivados dele.

        DEFEITO DE 15/09/2026: sem isto, 9 series apareceram como "11 series"
        e as linhas dos jogos varridos foram contadas em dobro. Numero inflado
        e pior que numero faltando -- parece certo.
        """
        pai = _matchup(id=1)
        d = normalizar([pai], [_mercado(matchupId=1)],
                       matchups_extra=[dict(pai, _paiId=1)],
                       mercados_extra=[_mercado(matchupId=1)])
        self.assertEqual(len(d["jogos"]), 1)
        self.assertEqual(len(d["linhas"]), 2, "a mesma aposta entrou duas vezes")

    def test_a_marca_de_pai_sobrevive_a_remocao_de_repetidos(self):
        """O derivado JA VEM na lista geral, so que sem saber de quem e filho.

        Ficar com a primeira copia jogava a marca fora e o mercado caia como
        "filho" -- o mesmo silencio, por outra porta.
        """
        pai = _matchup(id=1)
        filho_cru = _matchup(id=2, parentId=1)
        d = normalizar([pai, filho_cru], [_mercado(matchupId=1), _mercado(matchupId=2)],
                       matchups_extra=[dict(filho_cru, _paiId=1)],
                       mercados_extra=[_mercado(matchupId=2)])
        self.assertEqual(d["descartes"]["filho"], 0)
        self.assertEqual(len(d["jogos"]), 1)


class TestKillsComoElaRealmenteChega(unittest.TestCase):
    """A FORMA REAL, copiada da resposta ao vivo de 15/09/2026.

    Kills nao e um "special" nem mora dentro do jogo: e um confronto FILHO
    com unidade propria e os times renomeados. Vinha em toda leitura, de
    graca, e o filtro de "filho" jogava fora -- por isso eu rodava e via 274
    linhas sem uma unica de kills.

    Este teste guarda a forma. Se ela mudar, quebra aqui e nao no silencio.
    """

    def _kills(self):
        m = _matchup(id=2, parentId=1, units="Kills")
        m["participants"] = [
            {"alignment": "home", "name": "Fuego (Kills)"},
            {"alignment": "away", "name": "9z (Kills)"},
        ]
        m["periods"] = [{"period": n, "hasMoneyline": True, "hasSpread": True,
                         "hasTotal": True, "hasTeamTotal": True} for n in (1, 2, 3)]
        return m

    def test_o_mercado_de_kills_entra_e_diz_que_e_de_kills(self):
        total = _mercado(matchupId=2, type="total", period=1, prices=[
            {"designation": "over", "price": -110, "points": 33.5},
            {"designation": "under", "price": -110, "points": 33.5},
        ])
        d = normalizar([_matchup(id=1), self._kills()],
                       [_mercado(matchupId=1), total])
        k = [l for l in d["linhas"] if "Kills" in l["mercado"]]
        self.assertEqual(len(k), 2, "kills foi descartado")
        self.assertEqual(k[0]["mercado"], "Total - Kills - mapa 1")
        self.assertEqual({l["pontos"] for l in k}, {33.5})

    def test_kills_fica_pendurado_no_jogo_de_verdade(self):
        """Nao pode virar uma partida "Fuego (Kills)" x "9z (Kills)".

        Se virasse, a tela mostraria um confronto que nao existe e ele nao
        teria como ligar a linha de kills ao jogo que vai apostar.
        """
        d = normalizar([_matchup(id=1), self._kills()],
                       [_mercado(matchupId=1), _mercado(matchupId=2, period=1)])
        self.assertEqual(len(d["jogos"]), 1)
        self.assertEqual(d["jogos"][0]["casa"], "Shopify Rebellion")
        self.assertEqual({l["jogo"] for l in d["linhas"]}, {1})

    def test_os_tres_mapas_nao_viram_a_mesma_aposta(self):
        """A casa abre os 3 mapas com o MESMO preco antes do jogo.

        Visto ao vivo: mapa 1, 2 e 3 com odd identica. Se a chave nao separar
        o periodo, os tres viram uma linha so e o historico perde dois tercos
        do movimento.
        """
        mks = [_mercado(matchupId=2, type="total", period=n, prices=[
            {"designation": "over", "price": -110, "points": 33.5},
            {"designation": "under", "price": -110, "points": 33.5}]) for n in (1, 2, 3)]
        d = normalizar([_matchup(id=1), self._kills()], [_mercado(matchupId=1)] + mks)
        k = [l for l in d["linhas"] if "Kills" in l["mercado"]]
        self.assertEqual(len(k), 6)
        self.assertEqual(len({chave_da_linha(l) for l in k}), 6)

    def test_uma_unidade_NOVA_entra_sozinha(self):
        """Torre e tempo ainda nao abriram; quando abrirem, tem de passar.

        Nao sei como ela vai chamar a unidade. O rotulo repete o que ela
        disser, em vez de tentar adivinhar -- se vier "Towers", aparece
        "Towers".
        """
        m = self._kills()
        m["units"] = "Towers"
        d = normalizar([_matchup(id=1), m],
                       [_mercado(matchupId=1), _mercado(matchupId=2, type="total", period=1)])
        self.assertTrue(any("Towers" in l["mercado"] for l in d["linhas"]))


class TestFilhoSoEntraSeOPaiEntrou(unittest.TestCase):
    """Uma aposta pendurada num jogo que nao esta na lista nao serve.

    Achado revendo o proprio conserto, antes de subir: se o jogo pai estivesse
    AO VIVO (fora do escopo, que e pre-jogo) ou sem mercado nenhum, a linha de
    kills do filho ainda entrava -- apontando para um jogo que a tela nao
    mostra. Apareceria uma aposta orfa, sem como saber de quem e.

    Por isso a leitura passou a ser em DUAS passadas: primeiro quais partidas
    entram, so depois as apostas penduradas nelas.
    """

    def _filho(self):
        f = _matchup(id=2, parentId=1, units="Kills")
        f["participants"] = [{"alignment": "home", "name": "A (Kills)"},
                             {"alignment": "away", "name": "B (Kills)"}]
        return f

    def test_pai_AO_VIVO_leva_o_filho_junto(self):
        d = normalizar([_matchup(id=1, isLive=True), self._filho()],
                       [_mercado(matchupId=1), _mercado(matchupId=2, period=1)])
        self.assertEqual(d["jogos"], [])
        self.assertEqual(d["linhas"], [], "sobrou aposta de um jogo que nao esta na lista")

    def test_pai_SEM_MERCADO_leva_o_filho_junto(self):
        d = normalizar([_matchup(id=1), self._filho()],
                       [_mercado(matchupId=2, period=1)])
        self.assertEqual(d["jogos"], [])
        self.assertEqual(d["linhas"], [])

    def test_filho_AO_VIVO_nao_entra_mesmo_com_o_pai_certo(self):
        f = self._filho()
        f["isLive"] = True
        d = normalizar([_matchup(id=1), f],
                       [_mercado(matchupId=1), _mercado(matchupId=2, period=1)])
        self.assertEqual(len(d["jogos"]), 1)
        self.assertTrue(all("Kills" not in l["mercado"] for l in d["linhas"]))

    def test_filho_ANTES_do_pai_na_lista_ainda_e_adotado(self):
        """A resposta nao promete ordem, e ordem nao pode decidir isso."""
        d = normalizar([self._filho(), _matchup(id=1)],
                       [_mercado(matchupId=1), _mercado(matchupId=2, period=1)])
        self.assertEqual(len(d["jogos"]), 1)
        self.assertTrue(any("Kills" in l["mercado"] for l in d["linhas"]))


class TestACasaAvisaQuandoAbriu(unittest.TestCase):
    """`periods` diz, mapa a mapa, que mercados existem.

    MEDIDO EM 15/09/2026: 0 de 9 jogos de LoL tinham qualquer aposta dentro do
    mapa; 24 de 27 do CS2 tinham. Ou seja, ela abre kills/torre/tempo perto do
    inicio. Com este aviso nao gastamos requisicao batendo na porta para ouvir
    "nao tem" -- e no minuto em que abrir, a varredura vai atras sozinha.
    """

    def test_sem_nada_dentro_do_mapa_da_falso(self):
        from tools.coletar_odds import tem_mercado_no_mapa
        m = _matchup(periods=[
            {"period": 0, "hasMoneyline": True, "hasSpread": True, "hasTotal": True},
            {"period": 1, "hasMoneyline": True, "hasSpread": False, "hasTotal": False},
        ])
        self.assertFalse(tem_mercado_no_mapa(m))

    def test_total_no_mapa_1_da_verdadeiro(self):
        from tools.coletar_odds import tem_mercado_no_mapa
        m = _matchup(periods=[
            {"period": 0, "hasTotal": True},
            {"period": 1, "hasTotal": True},
        ])
        self.assertTrue(tem_mercado_no_mapa(m))

    def test_total_so_na_SERIE_nao_conta(self):
        """Total da serie e total de MAPAS -- e isso ele ja tinha."""
        from tools.coletar_odds import tem_mercado_no_mapa
        self.assertFalse(
            tem_mercado_no_mapa(_matchup(periods=[{"period": 0, "hasTotal": True}])))

    def test_sem_periods_nao_estoura(self):
        from tools.coletar_odds import tem_mercado_no_mapa
        self.assertFalse(tem_mercado_no_mapa({}))
        self.assertFalse(tem_mercado_no_mapa({"periods": None}))
        self.assertFalse(tem_mercado_no_mapa({"periods": ["lixo", 7]}))

    def test_o_aviso_chega_no_jogo(self):
        d = normalizar([_matchup(periods=[{"period": 1, "hasTotal": True}])], [_mercado()])
        self.assertTrue(d["jogos"][0]["tem_mercado_no_mapa"])

    def test_a_linha_guarda_o_NOME_do_mercado(self):
        """Sem isso, uma linha de kills fica igual a uma de mapas."""
        d = normalizar([_matchup()], [_mercado()])
        self.assertIn("mercado", d["linhas"][0])

    def test_o_rotulo_vem_do_special_quando_existe(self):
        from tools.coletar_odds import rotulo_do_derivado
        self.assertEqual(
            rotulo_do_derivado({"special": {"description": "Total Kills"}}), "Total Kills")

    def test_sem_special_usa_a_unidade(self):
        from tools.coletar_odds import rotulo_do_derivado
        self.assertEqual(rotulo_do_derivado({"units": "Maps"}), "Maps")

    def test_sem_nada_vira_interrogacao_e_nao_um_palpite(self):
        from tools.coletar_odds import rotulo_do_derivado
        self.assertEqual(rotulo_do_derivado({}), "?")


class TestSoOsJogosProximos(unittest.TestCase):
    """A linha de um jogo de semana que vem nem abriu; buscar e trafego a toa."""

    def _agora(self):
        from datetime import datetime, timezone
        return datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)

    def test_jogo_de_hoje_entra(self):
        from tools.coletar_odds import _comeca_em_breve
        self.assertTrue(_comeca_em_breve("2026-09-15T20:00:00Z", self._agora(), 36))

    def test_jogo_de_semana_que_vem_fica_de_fora(self):
        from tools.coletar_odds import _comeca_em_breve
        self.assertFalse(_comeca_em_breve("2026-09-25T20:00:00Z", self._agora(), 36))

    def test_jogo_que_ja_comecou_ha_pouco_ainda_entra(self):
        from tools.coletar_odds import _comeca_em_breve
        self.assertTrue(_comeca_em_breve("2026-09-15T10:00:00Z", self._agora(), 36))

    def test_jogo_de_ontem_fica_de_fora(self):
        from tools.coletar_odds import _comeca_em_breve
        self.assertFalse(_comeca_em_breve("2026-09-14T10:00:00Z", self._agora(), 36))

    def test_data_ilegivel_fica_de_fora_em_vez_de_estourar(self):
        from tools.coletar_odds import _comeca_em_breve
        self.assertFalse(_comeca_em_breve("nao e data", self._agora(), 36))


class TestQuandoACasaDizNAO(unittest.TestCase):
    """403/429 no meio da busca PARA tudo e guarda o que ja veio.

    Insistir depois de um bloqueio e o caminho mais curto para ser bloqueado de
    vez — e quem perde o acesso ao site e ELE, nao eu. Aconteceu comigo em
    15/09/2026, sondando demais: levei 403 e parei.
    """

    class _Resp:
        def __init__(self, code, data=None):
            self.status_code = code
            self._d = data if data is not None else []
        def json(self):
            return self._d

    class _Sessao:
        def __init__(self, respostas):
            self.respostas = list(respostas)
            self.pedidos = 0
        def get(self, url, **k):
            self.pedidos += 1
            return self.respostas.pop(0) if self.respostas else TestQuandoACasaDizNAO._Resp(200, [])

    def _jogo(self, i=1):
        return {"id": i, "inicio_utc": "2026-09-15T20:00:00Z",
                "tem_mercado_no_mapa": True}

    def _agora(self):
        from datetime import datetime, timezone
        return datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)

    def test_403_para_na_hora(self):
        from tools.coletar_odds import buscar_derivados
        se = self._Sessao([self._Resp(403)])
        r = buscar_derivados(se, {}, [self._jogo()], agora=self._agora(), pausa=0)
        self.assertTrue(r["parado"])
        self.assertEqual(se.pedidos, 1, "insistiu depois do NAO")

    def test_429_tambem_para(self):
        from tools.coletar_odds import buscar_derivados
        se = self._Sessao([self._Resp(429)])
        r = buscar_derivados(se, {}, [self._jogo()], agora=self._agora(), pausa=0)
        self.assertTrue(r["parado"])

    def test_o_que_ja_veio_ANTES_do_NAO_e_guardado(self):
        from tools.coletar_odds import buscar_derivados
        se = self._Sessao([
            self._Resp(200, [{"id": 10}]),      # related do jogo 1
            self._Resp(200, [{"type": "total"}]),  # mercados do derivado 10
            self._Resp(403),                    # e ai a casa diz nao
        ])
        jogos = [self._jogo(1), self._jogo(2)]
        r = buscar_derivados(se, {}, jogos, agora=self._agora(), pausa=0)
        self.assertTrue(r["parado"])
        self.assertEqual(len(r["matchups"]), 1, "perdeu o que ja tinha lido")

    def test_o_teto_de_chamadas_e_respeitado(self):
        from tools.coletar_odds import buscar_derivados
        se = self._Sessao([self._Resp(200, [{"id": i}]) for i in range(50)])
        jogos = [self._jogo(i) for i in range(30)]
        r = buscar_derivados(se, {}, jogos, agora=self._agora(), pausa=0, teto=6)
        self.assertLessEqual(r["chamadas"], 6)

    def test_ha_pausa_entre_as_chamadas(self):
        """Sem pausa, trinta chamadas seguidas viram enxurrada."""
        import inspect

        from tools import coletar_odds as m
        self.assertIn("sleep", inspect.getsource(m.buscar_derivados))


class TestEducacaoComACasa(unittest.TestCase):
    """E a porta dos fundos. Se eles nao quiserem, a resposta e PARAR.

    O risco nao e a conta de aposta dele (nada aqui usa login) — e bloqueio de
    IP, que tiraria o acesso ao site de onde ele aposta.
    """

    def test_403_e_429_param_na_hora(self):
        import inspect

        from tools import coletar_odds as m
        src = inspect.getsource(m.main)
        self.assertIn("403", src)
        self.assertIn("429", src)
        self.assertIn("RC_TEMPORARIO", src)

    def test_nao_ha_retentativa_nem_troca_de_identidade(self):
        """Insistir depois de um bloqueio e o caminho para ser bloqueado de vez."""
        import inspect

        from tools import coletar_odds as m
        src = inspect.getsource(m)
        self.assertNotIn("proxy", src.lower())
        self.assertNotIn("retry", src.lower())

    def test_o_agente_diz_quem_e_e_para_que(self):
        from tools.coletar_odds import AGENTE
        self.assertIn("gabinete", AGENTE.lower())
        self.assertIn("pessoal", AGENTE.lower())

    def test_sao_duas_requisicoes_por_corrida(self):
        import inspect

        from tools import coletar_odds as m
        self.assertEqual(inspect.getsource(m.main).count("requests.get"), 2)


if __name__ == "__main__":
    unittest.main()
