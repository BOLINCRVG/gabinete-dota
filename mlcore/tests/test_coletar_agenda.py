# -*- coding: utf-8 -*-
"""Testes do coletor de agenda.

O QUE ELE FAZ
Hoje o dono copia a lista de jogos de um site e cola no app. O coletor produz
a MESMA lista sozinho, a partir da API que o site oficial da Riot usa por tras.

POR QUE TANTO TESTE PARA UM PROGRAMA QUE SO BAIXA UM JSON
Porque a API nao e documentada nem prometida: pode mudar sem aviso. E o modo
de falhar mais caro aqui e SILENCIOSO — ler um campo que sumiu daria zero
jogos, e a tela diria "nenhum jogo hoje" com toda a confianca do mundo, num
dia cheio de jogos. Estes testes existem para que esse caso vire erro alto.

Os dois filtros (hora e TBD) nasceram de OLHAR A SAIDA em 15/09/2026, nao de
teoria: a primeira versao trouxe jogos de anteontem como "por vir" e chaves
"TBD x TBD" que nao dao para analisar.
"""
from __future__ import annotations

import json
import tempfile
import unittest
try:
    from mlcore.tests._repo import E_O_DOTA as _E_O_DOTA
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import E_O_DOTA as _E_O_DOTA
if _E_O_DOTA:
    raise unittest.SkipTest("so no LoL: importa uma ferramenta de tools/ que so existe no LoL")
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.coletar_agenda import (
    FOLGA_HORAS,
    _extrair_eventos,
    _gravar_atomico,
    _para_local,
    _quantos_ja_tem,
    normalizar,
)

AGORA = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)


def _ev(*, quando=None, a="Gen.G", b="T1", liga="LCK", bo=3, estado="unstarted"):
    return {
        "startTime": (quando or (AGORA + timedelta(hours=2))).isoformat().replace("+00:00", "Z"),
        "state": estado,
        "league": {"name": liga},
        "match": {"teams": [{"name": a}, {"name": b}],
                  "strategy": {"type": "bestOf", "count": bo}},
    }


def _norm(eventos, **k):
    k.setdefault("agora", AGORA)
    return normalizar(eventos, **k)


class TestOFormatoQueOAppJaSabeLer(unittest.TestCase):
    """A saida tem de casar com a de `parse_schedule_text`, campo por campo."""

    def test_devolve_os_mesmos_campos_do_parser_de_texto(self):
        j = _norm([_ev()])["jogos"][0]
        for campo in ("date", "time", "league", "teamA_raw", "teamB_raw"):
            self.assertIn(campo, j)

    def test_data_e_hora_no_formato_da_tela(self):
        j = _norm([_ev()])["jogos"][0]
        self.assertRegex(j["date"], r"^\d{2}/\d{2}$")
        self.assertRegex(j["time"], r"^\d{2}:\d{2}$")

    def test_traz_o_BO_REAL_de_cada_jogo(self):
        """Hoje o app poe um unico BO padrao em todos os jogos importados."""
        r = _norm([_ev(bo=5), _ev(bo=3), _ev(bo=1)])
        self.assertEqual([j["bo"] for j in r["jogos"]], [5, 3, 1])

    def test_formato_de_serie_desconhecido_NAO_vira_3_chutado(self):
        """PRINCIPIO IV: sem o dado, o campo fica 0 e quem le decide."""
        j = _norm([_ev(bo=7)])["jogos"][0]
        self.assertEqual(j["bo"], 0)

    def test_sem_strategy_tambem_nao_chuta(self):
        e = _ev()
        e["match"].pop("strategy")
        self.assertEqual(_norm([e])["jogos"][0]["bo"], 0)


class TestOFusoENoHorarioDele(unittest.TestCase):
    """A API responde em UTC; ele ve horario de Brasilia (UTC-3).

    Deixar em UTC criaria erro de 3 horas numa tela usada para apostar.
    """

    def test_converte_de_UTC_para_o_horario_de_brasilia(self):
        j = _norm([_ev(quando=datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc))])["jogos"][0]
        self.assertEqual(j["time"], "11:00")

    def test_a_virada_do_dia_tambem_e_convertida(self):
        """02:00Z de dia 16 e 23:00 do dia 15 para ele."""
        j = _norm([_ev(quando=datetime(2026, 9, 16, 2, 0, tzinfo=timezone.utc))],
                  agora=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc))["jogos"][0]
        self.assertEqual(j["time"], "23:00")
        self.assertEqual(j["date"], "15/09")

    def test_guarda_o_inicio_em_UTC_para_conferencia(self):
        self.assertTrue(_norm([_ev()])["jogos"][0]["inicio_utc"])

    def test_data_ilegivel_nao_estoura(self):
        e = _ev()
        e["startTime"] = "nao e data"
        self.assertEqual(_norm([e])["jogos"], [])
        self.assertIsNone(_para_local("nao e data"))


class TestJogoQueJaPassouNaoEAgenda(unittest.TestCase):
    """Defeito visto na saida real: jogos de 12/09 e 13/09 vinham como "por vir".

    A fonte nem sempre fecha o `state`, entao confiar so nele nao basta.
    """

    def test_jogo_de_anteontem_NAO_entra(self):
        r = _norm([_ev(quando=AGORA - timedelta(days=2), estado="unstarted")])
        self.assertEqual(r["jogos"], [])
        self.assertEqual(r["descartes"]["passados"], 1)

    def test_jogo_marcado_como_encerrado_NAO_entra(self):
        r = _norm([_ev(estado="completed")])
        self.assertEqual(r["jogos"], [])
        self.assertEqual(r["descartes"]["ja_jogados"], 1)

    def test_jogo_em_andamento_CONTINUA_valendo(self):
        r = _norm([_ev(quando=AGORA - timedelta(hours=1), estado="inProgress")])
        self.assertEqual(len(r["jogos"]), 1)

    def test_a_folga_para_tras_e_respeitada(self):
        dentro = _norm([_ev(quando=AGORA - timedelta(hours=FOLGA_HORAS - 0.5))])
        fora = _norm([_ev(quando=AGORA - timedelta(hours=FOLGA_HORAS + 0.5))])
        self.assertEqual(len(dentro["jogos"]), 1)
        self.assertEqual(len(fora["jogos"]), 0)

    def test_com_incluir_jogados_nada_e_filtrado(self):
        r = _norm([_ev(quando=AGORA - timedelta(days=5), estado="completed")],
                  incluir_jogados=True)
        self.assertEqual(len(r["jogos"]), 1)


class TestChaveIndefinidaNaoDaParaAnalisar(unittest.TestCase):

    def test_TBD_dos_dois_lados_fica_de_fora(self):
        r = _norm([_ev(a="TBD", b="TBD")])
        self.assertEqual(r["jogos"], [])
        self.assertEqual(r["descartes"]["indefinidos"], 1)

    def test_TBD_de_UM_lado_tambem_fica_de_fora(self):
        """Sem os dois times nao ha confronto para analisar."""
        self.assertEqual(_norm([_ev(a="TBD", b="Gen.G")])["jogos"], [])

    def test_o_descarte_e_CONTADO_e_nao_sumido(self):
        r = _norm([_ev(a="TBD"), _ev(), _ev(estado="completed")])
        self.assertEqual(r["descartes"]["indefinidos"], 1)
        self.assertEqual(r["descartes"]["ja_jogados"], 1)
        self.assertEqual(len(r["jogos"]), 1)

    def test_da_para_pedir_os_indefinidos_de_proposito(self):
        r = _norm([_ev(a="TBD", b="TBD")], incluir_indefinidos=True)
        self.assertEqual(len(r["jogos"]), 1)

    def test_evento_sem_dois_times_e_contado(self):
        e = _ev()
        e["match"]["teams"] = [{"name": "Gen.G"}]
        r = _norm([e])
        self.assertEqual(r["descartes"]["sem_dois_times"], 1)


class TestSeAAPIMudarELOUD(unittest.TestCase):
    """O pior resultado possivel seria devolver [] calado.

    A API nao e documentada. Se um campo sumir, "nenhum jogo hoje" apareceria
    com toda a confianca do mundo num dia cheio de jogos.
    """

    def test_resposta_que_nao_e_objeto_estoura(self):
        with self.assertRaises(ValueError):
            _extrair_eventos([1, 2, 3])

    def test_sem_data_estoura(self):
        with self.assertRaises(ValueError):
            _extrair_eventos({"outra_coisa": {}})

    def test_sem_schedule_estoura(self):
        with self.assertRaises(ValueError):
            _extrair_eventos({"data": {}})

    def test_events_que_nao_e_lista_estoura(self):
        with self.assertRaises(ValueError):
            _extrair_eventos({"data": {"schedule": {"events": "nao e lista"}}})

    def test_a_mensagem_de_erro_APONTA_a_causa(self):
        try:
            _extrair_eventos({"data": {}})
        except ValueError as e:
            self.assertIn("API mudou", str(e))

    def test_formato_certo_passa(self):
        self.assertEqual(_extrair_eventos({"data": {"schedule": {"events": []}}}), [])

    def test_lixo_no_meio_da_lista_nao_derruba_o_resto(self):
        r = _norm([None, "texto", 42, _ev()])
        self.assertEqual(len(r["jogos"]), 1)


class TestNuncaTrocaAgendaBoaPorVazia(unittest.TestCase):
    """Agenda vazia num dia de jogo e pior que agenda velha.

    A velha ao menos se denuncia (a data esta la); a vazia parece um fato.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.alvo = Path(self.tmp.name) / "agenda.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_conta_quantos_jogos_ja_havia(self):
        _gravar_atomico(self.alvo, {"jogos": [{"x": 1}, {"x": 2}]})
        self.assertEqual(_quantos_ja_tem(self.alvo), 2)

    def test_arquivo_ausente_conta_zero(self):
        self.assertEqual(_quantos_ja_tem(self.alvo), 0)

    def test_arquivo_corrompido_conta_zero_em_vez_de_estourar(self):
        self.alvo.write_text("{isso nao e json", encoding="utf-8")
        self.assertEqual(_quantos_ja_tem(self.alvo), 0)

    def test_a_escrita_e_atomica(self):
        """Ler arquivo pela metade ja quebrou o app duas vezes neste projeto."""
        _gravar_atomico(self.alvo, {"jogos": [{"a": 1}]})
        self.assertEqual(json.loads(self.alvo.read_text(encoding="utf-8"))["jogos"],
                         [{"a": 1}])
        # nenhum .tmp sobrando na pasta
        self.assertEqual([p.name for p in self.alvo.parent.glob("*.tmp")], [])

    def test_escrita_sobre_arquivo_existente_substitui_inteiro(self):
        _gravar_atomico(self.alvo, {"jogos": [{"a": 1}, {"a": 2}]})
        _gravar_atomico(self.alvo, {"jogos": [{"b": 1}]})
        self.assertEqual(_quantos_ja_tem(self.alvo), 1)


class TestOsNomesDeTimeCASAMComOApp(unittest.TestCase):
    """De nada adianta coletar se o app nao reconhece os times.

    Medido em 15/09/2026 contra o CSV real: dos 77 nomes distintos que a fonte
    devolve, o resolvedor do app acerta 74% sozinho — inclusive traduzindo
    ("BILIBILI GAMING" -> "Bilibili Gaming"). O resto ele ensina uma vez.
    """

    def test_o_resolvedor_do_app_aceita_o_que_o_coletor_produz(self):
        from app_core.schedule_import_core import resolve_schedule_team
        times = ["Gen.G", "T1", "Bilibili Gaming"]
        for cru, esperado in (("Gen.G", "Gen.G"),
                              ("BILIBILI GAMING", "Bilibili Gaming"),
                              ("T1", "T1")):
            alvo, achou = resolve_schedule_team(cru, times, {})
            self.assertTrue(achou, cru)
            self.assertEqual(alvo, esperado)

    def test_nome_desconhecido_nao_e_forcado(self):
        from app_core.schedule_import_core import resolve_schedule_team
        _, achou = resolve_schedule_team("Time Que Nao Existe", ["Gen.G", "T1"], {})
        self.assertFalse(achou)


if __name__ == "__main__":
    unittest.main()
