# -*- coding: utf-8 -*-
"""Testes da leitura da agenda coletada pelo robo.

O APP NAO VAI A REDE: quem coleta e o robo, o app so le o arquivo. Cada visita
ao site viraria uma ida ao alvo, e a hospedagem ja mostrou que nao e confiavel
para isso (o site caiu em 13/09/2026 por um problema dela).

O QUE ESTES TESTES PROTEGEM, em ordem de importancia:
  1. o formato entregue e o MESMO que a previa do app ja consome — se mudar,
     o botao novo para de conversar com o mapeamento de times que ja existe;
  2. ausencia nunca vira lista vazia calada — "nao tem jogo hoje" e "nao
     consegui coletar" sao coisas muito diferentes para quem vai apostar;
  3. o BO REAL de cada jogo vence o BO padrao da tela — um BO5 lido como BO3
     muda a odd da serie inteira.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app_core.agenda_coletada_core import (
    CAMPOS,
    bo_do_jogo,
    carregar_agenda_coletada,
)

UM_JOGO = {"date": "15/09", "time": "11:00", "league": "LCK",
           "teamA_raw": "Gen.G", "teamB_raw": "T1", "bo": 5,
           "estado": "unstarted", "inicio_utc": "2026-09-15T14:00:00Z"}


class _ComPasta(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _grava(self, conteudo) -> None:
        (self.raiz / "agenda_coletada.json").write_text(
            json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")


class TestOFormatoConversaComAPrevia(_ComPasta):
    """A previa do app consome a saida de `parse_schedule_text`."""

    def test_entrega_os_mesmos_campos_do_parser_de_texto(self):
        self._grava({"gerado_em": "", "jogos": [UM_JOGO]})
        j = carregar_agenda_coletada(self.raiz)["jogos"][0]
        for campo in CAMPOS:
            self.assertIn(campo, j)

    def test_os_campos_sao_os_que_o_parser_de_texto_produz(self):
        """Se um dos dois lados mudar, o botao novo para de conversar."""
        from app_core.schedule_import_core import parse_schedule_text
        texto = "LCK\n15/09\n11:00\nGen.G\nT1\n"
        do_texto = parse_schedule_text(texto)
        self.assertTrue(do_texto, "o parser de texto nao achou o jogo de exemplo")
        for campo in CAMPOS:
            self.assertIn(campo, do_texto[0])

    def test_os_valores_chegam_limpos(self):
        self._grava({"jogos": [dict(UM_JOGO, teamA_raw="  Gen.G  ")]})
        self.assertEqual(carregar_agenda_coletada(self.raiz)["jogos"][0]["teamA_raw"],
                         "Gen.G")


class TestAusenciaNaoViraSilencio(_ComPasta):
    """PRINCIPIO IV: sem dado, o motivo aparece — nunca uma lista vazia muda."""

    def test_arquivo_ausente_diz_que_o_robo_nao_coletou(self):
        d = carregar_agenda_coletada(self.raiz)
        self.assertEqual(d["jogos"], [])
        self.assertIn("ainda nao coletou", d["motivo"])

    def test_arquivo_corrompido_diz_que_esta_ilegivel(self):
        (self.raiz / "agenda_coletada.json").write_text("{nao e json", encoding="utf-8")
        d = carregar_agenda_coletada(self.raiz)
        self.assertEqual(d["jogos"], [])
        self.assertIn("ilegivel", d["motivo"])

    def test_arquivo_sem_a_lista_diz_isso(self):
        self._grava({"gerado_em": "2026-09-15T10:00:00"})
        self.assertIn("sem a lista", carregar_agenda_coletada(self.raiz)["motivo"])

    def test_coleta_vazia_diz_que_veio_vazia(self):
        self._grava({"gerado_em": "2026-09-15T10:00:00", "jogos": []})
        self.assertIn("nao trouxe jogo", carregar_agenda_coletada(self.raiz)["motivo"])

    def test_com_jogos_o_motivo_fica_vazio(self):
        self._grava({"jogos": [UM_JOGO]})
        self.assertEqual(carregar_agenda_coletada(self.raiz)["motivo"], "")

    def test_jogo_sem_um_dos_times_e_descartado(self):
        self._grava({"jogos": [dict(UM_JOGO, teamB_raw=""), UM_JOGO]})
        self.assertEqual(len(carregar_agenda_coletada(self.raiz)["jogos"]), 1)

    def test_lixo_no_meio_da_lista_nao_derruba_o_resto(self):
        self._grava({"jogos": [None, "texto", 42, UM_JOGO]})
        self.assertEqual(len(carregar_agenda_coletada(self.raiz)["jogos"]), 1)


class TestAIdadeVemJunto(_ComPasta):
    """Agenda velha repete os jogos de ontem com toda a confianca do mundo."""

    def test_calcula_a_idade_da_coleta(self):
        quando = datetime.now(timezone.utc) - timedelta(hours=5)
        self._grava({"gerado_em": quando.isoformat(), "jogos": [UM_JOGO]})
        self.assertAlmostEqual(carregar_agenda_coletada(self.raiz)["idade_horas"],
                               5.0, delta=0.2)

    def test_sem_carimbo_a_idade_e_None_e_nao_zero(self):
        """Zero diria "coletada agora", que e afirmacao sem base."""
        self._grava({"jogos": [UM_JOGO]})
        self.assertIsNone(carregar_agenda_coletada(self.raiz)["idade_horas"])

    def test_carimbo_ilegivel_tambem_e_None(self):
        self._grava({"gerado_em": "ontem de manha", "jogos": [UM_JOGO]})
        self.assertIsNone(carregar_agenda_coletada(self.raiz)["idade_horas"])

    def test_a_tela_avisa_quando_a_coleta_esta_velha(self):
        """O aviso MUDOU DE LUGAR em 16/09/2026, e quase se perdeu.

        Ele vivia no painel de importar jogos. Quando esse painel encolheu (a
        pedido dele: "isso aqui nao precisa mais, ate porque voce ja pega"), o
        aviso sumiu junto -- e foi ESTE TESTE que acusou. Sem ele, uma coleta
        travada ofereceria os jogos de ontem com toda a confianca do mundo, e
        o sintoma apareceria como "esse jogo nao existe".

        O aviso passou para o SELETOR DE CONFRONTO, que e onde os jogos
        coletados aparecem agora.
        """
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("idade_horas", src)
        self.assertIn("coleta do calendario tem", src.lower())


class TestOBODeCadaJogoVence(unittest.TestCase):
    """Antes o app carimbava um unico BO em todos os jogos importados."""

    def test_o_bo_do_jogo_vence_o_padrao_da_tela(self):
        self.assertEqual(bo_do_jogo({"bo": 5}, 3), 5)
        self.assertEqual(bo_do_jogo({"bo": 1}, 3), 1)

    def test_sem_bo_no_jogo_cai_no_padrao(self):
        self.assertEqual(bo_do_jogo({}, 5), 5)
        self.assertEqual(bo_do_jogo({"bo": 0}, 1), 1)

    def test_bo_invalido_no_jogo_cai_no_padrao(self):
        for ruim in ({"bo": 7}, {"bo": "tres"}, {"bo": None}):
            self.assertEqual(bo_do_jogo(ruim, 3), 3)

    def test_padrao_invalido_vira_3(self):
        self.assertEqual(bo_do_jogo({}, "nao e numero"), 3)
        self.assertEqual(bo_do_jogo({}, 99), 3)

    def test_entrada_que_nao_e_dicionario_nao_estoura(self):
        self.assertEqual(bo_do_jogo(None, 3), 3)
        self.assertEqual(bo_do_jogo("texto", 5), 5)

    def test_a_tela_usa_o_bo_por_jogo(self):
        """O BO de CADA jogo vence o plano B.

        O segundo argumento deixou de ser o seletor "BO padrao" da tela (que
        saiu em 16/09/2026 junto com a importacao por texto) e passou a ser um
        3 fixo: o plano B agora vale so quando NINGUEM disse qual e o formato,
        em vez de um seletor global carimbando todos os jogos iguais.
        """
        import inspect

        from app_core import schedule_agenda_ui_core as m
        src = inspect.getsource(m.render_schedule_agenda_expander)
        self.assertIn("_bo_do_jogo(g, ", src)
        self.assertNotIn("default_bo", src)


class TestOAppNaoVaiARede(unittest.TestCase):
    """Se um dia alguem puser a busca aqui, cada visita vira uma ida ao alvo."""

    def test_o_modulo_de_leitura_nao_importa_requests(self):
        import inspect

        from app_core import agenda_coletada_core as m
        self.assertNotIn("requests", inspect.getsource(m))

    def test_a_tela_da_agenda_nao_importa_requests(self):
        import inspect

        from app_core import schedule_agenda_ui_core as m
        self.assertNotIn("import requests", inspect.getsource(m))


if __name__ == "__main__":
    unittest.main()
