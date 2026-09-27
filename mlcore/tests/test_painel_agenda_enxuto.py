# -*- coding: utf-8 -*-
"""O painel "Jogos (agenda)" depois que a coleta automatica tomou o lugar dele.

ELE DISSE, olhando o painel (16/09/2026): "isso aqui nao precisa mais eu acho,
ate porque voce ja pega".

Tinha razao na maior parte. O robo baixa a agenda de 3 em 3 horas e os jogos
entram DIRETO no seletor de confronto, marcados "do calendario". Colar texto,
escolher um BO padrao e clicar em "Trazer os jogos coletados" viraram passos
que nao levam a lugar nenhum.

MAS NAO EM TUDO, e por isso este arquivo existe: duas coisas daquele painel
nao tem substituto em lugar nenhum do app. Tirar as duas junto com o resto
deixaria ele sem jeito de apagar jogos velhos -- que foi exatamente o que eu
mandei ele fazer horas antes -- e sem jeito de analisar um jogo que o
calendario da Riot nao traz.
"""
from __future__ import annotations

import inspect
import unittest

from app_core import schedule_agenda_ui_core as m

FONTE = inspect.getsource(m.render_schedule_agenda_expander)


def _sem_comentarios(src: str) -> str:
    """So o codigo. Os comentarios CITAM o que foi removido, de proposito."""
    linhas = []
    for ln in src.splitlines():
        corte = ln.split("#", 1)[0]
        linhas.append(corte)
    return "\n".join(linhas)


CODIGO = _sem_comentarios(FONTE)


class TestOQueSAIU(unittest.TestCase):

    def test_a_caixa_de_colar_jogos_saiu(self):
        self.assertNotIn("st.text_area", CODIGO)

    def test_o_BO_PADRAO_para_todos_saiu(self):
        """Era um unico formato carimbado em TODOS os jogos importados.

        Mesma familia do defeito que ele relatou no seletor: um BO5 tratado
        como BO3 muda a odd da serie inteira.
        """
        self.assertNotIn("plays_schedule_default_bo", CODIGO)
        self.assertNotIn("default_bo", CODIGO)

    def test_o_botao_de_trazer_os_coletados_saiu(self):
        """Virou redundante: os coletados ja entram sozinhos no seletor."""
        self.assertNotIn("plays_schedule_auto_btn", CODIGO)

    def test_processar_e_limpar_TEXTO_sairam(self):
        self.assertNotIn("plays_schedule_parse_btn", CODIGO)
        self.assertNotIn("plays_schedule_clear_btn", CODIGO)


class TestOQueFICOU(unittest.TestCase):
    """Estes dois nao existem em nenhum outro canto do app."""

    def test_LIMPAR_AGENDA_continua(self):
        """Unico jeito de apagar os jogos salvos antigos dele."""
        self.assertIn("plays_schedule_clear_list", CODIGO)
        self.assertIn("clear_schedule_games", CODIGO)

    def test_ADICIONAR_JOGO_MANUAL_continua(self):
        """Para o que o calendario da Riot nao traz -- chave que so definiu
        tarde, liga fora da cobertura."""
        self.assertIn("plays_manual_add", CODIGO)
        self.assertIn("plays_manual_teamA", CODIGO)

    def test_o_mapeamento_de_time_desconhecido_continua(self):
        """O jogo manual precisa dele: nome digitado por ele raramente bate
        letra por letra com o CSV."""
        self.assertIn("resolve_schedule_team", CODIGO)
        self.assertIn("save_schedule_aliases", CODIGO)

    def test_a_tela_DIZ_que_nao_precisa_importar(self):
        """Painel que encolhe sem explicar parece painel quebrado."""
        self.assertIn("do calendario", FONTE)


class TestOBODoJogoManualAgoraEUSADO(unittest.TestCase):
    """DEFEITO IRMAO do que ele relatou, achado ao enxugar o painel.

    O seletor de BO do jogo manual era DESENHADO E NUNCA LIDO -- havia um
    comentario no codigo dizendo isso em voz alta. O BO vinha do "BO padrao"
    la de cima, igual para todos os jogos. Ele podia escolher BO5 e o jogo
    entrava como BO3.

    Um BO5 analisado como BO3 muda a odd da serie inteira. Nao e detalhe.
    """

    def test_o_valor_escolhido_e_lido(self):
        self.assertIn("m_bo = st.selectbox", CODIGO)

    def test_e_vai_junto_com_o_jogo(self):
        self.assertIn('"bo": int(m_bo)', CODIGO)

    def test_o_plano_B_deixou_de_ser_um_seletor_global(self):
        """Cair em 3 so quando NINGUEM disse qual e -- e nao porque um
        seletor de tela carimbou o mesmo formato em todo mundo."""
        self.assertIn("_bo_do_jogo(g, 3)", CODIGO)


class TestONomeDoPainelNaoMENTE(unittest.TestCase):

    def test_nao_promete_mais_colar(self):
        """Chamava-se "colar / adicionar" e nao ha mais onde colar."""
        self.assertNotIn("colar / adicionar", FONTE)


if __name__ == "__main__":
    unittest.main()
