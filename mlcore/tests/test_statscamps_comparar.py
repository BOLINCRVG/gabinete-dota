# -*- coding: utf-8 -*-
"""Testes da comparacao do StatsCamps ja vir com os times do confronto.

Pedido do dono em 13/09/2026: a secao "Comparar times" comecava vazia, e ele
remarcava na mao os MESMOS dois times que ja estavam escolhidos no Confronto
logo acima (no print dele: FlyQuest e Sentinels).

O risco da facilidade e ela virar teimosia: seguir o confronto nao pode
apagar uma escolha feita a mao. Metade destes testes existe para isso.
"""
from __future__ import annotations

import unittest

from app_core.statscamps_tab_core import escolha_inicial_da_comparacao as escolher

OPCOES = ["FlyQuest", "Sentinels", "Cloud9", "Team Liquid"]


def _e(**k):
    base = {"opcoes": OPCOES, "time_a": None, "time_b": None,
            "ja_escolhido": None, "semeado_antes": None}
    base.update(k)
    return escolher(**base)


class TestOPedidoDoDono(unittest.TestCase):

    def test_vem_com_os_dois_times_do_confronto(self):
        self.assertEqual(_e(time_a="FlyQuest", time_b="Sentinels"),
                         ["FlyQuest", "Sentinels"])

    def test_mantem_a_ordem_do_confronto(self):
        """Time 1 primeiro. A tabela lado a lado fica na ordem que ele ve."""
        self.assertEqual(_e(time_a="Sentinels", time_b="FlyQuest"),
                         ["Sentinels", "FlyQuest"])

    def test_tolera_caixa_e_espaco(self):
        self.assertEqual(_e(time_a="  flyquest ", time_b="SENTINELS"),
                         ["FlyQuest", "Sentinels"])

    def test_um_time_so_ainda_e_marcado(self):
        """Metade e melhor que nada; a tela so pede 2+ para desenhar."""
        self.assertEqual(_e(time_a="Cloud9", time_b=None), ["Cloud9"])


class TestNaoAtropelaOQueODonoFez(unittest.TestCase):
    """A parte que importa mais: facilidade nao pode virar teimosia."""

    def test_escolha_feita_a_mao_nao_e_tocada(self):
        self.assertIsNone(_e(time_a="FlyQuest", time_b="Sentinels",
                             ja_escolhido=["Cloud9", "Team Liquid"],
                             semeado_antes=["FlyQuest", "Sentinels"]))

    def test_esvaziar_de_proposito_e_respeitado(self):
        """Ele tirou os dois; a tela nao pode recolocar no desenho seguinte."""
        self.assertIsNone(_e(time_a="FlyQuest", time_b="Sentinels",
                             ja_escolhido=[],
                             semeado_antes=["FlyQuest", "Sentinels"]))

    def test_acrescentar_um_terceiro_nao_e_desfeito(self):
        self.assertIsNone(_e(time_a="FlyQuest", time_b="Sentinels",
                             ja_escolhido=["FlyQuest", "Sentinels", "Cloud9"],
                             semeado_antes=["FlyQuest", "Sentinels"]))

    def test_nao_reescreve_quando_ja_esta_certo(self):
        """Devolver a mesma lista toda vez seria escrita a toa no estado."""
        self.assertIsNone(_e(time_a="FlyQuest", time_b="Sentinels",
                             ja_escolhido=["FlyQuest", "Sentinels"],
                             semeado_antes=["FlyQuest", "Sentinels"]))


class TestSegueOConfrontoEnquantoIntocada(unittest.TestCase):

    def test_trocar_o_confronto_arrasta_a_comparacao(self):
        self.assertEqual(_e(time_a="Cloud9", time_b="Team Liquid",
                            ja_escolhido=["FlyQuest", "Sentinels"],
                            semeado_antes=["FlyQuest", "Sentinels"]),
                         ["Cloud9", "Team Liquid"])

    def test_depois_de_mexido_para_de_seguir(self):
        """Uma vez que ele escolheu, o confronto nao manda mais."""
        self.assertIsNone(_e(time_a="Cloud9", time_b="Team Liquid",
                             ja_escolhido=["Cloud9"],
                             semeado_antes=["FlyQuest", "Sentinels"]))


class TestNaoForcaOImpossivel(unittest.TestCase):
    """PRINCIPIO IV: o que nao existe no recorte nao vira selecao."""

    def test_time_fora_do_recorte_nao_e_marcado(self):
        """Filtro de liga/ano diferente: o Streamlit recusaria o valor."""
        self.assertIsNone(_e(time_a="Gen.G", time_b="T1"))

    def test_so_o_que_existe_entra(self):
        self.assertEqual(_e(time_a="FlyQuest", time_b="T1"), ["FlyQuest"])

    def test_sem_opcoes_nao_estoura(self):
        self.assertIsNone(_e(opcoes=[], time_a="FlyQuest", time_b="Sentinels"))

    def test_confronto_vazio_nao_marca_nada(self):
        self.assertIsNone(_e(time_a="", time_b=None))

    def test_o_mesmo_time_dos_dois_lados_nao_duplica(self):
        self.assertEqual(_e(time_a="FlyQuest", time_b="FlyQuest"), ["FlyQuest"])


class TestNaTelaDeVerdade(unittest.TestCase):
    """Nao basta a regra: a tela tem de chamar ela ANTES de desenhar o widget.

    O Streamlit so aceita escrever na chave de um widget antes de ele existir
    no desenho. Se a semeadura for parar depois da chamada do multiselect, a
    regra fica certa e a tela continua vazia.
    """

    def test_a_semeadura_acontece_antes_do_widget(self):
        import inspect

        from app_core import statscamps_tab_core as m
        src = inspect.getsource(m.render_statscamps_tab)
        i_semeia = src.index("escolha_inicial_da_comparacao(")
        i_widget = src.index('key="sc_compare_sel"')
        self.assertLess(i_semeia, i_widget,
                        "a semeadura tem de vir antes do multiselect")

    def test_a_tela_le_as_chaves_do_confronto(self):
        import inspect

        from app_core import statscamps_tab_core as m
        src = inspect.getsource(m.render_statscamps_tab)
        self.assertIn('"teamA"', src)
        self.assertIn('"teamB"', src)


if __name__ == "__main__":
    unittest.main()
