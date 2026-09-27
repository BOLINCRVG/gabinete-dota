# -*- coding: utf-8 -*-
"""Testes do cartao "Confianca ML".

O DEFEITO (relatado por ele em 15/09/2026): o cartao mostrava "C (Baixa)
35/100" TODO DIA, em qualquer confronto.

A conta era 50 de base menos 15 de "nao tenho o relatorio de validacao".
50 - 15 = 35. O relatorio era procurado so na memoria da sessao — que zera a
cada reinicio do site — e a tela nunca lia o arquivo em disco. Nunca achava.

E ler o arquivo NAO era o conserto: o que estava la era de 04/03/2026 (seis
meses) e com uma liga so. Com ele o cartao pularia para "A (Alta) 90". Numero
falso e confiante e pior que numero falso e humilde.
"""
from __future__ import annotations

import unittest

from app_core.confianca_ml_core import (
    LIMITE_DIAS,
    escolher_entre_ligas,
    montar_confianca,
    pontos_da_liga,
)

LCK = {"liga": "LCK", "n": 490, "nota": 7.15, "acuracia": 0.64, "ece": 0.058}
CBLOL = {"liga": "CBLOL", "n": 239, "nota": 7.1, "acuracia": 0.69, "ece": 0.124}
HC = {"liga": "HC", "n": 99, "nota": 9.61, "acuracia": 0.75, "ece": 0.105}


def _c(**k):
    base = {"linha_liga": LCK, "idade_dias": 9.0, "liga": "LCK",
            "ajuste_do_confronto": 0.0, "n_jogos_dos_times": float("nan")}
    base.update(k)
    return montar_confianca(**base)


class TestNaoMostraMaisONumeroDeNaoSei(unittest.TestCase):
    """O 35/100 fixo era ausencia virando veredito."""

    def test_sem_medicao_da_liga_NAO_da_nota(self):
        c = _c(linha_liga=None)
        self.assertFalse(c["disponivel"])
        self.assertEqual(c["grade"], "—")
        self.assertEqual(c["score_text"], "—")

    def test_e_diz_o_motivo(self):
        self.assertIn("medicao", _c(linha_liga=None)["motivo"])

    def test_medicao_velha_NAO_vale(self):
        c = _c(idade_dias=LIMITE_DIAS + 1)
        self.assertFalse(c["disponivel"])
        self.assertIn("velha", c["motivo"])

    def test_medicao_no_limite_ainda_vale(self):
        self.assertTrue(_c(idade_dias=LIMITE_DIAS)["disponivel"])

    def test_medicao_incompleta_NAO_vira_meia_nota(self):
        for ruim in ({"n": 400}, {"ece": 0.05}, {"ece": "x", "n": 400}):
            self.assertFalse(pontos_da_liga(ruim)["usavel"], ruim)

    def test_nunca_devolve_35_por_falta_de_dado(self):
        """O numero exato do defeito. Se voltar, foi por outro caminho."""
        c = _c(linha_liga=None)
        self.assertNotIn("35", str(c.get("score_text")))


class TestACalibracaoManda(unittest.TestCase):
    """"Confianca" e sobre o % ser honesto, nao sobre haver edge."""

    def test_liga_com_percentual_honesto_pontua_alto(self):
        self.assertEqual(_c(linha_liga=LCK)["grade"], "A (Alta)")

    def test_liga_com_percentual_DESONESTO_cai_para_C(self):
        """CBLOL, ECE 0,124: o EV calculado ali nao e real."""
        c = _c(linha_liga=CBLOL, liga="CBLOL")
        self.assertEqual(c["grade"], "C (Baixa)")
        self.assertIn("calibracao ruim", c["porque"])

    def test_previsibilidade_alta_NAO_salva_calibracao_ruim(self):
        """HC tem nota 9,61 e mesmo assim nao e confiavel."""
        c = _c(linha_liga=HC, liga="HC")
        self.assertEqual(c["grade"], "C (Baixa)")

    def test_amostra_pequena_e_dita_na_explicacao(self):
        self.assertIn("poucos jogos", _c(linha_liga=HC, liga="HC")["porque"])

    def test_a_calibracao_pesa_mais_que_a_previsibilidade(self):
        honesta_chata = {"n": 490, "nota": 6.6, "ece": 0.02}
        desonesta_previsivel = {"n": 490, "nota": 10.0, "ece": 0.15}
        self.assertGreater(pontos_da_liga(honesta_chata)["pontos"],
                           pontos_da_liga(desonesta_previsivel)["pontos"])


class TestOConfrontoEntraNaConta(unittest.TestCase):
    """A escolha dele: o cartao junta a liga COM o que houve neste confronto."""

    def test_ajuste_do_confronto_move_a_nota(self):
        alto = _c(ajuste_do_confronto=+15.0)["score"]
        baixo = _c(ajuste_do_confronto=-15.0)["score"]
        self.assertGreater(alto, baixo)

    def test_confronto_ruim_pode_derrubar_liga_boa(self):
        """Escalacao sem cobertura e guarda disparando valem mais que a liga."""
        self.assertEqual(_c(ajuste_do_confronto=-25.0)["grade"], "C (Baixa)")

    def test_o_resumo_do_confronto_aparece_na_explicacao(self):
        c = _c(resumo_do_confronto="C treinado | A-real ativo")
        self.assertIn("C treinado", c["porque"])

    def test_poucos_jogos_dos_times_descontam(self):
        muitos = _c(n_jogos_dos_times=40)["score"]
        poucos = _c(n_jogos_dos_times=5)["score"]
        self.assertGreater(muitos, poucos)

    def test_a_nota_fica_entre_0_e_100(self):
        for aj in (-500.0, 500.0):
            s = _c(ajuste_do_confronto=aj)["score"]
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 100.0)


class TestNaoDiscordaDaLinhaDeBaixo(unittest.TestCase):
    """A linha da liga e o cartao leem a MESMA medicao e os MESMOS limiares.

    Se divergissem, a tela diria "calibracao honesta" embaixo e "C (Baixa)"
    em cima, sobre a mesma liga, no mesmo instante.
    """

    def test_os_limiares_vem_do_modulo_da_linha_da_liga(self):
        import inspect

        from app_core import confianca_ml_core as m
        src = inspect.getsource(m)
        self.assertIn("from app_core.nota_liga_core import", src)
        self.assertIn("_ECE_BOA", src)

    def test_honesta_na_linha_nunca_e_calibracao_ruim_no_cartao(self):
        from app_core.nota_liga_core import _ECE_BOA, _calib_label
        linha = {"n": 400, "nota": 8.0, "ece": _ECE_BOA}
        self.assertEqual(_calib_label(_ECE_BOA), "honesta")
        self.assertIn("calibracao honesta", pontos_da_liga(linha)["notas"])

    def test_o_mesmo_prazo_de_frescor_dos_dois_lados(self):
        """A linha avisa "velha" acima de 14 dias; o cartao para de valer ali."""
        self.assertEqual(LIMITE_DIAS, 14.0)


class TestConfrontoEntreLigas(unittest.TestCase):
    """O cartao apagava em confronto entre ligas (relatado por ele, 15/09/2026).

    `_lg_hint` so vem preenchido quando as DUAS ligas sao a mesma. Num
    PRM x TCL ele vem vazio, `nota_for_league("")` devolve None, e o cartao
    dizia "nao medida" — justamente onde confianca mais importa, porque ligar
    duas escalas de rating e a conta mais dificil que o modelo faz.
    """

    def test_escolhe_a_liga_PIOR_e_nao_a_media(self):
        """Media deixaria a liga boa maquiar a ruim.

        Numeros reais de 15/09: PRM ece 0,070 nota 7,82 · TCL ece 0,060
        nota 10,0. A odd do confronto herda a desonestia do lado pior.
        """
        prm = {"liga": "PRM", "n": 306, "nota": 7.82, "ece": 0.070}
        tcl = {"liga": "TCL", "n": 204, "nota": 10.0, "ece": 0.060}
        e = escolher_entre_ligas(prm, tcl)
        self.assertEqual(e["linha"]["liga"], "PRM")
        self.assertTrue(e["entre_ligas"])

    def test_a_ordem_dos_times_nao_muda_a_escolha(self):
        prm = {"liga": "PRM", "n": 306, "nota": 7.82, "ece": 0.070}
        tcl = {"liga": "TCL", "n": 204, "nota": 10.0, "ece": 0.060}
        self.assertEqual(escolher_entre_ligas(prm, tcl)["linha"]["liga"],
                         escolher_entre_ligas(tcl, prm)["linha"]["liga"])

    def test_a_calibracao_ruim_vence_a_previsibilidade_alta(self):
        """Liga previsivel mas desonesta nao pode carregar o confronto."""
        honesta = {"liga": "A", "n": 400, "nota": 6.5, "ece": 0.03}
        desonesta = {"liga": "B", "n": 400, "nota": 10.0, "ece": 0.15}
        self.assertEqual(escolher_entre_ligas(honesta, desonesta)["linha"]["liga"], "B")

    def test_so_uma_liga_medida_usa_ela_e_NAO_marca_entre_ligas(self):
        """Sem as duas nao da para dizer qual e o lado fraco."""
        e = escolher_entre_ligas({"liga": "A", "n": 400, "nota": 8.0, "ece": 0.04}, None)
        self.assertEqual(e["linha"]["liga"], "A")
        self.assertFalse(e["entre_ligas"])

    def test_nenhuma_medida_devolve_nada(self):
        e = escolher_entre_ligas(None, None)
        self.assertIsNone(e["linha"])
        self.assertEqual(e["ligas"], [])

    def test_o_cartao_DIZ_que_e_entre_ligas(self):
        """Sem isso o numero pareceria a medicao de uma liga so."""
        c = montar_confianca(
            linha_liga={"liga": "PRM", "n": 306, "nota": 7.82, "ece": 0.070},
            idade_dias=9.0, liga="PRM+TCL", entre_ligas=True)
        self.assertIn("entre ligas", c["porque"])
        self.assertIn("mais fraco", c["porque"])

    def test_as_duas_ligas_aparecem_na_fonte(self):
        c = montar_confianca(
            linha_liga={"liga": "PRM", "n": 306, "nota": 7.82, "ece": 0.070},
            idade_dias=9.0, liga="PRM+TCL", entre_ligas=True)
        self.assertIn("PRM", c["source"])
        self.assertIn("TCL", c["source"])

    def test_confronto_de_UMA_liga_nao_ganha_o_aviso(self):
        c = montar_confianca(linha_liga=LCK, idade_dias=9.0, liga="LCK")
        self.assertNotIn("entre ligas", c["porque"])

    def test_nenhuma_variavel_atropela_a_funcao_de_escapar_HTML(self):
        """ERRO MEU, 15/09/2026: chamei uma variavel local de `_esc`.

        `_esc` ja e `html.escape`, importada no topo do arquivo. O dicionario
        atropelou a funcao e a Visao Geral inteira caiu 300 linhas abaixo, com
        "'dict' object is not callable" — longe do ponto onde eu errei.
        A blindagem por aba segurou o site e o teste de tela acusou; mas so o
        teste que diz ONDE quebrou tornou a caçada rapida.
        """
        import re
        import inspect

        from app_core import visao_tab_core as m
        src = inspect.getsource(m)
        # a unica linha com `_esc =` legitima e o import do topo
        atribuicoes = re.findall(r"^\s*_esc\s*=", src, re.M)
        self.assertEqual(atribuicoes, [],
                         "alguem atribuiu a `_esc`, que e html.escape")

    def test_a_tela_trata_o_caso_sem_liga_compartilhada(self):
        import inspect

        from app_core import visao_tab_core as m
        src = inspect.getsource(m.render_visao_tab)
        self.assertIn("escolher_entre_ligas", src)
        self.assertIn("_lgA", src)
        self.assertIn("_lgB", src)


class TestATelaUsaOModuloNovo(unittest.TestCase):

    def test_a_visao_geral_chama_montar_confianca(self):
        import inspect

        from app_core import visao_tab_core as m
        src = inspect.getsource(m.render_visao_tab)
        self.assertIn("montar_confianca", src)

    def test_a_visao_geral_nao_usa_mais_o_grade_do_relatorio_velho(self):
        import inspect

        from app_core import visao_tab_core as m
        src = inspect.getsource(m.render_visao_tab)
        self.assertNotIn('_conf_summary.get("grade"', src)


if __name__ == "__main__":
    unittest.main()
