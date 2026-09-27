# -*- coding: utf-8 -*-
"""Quao velho e o dado do time, na hora de escolher o time.

ELE PEDIU (21/09/2026), logo depois de conferirmos 16 apelidos um a um contra
a Leaguepedia: "vao ter jogos que os dados vao ser bem velhos, e seria legal
avisar em algum lugar na mesma escolha mesmo ao lado assim 'dado mais recente
- (e aqui a data)' e uma cor diferente para ficar facil identificar".

OS DOIS CASOS REAIS DAQUELA CONFERENCIA, que explicam o desenho:

  - "Bilibili Junior" da Pinnacle E o "Bilibili Gaming Junior" do CSV -- a
    Leaguepedia confirma. Mas o ultimo jogo dele la e de 08/10/2025. Apelido
    CERTO, analise FRACA, e as duas coisas eram indistinguiveis na lista.

  - "T1 Academy" tinha dois candidatos, e o que jogou MAIS RECENTE ("T1
    Esports Academy Rookies", 20/09) era o ERRADO: quem disputa a WSCI e o
    "T1 Esports Academy". Por isso a data e INFORMACAO, nunca criterio
    automatico -- ela ajuda a decidir, nao decide, e nada aqui reordena ou
    esconde candidato por causa dela.

E O PERIGO PROPRIO DESTA MUDANCA: o rotulo deixa de ser o nome do time. Se o
rotulo for gravado como apelido, o arquivo guarda "Team Liquid  -  dado mais
recente: 20/09/2026 (ontem)" e nada casa depois. Metade destes testes existe
so para isso.
"""
from __future__ import annotations

import unittest
from datetime import date

from app_core import frescor_do_time_core as f
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol


class TestOTextoQueElePediu(unittest.TestCase):

    HOJE = date(2026, 9, 21)

    def test_diz_DADO_MAIS_RECENTE_com_a_data(self):
        t = f.texto_do_frescor(date(2026, 9, 20), self.HOJE)
        self.assertIn("dado mais recente", t)
        self.assertIn("20/09/2026", t)

    def test_ontem_e_hoje_saem_por_extenso(self):
        self.assertIn("ontem", f.texto_do_frescor(date(2026, 9, 20), self.HOJE))
        self.assertIn("hoje", f.texto_do_frescor(date(2026, 9, 21), self.HOJE))

    def test_meses_e_mais_de_um_ano(self):
        self.assertIn("meses", f.texto_do_frescor(date(2026, 6, 1), self.HOJE))
        self.assertIn("mais de um ano", f.texto_do_frescor(date(2025, 1, 1), self.HOJE))

    def test_o_caso_BILIBILI_aparece_como_quase_um_ano(self):
        """O time que motivou o pedido."""
        t = f.texto_do_frescor(date(2025, 10, 8), self.HOJE)
        self.assertIn("08/10/2025", t)
        self.assertIn("meses", t)

    def test_sem_jogo_nenhum_NAO_vira_data(self):
        """PRINCIPIO IV: ausencia nao vira numero."""
        self.assertEqual(f.texto_do_frescor(None, self.HOJE), "sem jogo no CSV")


class TestACorSeparaOQueImporta(unittest.TestCase):

    HOJE = date(2026, 9, 21)

    def _s(self, ultimo):
        return f.simbolo(f.dias_parado(ultimo, self.HOJE))

    def test_tres_cores_diferentes_mais_a_ausencia(self):
        cores = {self._s(date(2026, 9, 20)),      # ontem
                 self._s(date(2026, 8, 1)),       # ~7 semanas
                 self._s(date(2025, 10, 8)),      # quase um ano
                 f.simbolo(None)}
        self.assertEqual(len(cores), 4, "cores repetidas nao separam nada")

    def test_time_em_atividade_e_verde(self):
        self.assertEqual(self._s(date(2026, 9, 20)), f.VERDE)

    def test_bilibili_e_vermelho(self):
        self.assertEqual(self._s(date(2025, 10, 8)), f.VERMELHO)

    def test_sem_dado_e_branco_e_nao_vermelho(self):
        """"Nao sei" e diferente de "sei que esta velho"."""
        self.assertEqual(f.simbolo(None), f.BRANCO)
        self.assertNotEqual(f.simbolo(None), f.VERMELHO)

    def test_a_legenda_explica_todas_as_cores(self):
        leg = f.legenda()
        for cor in (f.VERDE, f.AMARELO, f.VERMELHO, f.BRANCO):
            self.assertIn(cor, leg)


class TestORotuloNAOEhONome(unittest.TestCase):
    """O perigo proprio desta mudanca."""

    HOJE = date(2026, 9, 21)
    ULTIMOS = {"Team Liquid": date(2026, 9, 20),
               "Bilibili Gaming Junior": date(2025, 10, 8)}

    def test_o_rotulo_carrega_nome_cor_e_data(self):
        r = f.rotulo("Team Liquid", self.ULTIMOS, self.HOJE)
        self.assertIn("Team Liquid", r)
        self.assertIn("20/09/2026", r)
        self.assertTrue(r.startswith(f.VERDE))

    def test_da_para_VOLTAR_do_rotulo_ao_nome(self):
        rot, volta = f.opcoes(["Team Liquid", "Bilibili Gaming Junior"],
                              self.ULTIMOS, self.HOJE)
        self.assertEqual(len(rot), 2)
        for r in rot:
            self.assertIn(volta[r], ("Team Liquid", "Bilibili Gaming Junior"))
        self.assertEqual(volta[rot[0]], "Team Liquid")

    def test_o_nome_de_volta_e_EXATO_sem_a_data_grudada(self):
        rot, volta = f.opcoes(["Team Liquid"], self.ULTIMOS, self.HOJE)
        self.assertEqual(volta[rot[0]], "Team Liquid")
        self.assertNotIn("dado mais recente", volta[rot[0]])

    def test_time_sem_data_tambem_volta_inteiro(self):
        rot, volta = f.opcoes(["Time Novo"], {}, self.HOJE)
        self.assertEqual(volta[rot[0]], "Time Novo")

    def test_nome_vazio_nao_vira_opcao(self):
        rot, volta = f.opcoes(["", None, "Team Liquid"], self.ULTIMOS, self.HOJE)
        self.assertEqual(len(rot), 1)

    def test_rotulo_repetido_nao_engole_o_primeiro(self):
        rot, volta = f.opcoes(["A", "A"], {}, self.HOJE)
        self.assertEqual(len(volta), len(rot))


class TestADataNaoDECIDE(unittest.TestCase):
    """O caso T1: o candidato mais recente era o errado.

    A data informa; quem escolhe e ele. Nada aqui pode reordenar nem sumir
    com candidato por causa do frescor.
    """

    def test_a_ordem_dos_candidatos_e_preservada(self):
        ultimos = {"T1 Esports Academy": date(2026, 9, 15),
                   "T1 Esports Academy Rookies": date(2026, 9, 20)}
        entrada = ["T1 Esports Academy", "T1 Esports Academy Rookies"]
        rot, volta = f.opcoes(entrada, ultimos, date(2026, 9, 21))
        self.assertEqual([volta[r] for r in rot], entrada)

    def test_nenhum_candidato_some_por_ser_velho(self):
        rot, _ = f.opcoes(["Velho", "Novo"],
                          {"Velho": date(2020, 1, 1), "Novo": date(2026, 9, 20)},
                          date(2026, 9, 21))
        self.assertEqual(len(rot), 2)


class TestOAvisoDepoisDaEscolha(unittest.TestCase):

    HOJE = date(2026, 9, 21)

    def test_time_em_dia_NAO_gera_aviso(self):
        self.assertEqual(
            f.avisar_se_velho("X", {"X": date(2026, 9, 20)}, self.HOJE), "")

    def test_time_velho_avisa_que_a_analise_sai_fraca(self):
        av = f.avisar_se_velho("Bilibili Gaming Junior",
                               {"Bilibili Gaming Junior": date(2025, 10, 8)}, self.HOJE)
        self.assertIn("08/10/2025", av)
        self.assertIn("fraca", av)

    def test_time_sem_dado_diz_que_nao_ha_o_que_analisar(self):
        av = f.avisar_se_velho("Fantasma", {}, self.HOJE)
        self.assertIn("nao tem nenhum jogo", av)


class TestLerOQuadroDeJogos(unittest.TestCase):

    def _quadro(self):
        import pandas as pd
        return pd.DataFrame({
            "team": ["A", "A", "B", "", None, "C"],
            "date": [pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-20"),
                     pd.Timestamp("2025-10-08"), pd.Timestamp("2026-01-01"),
                     pd.Timestamp("2026-01-01"), None],
        })

    def test_pega_o_jogo_MAIS_RECENTE_de_cada_time(self):
        u = f.ultimo_jogo_por_time(self._quadro())
        self.assertEqual(u["A"], date(2026, 9, 20))
        self.assertEqual(u["B"], date(2025, 10, 8))

    def test_time_sem_data_valida_simplesmente_NAO_entra(self):
        self.assertNotIn("C", f.ultimo_jogo_por_time(self._quadro()))

    def test_nome_vazio_nao_entra(self):
        u = f.ultimo_jogo_por_time(self._quadro())
        self.assertNotIn("", u)

    def test_quadro_ausente_ou_torto_devolve_vazio_e_nao_estoura(self):
        import pandas as pd
        self.assertEqual(f.ultimo_jogo_por_time(None), {})
        self.assertEqual(f.ultimo_jogo_por_time(pd.DataFrame({"x": [1]})), {})

    def test_aceita_data_em_texto(self):
        import pandas as pd
        u = f.ultimo_jogo_por_time(pd.DataFrame(
            {"team": ["A"], "date": ["2026-09-20T18:00:00"]}))
        self.assertEqual(u["A"], date(2026, 9, 20))


class TestAsTelasUsamAMesmaPeca(unittest.TestCase):
    """Tres telas escolhem time. Se cada uma montar o rotulo do seu jeito,
    elas divergem -- e uma delas vai acabar gravando o rotulo como apelido."""

    def _fonte(self, caminho):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        return (raiz / caminho).read_text(encoding="utf-8")

    def test_a_tela_de_jogos_usa(self):
        self.assertIn("frescor_do_time_core",
                      self._fonte("app_core/matchup_selection_core.py"))

    def test_a_tela_de_apelidos_usa(self):
        self.assertIn("frescor_do_time_core",
                      self._fonte("app_core/schedule_alias_ui_core.py"))

    def test_a_tela_da_agenda_usa(self):
        self.assertIn("frescor_do_time_core",
                      self._fonte("app_core/schedule_agenda_ui_core.py"))

    def test_a_aba_oportunidades_usa(self):
        self.assertIn("frescor_do_time_core",
                      self._fonte("app_core/oportunidades_tab_core.py"))

    def test_todas_gravam_o_NOME_e_nao_o_rotulo(self):
        """A volta pelo dicionario tem de estar em todas."""
        for caminho in ("app_core/matchup_selection_core.py",
                        "app_core/schedule_alias_ui_core.py",
                        "app_core/schedule_agenda_ui_core.py",
                        "app_core/oportunidades_tab_core.py"):
            fonte = self._fonte(caminho)
            self.assertTrue(
                ".get(" in fonte and "opcoes" in fonte,
                "%s precisa voltar do rotulo para o nome" % caminho)

    @so_no_lol("usa o plays_app.py, o app do LoL")
    def test_o_app_monta_a_tabela_de_datas_uma_vez_so(self):
        fonte = self._fonte("plays_app.py")
        self.assertIn("_ultimo_jogo_por_time", fonte)
        self.assertIn('st.session_state["_ultimo_jogo_por_time"]', fonte)


if __name__ == "__main__":
    unittest.main()
