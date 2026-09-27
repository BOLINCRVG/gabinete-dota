# -*- coding: utf-8 -*-
"""O preco justo da Pinnacle, e o quadro com o mercado inteiro dela.

ELE DISSE (23/09/2026) como usa a casa: a Pinnacle e a referencia do preco
justo. A odd dela tem margem dentro -- e nos esports a margem e grande
(medido: 4,8% no LPL, 14% no vencedor de kills, 16% no placar exato). O
metodo de tirar a margem e o da potencia (fontes em
app_core/preco_justo_casa_core.py).
"""
from __future__ import annotations

import unittest
from pathlib import Path

from app_core.preco_justo_casa_core import anexar_preco_justo, chave_do_mercado, sem_margem


class TestSemMargem(unittest.TestCase):

    def test_mercado_simetrico_fica_meio_a_meio(self):
        p = sem_margem([1.87, 1.87])
        self.assertAlmostEqual(p[0], 0.5, places=9)
        self.assertAlmostEqual(p[1], 0.5, places=9)

    def test_soma_um_sempre(self):
        for odds in ([1.44, 2.79], [1.13, 5.03], [5.14, 4.38, 2.41, 3.11], [1.2, 4.5]):
            self.assertAlmostEqual(sum(sem_margem(odds)), 1.0, places=9)

    def test_o_favorito_fica_com_MENOS_margem_que_na_divisao_simples(self):
        """E isso que a potencia faz de diferente: a margem pesa mais no
        azarao (vies favorito-azarao; Clarke et al. 2017)."""
        odds = [1.13, 5.03]
        p = sem_margem(odds)
        simples = [(1 / o) / sum(1 / x for x in odds) for o in odds]
        self.assertGreater(p[0], simples[0])
        self.assertLess(p[1], simples[1])

    def test_confere_com_a_conta_de_mao(self):
        """k tal que 0,8850^k + 0,1988^k = 1 -- conferido por fora."""
        p = sem_margem([1.13, 5.03])
        a, b = 1 / 1.13, 1 / 5.03
        # a solucao satisfaz p_i = (1/odd_i)^k para o MESMO k
        import math
        k0 = math.log(p[0]) / math.log(a)
        k1 = math.log(p[1]) / math.log(b)
        self.assertAlmostEqual(k0, k1, places=6)

    def test_sem_par_nao_inventa(self):
        self.assertIsNone(sem_margem([1.9]))
        self.assertIsNone(sem_margem([1.9, None]))
        self.assertIsNone(sem_margem([1.9, 1.0]))

    def test_casa_pagando_acima_do_justo_nao_vira_margem(self):
        self.assertIsNone(sem_margem([2.2, 2.2]))


def _l(**k):
    base = {"jogo": 1, "mercado": "Handicap - serie", "tipo": "spread", "periodo": 0}
    base.update(k)
    return base


class TestQuemEParDeQuem(unittest.TestCase):

    def test_handicap_as_DUAS_versoes_nao_se_misturam(self):
        """Visto ao vivo: mandante -1,5 / visitante +1,5 E mandante +1,5 /
        visitante -1,5, no mesmo jogo. Valor absoluto juntaria as quatro."""
        ls = [_l(lado="home", pontos=-1.5, odd=2.44), _l(lado="away", pontos=1.5, odd=1.5618),
              _l(lado="home", pontos=1.5, odd=1.13), _l(lado="away", pontos=-1.5, odd=5.03)]
        out = anexar_preco_justo(ls)
        self.assertFalse(any(x["sem_par"] for x in out))
        self.assertAlmostEqual(out[0]["p_casa_justa"] + out[1]["p_casa_justa"], 1.0)
        self.assertAlmostEqual(out[2]["p_casa_justa"] + out[3]["p_casa_justa"], 1.0)

    def test_total_do_time_separa_mandante_e_visitante(self):
        base = dict(mercado="Total do time - Kills - mapa 1", tipo="team_total", periodo=1)
        ls = [_l(**base, time="home", lado="over", pontos=28.5, odd=1.86),
              _l(**base, time="home", lado="under", pontos=28.5, odd=1.88),
              _l(**base, time="away", lado="over", pontos=28.5, odd=1.89),
              _l(**base, time="away", lado="under", pontos=28.5, odd=1.85)]
        self.assertNotEqual(chave_do_mercado(ls[0]), chave_do_mercado(ls[2]))
        out = anexar_preco_justo(ls)
        self.assertFalse(any(x["sem_par"] for x in out))

    def test_lado_repetido_fica_sem_preco_em_vez_de_errado(self):
        base = dict(mercado="Total do time - Kills - mapa 1", tipo="team_total", periodo=1)
        ls = [_l(**base, lado="over", pontos=28.5, odd=1.86),
              _l(**base, lado="over", pontos=28.5, odd=1.89)]
        out = anexar_preco_justo(ls)
        self.assertTrue(all(x["sem_par"] and x["p_casa_justa"] is None for x in out))

    def test_placar_exato_usa_todos_os_resultados(self):
        placar = ["A 2, B 0", "A 2, B 1", "A 1, B 2", "A 0, B 2"]
        ls = [_l(mercado="Correct Score", tipo="moneyline", lado=n, odd=o)
              for n, o in zip(placar, [2.41, 3.11, 4.38, 5.14])]
        out = anexar_preco_justo(ls)
        self.assertAlmostEqual(sum(x["p_casa_justa"] for x in out), 1.0)

    def test_nao_mexe_na_lista_original(self):
        ls = [_l(lado="home", pontos=-1.5, odd=2.44), _l(lado="away", pontos=1.5, odd=1.56)]
        anexar_preco_justo(ls)
        self.assertNotIn("p_casa_justa", ls[0])


class TestOQuadroInteiro(unittest.TestCase):
    """A aba mostra TODA linha da casa, com o justo dela e o do modelo."""

    def _odds(self):
        linhas = [
            {"jogo": 1, "tipo": "total", "periodo": 1, "lado": "over", "pontos": 27.5,
             "odd": 1.72, "unidade": "Kills", "mercado": "Total - Kills - mapa 1"},
            {"jogo": 1, "tipo": "total", "periodo": 1, "lado": "under", "pontos": 27.5,
             "odd": 2.10, "unidade": "Kills", "mercado": "Total - Kills - mapa 1"},
            {"jogo": 1, "tipo": "moneyline", "periodo": 1, "lado": "A", "odd": 1.66,
             "unidade": "1st Roshan", "mercado": "(Map 1) 1st Roshan"},
            {"jogo": 1, "tipo": "moneyline", "periodo": 1, "lado": "B", "odd": 2.13,
             "unidade": "1st Roshan", "mercado": "(Map 1) 1st Roshan"},
        ]
        return {"jogos": [{"id": 1, "casa": "A", "fora": "B", "liga": "PGL"}],
                "linhas": linhas}

    def _montar(self, totais):
        from app_core.oportunidades_core import montar_oportunidades
        return montar_oportunidades(self._odds(), lambda a, b, m: totais,
                                    prob_over=lambda t, x: 0.70)

    def test_toda_linha_da_casa_aparece(self):
        r = self._montar({"kills": object()})
        self.assertEqual(len(r["mercado_completo"]), 4)
        for x in r["mercado_completo"]:
            self.assertIsNotNone(x["odd_casa_justa"], x)

    def test_o_exemplo_DELE_fair_1_40_contra_1_72(self):
        """"se la no resumo ele aparece com uma fair de 1.40 e na pinnacle ele
        esta com 1,72 de odd, e ele que eu vou girar": o modelo da 70% ao
        over 27,5, a casa paga 1,72."""
        r = self._montar({"kills": object()})
        over = [x for x in r["mercado_completo"] if x["lado"] == "over"][0]
        self.assertAlmostEqual(over["odd_modelo"], 1 / 0.70)
        self.assertAlmostEqual(over["vantagem_pct"], (0.70 * 1.72 - 1) * 100)
        self.assertGreater(over["diferenca_pp"], 0)
        achado = [a for a in r["todas"] if a["lado"] == "over"][0]
        self.assertIsNotNone(achado["odd_casa_justa"])

    def test_linha_sem_modelo_mostra_o_MOTIVO_e_nenhum_numero_do_modelo(self):
        r = self._montar({"kills": object()})
        rosh = [x for x in r["mercado_completo"] if "Roshan" in x["mercado"]]
        for x in rosh:
            self.assertIsNone(x["odd_modelo"])
            self.assertIsNone(x["vantagem_pct"])
            self.assertTrue(x["motivo"])

    def test_o_motivo_e_do_PROPRIO_mercado(self):
        """Visto na tela do Dota: par ou impar aparecia com o motivo do
        handicap ("o motor calcula o total, nao a diferenca")."""
        from app_core.oportunidades_core import _motivo_no_quadro
        par = {"tipo": "moneyline", "unidade": "Kills", "lado": "Odd",
               "mercado": "(Map 1) Total Kills Odd/Even"}
        self.assertIn("par ou impar", _motivo_no_quadro(par))
        rosh = {"tipo": "moneyline", "unidade": "1st Roshan", "lado": "A",
                "mercado": "(Map 1) 1st Roshan"}
        self.assertIn("quem faz primeiro", _motivo_no_quadro(rosh))
        hc = {"tipo": "spread", "unidade": "Kills", "lado": "home", "pontos": -5.5,
              "mercado": "Handicap - Kills - mapa 1"}
        self.assertIn("diferenca", _motivo_no_quadro(hc))

    def test_total_que_o_modelo_sabe_mas_nao_conseguiu_diz_ISSO(self):
        r = self._montar(None)
        over = [x for x in r["mercado_completo"] if x["lado"] == "over"][0]
        self.assertIn("nao conseguiu este confronto", over["motivo"])

    def test_a_tabela_da_tela_tem_as_colunas(self):
        from app_core.oportunidades_tab_core import tabela_do_mercado_completo
        r = self._montar({"kills": object()})
        t = tabela_do_mercado_completo(r["mercado_completo"])
        self.assertEqual(len(t), 4)
        for c in ("casa paga", "pinnacle sem margem", "modelo", "vantagem %",
                  "por que sem modelo"):
            self.assertIn(c, t[0])
        over = [x for x in t if x["aposta"] == "acima de 27,5"][0]
        self.assertEqual(over["casa paga"], 1.72)


class TestOJogoAntesDoMercado(unittest.TestCase):
    """24/09/2026: com cinco jogos na tela, a tabela de um mercado misturava
    os jogos. Agora o jogo e escolhido antes, e o aberto vem primeiro."""

    def test_o_confronto_aberto_vem_primeiro(self):
        from app_core.oportunidades_tab_core import ordem_dos_jogos
        por_jogo = {
            1: [{"vantagem_pct": 40.0}],
            2: [{"vantagem_pct": 5.0, "com_vitoria": True}],
            3: [{"vantagem_pct": 12.0}],
        }
        self.assertEqual(ordem_dos_jogos(por_jogo), [2, 1, 3])

    def test_a_tabela_linha_por_linha_SAIU(self):
        """25/09/2026: "acho que nao precisa mais disso" -- a tabela no
        formato da casa mostra o mesmo em qualquer linha."""
        fonte = (Path(__file__).resolve().parents[2] / "app_core" /
                 "oportunidades_tab_core.py").read_text(encoding="utf-8")
        self.assertNotIn("Qual mercado voce quer comparar", fonte)
        self.assertIn("_tabelas_da_casa(st, [x for x in justas if x.get(\"jogo\") == jogo_cmp]", fonte)


class TestALinhaJusta(unittest.TestCase):
    """Pedido dele (25/09/2026): "caso o seu calculo de 25.5, eu quero ver como
    e a odd que o nosso modelo da o 25.5 e como a pinnacle aponta"."""

    def _odds(self, linhas_pin):
        linhas = []
        for pts, o_over, o_under in linhas_pin:
            for lado, odd in (("over", o_over), ("under", o_under)):
                linhas.append({"jogo": 1, "tipo": "total", "periodo": 1, "lado": lado,
                               "pontos": pts, "odd": odd, "unidade": "Kills",
                               "mercado": "Total - Kills - mapa 1"})
        return {"jogos": [{"id": 1, "casa": "FlyQuest", "fora": "Shopify", "liga": "LCS"}],
                "linhas": linhas}

    def _montar(self, odds):
        import math
        from app_core.oportunidades_core import montar_linhas_justas
        # modelo: mediana em 24,9 kills
        prob = lambda t, L: 1.0 / (1.0 + math.exp((L - 24.9) / 3.0))
        return montar_linhas_justas(odds, lambda a, b, m: {"kills": object()}, prob_over=prob)

    def test_linha_do_modelo_da_pinnacle_e_a_distancia(self):
        r = self._montar(self._odds([(24.5, 1.68, 2.08), (25.5, 1.87, 1.87), (26.5, 2.08, 1.68)]))
        self.assertEqual(len(r), 1)
        x = r[0]
        self.assertEqual(x["linha_modelo"], 24.5)
        self.assertEqual(x["linha_pinnacle"], 25.5)
        self.assertEqual(x["diferenca"], -1.0)
        self.assertTrue(x["pinnacle_oferece"])
        self.assertEqual((x["casa_a"], x["casa_b"]), (1.68, 2.08))
        self.assertAlmostEqual(x["vantagem_b"], ((1 - x["p_modelo_a"]) * 2.08 - 1) * 100)

    def test_sem_a_linha_do_modelo_compara_na_linha_DELA(self):
        r = self._montar(self._odds([(26.5, 1.90, 1.90), (27.5, 2.10, 1.72)]))
        x = r[0]
        self.assertFalse(x["pinnacle_oferece"])
        self.assertEqual(x["linha_pinnacle"], 26.5)
        self.assertIsNotNone(x["vantagem_pin_b"])
        self.assertEqual((x["pin_casa_a"], x["pin_casa_b"]), (1.90, 1.90))

    def test_a_meia_linha_mais_perto_de_meio(self):
        from app_core.oportunidades_core import linha_mais_perto_de_meio
        L, p = linha_mais_perto_de_meio(lambda L: 1 - L / 10.0, [x + 0.5 for x in range(10)])
        self.assertEqual(L, 4.5)


if __name__ == "__main__":
    unittest.main()
