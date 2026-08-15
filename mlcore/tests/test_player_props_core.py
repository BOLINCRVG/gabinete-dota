# -*- coding: utf-8 -*-
"""Motor de props de jogador: projeção por componentes + filtro de atropelo.

O ponto central (pedido dele 22/07): em Gen.G × Pain, boa parte das linhas de
jogador bate só porque é massacre. Para o resumo poder ESCONDER essas, a
projeção é feita em componentes separados e o quanto veio do desequilíbrio vira
um número (stomp_share) — não uma opinião.
"""
from __future__ import annotations

import math
import unittest

from app_core.player_props_core import (
    GAP_SENSITIVITY,
    STOMP_SHARE_LIMIAR,
    build_prop_row,
    divergencia_do_mercado,
    edge_vs_odd,
    fair_odd,
    prob_over,
    project_player_stat,
    remover_vig,
    separar_por_atropelo,
)


class NoVigTests(unittest.TestCase):
    """Tirar a margem da casa (padrão do mercado — pesquisa 22/07)."""

    def test_mercado_equilibrado_com_margem(self):
        # 1.90/1.90: implícitas somam 1.0526 -> vig ~5.26%, justa 50/50
        r = remover_vig(1.90, 1.90)
        self.assertAlmostEqual(r["p_casa_over"], 0.5, places=9)
        self.assertAlmostEqual(r["p_casa_under"], 0.5, places=9)
        self.assertAlmostEqual(r["vig_pct"], 5.263, places=2)

    def test_probabilidades_sem_vig_somam_um(self):
        r = remover_vig(1.50, 2.40)
        self.assertAlmostEqual(r["p_casa_over"] + r["p_casa_under"], 1.0, places=9)

    def test_favorito_tem_probabilidade_maior(self):
        r = remover_vig(1.40, 3.00)
        self.assertGreater(r["p_casa_over"], r["p_casa_under"])

    def test_sem_os_dois_lados_nao_da_p_tirar_vig(self):
        self.assertEqual(remover_vig(1.90, None), {})
        self.assertEqual(remover_vig(1.0, 1.90), {})
        self.assertEqual(remover_vig("x", "y"), {})


class DivergenciaTests(unittest.TestCase):
    def test_modelo_mais_otimista_da_positivo(self):
        self.assertAlmostEqual(divergencia_do_mercado(0.60, 0.50), 10.0, places=9)

    def test_modelo_mais_pessimista_da_negativo(self):
        self.assertLess(divergencia_do_mercado(0.45, 0.50), 0.0)

    def test_entrada_invalida_vira_nan(self):
        self.assertTrue(math.isnan(divergencia_do_mercado(None, 0.5)))


class ProjecaoTests(unittest.TestCase):
    def test_sem_ajuste_nenhum_devolve_a_taxa(self):
        r = project_player_stat(base_rate=3.0, stat="kills")
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["proj"], 3.0)
        self.assertAlmostEqual(r["stomp_share"], 0.0)

    def test_oponente_que_conde_mais_sobe_a_projecao(self):
        r = project_player_stat(base_rate=3.0, stat="kills", opp_concede_ratio=1.2)
        self.assertAlmostEqual(r["proj"], 3.6)

    def test_jogo_mais_longo_sobe_tudo(self):
        r = project_player_stat(base_rate=3.0, stat="kills", duration_ratio=1.1)
        self.assertAlmostEqual(r["proj"], 3.3)

    def test_time_mais_forte_mata_mais_e_morre_menos(self):
        k = project_player_stat(base_rate=3.0, stat="kills", mu_gap=+1.0)
        d = project_player_stat(base_rate=3.0, stat="deaths", mu_gap=+1.0)
        self.assertGreater(k["proj"], 3.0)   # mata mais
        self.assertLess(d["proj"], 3.0)      # morre menos

    def test_time_mais_fraco_e_o_espelho(self):
        k = project_player_stat(base_rate=3.0, stat="kills", mu_gap=-1.0)
        d = project_player_stat(base_rate=3.0, stat="deaths", mu_gap=-1.0)
        self.assertLess(k["proj"], 3.0)
        self.assertGreater(d["proj"], 3.0)

    def test_gap_extremo_nao_inverte_nem_explode(self):
        for gap in (-50.0, +50.0):
            for stat in GAP_SENSITIVITY:
                r = project_player_stat(base_rate=3.0, stat=stat, mu_gap=gap)
                self.assertGreaterEqual(r["proj"], 0.0)
                self.assertLess(r["proj"], 3.0 * 2.6)

    def test_entrada_invalida_nao_inventa_numero(self):
        self.assertFalse(project_player_stat(base_rate="lixo", stat="kills")["ok"])
        self.assertFalse(project_player_stat(base_rate=3.0, stat="pentakills")["ok"])
        self.assertFalse(project_player_stat(base_rate=-1.0, stat="kills")["ok"])


class StompShareTests(unittest.TestCase):
    """O coração do filtro: quanto da projeção veio do massacre."""

    def test_sem_gap_a_parcela_de_atropelo_e_zero(self):
        r = project_player_stat(base_rate=3.0, stat="kills", mu_gap=0.0, opp_concede_ratio=1.3)
        self.assertAlmostEqual(r["stomp_share"], 0.0)
        self.assertFalse(r["movido_por_atropelo"])

    def test_confronto_desequilibrado_marca_a_linha(self):
        r = project_player_stat(base_rate=3.0, stat="kills", mu_gap=+3.0)
        self.assertGreater(r["stomp_share"], STOMP_SHARE_LIMIAR)
        self.assertTrue(r["movido_por_atropelo"])

    def test_proj_sem_gap_ignora_o_desequilibrio(self):
        r = project_player_stat(base_rate=3.0, stat="kills", opp_concede_ratio=1.2, mu_gap=+2.0)
        self.assertAlmostEqual(r["proj_sem_gap"], 3.6)  # só jogador × oponente
        self.assertNotAlmostEqual(r["proj"], r["proj_sem_gap"])

    def test_gap_maior_aumenta_a_parcela(self):
        p = project_player_stat(base_rate=3.0, stat="kills", mu_gap=+0.5)["stomp_share"]
        g = project_player_stat(base_rate=3.0, stat="kills", mu_gap=+2.0)["stomp_share"]
        self.assertLess(p, g)


class ProbabilidadeEOddTests(unittest.TestCase):
    def test_linha_na_projecao_fica_perto_de_meio(self):
        # Poisson(3.0), over 2.5 => P(X>=3) — acima de 50% por ser discreta
        p = prob_over(3.0, 2.5)
        self.assertGreater(p, 0.35)
        self.assertLess(p, 0.65)

    def test_linha_mais_alta_da_probabilidade_menor(self):
        self.assertLess(prob_over(3.0, 5.5), prob_over(3.0, 2.5))

    def test_odd_justa_e_o_inverso(self):
        self.assertAlmostEqual(fair_odd(0.5), 2.0)
        self.assertAlmostEqual(fair_odd(0.25), 4.0)
        self.assertTrue(math.isnan(fair_odd(0.0)))

    def test_edge_positivo_quando_a_casa_paga_acima_do_justo(self):
        # p=0.60 -> justa 1.667; casa pagando 2.00 => edge = +20%
        self.assertAlmostEqual(edge_vs_odd(0.60, 2.00), 20.0, places=6)
        self.assertLess(edge_vs_odd(0.60, 1.50), 0.0)

    def test_odd_invalida_vira_nan(self):
        self.assertTrue(math.isnan(edge_vs_odd(0.6, 1.0)))
        self.assertTrue(math.isnan(edge_vs_odd(0.6, "x")))


class LinhaDaTabelaTests(unittest.TestCase):
    def _row(self, **kw):
        base = dict(player="Chovy", role="MID", team="Gen.G", stat="kills",
                    line=4.5, base_rate=5.0)
        base.update(kw)
        return build_prop_row(**base)

    def test_traz_projecao_probabilidade_e_odd(self):
        r = self._row()
        self.assertTrue(r["ok"])
        for k in ("proj", "p_over", "p_under", "odd_justa_over", "stomp_share"):
            self.assertIn(k, r)

    def test_com_odd_da_casa_calcula_o_edge(self):
        r = self._row(odd_casa=2.0)
        self.assertIn("edge_pct", r)
        self.assertTrue(math.isfinite(r["edge_pct"]))

    def test_over_e_under_somam_um(self):
        r = self._row()
        self.assertAlmostEqual(r["p_over"] + r["p_under"], 1.0, places=9)

    def test_stat_invalido_nao_quebra_a_tabela(self):
        r = self._row(stat="mvp")
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "stat_desconhecido")


class SepararPorAtropeloTests(unittest.TestCase):
    """O resumo abre limpo; as movidas pelo atropelo ficam atrás do botão."""

    def test_separa_as_duas_listas(self):
        limpa = build_prop_row(player="A", role="MID", team="X", stat="kills",
                                line=3.5, base_rate=3.0, mu_gap=0.0)
        suja = build_prop_row(player="B", role="ADC", team="X", stat="kills",
                               line=3.5, base_rate=3.0, mu_gap=+3.0)
        ok, atropelo = separar_por_atropelo([limpa, suja])
        self.assertEqual([r["player"] for r in ok], ["A"])
        self.assertEqual([r["player"] for r in atropelo], ["B"])

    def test_ignora_linhas_invalidas(self):
        ruim = build_prop_row(player="C", role="TOP", team="X", stat="xyz",
                               line=1.5, base_rate=1.0)
        ok, atropelo = separar_por_atropelo([ruim])
        self.assertEqual(ok, [])
        self.assertEqual(atropelo, [])

    def test_lista_vazia(self):
        self.assertEqual(separar_por_atropelo([]), ([], []))
        self.assertEqual(separar_por_atropelo(None), ([], []))


if __name__ == "__main__":
    unittest.main()
