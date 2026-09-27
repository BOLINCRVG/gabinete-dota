# -*- coding: utf-8 -*-
"""A REGUA honesta: invariantes que, se quebrarem, devolvem a nota inflada.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ POR QUE ESTE ARQUIVO EXISTE: a nota oficial (ml_fusion_scorecard.py) pontua ║
║ a temporada com o artefato JA TREINADO e sem atualizar nada — medido em     ║
║ 09/09/2026: 7.175 dos 8.452 jogos (84,9%) estavam no treino. Resultado:     ║
║ nota 8,91 contra 7,06 medida honestamente. A regua nova so vale enquanto    ║
║ os invariantes abaixo valerem, e nenhum deles e obvio de olhar o codigo.    ║
║                                                                             ║
║ 1. CORTE ESTRITAMENTE ANTERIOR (point-in-time correctness). "O exemplo so   ║
║    pode ver o que existia no momento dele"; juncoes fora do ponto no tempo  ║
║    "inflam a AUC offline em 5 a 20%". A regra do backtest: "em cada passo,  ║
║    as decisoes sao tomadas usando apenas a informacao disponivel naquele    ║
║    momento".                                                                ║
║    https://www.systemoverflow.com/learn/ml-feature-stores/feature-store-architecture/point-in-time-correctness-and-time-travel
║    https://analystprep.com/study-notes/cfa-level-2/problems-in-backtesting/  ║
║    (ambas abertas em 09/09/2026)                                            ║
║    O TESTE QUE IMPORTA e o do MESMO DIA: `side="right"` no searchsorted     ║
║    incluiria o proprio jogo no historico que o preve. Passaria despercebido ║
║    em qualquer inspecao e inflaria tudo.                                    ║
║                                                                             ║
║ 2. ORIGEM ROLANTE, e nao um holdout so: "no arranjo de origem fixa, uma     ║
║    serie da apenas UMA previsao por passo", e os erros podem refletir       ║
║    "padroes observaveis apenas naquela regiao especifica".                   ║
║    Hewamalage/Bergmeir, https://arxiv.org/abs/2203.10716 (aberta 09/09/2026)║
║                                                                             ║
║ 3. NADA DE REIMPLEMENTAR A MATEMATICA. A cura para divergencia treino/      ║
║    servico e "definir a transformacao UMA VEZ". Por isso a regua CHAMA      ║
║    mlcore.fusion_core.hybrid_momentum_blend em vez de copiar a formula —    ║
║    e ha teste para isso, porque copiar seria facil e silencioso.            ║
║    https://chalk.ai/blog/what-is-a-feature-store  (aberta em 09/09/2026)    ║
║                                                                             ║
║ 4. O QUE ESTA REGUA NAO MEDE, de proposito: `season_form_blend` e           ║
║    `patch_meta_guard` leem o params_fitted.json (18/07/2026). Nas dobras    ║
║    anteriores a essa data isso e look-ahead — os parametros viram uma       ║
║    "decisao informada pelo teste", que e exatamente o vies que a validacao  ║
║    cruzada ANINHADA existe para eliminar: "os hiperparametros sao           ║
║    escolhidos por validacao interna inteiramente sobre o treino da dobra    ║
║    externa". Aninhar aqui exigiria refazer o fit de ~6h por dobra.          ║
║    https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
║    (aberta em 09/09/2026)                                                   ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import inspect
import unittest

import pandas as pd

from tools.nota_honesta import HistoricoPorTime, nota_de, pontuar_janela_prequencial


def _jogos(datas, time="A"):
    return pd.DataFrame({
        "team": [time] * len(datas),
        "date": pd.to_datetime(list(datas)),
        "win": [1] * len(datas),
    })


class TestCorteEstritamenteAnterior(unittest.TestCase):
    """O invariante mais caro de perder: um jogo nao pode entrar no proprio historico."""

    def setUp(self):
        self.h = HistoricoPorTime(_jogos(["2026-01-01", "2026-01-05", "2026-01-10"]))

    def test_nao_inclui_o_jogo_do_MESMO_DIA(self):
        # side="right" devolveria 2 aqui. E o bug que ninguem ve.
        self.assertEqual(len(self.h.antes_de("A", "2026-01-05")), 1)

    def test_inclui_tudo_que_e_anterior(self):
        self.assertEqual(len(self.h.antes_de("A", "2026-01-06")), 2)
        self.assertEqual(len(self.h.antes_de("A", "2026-02-01")), 3)

    def test_antes_do_primeiro_jogo_e_vazio(self):
        self.assertTrue(self.h.antes_de("A", "2025-12-31").empty)

    def test_time_desconhecido_devolve_vazio_e_nao_estoura(self):
        self.assertTrue(self.h.antes_de("NAO_EXISTE", "2026-01-05").empty)

    def test_o_historico_devolvido_e_so_daquele_time(self):
        h = HistoricoPorTime(pd.concat([_jogos(["2026-01-01"], "A"),
                                        _jogos(["2026-01-02"], "B")], ignore_index=True))
        d = h.antes_de("A", "2026-01-03")
        self.assertEqual(set(d["team"].unique()), {"A"})

    def test_ordem_de_entrada_bagunçada_nao_quebra_o_corte(self):
        h = HistoricoPorTime(_jogos(["2026-01-10", "2026-01-01", "2026-01-05"]))
        self.assertEqual(len(h.antes_de("A", "2026-01-06")), 2)


class TestAReguaNaoReimplementaAMatematica(unittest.TestCase):
    """Copiar a formula seria facil, silencioso, e mataria o proposito da regua."""

    def test_usa_a_funcao_de_producao_do_hybrid(self):
        src = inspect.getsource(pontuar_janela_prequencial)
        self.assertIn("from mlcore.fusion_core import hybrid_momentum_blend", src,
                      "a regua deve CHAMAR o hybrid de producao, nunca copiar a formula")

    def test_usa_o_update_de_producao(self):
        src = inspect.getsource(pontuar_janela_prequencial)
        self.assertIn("art.model.update(", src,
                      "a cadencia tem que usar o update do modelo de producao")

    def test_pontua_ANTES_de_atualizar(self):
        """Se inverter, o modelo ve o jogo que esta prevendo — nota inflada e
        indistinguivel de progresso real."""
        src = inspect.getsource(pontuar_janela_prequencial)
        i_p = src.index("art.model.p_blue(")
        i_u = src.index("art.model.update(")
        self.assertLess(i_p, i_u, "p_blue tem que vir ANTES de update no laco")


class TestANotaUsaAMesmaEscalaDaOficial(unittest.TestCase):
    """Se a escala divergir, comparar honesta x oficial vira comparar reguas
    diferentes — e a inflacao medida (8,91 -> 7,06) perde o sentido."""

    def test_mistura_45_55(self):
        src = inspect.getsource(nota_de)
        self.assertIn("0.45", src)
        self.assertIn("0.55", src)

    def test_segmento_vazio_devolve_nan_em_vez_de_inventar(self):
        import math
        self.assertTrue(math.isnan(nota_de([], [(0.6, 1)])))
        self.assertTrue(math.isnan(nota_de([(0.6, 1)], [])))

    def test_previsao_perfeita_bate_o_teto_e_pessima_o_piso(self):
        otimo = [(0.999, 1)] * 50
        pessimo = [(0.001, 1)] * 50
        self.assertGreater(nota_de(otimo, otimo), 9.0)
        self.assertLess(nota_de(pessimo, pessimo), 1.0)


if __name__ == "__main__":
    unittest.main()
