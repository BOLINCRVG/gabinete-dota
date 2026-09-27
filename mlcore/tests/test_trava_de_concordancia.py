# -*- coding: utf-8 -*-
"""A trava que cala a forma justo quando ela discorda do modelo.

O ACHADO (21/09/2026). Em mlcore/fusion_core.py, o hybrid_momentum_blend decide
quanto peso dar a forma recente assim:

    agree = media(1 - |forma - modelo| / 0.5)
    peso  = teto * fator_amostra * (0.35 + 0.65 * agree)

O peso CRESCE quando a forma concorda com o modelo e ENCOLHE quando discorda.
Medindo fora da amostra (tools/sobrou_sinal.py), o sinal que a forma ainda tem
a dar depois deste estagio esta concentrado exatamente onde ela discorda:

    mapa 1, forma CONCORDA ... +0,004 [-0,004, +0,012]  nada
    mapa 1, forma DISCORDA ... +0,037 [+0,018, +0,058]  real

Entrou um botao, `agree_damp`, para MEDIR isso. E o que estes testes protegem
e o que ainda nao foi medido: **o padrao tem de ser exatamente o de hoje**.
Um botao novo que ja muda o preco servido seria mudanca disfarcada de medicao.
"""
from __future__ import annotations

import math
import unittest

import pandas as pd

from mlcore.fusion_core import hybrid_momentum_blend
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol


def _quadro(n, vitorias):
    return pd.DataFrame({
        "win": [1] * vitorias + [0] * (n - vitorias),
        "firstblood": [1] * vitorias + [0] * (n - vitorias),
        "firsttower": [1] * vitorias + [0] * (n - vitorias),
        "firstdragon": [1] * vitorias + [0] * (n - vitorias),
        "dragons_for": [3] * n, "dragons_against": [2] * n,
        "barons_for": [1] * n, "barons_against": [0] * n,
    })


class TestOPadraoNaoMudouNada(unittest.TestCase):
    """Enquanto nao houver medicao, o preco servido tem de ser o mesmo."""

    def test_sem_passar_o_botao_o_peso_e_a_formula_antiga(self):
        for pm, a, b in ((0.70, (20, 4), (20, 16)), (0.55, (14, 9), (14, 6)),
                         (0.40, (30, 15), (30, 15)), (0.85, (25, 22), (25, 3))):
            r = hybrid_momentum_blend(p_model=pm, df_team_a=_quadro(*a),
                                      df_team_b=_quadro(*b))
            self.assertTrue(r["applied"], "o estagio nem rodou neste caso")
            esperado = (r["w_hybrid"] / (0.35 + 0.65 * r["agree_to_model"]))
            r0 = hybrid_momentum_blend(p_model=pm, df_team_a=_quadro(*a),
                                       df_team_b=_quadro(*b), agree_damp=0.0)
            self.assertAlmostEqual(r0["w_hybrid"], esperado, places=9,
                                   msg="agree_damp=0 devia ser o peso sem a trava")

    def test_agree_damp_1_e_IDENTICO_ao_padrao(self):
        for pm, a, b in ((0.70, (20, 4), (20, 16)), (0.30, (18, 12), (18, 5))):
            padrao = hybrid_momentum_blend(p_model=pm, df_team_a=_quadro(*a),
                                           df_team_b=_quadro(*b))
            um = hybrid_momentum_blend(p_model=pm, df_team_a=_quadro(*a),
                                       df_team_b=_quadro(*b), agree_damp=1.0)
            self.assertEqual(padrao["p_blended"], um["p_blended"])
            self.assertEqual(padrao["w_hybrid"], um["w_hybrid"])

    def test_o_caminho_SERVIDO_usa_1_0_quando_ninguem_configurou(self):
        import inspect

        from app_core import ml_pre_live_pipeline_core as m
        fonte = inspect.getsource(m)
        self.assertIn('ss.get("ml_hybrid_agree_damp", 1.0)', fonte,
                      "o caminho servido precisa cair em 1.0 por omissao")

    @so_no_lol("so vale para o LoL")
    def test_a_REGUA_tambem_cai_em_1_0(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        fonte = (raiz / "tools" / "nota_honesta.py").read_text(encoding="utf-8")
        self.assertIn('_ss.get("ml_hybrid_agree_damp", 1.0)', fonte)


class TestOBotaoFazOQueDiz(unittest.TestCase):

    def test_desligar_a_trava_da_MAIS_peso_a_forma(self):
        """E o ponto todo: hoje a discordancia encolhe o peso."""
        a, b, pm = _quadro(20, 4), _quadro(20, 16), 0.70   # forma discorda forte
        com = hybrid_momentum_blend(p_model=pm, df_team_a=a, df_team_b=b, agree_damp=1.0)
        sem = hybrid_momentum_blend(p_model=pm, df_team_a=a, df_team_b=b, agree_damp=0.0)
        self.assertGreater(sem["w_hybrid"], com["w_hybrid"])
        self.assertLess(sem["p_blended"], com["p_blended"],
                        "sem a trava, o preco devia ceder mais a forma")

    def test_quando_a_forma_CONCORDA_o_botao_quase_nao_muda_nada(self):
        """Se mudasse muito no caso que concorda, o botao nao estaria mexendo
        no que o achado apontou."""
        a, b, pm = _quadro(20, 16), _quadro(20, 4), 0.80   # forma concorda
        com = hybrid_momentum_blend(p_model=pm, df_team_a=a, df_team_b=b, agree_damp=1.0)
        sem = hybrid_momentum_blend(p_model=pm, df_team_a=a, df_team_b=b, agree_damp=0.0)
        movimento_concordando = abs(sem["p_blended"] - com["p_blended"])
        a2, b2 = _quadro(20, 4), _quadro(20, 16)
        com2 = hybrid_momentum_blend(p_model=0.80, df_team_a=a2, df_team_b=b2, agree_damp=1.0)
        sem2 = hybrid_momentum_blend(p_model=0.80, df_team_a=a2, df_team_b=b2, agree_damp=0.0)
        movimento_discordando = abs(sem2["p_blended"] - com2["p_blended"])
        self.assertGreater(movimento_discordando, movimento_concordando)

    def test_valor_fora_da_faixa_nao_estoura_e_e_aparado(self):
        a, b = _quadro(20, 4), _quadro(20, 16)
        for v in (-5.0, 7.0, float("nan"), None, "oi"):
            r = hybrid_momentum_blend(p_model=0.7, df_team_a=a, df_team_b=b, agree_damp=v)
            self.assertTrue(math.isfinite(float(r["p_blended"])))
            self.assertGreaterEqual(r["agree_damp"], 0.0)
            self.assertLessEqual(r["agree_damp"], 1.0)

    def test_o_traco_conta_o_que_foi_usado(self):
        """Sem isso, uma odd estranha na tela nao teria como ser explicada."""
        r = hybrid_momentum_blend(p_model=0.7, df_team_a=_quadro(20, 4),
                                  df_team_b=_quadro(20, 16), agree_damp=0.25)
        self.assertAlmostEqual(r["agree_damp"], 0.25)
        self.assertIn("fator_concordancia", r)


class TestAVarreduraNaoReimplementaOEstagio(unittest.TestCase):
    """PRINCIPIO V: medir chamando a producao, nunca copiando-a.

    A varredura reaplica so o ultimo estagio para nao gastar 8 horas de
    maquina (uma dobra da regua leva 12m22s). O risco dessa economia e
    recriar a conta do amortecimento interregional aqui -- duas versoes dela
    divergem. Por isso a regua GRAVA os pesos ja amortecidos.
    """

    def setUp(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        self.fonte = (raiz / "tools" / "varrer_agree.py").read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_chama_a_funcao_de_producao(self):
        self.assertIn("from mlcore.fusion_core import hybrid_momentum_blend", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_usa_os_pesos_GRAVADOS_e_nao_recalcula_o_interregional(self):
        self.assertIn('max_weight=float(x["w"])', self.fonte)
        self.assertNotIn("ml_interregional_hybrid_max_weight_mult", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_a_nota_sai_da_funcao_oficial(self):
        self.assertIn("nota_de", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_a_regua_grava_o_entrado_do_ultimo_estagio(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        fonte = (raiz / "tools" / "nota_honesta.py").read_text(encoding="utf-8")
        i_grava = fonte.index("antes_hybrid.append({")
        i_aplica = fonte.index("_hy = hybrid_momentum_blend(")
        self.assertLess(i_grava, i_aplica,
                        "o preco gravado tem de ser o que ENTRA no estagio")


if __name__ == "__main__":
    unittest.main()
