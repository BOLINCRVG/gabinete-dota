# -*- coding: utf-8 -*-
"""Testes da nota de habilidade por confronto de todos contra todos.

O teste que mais vale aqui e o de PARIDADE com a biblioteca openskill. Ele
so roda onde a biblioteca esta disponivel (ela NAO entra no requirements do
projeto de proposito), mas foi rodado de verdade e pegou um defeito que
nenhum dos outros teria pego: a condicao de empate estava `q == i` em vez de
`posicao(q) == posicao(i)`. Sem empate as duas coincidem e tudo batia; com
empate o erro chegava a 2,1 de mu.

Os demais testes travam as propriedades que precisam valer mesmo sem a
biblioteca por perto.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mlcore.players_skill_ffa import (
    BETA,
    MU0,
    SIGMA0,
    _posicoes_por_nota,
    atualizar_ffa,
    construir_notas,
    deltas_por_jogo,
)


def _tem_openskill():
    """A biblioteca de referencia, se instalada em qualquer lugar do sys.path."""
    try:
        from openskill.models import PlackettLuce  # noqa: F401
        return True
    except Exception:
        return False


class TestPosicoes(unittest.TestCase):

    def test_maior_nota_vira_melhor_posicao(self):
        self.assertEqual(_posicoes_por_nota([1.0, 3.0, 2.0]), [2, 0, 1])

    def test_empate_vira_a_MESMA_posicao(self):
        """Dois desempenhos iguais nao podem render atualizacoes diferentes."""
        self.assertEqual(_posicoes_por_nota([5.0, 5.0, 1.0]), [0, 0, 2])

    def test_todos_empatados(self):
        self.assertEqual(_posicoes_por_nota([2.0] * 4), [0, 0, 0, 0])


class TestAtualizacao(unittest.TestCase):

    def test_quem_fica_na_frente_SOBE_e_quem_fica_atras_desce(self):
        novos = atualizar_ffa([(MU0, SIGMA0)] * 3, [0, 1, 2])
        self.assertGreater(novos[0][0], MU0)
        self.assertLess(novos[2][0], MU0)

    def test_a_incerteza_sempre_encolhe(self):
        novos = atualizar_ffa([(MU0, SIGMA0)] * 5, [0, 1, 2, 3, 4])
        for _, sg in novos:
            self.assertLess(sg, SIGMA0)
            self.assertGreater(sg, 0.0)

    def test_empate_geral_move_TODO_MUNDO_igual_para_cima(self):
        """SURPRESA REGISTRADA: empate geral NAO deixa o rating parado.

        Eu esperava que 6 jogadores identicos, todos empatados, saissem com o
        rating intacto. Nao e o que acontece: todos SOBEM, e a mesma coisa. A
        conta explica — com todos na mesma posicao, cada um e comparado com o
        campo inteiro, ele proprio incluso, e o termo (1 - 1/n) sobra positivo.

        Isto NAO e defeito meu: conferido contra a biblioteca openskill, que
        devolve exatamente os mesmos numeros. O teste antigo afirmava a minha
        expectativa em vez do comportamento real, e era o teste que estava
        errado.

        Na pratica nao afeta nada: a nota de desempenho e continua, entao
        empate perfeito entre os 10 nao acontece.
        """
        novos = atualizar_ffa([(MU0, SIGMA0)] * 6, [0] * 6)
        primeiro = novos[0][0]
        self.assertGreater(primeiro, MU0)
        for mu, _ in novos:
            self.assertAlmostEqual(mu, primeiro, places=12)

    def test_empatados_recebem_EXATAMENTE_o_mesmo(self):
        """Foi aqui que o defeito da condicao de empate aparecia."""
        novos = atualizar_ffa([(MU0, SIGMA0)] * 4, [0, 0, 2, 2])
        self.assertAlmostEqual(novos[0][0], novos[1][0], places=12)
        self.assertAlmostEqual(novos[2][0], novos[3][0], places=12)
        self.assertGreater(novos[0][0], novos[2][0])

    def test_vencer_gente_FORTE_paga_mais_que_vencer_gente_fraca(self):
        """E a razao de ser desta camada: a media acumulada nao sabe disso."""
        contra_fortes = atualizar_ffa(
            [(MU0, SIGMA0), (40.0, SIGMA0), (40.0, SIGMA0)], [0, 1, 2])
        contra_fracos = atualizar_ffa(
            [(MU0, SIGMA0), (10.0, SIGMA0), (10.0, SIGMA0)], [0, 1, 2])
        self.assertGreater(contra_fortes[0][0] - MU0, contra_fracos[0][0] - MU0)

    def test_um_jogador_so_nao_muda_nada(self):
        self.assertEqual(atualizar_ffa([(MU0, SIGMA0)], [0]), [(MU0, SIGMA0)])

    def test_tamanhos_diferentes_sao_recusados(self):
        with self.assertRaises(ValueError):
            atualizar_ffa([(MU0, SIGMA0)] * 3, [0, 1])


@unittest.skipUnless(_tem_openskill(),
                     "openskill nao instalada — ela NAO entra no requirements "
                     "de proposito; este teste roda onde ela existe")
class TestParidadeComOpenskill(unittest.TestCase):
    """A conta daqui tem de bater com a biblioteca, digito a digito."""

    CENARIOS = [
        ("10 novatos, ordem estrita", [(MU0, SIGMA0)] * 10, list(range(10))),
        ("10 novatos, empate geral", [(MU0, SIGMA0)] * 10, [0] * 10),
        ("empates parciais", [(MU0, SIGMA0)] * 6, [0, 0, 2, 2, 4, 4]),
        ("ratings ja diferentes",
         [(30.0, 4.0), (25.0, 8.3), (20.0, 2.0), (28.0, 6.0), (22.0, 5.0)],
         [2, 0, 4, 1, 3]),
        ("o pior vence", [(35.0, 2.0), (30.0, 3.0), (15.0, 8.0)], [2, 1, 0]),
        ("incerteza muito diferente",
         [(25.0, 1.0), (25.0, 8.3), (25.0, 4.0), (25.0, 0.5)], [0, 1, 2, 3]),
    ]

    def test_bate_com_a_biblioteca(self):
        from openskill.models import PlackettLuce
        from mlcore.players_skill_ffa import KAPPA

        modelo = PlackettLuce(mu=MU0, sigma=SIGMA0, beta=BETA, kappa=KAPPA, tau=0.0)
        for nome, ratings, posicoes in self.CENARIOS:
            with self.subTest(cenario=nome):
                meus = atualizar_ffa(ratings, posicoes)
                times = [[modelo.rating(mu=m, sigma=s)] for m, s in ratings]
                deles = [(t[0].mu, t[0].sigma)
                         for t in modelo.rate(times, ranks=list(posicoes))]
                for (m1, s1), (m2, s2) in zip(meus, deles):
                    self.assertAlmostEqual(m1, m2, places=9)
                    self.assertAlmostEqual(s1, s2, places=9)


def _quadro(n_jogos: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    linhas = []
    d0 = pd.Timestamp("2025-01-01")
    for g in range(n_jogos):
        for lado in ("blue", "red"):
            for i, rota in enumerate(["Top", "Jungle", "Mid", "Bot", "Support"]):
                linhas.append({
                    "gameid": "g{:03d}".format(g),
                    "date": d0 + pd.Timedelta(days=g),
                    "player_key": "p{}".format(int(rng.integers(0, 14))),
                    "pos_can": rota,
                    "lado": lado,
                })
    df = pd.DataFrame(linhas)
    return df.sort_values(["date", "gameid"]).reset_index(drop=True)


class TestConstrucao(unittest.TestCase):

    def test_a_nota_do_PRIMEIRO_jogo_e_a_inicial(self):
        """Ponto no tempo: antes de jogar, ninguem tem historico."""
        df = _quadro()
        ps = pd.Series(np.random.default_rng(1).normal(size=len(df)), index=df.index)
        antes, _ = construir_notas(df, ps)
        primeira = antes.iloc[0]
        self.assertAlmostEqual(float(primeira), MU0 - 3.0 * SIGMA0, places=9)

    def test_mexer_no_ULTIMO_jogo_nao_muda_nota_nenhuma_anterior(self):
        df = _quadro()
        rng = np.random.default_rng(5)
        ps = pd.Series(rng.normal(size=len(df)), index=df.index)
        a1, _ = construir_notas(df, ps)

        ps2 = ps.copy()
        ultimos = df.index[df["gameid"] == df["gameid"].iloc[-1]]
        ps2.loc[ultimos] = 999.0
        a2, _ = construir_notas(df, ps2)

        anteriores = df.index[df["gameid"] != df["gameid"].iloc[-1]]
        np.testing.assert_allclose(
            a1.loc[anteriores].to_numpy(), a2.loc[anteriores].to_numpy(),
            err_msg="VAZAMENTO: o passado mudou quando o futuro mudou")

    def test_a_nota_final_desconta_a_incerteza(self):
        df = _quadro()
        ps = pd.Series(np.random.default_rng(2).normal(size=len(df)), index=df.index)
        _, finais = construir_notas(df, ps)
        self.assertTrue(finais)
        for reg in finais.values():
            self.assertAlmostEqual(reg["n"], reg["mu"] - 3.0 * reg["sigma"], places=9)

    def test_delta_por_jogo_e_azul_menos_vermelho(self):
        df = _quadro(n_jogos=5)
        antes = pd.Series(1.0, index=df.index)
        antes[df["lado"].to_numpy() == "red"] = -1.0
        d = deltas_por_jogo(df, antes)
        for v in d.values():
            self.assertAlmostEqual(v, 10.0)


if __name__ == "__main__":
    unittest.main()
