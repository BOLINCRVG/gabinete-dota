# -*- coding: utf-8 -*-
"""Testes do WHR — o CORE do modelo (alavanca N, +1.25 de nota em 07/07).

Buraco encontrado em 19/07: mlcore/whr_core.py, que calcula a trajetória de
rating de TODOS os times (e portanto a odd que ele aposta), não tinha teste
próprio. Estes testes travam as PROPRIEDADES matemáticas do fitter — não
números mágicos — para que qualquer mexida futura acuse regressão.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from mlcore.whr_core import (  # noqa: E402
    NAT,
    WHRFitter,
    _solve_tridiag,
)


class TestSolverTridiagonal(unittest.TestCase):
    """O solver é o núcleo do passo de Newton: se ele erra, todo rating erra."""

    def test_resolve_sistema_1x1(self):
        x = _solve_tridiag(np.array([2.0]), np.array([]), np.array([6.0]))
        self.assertAlmostEqual(x[0], 3.0)

    def test_resolve_sistema_conhecido(self):
        # [[2,1],[1,2]] x = [3,3]  ->  x = [1,1]
        x = _solve_tridiag(np.array([2.0, 2.0]), np.array([1.0]), np.array([3.0, 3.0]))
        self.assertAlmostEqual(x[0], 1.0, places=9)
        self.assertAlmostEqual(x[1], 1.0, places=9)

    def test_bate_com_numpy_em_matriz_aleatoria(self):
        """Contra a referência: mesma resposta que o solver denso do numpy."""
        rng = np.random.default_rng(20260719)
        n = 6
        d = rng.uniform(4.0, 6.0, n)          # diagonal dominante = estável
        o = rng.uniform(-1.0, 1.0, n - 1)
        rhs = rng.uniform(-3.0, 3.0, n)
        A = np.diag(d) + np.diag(o, 1) + np.diag(o, -1)
        esperado = np.linalg.solve(A, rhs)
        obtido = _solve_tridiag(d.copy(), o.copy(), rhs.copy())
        np.testing.assert_allclose(obtido, esperado, rtol=1e-9, atol=1e-9)


class TestWHRFitterPropriedades(unittest.TestCase):
    def _fitter(self, **kw) -> WHRFitter:
        return WHRFitter(**kw)

    def test_sem_jogos_probabilidade_e_meio(self):
        f = self._fitter()
        self.assertAlmostEqual(f.p_blue("A", "B"), 0.5, places=6)

    def test_quem_ganha_sempre_fica_com_rating_maior(self):
        f = self._fitter()
        for d in range(10):
            f.add_game("A", "B", True, d)
        f.full_sweeps(sweeps=5)
        self.assertGreater(f.rating_nat("A"), f.rating_nat("B"))
        self.assertGreater(f.p_blue("A", "B"), 0.5)

    def test_probabilidade_e_simetrica(self):
        """p(A vence B) + p(B vence A) = 1 — invariante do Bradley-Terry."""
        f = self._fitter()
        for d in range(6):
            f.add_game("A", "B", d % 3 != 0, d)
        f.full_sweeps(sweeps=4)
        self.assertAlmostEqual(f.p_blue("A", "B") + f.p_blue("B", "A"), 1.0, places=9)

    def test_empate_de_resultados_mantem_times_proximos(self):
        f = self._fitter()
        for d in range(12):
            f.add_game("A", "B", d % 2 == 0, d)  # 50/50
        f.full_sweeps(sweeps=6)
        self.assertLess(abs(f.rating_nat("A") - f.rating_nat("B")), 0.35)
        self.assertAlmostEqual(f.p_blue("A", "B"), 0.5, delta=0.1)

    def test_mais_vitorias_gera_rating_maior(self):
        """Monotonicidade: 9-1 tem que ficar acima de 6-4."""
        def rating_de(vitorias: int, total: int = 10) -> float:
            f = self._fitter()
            for d in range(total):
                f.add_game("X", f"opp{d}", d < vitorias, d)
            f.full_sweeps(sweeps=5)
            return f.rating_nat("X")

        self.assertGreater(rating_de(9), rating_de(6))
        self.assertGreater(rating_de(6), rating_de(3))

    def test_ratings_elo_converte_de_natural_para_elo(self):
        f = self._fitter()
        for d in range(8):
            f.add_game("A", "B", True, d)
        f.full_sweeps(sweeps=5)
        elo = f.ratings_elo()
        self.assertIn("A", elo)
        # NAT converte Elo->natural; a volta tem que ser consistente
        self.assertAlmostEqual(elo["A"], f.rating_nat("A") / NAT, places=6)

    def test_w2_controla_a_memoria_do_rating(self):
        """w2 = variância do passeio por dia = quanto o rating pode se mover.

        Cenário: A domina os 10 primeiros dias e DESABA nos 10 seguintes
        (histórico total 50/50). O rating FINAL revela a memória:
          - w2 ALTO  -> segue o período recente (A perdendo) => A fica ABAIXO
          - w2 BAIXO -> trajetória rígida ~ média de toda a história => empate
        (Contra-intuitivo: escrevi a asserção invertida na 1ª tentativa e o
        teste me corrigiu — por isso ela fica documentada aqui.)"""
        def delta_final(w2: float) -> float:
            f = WHRFitter(w2_elo_per_day=w2)
            for d in range(10):
                f.add_game("A", "B", True, d)        # A domina no início
            for d in range(10, 20):
                f.add_game("A", "B", False, d)       # e desaba depois
            f.full_sweeps(sweeps=6)
            return f.rating_nat("A") - f.rating_nat("B")

        recente, rigido = delta_final(600.0), delta_final(1.0)
        self.assertLess(recente, -0.1, "w2 alto deveria seguir o período recente (A perdendo)")
        self.assertAlmostEqual(rigido, 0.0, delta=0.05,
                               msg="w2 baixo deveria ficar perto da média histórica (50/50)")
        self.assertLess(recente, rigido)

    def test_prior_de_liga_puxa_time_sem_historico(self):
        """Time novo herda o patamar da liga (prior SOS) — o que evita odd
        absurda em time sem jogos."""
        f = WHRFitter(league_prior_points={"FORTE": 300.0, "FRACA": -300.0})
        f.add_game("novoA", "novoB", True, 0, blue_league="FORTE")
        f.full_sweeps(sweeps=3)
        self.assertGreater(f.rating_nat("novoA", "FORTE"), f.rating_nat("novoB", "FRACA"))

    def test_valores_sao_finitos_apos_sequencia_extrema(self):
        """Sequência degenerada (100% vitórias) não pode gerar NaN/inf: o
        clamp do passo de Newton existe para isso."""
        f = self._fitter()
        for d in range(40):
            f.add_game("A", "B", True, d)
        f.full_sweeps(sweeps=8)
        for t in ("A", "B"):
            r = f.rating_nat(t)
            self.assertTrue(math.isfinite(r), f"{t} virou {r}")
        p = f.p_blue("A", "B")
        self.assertTrue(math.isfinite(p) and 0.0 < p < 1.0)

    def test_mesma_entrada_gera_mesmo_resultado(self):
        """Determinismo: o fit não pode depender de ordem de dict/aleatório."""
        def roda() -> float:
            f = self._fitter()
            for d in range(15):
                f.add_game("A", "B", d % 3 != 0, d)
            f.full_sweeps(sweeps=5)
            return f.rating_nat("A")

        self.assertEqual(roda(), roda())


if __name__ == "__main__":
    unittest.main()
