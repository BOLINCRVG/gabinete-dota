# -*- coding: utf-8 -*-
"""WHR com margem de vitoria (resultado fracionario) — contrato travado.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ O QUE ESTE ARQUIVO PROTEGE, em uma frase: que ligar a margem seja uma        ║
║ ESCOLHA, e que desligada o motor seja BIT A BIT o que estava em producao.    ║
║                                                                             ║
║ 1. A FORMA (resultado fracionario em vez de empurrao no rating) e o que      ║
║    mantem a matematica do WHR de pe: o gradiente vira (ỹ - p) e a Hessiana  ║
║    segue -p(1-p), porque a Hessiana NAO depende do resultado. Vitoria        ║
║    fracionaria e pratica estabelecida em Bradley-Terry — no hoquei           ║
║    universitario a vitoria na prorrogacao vale 1/3 de vitoria.               ║
║    https://arxiv.org/abs/2112.01267  (aberta em 08/09/2026)                  ║
║                                                                             ║
║ 2. SO A SURPRESA DA MARGEM ENTRA: E[MOV|Δr] = a·tanh(b·Δr) + c. Margem crua ║
║    confundiria "dominou porque e melhor" (ja no rating) com "dominou acima   ║
║    do esperado" (informacao nova), e a tanh satura, o que trata placar       ║
║    inflado no fim de jogo. https://arxiv.org/html/2506.00348 (aberta 08/09)  ║
║                                                                             ║
║ 3. A PROVA que autorizou isto e CEGA, nao teorica: tools/eval_whr_mov.py,    ║
║    ajuste em 2025 e teste em 2026 sem nunca tocar o ano cego. Ouro aos 15    ║
║    min, lam=0.35, folga=0.15, w2=90: logloss +0.0117 (IC95 [+0.0078,        ║
║    +0.0157]), acerto 65.0%->65.7%, ganho em 9/9 meses e 28/34 ligas.         ║
║    evidence/eval_whr_mov_2026.log e evidence/eval_whr_mov_gd15_folga.log     ║
║                                                                             ║
║ 4. O RISCO QUE ESTES TESTES VIGIAM e o da familia D8/D9/D5: um estagio que   ║
║    parece ligado e nao faz nada, ou ausencia de dado que vira numero         ║
║    plausivel. Por isso ha teste para "sem MOV cai no binario E CONTA", e     ║
║    teste que exige que ligado MUDE o resultado — sem esse par, um bug que    ║
║    matasse a margem passaria despercebido com todos os outros verdes.        ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import unittest

from mlcore.whr_core import WHRFitter

# a=0 faz E[MOV|Δr] = c, constante: separa o que o teste quer medir do formato
# da tanh. Os testes que precisam da tanh usam _FIT_TANH.
_FIT = {"a": 0.0, "b": 1.0, "c": 0.0, "s": 1.0}
_FIT_TANH = {"a": 2.0, "b": 1.0, "c": 0.0, "s": 1.0}


def _jogos(f, n=6):
    """Um calendario minimo, sempre igual, para comparar dois fitters."""
    for i in range(n):
        f.add_game("A", "B", i % 2, 100 + i, "LCK", "LCK")
        f.add_game("A", "C", 1, 100 + i, "LCK", "LCK")
        f.add_game("B", "C", 1, 100 + i, "LCK", "LCK")


class TestDesligadoEIdenticoAProducao(unittest.TestCase):
    """A garantia que permite mexer no motor sem mexer em producao."""

    def test_default_e_lambda_zero(self):
        f = WHRFitter()
        self.assertEqual(f.mov_lambda, 0.0)
        self.assertEqual(f.mov_folga, 0.0)

    def test_desligado_guarda_resultado_binario_exato(self):
        f = WHRFitter()
        f.add_game("A", "B", 1, 100, "LCK", "LCK")
        (_, _, _, yf_a), = f.games["A"]
        (_, _, _, yf_b), = f.games["B"]
        self.assertEqual(yf_a, 1.0)
        self.assertEqual(yf_b, 0.0)

    def test_desligado_IGNORA_a_margem(self):
        """Passar MOV com a margem desligada nao pode mudar nada."""
        sem = WHRFitter()
        com = WHRFitter()
        for i in range(4):
            sem.add_game("A", "B", 1, 100 + i, "LCK", "LCK")
            com.add_game("A", "B", 1, 100 + i, "LCK", "LCK", mov=9999.0)
        sem.full_sweeps(sweeps=6)
        com.full_sweeps(sweeps=6)
        self.assertEqual(sem.ratings_elo(), com.ratings_elo())

    def test_ratings_identicos_com_e_sem_os_parametros_novos(self):
        """Construir o fitter com os argumentos novos (mas neutros) nao move
        um digito do rating."""
        antigo = WHRFitter(w2_elo_per_day=60.0)
        novo = WHRFitter(w2_elo_per_day=60.0, mov_lambda=0.0, mov_folga=0.0,
                         mov_fit=_FIT_TANH)
        _jogos(antigo)
        _jogos(novo)
        antigo.full_sweeps(sweeps=8)
        novo.full_sweeps(sweeps=8)
        self.assertEqual(antigo.ratings_elo(), novo.ratings_elo())


class TestOResultadoFracionario(unittest.TestCase):

    def _f(self, lam=0.5, folga=0.0, fit=None):
        return WHRFitter(mov_lambda=lam, mov_folga=folga, mov_fit=fit or _FIT)

    def test_margem_igual_a_esperada_nao_muda_nada(self):
        """O ponto do desenho: so a SURPRESA conta. MOV == E[MOV] -> ỹ == y."""
        f = self._f()
        self.assertAlmostEqual(
            f.resultado_fracionario("A", "B", 1, 0.0, "LCK", "LCK"), 1.0, places=12)
        self.assertAlmostEqual(
            f.resultado_fracionario("A", "B", 0, 0.0, "LCK", "LCK"), 0.0, places=12)

    def test_vencer_abaixo_do_esperado_DESCONTA(self):
        f = self._f(folga=0.2)
        y = f.resultado_fracionario("A", "B", 1, -1.0, "LCK", "LCK")
        self.assertLess(y, 0.8)          # base da vitoria com folga 0.2
        self.assertGreaterEqual(y, 0.0)

    def test_vencer_acima_do_esperado_PREMIA_quando_ha_folga(self):
        """Sem folga isto seria impossivel: o teto de 1 come o premio. Foi por
        isso que a folga entrou — medido, 72,5% dos jogos batiam no teto."""
        sem_folga = self._f(folga=0.0)
        com_folga = self._f(folga=0.2)
        self.assertEqual(sem_folga.resultado_fracionario("A", "B", 1, 5.0), 1.0)
        y = com_folga.resultado_fracionario("A", "B", 1, 1.0)
        self.assertGreater(y, 0.8)
        self.assertLessEqual(y, 1.0)

    def test_perder_jogando_bem_e_recompensado(self):
        f = self._f(folga=0.2)
        y = f.resultado_fracionario("A", "B", 0, 1.0, "LCK", "LCK")
        self.assertGreater(y, 0.2)       # base da derrota com folga 0.2

    def test_simetria_o_oponente_recebe_o_complemento(self):
        f = self._f(folga=0.15)
        f.add_game("A", "B", 1, 100, "LCK", "LCK", mov=0.7)
        (_, _, _, ya), = f.games["A"]
        (_, _, _, yb), = f.games["B"]
        self.assertAlmostEqual(ya + yb, 1.0, places=12)

    def test_sempre_dentro_de_zero_e_um(self):
        f = self._f(lam=2.0, folga=0.1)
        for mov in (-1e6, -3.0, 0.0, 3.0, 1e6):
            y = f.resultado_fracionario("A", "B", 1, mov)
            self.assertTrue(0.0 <= y <= 1.0, f"mov={mov} -> {y}")

    def test_a_expectativa_usa_a_tanh_do_rating(self):
        """Contra um adversario mais fraco espera-se margem maior, entao a MESMA
        margem crua vale MENOS. Se isto quebrar, a margem virou margem crua."""
        f = WHRFitter(mov_lambda=0.5, mov_fit=_FIT_TANH,
                      league_prior_points={"FORTE": 400.0, "FRACA": -400.0})
        contra_fraco = f.resultado_fracionario("A", "Z", 1, 1.0, "FORTE", "FRACA")
        contra_forte = f.resultado_fracionario("A", "Z", 1, 1.0, "FRACA", "FORTE")
        self.assertLess(contra_fraco, contra_forte)


class TestAusenciaDeDadoNaoViraNumero(unittest.TestCase):
    """A licao das dividas D5/D8/D9: ou se mede, ou se conta — nunca se inventa."""

    def test_sem_mov_cai_no_binario_E_CONTA(self):
        f = WHRFitter(mov_lambda=0.5, mov_fit=_FIT)
        self.assertEqual(f.resultado_fracionario("A", "B", 1, None), 1.0)
        self.assertEqual(f.n_mov_faltando, 1)
        self.assertEqual(f.n_mov_usados, 0)

    def test_mov_nao_finito_conta_como_faltando(self):
        f = WHRFitter(mov_lambda=0.5, mov_fit=_FIT)
        for ruim in (float("nan"), float("inf"), "vazio", None):
            f.resultado_fracionario("A", "B", 1, ruim)
        self.assertEqual(f.n_mov_faltando, 4)

    def test_ajuste_sem_desvio_e_recusado_nao_estoura(self):
        """Bloco de ajuste malformado desliga a margem em vez de dividir por 0."""
        for fit in ({}, {"a": 1.0, "b": 1.0, "c": 0.0, "s": 0.0}, {"s": float("nan")}):
            f = WHRFitter(mov_lambda=0.5, mov_fit=fit)
            self.assertEqual(f.resultado_fracionario("A", "B", 1, 5.0), 1.0)
            self.assertEqual(f.n_mov_faltando, 1)

    def test_mov_usado_e_contado(self):
        f = WHRFitter(mov_lambda=0.5, mov_fit=_FIT)
        f.add_game("A", "B", 1, 100, "LCK", "LCK", mov=0.5)
        self.assertEqual(f.n_mov_usados, 1)
        self.assertEqual(f.n_mov_faltando, 0)


class TestOMotorContinuaFuncionando(unittest.TestCase):
    """Nao basta a conta do ỹ estar certa: o Newton tem que seguir convergindo
    e o rating tem que ir para o lado certo."""

    def test_ligado_MUDA_o_rating(self):
        """Guarda contra o estagio virar inerte com a bandeira ligada (D8).

        CUIDADO QUE ME PEGOU AO ESCREVER ISTO (08/09/2026): a primeira versao
        usava mov=2.0, que com lam=0.5 e folga=0.15 da 0.85+1.0 -> RECORTA em
        1.0, ou seja, exatamente o binario. O teste falhava e estava CERTO: o
        cenario e que era degenerado. A margem so muda alguma coisa na faixa
        em que o recorte nao morde — por isso mov=0.2 aqui. E a mesma razao
        pela qual a folga existe: sem ela, quase tudo satura."""
        base = WHRFitter(mov_lambda=0.0)
        mov = WHRFitter(mov_lambda=0.5, mov_folga=0.15, mov_fit=_FIT)
        for i in range(6):
            base.add_game("A", "B", 1, 100 + i, "LCK", "LCK", mov=0.2)
            mov.add_game("A", "B", 1, 100 + i, "LCK", "LCK", mov=0.2)
        base.full_sweeps(sweeps=8)
        mov.full_sweeps(sweeps=8)
        self.assertNotAlmostEqual(base.ratings_elo()["A"], mov.ratings_elo()["A"],
                                 places=3, msg="margem ligada nao mexeu no rating")

    def test_quem_domina_alem_do_esperado_sobe_mais(self):
        """O comportamento que justifica a mudanca inteira."""
        def _rating(mov_por_jogo):
            f = WHRFitter(mov_lambda=0.5, mov_folga=0.2, mov_fit=_FIT)
            for i in range(10):
                f.add_game("A", "B", 1, 100 + i, "LCK", "LCK", mov=mov_por_jogo)
            f.full_sweeps(sweeps=12)
            return f.ratings_elo()["A"]
        self.assertGreater(_rating(1.0), _rating(-1.0),
                           "vencer dominando devia valer mais que vencer apertado")

    def test_converge_com_resultado_fracionario(self):
        f = WHRFitter(mov_lambda=0.5, mov_folga=0.15, mov_fit=_FIT_TANH)
        for i in range(12):
            f.add_game("A", "B", i % 2, 100 + i, "LCK", "LCK", mov=(i % 3) - 1.0)
            f.add_game("A", "C", 1, 100 + i, "LCK", "LCK", mov=0.5)
        f.full_sweeps(sweeps=20, tol_elo=0.5)
        for t, v in f.ratings_elo().items():
            self.assertTrue(-5000.0 < v < 5000.0, f"{t} divergiu: {v}")


if __name__ == "__main__":
    unittest.main()
