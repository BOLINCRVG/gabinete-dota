# -*- coding: utf-8 -*-
"""A regua do handicap de abates -- e o erro de medicao que ela quase escondeu.

POR QUE ESTE MERCADO (21/09/2026). Medindo margem e LIMITE de cada mercado nas
14.937 leituras da Pinnacle, o handicap de abates tem limite 950 (o segundo
maior) e 3.700 linhas. Os mercados exoticos que pareciam atraentes -- dragao,
barao, inibidor, Elder -- tem limite 100: acertar neles rende pouco dinheiro.

QUATRO SUPOSICOES MINHAS, TODAS MEDIDAS, TODAS ERRADAS:

  1. "abates dos dois times andam juntos em jogo sangrento" -> correlacao
     -0,229. Atropelo pesa mais. Supor independencia encurta a largura 10%.
  2. "a diferenca e quase normal" -> curtose -1,14. A forma empirica erra
     3,6pp onde a normal erra 5,3pp.
  3. "a media movel exagera a diferenca" -> ela e TIMIDA (inclinacao 1,29).
  4. "esticar a previsao para o tamanho certo vai ajudar" -> PIOROU.

E O ERRO QUE DUROU MAIS, que era meu, de MEDICAO e nao de modelo:

eu olhava a calibracao MARGINAL -- media das probabilidades previstas contra a
frequencia observada, linha por linha. Com ela, o Dixon-Coles (que preve
MELHOR: correlacao 0,348 contra 0,307, erro 10,30 contra 10,74) aparecia como
PIOR: 5,6pp contra 3,2pp.

A media marginal pune o modelo melhor. P(dif > linha) e funcao curva da media
prevista; espalhar as medias -- que e o que um modelo informativo faz --
desloca a media das probabilidades mesmo com cada previsao certa. Jensen.

Com o teste certo (calibracao CONDICIONAL, por faixa de probabilidade
prevista), o mesmo modelo da ECE de 1,49pp; com o calibrador, 0,31pp.

E O QUE ESTES TESTES NAO DEIXAM ESQUECER: calibrado NAO e o mesmo que
lucrativo. A casa tambem e calibrada, e cobra 8,41%. Vantagem exige acertar
MAIS que ela, e isso ainda nao foi medido.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol

RAIZ = Path(__file__).resolve().parents[2]
ARQ = RAIZ / "tools" / "handicap_de_abates.py"


def _carregar():
    spec = importlib.util.spec_from_file_location("handicap_de_abates", ARQ)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOCalibrador(unittest.TestCase):

    def setUp(self):
        self.m = _carregar()

    def _pares(self, desvio):
        """Probabilidades cruas + resultado, com um erro conhecido.

        `desvio` positivo = o modelo cru exagera (diz mais do que acontece).
        """
        import random
        rng = random.Random(5)
        fora = []
        for _ in range(4000):
            p = rng.random()
            real = max(0.0, min(1.0, p - desvio * (p - 0.5) * 2))
            fora.append((p, 1.0 if rng.random() < real else 0.0))
        return fora

    @so_no_lol("so vale para o LoL")
    def test_corrige_o_exagero_das_pontas(self):
        cal = self.m.calibrador(self._pares(0.30))
        # onde o modelo cru dizia 10%, a verdade e mais perto de 26%
        self.assertGreater(cal(0.10), 0.16)
        self.assertLess(cal(0.90), 0.84)

    @so_no_lol("so vale para o LoL")
    def test_NAO_inverte_a_ordem_das_apostas(self):
        """Corrigir a conta e estragar a escolha seria trocar um problema por
        outro pior: a ordem e o que decide em qual linha apostar."""
        cal = self.m.calibrador(self._pares(0.25))
        vals = [cal(x / 20.0) for x in range(1, 20)]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(a, b + 1e-9, "o calibrador inverteu a ordem")

    @so_no_lol("so vale para o LoL")
    def test_modelo_ja_certo_quase_nao_e_mexido(self):
        cal = self.m.calibrador(self._pares(0.0))
        for v in (0.2, 0.4, 0.6, 0.8):
            self.assertAlmostEqual(cal(v), v, delta=0.06)

    @so_no_lol("so vale para o LoL")
    def test_sem_base_suficiente_NAO_inventa_correcao(self):
        """Poucos jogos e melhor nao corrigir que corrigir por ruido."""
        cal = self.m.calibrador([(0.5, 1.0), (0.4, 0.0)])
        self.assertEqual(cal(0.37), 0.37)

    @so_no_lol("so vale para o LoL")
    def test_fora_da_faixa_conhecida_nao_extrapola(self):
        cal = self.m.calibrador(self._pares(0.20))
        self.assertGreaterEqual(cal(0.0), 0.0)
        self.assertLessEqual(cal(1.0), 1.0)


class TestAFormaEmpirica(unittest.TestCase):
    """A diferenca de abates tem curtose -1,14: nao e normal."""

    def setUp(self):
        self.m = _carregar()

    @so_no_lol("so vale para o LoL")
    def test_padroniza_dividindo_pela_largura(self):
        z = self.m.forma_empirica([10.0, -10.0, 0.0], [10.0, 10.0, 10.0])
        self.assertEqual(z, [-1.0, 0.0, 1.0])

    @so_no_lol("so vale para o LoL")
    def test_descarta_largura_invalida_em_vez_de_estourar(self):
        self.assertEqual(self.m.forma_empirica([5.0, 5.0], [0.0, None]), [])

    @so_no_lol("so vale para o LoL")
    def test_a_probabilidade_sai_da_forma_e_nao_de_uma_curva(self):
        forma = [float(i) / 100.0 for i in range(-200, 201)]
        p = self.m.prob_acima(0.0, 0.0, 1.0, forma)
        self.assertAlmostEqual(p, 0.5, delta=0.01)

    @so_no_lol("so vale para o LoL")
    def test_SEM_forma_devolve_None_e_nao_cai_na_normal(self):
        """Aproximacao silenciosa aqui viraria preco na tela."""
        self.assertIsNone(self.m.prob_acima(0.0, 0.0, 1.0, []))
        self.assertIsNone(self.m.prob_acima(0.0, 0.0, 0.0, [1.0, 2.0]))

    @so_no_lol("so vale para o LoL")
    def test_linha_mais_alta_da_probabilidade_MENOR(self):
        forma = [float(i) / 50.0 for i in range(-100, 101)]
        self.assertGreater(self.m.prob_acima(-5.0, 0.0, 3.0, forma),
                           self.m.prob_acima(+5.0, 0.0, 3.0, forma))


class TestOsAchadosFicamEscritos(unittest.TestCase):
    """Cada suposicao derrubada custou uma rodada de medicao. Se o motivo
    sumir do arquivo, alguem tenta de novo."""

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_o_ganho_esta_desligado_e_o_motivo_escrito(self):
        m = _carregar()
        self.assertFalse(m.GANHO_LIGADO)
        self.assertIn("PIOROU", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_botao_de_largura_esta_marcado_como_INERTE(self):
        """Medido: 5,6pp em 1.00, 1.05, 1.10, 1.15, 1.20 e 1.30. A forma
        empirica e montada dividindo pela mesma largura, e o fator cancela."""
        self.assertIn("NAO FAZ NADA", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_erro_de_MEDICAO_esta_documentado(self):
        self.assertIn("ONDE EU ERREI A MEDICAO", self.fonte)
        self.assertIn("Jensen", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_a_calibracao_medida_e_a_CONDICIONAL(self):
        self.assertIn("CALIBRACAO CONDICIONAL", self.fonte)
        self.assertIn("faixa prevista", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_usa_o_Dixon_Coles_de_PRODUCAO(self):
        self.assertIn("from mlcore.lines_engine import DC_BLEND_HL_DAYS", self.fonte)


class TestNadaOlhaOFuturo(unittest.TestCase):

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_percorre_jogo_a_jogo_com_desempate_estavel(self):
        """Percorrer linha a linha ja me custou um vazamento em 21/09/2026."""
        self.assertIn('kind="mergesort"', self.fonte)
        self.assertIn('side"].astype(str).str.lower() == "blue"', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_jogo_entra_no_historico_DEPOIS_de_virar_previsao(self):
        i_prev = self.fonte.index('linhas.append({"date": quando')
        i_aprende = self.fonte.index('hist[t] = nova if d is None')
        self.assertLess(i_prev, i_aprende)

    @so_no_lol("so vale para o LoL")
    def test_vies_largura_e_calibrador_saem_SO_do_passado(self):
        i = self.fonte.index("passado = (treino + teste[:ini])")
        trecho = self.fonte[i:i + 1400]
        self.assertIn("_ajustar(passado)", trecho)
        self.assertIn("calibrador(cru)", trecho)


class TestCalibradoNAOEhLucrativo(unittest.TestCase):
    """A confusao mais cara possivel, e a mais facil de cometer.

    Calibracao diz que 60% acontece 60% das vezes. Nao diz nada sobre ganhar
    da casa: a Pinnacle tambem e calibrada, e cobra 8,41% neste mercado. Ter
    vantagem exige acertar MAIS que ela, jogo a jogo -- e isso so a comparacao
    com o preco dela responde.
    """

    @so_no_lol("so vale para o LoL")
    def test_o_arquivo_avisa_que_calibrado_nao_basta(self):
        fonte = ARQ.read_text(encoding="utf-8")
        self.assertIn("margem da casa", fonte)
        self.assertIn("8,41", fonte)

    def test_nenhum_preco_deste_arquivo_vai_para_a_tela(self):
        """Ele e regua, nao motor: o app nao pode importar daqui."""
        alvos = list((RAIZ / "app_core").glob("*.py")) + [RAIZ / "plays_app.py"]
        for a in alvos:
            if not a.exists():
                continue
            self.assertNotIn("handicap_de_abates", a.read_text(encoding="utf-8"),
                             "%s importou a regua" % a.name)


if __name__ == "__main__":
    unittest.main()
