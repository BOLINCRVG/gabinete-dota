# -*- coding: utf-8 -*-
"""A régua oficial mede o que a tela serve? — dívida D5.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ A D5 diz que "a nota oficial mede um caminho que não é o servido", com      ║
║ divergência média de 3,36 pp. Em 08/09/2026, ao medir isso com a D8 e a D9  ║
║ já consertadas, apareceu que o NÚMERO NUNCA FOI CONFIÁVEL:                  ║
║                                                                             ║
║   `Amostra.delta_eff` chega ZERO em 100% dos jogos, porque o replay monta   ║
║   o campo com `getattr(s, "delta_players_eff", 0.0)` e a classe `Sample`    ║
║   (mlcore/fit_samples_core.py) não tem campo de delta nenhum. O delta é     ║
║   calculado durante a montagem e descartado. Com delta = 0, `p_regua`       ║
║   devolve o ELO SOZINHO — conferido: bate com `p_team` em 1,11e-16.         ║
║                                                                             ║
║ Ou seja: a "divergência régua × servido" comparava o Elo puro contra a      ║
║ cadeia completa. Isso mede alguma coisa, mas não o que a dívida afirma.     ║
║                                                                             ║
║ POR QUE O DEFEITO SOBREVIVEU: `getattr` com default é falha silenciosa —    ║
║ devolve um número plausível para sempre, e nada quebra. É a mesma família   ║
║ da D8 (estágio inerte com bandeira ligada) e da D9 (cache que não vence).   ║
║ A regra que este projeto já aplicou nas duas: ausência de dado NÃO pode     ║
║ virar um número; ou se mede, ou se FALHA ALTO.                              ║
║ https://thicket.sh/blog/when-your-metric-is-broken-not-your-business        ║
║ (conferida abrindo a página em 04/09/2026)                                  ║
║                                                                             ║
║ O CONSERTO DE VERDADE é levar o delta até a `Sample`, que o TREINO usa —    ║
║ frente própria, não coube aqui. Até lá, `p_regua` AVISA em vez de mentir    ║
║ calada, e estes testes travam esse aviso.                                   ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from mlcore.replay_pre_jogo import Amostra, p_regua


def _amostra(p_team=0.6, delta=0.0):
    return Amostra(gameid="g", league="LCK", split="", side="blue", bo="3",
                   phase="regular", month="2026-05", y=1, p_team=p_team,
                   p_players=0.55, n_team=100.0, n_players=50.0,
                   delta_eff=delta)


class TestReguaAvisaQuandoODeltaSumiu(unittest.TestCase):

    def _saida(self, amostras, w=0.85):
        buf = io.StringIO()
        with redirect_stdout(buf):
            r = p_regua(amostras, w)
        return r, buf.getvalue()

    def test_delta_zero_em_todos_AVISA(self):
        """O caso real: 100% dos jogos com delta 0, e ninguem sabia."""
        _, saida = self._saida([_amostra(delta=0.0) for _ in range(5)])
        self.assertIn("AVISO", saida)
        self.assertIn("Elo SOZINHO", saida)

    def test_com_delta_de_verdade_NAO_avisa(self):
        """O aviso nao pode virar ruido quando o dado estiver certo."""
        amostras = [_amostra(delta=0.3), _amostra(delta=-0.2), _amostra(delta=0.0)]
        _, saida = self._saida(amostras)
        self.assertNotIn("AVISO", saida)

    def test_com_delta_zero_a_regua_E_o_elo_puro(self):
        """A consequencia que o aviso descreve, travada em numero."""
        a = _amostra(p_team=0.73, delta=0.0)
        r, _ = self._saida([a])
        self.assertAlmostEqual(r[0], 0.73, delta=1e-12)

    def test_com_delta_a_regua_SAI_do_elo(self):
        """Prova que a formula usa o delta — se parar de usar, isto pega."""
        r, _ = self._saida([_amostra(p_team=0.6, delta=0.5)])
        self.assertGreater(r[0], 0.6 + 1e-6)

    def test_w_lineup_zero_anula_o_delta(self):
        r, _ = self._saida([_amostra(p_team=0.6, delta=0.5)], w=0.0)
        self.assertAlmostEqual(r[0], 0.6, delta=1e-12)

    def test_lista_vazia_nao_avisa_nem_estoura(self):
        r, saida = self._saida([])
        self.assertEqual(r, [])
        self.assertNotIn("AVISO", saida)


if __name__ == "__main__":
    unittest.main()
