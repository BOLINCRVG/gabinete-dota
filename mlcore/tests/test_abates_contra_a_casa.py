# -*- coding: utf-8 -*-
"""A fair de ABATES da tela contra o preco da Pinnacle.

ELE EXPLICOU O QUE PRECISA (22/09/2026): "eu entendo o modelo ele ser afastado
do ml da pinnacle, ate porque e a casa mais sharp que nos temos, o que eu
preciso e realmente a vantagem em relacao as kills... se la no resumo ele
aparece com uma fair de 1.40 e na pinnacle ele esta com 1,72 de odd, e ele que
eu vou girar".

Isso mudou a pergunta. Nao interessa empatar com a Pinnacle no vencedor --
interessa se a fair de ABATES que a tela mostra ganha dinheiro.

DUAS ARMADILHAS QUE ESTES TESTES IMPEDEM, e a segunda e a mais cara:

1. MEDIR OUTRO MODELO. A ferramenta chama o caminho da TELA (build_profile ->
   _expected_totals_dc -> prob_over_from_total). Medir um modelo melhor diria
   o que seria possivel, nao o que ele tem na mao.

2. CONTAR APOSTA COMO PARTIDA NO RETORNO. Um mapa costuma ter varias linhas
   (24.5, 26.5, 28.5...). Se o modelo acerta o total daquele mapa, TODAS
   ganham juntas. Na primeira medicao deram 51 apostas com +23,1% de retorno
   -- numero que da vontade de apostar. Reamostrando PARTIDAS, o intervalo e
   [-7%, +55%]: inclui perder dinheiro. A barra de erro por aposta teria
   escondido isso.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol

RAIZ = Path(__file__).resolve().parents[2]
ARQ = RAIZ / "tools" / "abates_contra_a_casa.py"


def _carregar():
    spec = importlib.util.spec_from_file_location("abates_contra_a_casa", ARQ)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _l(jogo, mapa, lado, pontos, odd, lido="t", alt=False, unidade="Kills"):
    return {"jogo": jogo, "periodo": mapa, "lado": lado, "pontos": pontos,
            "odd": odd, "lido_em": lido, "alternativa": alt, "tipo": "total",
            "unidade": unidade, "limite": 250}


class TestAsApostasDeTotal(unittest.TestCase):

    def _ler(self, linhas):
        m = _carregar()
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "h.jsonl"
            alvo.write_text("\n".join(json.dumps(x) for x in linhas), encoding="utf-8")
            m.PRECOS = alvo
            return m.apostas_de_total()

    @so_no_lol("so vale para o LoL")
    def test_o_par_over_under_vira_UMA_aposta(self):
        r = self._ler([_l(1, 1, "over", 27.5, 1.9), _l(1, 1, "under", 27.5, 1.9)])
        self.assertEqual(len(r), 1)
        self.assertAlmostEqual(r[0]["linha"], 27.5)

    @so_no_lol("so vale para o LoL")
    def test_as_probabilidades_somam_um(self):
        r = self._ler([_l(1, 1, "over", 27.5, 1.72), _l(1, 1, "under", 27.5, 2.10)])
        self.assertAlmostEqual(r[0]["p_over"] + r[0]["p_under"], 1.0, places=9)

    @so_no_lol("so vale para o LoL")
    def test_linhas_diferentes_sao_apostas_diferentes(self):
        r = self._ler([_l(1, 1, "over", 26.5, 1.9), _l(1, 1, "under", 26.5, 1.9),
                       _l(1, 1, "over", 28.5, 1.9), _l(1, 1, "under", 28.5, 1.9)])
        self.assertEqual(sorted(x["linha"] for x in r), [26.5, 28.5])

    @so_no_lol("so vale para o LoL")
    def test_a_alternativa_fica_de_fora(self):
        self.assertEqual(self._ler([_l(1, 1, "over", 27.5, 1.9, alt=True),
                                    _l(1, 1, "under", 27.5, 1.9, alt=True)]), [])

    @so_no_lol("so vale para o LoL")
    def test_outra_unidade_nao_entra(self):
        self.assertEqual(self._ler([_l(1, 1, "over", 11.5, 1.9, unidade="Turrets"),
                                    _l(1, 1, "under", 11.5, 1.9, unidade="Turrets")]), [])

    @so_no_lol("so vale para o LoL")
    def test_um_lado_so_NAO_vira_aposta(self):
        """Sem os dois lados nao da para tirar a margem."""
        self.assertEqual(self._ler([_l(1, 1, "over", 27.5, 1.9)]), [])

    @so_no_lol("so vale para o LoL")
    def test_a_SERIE_nao_entra_porque_o_motor_fala_de_MAPA(self):
        self.assertEqual(self._ler([_l(1, 0, "over", 27.5, 1.9),
                                    _l(1, 0, "under", 27.5, 1.9)]), [])

    @so_no_lol("so vale para o LoL")
    def test_usa_a_leitura_mais_RECENTE(self):
        r = self._ler([_l(1, 1, "over", 27.5, 3.0, "2026-09-20T08:00"),
                       _l(1, 1, "over", 27.5, 1.8, "2026-09-20T19:00"),
                       _l(1, 1, "under", 27.5, 1.4, "2026-09-20T08:00"),
                       _l(1, 1, "under", 27.5, 2.0, "2026-09-20T19:00")])
        self.assertAlmostEqual(r[0]["odd_over"], 1.8)


class TestMedeOModeloDATELA(unittest.TestCase):
    """PRINCIPIO V: medir uma segunda conta responderia outra pergunta."""

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_usa_o_caminho_de_producao(self):
        self.assertIn("from app_core.runtime_wrappers_core import _expected_totals_dc",
                      self.fonte)
        self.assertIn("build_profile", self.fonte)
        self.assertIn("prob_over_from_total", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_usa_as_regulagens_PADRAO_da_tela(self):
        m = _carregar()
        self.assertEqual(m.MEIA_VIDA, 180.0)
        self.assertIn('league_mode="all"', self.fonte)
        self.assertIn("Filters(year=None, league=None, split=None, playoffs=None)",
                      self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_quadro_e_cortado_ANTES_da_partida(self):
        """`apply_dc_blend` usa a data mais recente do quadro que recebe."""
        self.assertIn('antes = tg[tg["date"] < r.date]', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_usa_o_resolvedor_de_producao_e_nao_adivinha(self):
        self.assertIn("resolve_schedule_team", self.fonte)


class TestORetornoNAOMenteSobreACerteza(unittest.TestCase):
    """A armadilha mais cara: +23% com barra de erro de aposta pareceria
    dinheiro na mesa. Reamostrando partidas, o intervalo inclui perder."""

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_reamostra_PARTIDAS_e_nao_apostas(self):
        i = self.fonte.index("SE ELE APOSTASSE SO ONDE A TELA APONTA VANTAGEM")
        trecho = self.fonte[i:i + 2500]
        self.assertIn("por_p.setdefault(c[\"partida\"]", trecho)
        self.assertIn("rng.integers(0, len(chaves)", trecho)

    @so_no_lol("so vale para o LoL")
    def test_o_intervalo_aparece_junto_do_retorno(self):
        self.assertIn("intervalo de 95% (reamostrando PARTIDAS)", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_existe_piso_de_partidas_para_qualquer_conclusao(self):
        m = _carregar()
        self.assertGreaterEqual(m.PARTIDAS_MINIMAS, 30)
        self.assertIn("AMOSTRA PEQUENA DEMAIS", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_a_incerteza_geral_tambem_conta_partida(self):
        self.assertIn("contando PARTIDA e nao aposta", self.fonte)


class TestOLadoCertoEOResultadoCerto(unittest.TestCase):

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_o_resultado_compara_o_total_com_a_linha(self):
        self.assertIn('1.0 if float(real) > float(a["linha"]) else 0.0', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_under_usa_a_probabilidade_COMPLEMENTAR(self):
        """Apostar no under com a probabilidade do over inverteria tudo."""
        self.assertIn('(1 - c["p_modelo"], c["odd_under"], 1 - c["y"])', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_mapa_da_aposta_casa_com_o_mapa_do_jogo(self):
        self.assertIn('indice.get((dd, par, a["mapa"]), [])', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_perfil_e_montado_para_o_MAPA_da_aposta(self):
        self.assertIn('mapa = "m%d" % a["mapa"]', self.fonte)


if __name__ == "__main__":
    unittest.main()
