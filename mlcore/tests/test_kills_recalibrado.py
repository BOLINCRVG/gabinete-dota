# -*- coding: utf-8 -*-
"""O total de kills reajustado toda semana (25/09/2026).

Medido (tools/kills_total_com_vitoria.py e tools/gerar_kills_recalibrado.py):
a conta servida estava descalibrada; uma reta + a forma real do erro, refeita
TODA SEMANA com os ultimos 60 dias (janela escolhida antes da prova cega),
melhora a prova cega de 23/06-21/09/2026 em +0,0030 [+0,0006, +0,0054].
Ajustado uma vez so, o mesmo conserto piorava: por isso a trava de idade.
"""
from __future__ import annotations

import unittest
from datetime import date, timedelta

import mlcore.lines_engine as le
from mlcore.lines_engine import MatchupTotals, prob_over_from_total
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol


def _art(dias_atras=0, a=8.0, b=0.75):
    q = [float(x) for x in range(-20, 21)] + [float(x) + 0.5 for x in range(-20, 20)]
    return {"coef": [a, b], "quantis": sorted(q),
            "gerado_em": (date.today() - timedelta(days=dias_atras)).isoformat()}


class _ComArtefato(unittest.TestCase):
    def setUp(self):
        self._guardado = dict(le._KILLS_RECAL_CACHE)
        le._KILLS_RECAL_CACHE.clear()

    def tearDown(self):
        le._KILLS_RECAL_CACHE.clear()
        le._KILLS_RECAL_CACHE.update(self._guardado)

    def usar(self, art):
        le._KILLS_RECAL_CACHE["art"] = art


class TestOReajuste(_ComArtefato):

    def test_kills_usa_a_reta_e_a_forma(self):
        self.usar(_art())
        t = MatchupTotals(mean=30.0, sd=10.0, dist="normal", metric="kills")
        # media reajustada: 8 + 0,75 x 30 = 30,5; linha 30,5 -> metade
        self.assertAlmostEqual(prob_over_from_total(t, 30.5), 0.5, delta=0.02)

    def test_linha_maior_chance_menor(self):
        self.usar(_art())
        t = MatchupTotals(mean=28.0, sd=10.0, dist="normal", metric="kills")
        ps = [prob_over_from_total(t, x + 0.5) for x in range(15, 40)]
        for a, b in zip(ps, ps[1:]):
            self.assertGreaterEqual(a, b)

    def test_so_kills_muda(self):
        t = MatchupTotals(mean=32.0, sd=4.0, dist="normal", metric="time")
        self.usar(_art())
        com = prob_over_from_total(t, 33.5)
        self.usar(None)
        sem = prob_over_from_total(t, 33.5)
        self.assertEqual(com, sem)

class TestATravaDeIdade(_ComArtefato):

    def test_velho_demais_volta_a_curva_de_sempre(self):
        self.usar(_art(dias_atras=le.KILLS_RECAL_MAX_DIAS + 1))
        t = MatchupTotals(mean=30.4, sd=10.4, dist="normal", metric="kills")
        esperado = 1.0 - le._norm_cdf(31.5, 30.4, 10.4)
        self.assertAlmostEqual(prob_over_from_total(t, 31.5), esperado, places=12)

    def test_no_limite_ainda_vale(self):
        self.usar(_art(dias_atras=le.KILLS_RECAL_MAX_DIAS))
        self.assertIsNotNone(le._kills_recalibrado())

    def test_sem_arquivo_volta_a_curva(self):
        self.usar(None)
        t = MatchupTotals(mean=30.4, sd=10.4, dist="normal", metric="kills")
        esperado = 1.0 - le._norm_cdf(31.5, 30.4, 10.4)
        self.assertAlmostEqual(prob_over_from_total(t, 31.5), esperado, places=12)


class TestORoboRefazTodaSemana(unittest.TestCase):

    @so_no_lol("so vale para o LoL")
    def test_o_robo_semanal_gera_o_reajuste(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        yml = (raiz / ".github" / "workflows" / "reguas_pinnacle.yml").read_text(encoding="utf-8")
        self.assertIn("python tools/gerar_kills_recalibrado.py", yml)
        self.assertIn("kills_recalibrado.json", yml)


if __name__ == "__main__":
    unittest.main()
