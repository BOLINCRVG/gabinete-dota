# -*- coding: utf-8 -*-
"""Os tres calibradores sao MODELOS ANINHADOS — e a beta consegue nao fazer nada.

Este teste existe para travar as duas propriedades que motivaram a familia beta
(Kull, Silva Filho & Flach, AISTATS 2017) e que decidem o gate:

1. ANINHAMENTO: temperature (1 par.) subset Platt (2) subset beta (3). Como os
   tres sao ajustados por maxima verossimilhanca, no PROPRIO conjunto de ajuste
   a logloss tem que ser monotona nessa ordem. Se um dia isso quebrar, o
   ajustador esta com bug — nao e questao de dado.

2. IDENTIDADE: se o modelo ja sai calibrado, a beta consegue ficar parada
   (a=b=1, c=0 pertence a familia) e o Platt NAO consegue. E o argumento
   central de por que a beta e a aposta segura para LightGBM, que nao e rede
   neural superconfiante e pode ja sair perto de calibrado.
"""
from __future__ import annotations

import math
import random
import unittest

from mlcore.calibration import (
    BetaCalibrator,
    TemperatureCalibrator,
    fit_beta_from_raw_probs,
    fit_platt_corrigido,
    fit_temperature_from_raw_probs,
    platt_targets,
)
from mlcore.utils import sigmoid


def _logloss(cal, ps, ys) -> float:
    tot = 0.0
    for p, y in zip(ps, ys):
        q = min(max(cal.predict_proba(p), 1e-12), 1 - 1e-12)
        tot += -(math.log(q) if y == 1 else math.log(1.0 - q))
    return tot / len(ys)


class _Identidade:
    @staticmethod
    def predict_proba(p):
        return p


class CalibradoresAninhadosTest(unittest.TestCase):

    def _amostra(self, n, distorcao, semente=7):
        """Gera escores e rotulos onde a probabilidade VERDADEIRA e p_true e o
        escore observado sai distorcido por `distorcao` (no espaco do logit)."""
        rnd = random.Random(semente)
        ps, ys = [], []
        for _ in range(n):
            z = rnd.gauss(0.0, 1.6)
            p_true = sigmoid(z)
            ps.append(min(max(sigmoid(distorcao(z)), 1e-9), 1 - 1e-9))
            ys.append(1 if rnd.random() < p_true else 0)
        return ps, ys

    def test_alvos_regularizados_do_platt(self):
        alvo = platt_targets([1, 1, 1, 0])
        # y+ = (3+1)/(3+2) = 0.8 ; y- = 1/(1+2) = 0.333...
        self.assertAlmostEqual(alvo[0], 0.8, places=9)
        self.assertAlmostEqual(alvo[3], 1.0 / 3.0, places=9)
        self.assertTrue(all(0.0 < v < 1.0 for v in alvo),
                        "alvo regularizado nunca pode ser 0 nem 1 exatos")

    def test_aninhamento_da_logloss_no_conjunto_de_ajuste(self):
        # escore comprimido para o meio (z/2) — a distorcao que a literatura
        # descreve para arvores impulsionadas.
        ps, ys = self._amostra(4000, lambda z: z / 2.0)
        t = fit_temperature_from_raw_probs(ys, ps)
        p = fit_platt_corrigido(ys, ps)
        b = fit_beta_from_raw_probs(ys, ps)
        ll_t, ll_p, ll_b = _logloss(t, ps, ys), _logloss(p, ps, ys), _logloss(b, ps, ys)
        self.assertLessEqual(ll_p, ll_t + 1e-6, "Platt nao pode perder para temperature")
        self.assertLessEqual(ll_b, ll_p + 1e-6, "beta nao pode perder para Platt")

    def test_recalibrar_melhora_quando_ha_distorcao(self):
        ps, ys = self._amostra(4000, lambda z: z / 2.0)
        base = _logloss(_Identidade, ps, ys)
        for cal in (fit_temperature_from_raw_probs(ys, ps),
                    fit_platt_corrigido(ys, ps),
                    fit_beta_from_raw_probs(ys, ps)):
            self.assertLess(_logloss(cal, ps, ys), base)

    def test_beta_fica_parada_quando_ja_esta_calibrado(self):
        """O argumento central: com escore JA calibrado, a beta encosta na
        identidade. O Platt nao tem a identidade na familia dele."""
        ps, ys = self._amostra(6000, lambda z: z)   # sem distorcao
        b = fit_beta_from_raw_probs(ys, ps)
        base = _logloss(_Identidade, ps, ys)
        self.assertAlmostEqual(b.a, 1.0, delta=0.15)
        self.assertAlmostEqual(b.b, 1.0, delta=0.15)
        self.assertAlmostEqual(b.c, 0.0, delta=0.20)
        self.assertLess(_logloss(b, ps, ys), base + 0.002,
                        "beta nao pode PIORAR um escore que ja estava calibrado")

    def test_monotonicidade_garantida(self):
        """a, b >= 0 sempre: recalibracao tem que preservar a ordenacao, senao
        deixa de ser recalibracao e a decisao por logloss perde o sentido."""
        for semente in range(6):
            ps, ys = self._amostra(600, lambda z: -z, semente=semente)  # invertido de proposito
            b = fit_beta_from_raw_probs(ys, ps)
            t = fit_temperature_from_raw_probs(ys, ps)
            self.assertGreaterEqual(b.a, 0.0)
            self.assertGreaterEqual(b.b, 0.0)
            self.assertGreater(t.a, 0.0)

    def test_ida_e_volta_do_dicionario(self):
        for cal in (TemperatureCalibrator(a=1.7), BetaCalibrator(a=0.9, b=1.2, c=-0.3)):
            volta = type(cal).from_dict(cal.to_dict())
            self.assertEqual(cal.to_dict(), volta.to_dict())
            self.assertAlmostEqual(cal.predict_proba(0.62), volta.predict_proba(0.62))


if __name__ == "__main__":
    unittest.main()
