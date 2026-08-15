"""Tests for mlcore/fit_form_core.py — form blend helper."""
import math
import unittest

import numpy as np

from mlcore.fit_samples_core import Sample
from mlcore.fit_form_core import (
    _apply_form_blend_on_sample,
    _fit_logreg_numpy,
    _meta_recent_evidence_from_sample,
    _np_sigmoid,
    _predict_logreg_numpy,
)


def _make_sample(
    *,
    form_n_a: int = 10,
    form_n_b: int = 10,
    form_wins_a: int = 7,
    form_wins_b: int = 3,
    same_league: bool = True,
) -> Sample:
    return Sample(
        month="2025-03",
        gameid="test001",
        league="LCK",
        patch="14.1",
        split="Spring",
        macro="KR",
        side="blue",
        bo_bucket="bo1",
        phase="regular",
        y=1,
        p_team=0.6,
        p_players=0.55,
        n_team=500.0,
        n_players=300.0,
        form_n_a=form_n_a,
        form_n_b=form_n_b,
        form_wins_a=form_wins_a,
        form_wins_b=form_wins_b,
        same_league=same_league,
    )


class TestApplyFormBlendOnSample(unittest.TestCase):

    def test_disabled_returns_p_model(self):
        s = _make_sample()
        result = _apply_form_blend_on_sample(
            p_model=0.6,
            sample=s,
            enabled=False,
            max_weight=0.5,
            knee_games=6.0,
            beta_prior=2.0,
            signal_power=1.0,
        )
        self.assertAlmostEqual(result, 0.6, places=5)

    def test_returns_float(self):
        s = _make_sample()
        result = _apply_form_blend_on_sample(
            p_model=0.6, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertIsInstance(result, float)

    def test_result_strictly_between_0_and_1(self):
        s = _make_sample()
        result = _apply_form_blend_on_sample(
            p_model=0.6, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_result_is_finite(self):
        s = _make_sample()
        result = _apply_form_blend_on_sample(
            p_model=0.6, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertTrue(math.isfinite(result))

    def test_strong_form_a_shifts_up(self):
        s = _make_sample(form_n_a=10, form_n_b=10, form_wins_a=9, form_wins_b=1)
        result = _apply_form_blend_on_sample(
            p_model=0.5, sample=s, enabled=True,
            max_weight=0.75, knee_games=4.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertGreater(result, 0.5)

    def test_strong_form_b_shifts_down(self):
        s = _make_sample(form_n_a=10, form_n_b=10, form_wins_a=1, form_wins_b=9)
        result = _apply_form_blend_on_sample(
            p_model=0.5, sample=s, enabled=True,
            max_weight=0.75, knee_games=4.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertLess(result, 0.5)

    def test_zero_form_games_returns_p_model(self):
        s = _make_sample(form_n_a=0, form_n_b=0, form_wins_a=0, form_wins_b=0)
        result = _apply_form_blend_on_sample(
            p_model=0.65, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertAlmostEqual(result, 0.65, places=5)

    def test_equal_form_stays_near_p_model(self):
        s = _make_sample(form_n_a=10, form_n_b=10, form_wins_a=5, form_wins_b=5)
        result = _apply_form_blend_on_sample(
            p_model=0.6, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertAlmostEqual(result, 0.6, places=2)

    def test_high_max_weight_allows_larger_shift(self):
        s = _make_sample(form_n_a=20, form_n_b=20, form_wins_a=18, form_wins_b=2)
        r_low = _apply_form_blend_on_sample(
            p_model=0.5, sample=s, enabled=True,
            max_weight=0.1, knee_games=2.0, beta_prior=1.5, signal_power=1.0,
        )
        r_high = _apply_form_blend_on_sample(
            p_model=0.5, sample=s, enabled=True,
            max_weight=0.9, knee_games=2.0, beta_prior=1.5, signal_power=1.0,
        )
        self.assertGreater(r_high, r_low)


class TestMetaRecentEvidence(unittest.TestCase):
    """Sinal de evidência recente (0..1) que decide o peso do meta-blend."""

    def test_sem_jogos_recentes_da_zero(self):
        self.assertEqual(_meta_recent_evidence_from_sample(_make_sample(form_n_a=0)), 0.0)
        self.assertEqual(_meta_recent_evidence_from_sample(_make_sample(form_n_b=0)), 0.0)

    def test_formas_iguais_dao_sinal_fraco(self):
        s = _make_sample(form_wins_a=5, form_wins_b=5, form_n_a=10, form_n_b=10)
        self.assertAlmostEqual(_meta_recent_evidence_from_sample(s), 0.0, places=9)

    def test_formas_opostas_dao_sinal_forte(self):
        s = _make_sample(form_wins_a=10, form_wins_b=0, form_n_a=10, form_n_b=10)
        opostas = _meta_recent_evidence_from_sample(s)
        parecidas = _meta_recent_evidence_from_sample(
            _make_sample(form_wins_a=6, form_wins_b=4, form_n_a=10, form_n_b=10))
        self.assertGreater(opostas, parecidas)

    def test_amostra_maior_da_mais_confianca(self):
        pouco = _meta_recent_evidence_from_sample(
            _make_sample(form_wins_a=3, form_wins_b=0, form_n_a=3, form_n_b=3))
        muito = _meta_recent_evidence_from_sample(
            _make_sample(form_wins_a=20, form_wins_b=0, form_n_a=20, form_n_b=20))
        self.assertLess(pouco, muito)

    def test_sempre_entre_0_e_1(self):
        for wa, wb, n in ((0, 10, 10), (10, 0, 10), (1, 1, 2), (50, 0, 50)):
            s = _make_sample(form_wins_a=wa, form_wins_b=wb, form_n_a=n, form_n_b=n)
            v = _meta_recent_evidence_from_sample(s)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestLogRegNumpy(unittest.TestCase):
    """Regressão logística implementada à mão em numpy — usada no meta-blend.
    Sem teste, um sinal de gradiente invertido passaria despercebido."""

    @staticmethod
    def _dados_separaveis(n=200, seed=7):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)  # regra linear limpa
        return X, y

    def test_aprende_problema_separavel(self):
        X, y = self._dados_separaveis()
        m = _fit_logreg_numpy(X, y)
        self.assertNotIn("error", m)
        # melhor que moeda com folga: dados são linearmente separáveis
        self.assertLess(m["train_logloss"], 0.45)
        p = _predict_logreg_numpy(m, X)
        acc = float(np.mean((p > 0.5) == (y > 0.5)))
        self.assertGreater(acc, 0.9)

    def test_coeficiente_aponta_na_direcao_certa(self):
        X, y = self._dados_separaveis()
        m = _fit_logreg_numpy(X, y)
        # y cresce com x0: o peso de x0 tem que ser positivo e dominante
        self.assertGreater(m["coef"][0], 0.0)
        self.assertGreater(m["coef"][0], abs(m["coef"][1]) * 0.5)

    def test_poucos_dados_recusa_em_vez_de_fingir(self):
        X = np.zeros((10, 2))
        y = np.zeros(10)
        self.assertEqual(_fit_logreg_numpy(X, y).get("error"), "invalid_train_data")

    def test_l2_maior_encolhe_os_pesos(self):
        X, y = self._dados_separaveis()
        livre = _fit_logreg_numpy(X, y, l2=0.0)
        preso = _fit_logreg_numpy(X, y, l2=50.0)
        self.assertLess(np.abs(preso["coef"]).sum(), np.abs(livre["coef"]).sum())

    def test_feature_constante_nao_gera_divisao_por_zero(self):
        X, y = self._dados_separaveis()
        X = np.hstack([X, np.full((len(X), 1), 3.0)])  # coluna constante
        m = _fit_logreg_numpy(X, y)
        self.assertNotIn("error", m)
        self.assertTrue(all(math.isfinite(c) for c in m["coef"]))

    def test_predicao_com_modelo_invalido_devolve_vazio(self):
        """Modelo quebrado -> vazio (recusa), nunca uma probabilidade inventada."""
        X = np.zeros((5, 2))
        self.assertEqual(len(_predict_logreg_numpy({}, X)), 0)              # sem coef
        self.assertEqual(len(_predict_logreg_numpy({"coef": [1.0]}, X)), 0)  # dimensão errada
        self.assertEqual(len(_predict_logreg_numpy({"coef": "x"}, X)), 0)    # tipo errado

    def test_sigmoid_e_estavel_em_z_extremo(self):
        out = _np_sigmoid(np.array([-1e6, 0.0, 1e6]))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(float(out[1]), 0.5, places=12)
        self.assertLess(float(out[0]), 1e-12)
        self.assertGreater(float(out[2]), 1.0 - 1e-12)


if __name__ == "__main__":
    unittest.main()
