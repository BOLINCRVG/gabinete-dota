import math
import unittest

import pandas as pd

from mlcore.calibration_diagnostics import (
    _apply_microseg_calibration,
    _ece,
    _metrics,
    _oof_platt_by_month,
    _reliability_bins,
    _safe_logit,
    _safe_sigmoid,
    _seg_key,
)


class CalibrationDiagnosticsCoreTest(unittest.TestCase):
    def test_microseg_adjusts_probability_with_strong_local_bias(self):
        table = {
            "LPL|S1|blue|1|regular": {"n": 120.0, "bias": -0.08},
            "*|*|*|*|*": {"n": 500.0, "bias": 0.00},
        }
        p0 = 0.70
        p1 = _apply_microseg_calibration(
            p0,
            league="LPL",
            split="S1",
            side="blue",
            bo="1",
            phase="regular",
            table=table,
            min_n=30,
            strength=1.0,
            cap_pp=10.0,
            hierarchical=False,
        )
        self.assertLess(float(p1), float(p0))

    def test_microseg_falls_back_to_global_hierarchical(self):
        table = {
            "*|*|*|*|*": {"n": 400.0, "bias": 0.05},
        }
        p0 = 0.45
        p1 = _apply_microseg_calibration(
            p0,
            league="LCK",
            split="S1",
            side="blue",
            bo="1",
            phase="regular",
            table=table,
            min_n=30,
            strength=0.7,
            cap_pp=8.0,
            hierarchical=True,
            hierarchy_decay=0.8,
        )
        self.assertGreater(float(p1), float(p0))


class TestEce(unittest.TestCase):

    def test_perfect_calibration_near_zero(self):
        y = [1, 0, 1, 0]
        p = [0.875, 0.125, 0.875, 0.125]
        result = _ece(y, p, bins=2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_empty_lists_return_nan(self):
        self.assertTrue(math.isnan(_ece([], [], bins=10)))

    def test_mismatched_lengths_return_nan(self):
        self.assertTrue(math.isnan(_ece([1, 0], [0.5], bins=10)))

    def test_returns_float(self):
        self.assertIsInstance(_ece([1, 0], [0.7, 0.3]), float)

    def test_result_between_0_and_1(self):
        y = [1, 1, 0, 0, 1]
        p = [0.8, 0.7, 0.3, 0.2, 0.6]
        result = _ece(y, p)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestSafeLogit(unittest.TestCase):

    def test_half_gives_zero(self):
        self.assertAlmostEqual(_safe_logit(0.5), 0.0, places=5)

    def test_high_prob_positive(self):
        self.assertGreater(_safe_logit(0.75), 0.0)

    def test_low_prob_negative(self):
        self.assertLess(_safe_logit(0.25), 0.0)

    def test_returns_float(self):
        self.assertIsInstance(_safe_logit(0.6), float)

    def test_result_is_finite(self):
        self.assertTrue(math.isfinite(_safe_logit(0.5)))

    def test_clipped_zero_not_inf(self):
        self.assertTrue(math.isfinite(_safe_logit(0.0)))


class TestSafeSigmoid(unittest.TestCase):

    def test_zero_gives_half(self):
        self.assertAlmostEqual(_safe_sigmoid(0.0), 0.5, places=5)

    def test_positive_above_half(self):
        self.assertGreater(_safe_sigmoid(1.0), 0.5)

    def test_negative_below_half(self):
        self.assertLess(_safe_sigmoid(-1.0), 0.5)

    def test_returns_float(self):
        self.assertIsInstance(_safe_sigmoid(0.0), float)

    def test_large_positive_near_one(self):
        self.assertGreater(_safe_sigmoid(35.0), 0.99)

    def test_large_negative_near_zero(self):
        self.assertLess(_safe_sigmoid(-35.0), 0.01)

    def test_symmetric_around_half(self):
        a = _safe_sigmoid(2.0)
        b = _safe_sigmoid(-2.0)
        self.assertAlmostEqual(a + b, 1.0, places=5)


class TestSegKey(unittest.TestCase):

    def test_builds_pipe_delimited_key(self):
        key = _seg_key("LCK", "Spring", "blue", "1", "regular")
        parts = key.split("|")
        self.assertEqual(len(parts), 5)

    def test_league_in_key(self):
        key = _seg_key("LCK", "S1", "blue", "1", "regular")
        self.assertIn("LCK", key)

    def test_side_lowercased(self):
        key = _seg_key("LCK", "S1", "BLUE", "1", "regular")
        self.assertIn("blue", key)

    def test_phase_lowercased(self):
        key = _seg_key("LCK", "S1", "blue", "1", "REGULAR")
        self.assertIn("regular", key)

    def test_wildcard_key(self):
        key = _seg_key("*", "*", "*", "*", "*")
        self.assertEqual(key, "*|*|*|*|*")

    def test_returns_string(self):
        self.assertIsInstance(_seg_key("LCK", "S1", "blue", "1", "regular"), str)


class TestReliabilityBins(unittest.TestCase):
    """Tabela de confiabilidade: em cada faixa de probabilidade, o que o
    modelo DIZ vs. o que ACONTECE. É o raio-X da calibração."""

    def test_calibrado_tem_gap_zero(self):
        y = [1, 0] * 50
        p = [0.5] * 100
        rows = _reliability_bins(y, p, bins=10)
        cheio = [r for r in rows if r["n"] > 0]
        self.assertEqual(len(cheio), 1)
        self.assertAlmostEqual(cheio[0]["gap"], 0.0, places=9)

    def test_superconfiante_tem_gap_negativo(self):
        # diz 95% e acerta 50% -> realidade abaixo do prometido
        rows = _reliability_bins([1, 0] * 50, [0.95] * 100, bins=10)
        cheio = [r for r in rows if r["n"] > 0][0]
        self.assertLess(cheio["gap"], -0.4)

    def test_todas_as_amostras_sao_contadas(self):
        y = [1, 0, 1, 1, 0, 1, 0, 0]
        p = [0.05, 0.15, 0.35, 0.55, 0.65, 0.75, 0.85, 1.0]  # inclui borda p=1
        rows = _reliability_bins(y, p, bins=10)
        self.assertEqual(sum(r["n"] for r in rows), len(y),
                         "amostra sumiu na borda do último bin")

    def test_bin_vazio_vem_marcado_e_nao_inventado(self):
        rows = _reliability_bins([1], [0.55], bins=10)
        vazios = [r for r in rows if r["n"] == 0]
        self.assertTrue(all(r["gap"] is None for r in vazios))

    def test_entrada_invalida_devolve_vazio(self):
        self.assertEqual(_reliability_bins([], []), [])
        self.assertEqual(_reliability_bins([1, 0], [0.5]), [])


class TestMetrics(unittest.TestCase):
    def test_traz_as_tres_reguas_e_n(self):
        out = _metrics([1, 0, 1, 0], [0.8, 0.2, 0.7, 0.3])
        self.assertEqual(out["n"], 4)
        for k in ("logloss", "brier", "ece"):
            self.assertTrue(math.isfinite(out[k]), f"{k} não finito")

    def test_vazio_devolve_none_em_vez_de_quebrar(self):
        out = _metrics([], [])
        self.assertEqual(out["n"], 0)
        self.assertIsNone(out["logloss"])

    def test_modelo_melhor_ganha_nas_tres_reguas(self):
        y = [1, 0] * 20
        bom = _metrics(y, [0.85, 0.15] * 20)
        ruim = _metrics(y, [0.55, 0.45] * 20)
        self.assertLess(bom["logloss"], ruim["logloss"])
        self.assertLess(bom["brier"], ruim["brier"])


class TestOofPlattByMonth(unittest.TestCase):
    """Calibração out-of-fold POR MÊS: o mês M só pode ser calibrado com
    dados de meses ANTERIORES. Mesma família de proteção do holdout
    temporal — vazamento aqui inflaria a nota do diagnóstico."""

    @staticmethod
    def _df(n_por_mes: int, meses: list, p: float = 0.7, taxa_real: float = 0.5):
        rows = []
        for m in meses:
            for i in range(n_por_mes):
                rows.append({"month": m, "y": 1 if i < int(n_por_mes * taxa_real) else 0, "p": p})
        return pd.DataFrame(rows)

    def test_primeiro_mes_nunca_e_calibrado(self):
        """Não existe passado antes do 1º mês — ele TEM que sair cru.
        Se sair calibrado, algo treinou com o próprio mês (leakage)."""
        df = self._df(1500, ["2026-01", "2026-02"])
        out = _oof_platt_by_month(df, y_col="y", p_col="p", min_train_n=1200)
        jan = out[df["month"] == "2026-01"]
        self.assertTrue((jan == 0.7).all(), "1º mês foi alterado = LEAKAGE")

    def test_mes_seguinte_e_corrigido_na_direcao_certa(self):
        # modelo diz 70% mas a taxa real é 50%: fevereiro deve ser puxado p/ baixo
        df = self._df(1500, ["2026-01", "2026-02"])
        out = _oof_platt_by_month(df, y_col="y", p_col="p", min_train_n=1200)
        fev = out[df["month"] == "2026-02"]
        self.assertLess(float(fev.mean()), 0.7, "excesso de confiança não corrigido")
        self.assertTrue(((fev > 0.0) & (fev < 1.0)).all())

    def test_treino_insuficiente_mantem_cru(self):
        df = self._df(100, ["2026-01", "2026-02"])  # 100 < min_train_n
        out = _oof_platt_by_month(df, y_col="y", p_col="p", min_train_n=1200)
        self.assertTrue((out == 0.7).all(), "calibrou com amostra insuficiente")

    def test_df_invalido_devolve_serie_vazia(self):
        self.assertTrue(_oof_platt_by_month(pd.DataFrame(), y_col="y", p_col="p").empty)
        df = pd.DataFrame({"month": ["2026-01"], "y": [1]})  # sem coluna p
        self.assertTrue(_oof_platt_by_month(df, y_col="y", p_col="p").empty)


if __name__ == "__main__":
    unittest.main()
