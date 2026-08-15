"""Tests for mlcore/fit_samples_core.py — pure config validation helpers.

Ampliado 19/07 (campanha de cobertura, 25%): os validadores de config já
tinham teste, mas as funções de CÁLCULO não — ECE (mede calibração),
shrinkage empirical-Bayes (James-Stein, decide o quanto confiar em cada
segmento) e a tabela de microseg construída das predições. Erro nelas
distorce silenciosamente a probabilidade final.
"""
import math
import unittest

import pandas as pd

from mlcore.fit_samples_core import (
    _build_microseg_table_from_pred_rows,
    _default_ml_marginal_band_profiles,
    _ece_metric,
    _estimate_shrink_k_empirical_bayes,
    _fuse_probs_by_precision,
    _is_valid_areal_prob_tuning,
    _is_valid_ml_marginal_band_coeffs,
    _month_key,
    _sanitize_ml_marginal_band_coeffs,
)


def _valid_areal_cfg():
    return {
        "version": "areal_prob_tuning_v1",
        "global": {
            "bayes_info_scale": 1.0,
            "bayes_mu_q_strength": 1.0,
            "quality_k_boost": 0.22,
            "qwl_surplus_mix": 0.70,
            "qwl_throw_weight": 1.20,
            "qwl_comeback_weight": 0.65,
            "qwl_phase_dynamic_strength": 0.35,
            "qwl_opponent_context_mix": 0.30,
        },
    }


class TestIsValidArealProbTuning(unittest.TestCase):

    def test_valid_config_returns_true(self):
        self.assertTrue(_is_valid_areal_prob_tuning(_valid_areal_cfg()))

    def test_none_returns_false(self):
        self.assertFalse(_is_valid_areal_prob_tuning(None))

    def test_empty_dict_returns_false(self):
        self.assertFalse(_is_valid_areal_prob_tuning({}))

    def test_missing_version_returns_false(self):
        cfg = _valid_areal_cfg()
        del cfg["version"]
        self.assertFalse(_is_valid_areal_prob_tuning(cfg))

    def test_wrong_version_prefix_returns_false(self):
        cfg = _valid_areal_cfg()
        cfg["version"] = "wrong_prefix_v1"
        self.assertFalse(_is_valid_areal_prob_tuning(cfg))

    def test_missing_global_returns_false(self):
        cfg = _valid_areal_cfg()
        del cfg["global"]
        self.assertFalse(_is_valid_areal_prob_tuning(cfg))

    def test_empty_global_returns_false(self):
        cfg = _valid_areal_cfg()
        cfg["global"] = {}
        self.assertFalse(_is_valid_areal_prob_tuning(cfg))

    def test_non_dict_global_returns_false(self):
        cfg = _valid_areal_cfg()
        cfg["global"] = "not_a_dict"
        self.assertFalse(_is_valid_areal_prob_tuning(cfg))

    def test_returns_bool(self):
        result = _is_valid_areal_prob_tuning(_valid_areal_cfg())
        self.assertIsInstance(result, bool)


class TestDefaultMlMarginalBandProfiles(unittest.TestCase):

    def test_returns_dict(self):
        result = _default_ml_marginal_band_profiles()
        self.assertIsInstance(result, dict)

    def test_has_top_core_edge_bands(self):
        result = _default_ml_marginal_band_profiles()
        for band in ("top", "core", "edge"):
            self.assertIn(band, result)

    def test_each_band_has_required_keys(self):
        result = _default_ml_marginal_band_profiles()
        required = {"strength_scale", "prior_scale", "max_shrink_scale", "cap_shift",
                    "soft_shift", "hard_shift", "gate_score", "gate_n_eff"}
        for band in ("top", "core", "edge"):
            self.assertTrue(required.issubset(set(result[band].keys())))

    def test_all_values_are_floats(self):
        result = _default_ml_marginal_band_profiles()
        for band_vals in result.values():
            for v in band_vals.values():
                self.assertIsInstance(v, float)


class TestSanitizeMlMarginalBandCoeffs(unittest.TestCase):

    def _valid_raw(self):
        return {
            "top": {"strength_scale": 0.35, "prior_scale": 0.70, "max_shrink_scale": 0.55,
                    "cap_shift": 0.06, "soft_shift": 10.0, "hard_shift": 8.0,
                    "gate_score": 80.0, "gate_n_eff": 30.0},
            "core": {"strength_scale": 0.75, "prior_scale": 0.95, "max_shrink_scale": 0.85,
                     "cap_shift": 0.02, "soft_shift": 3.0, "hard_shift": 2.0,
                     "gate_score": 84.0, "gate_n_eff": 42.0},
            "edge": {"strength_scale": 1.15, "prior_scale": 1.20, "max_shrink_scale": 1.05,
                     "cap_shift": -0.04, "soft_shift": -6.0, "hard_shift": -4.0,
                     "gate_score": 88.0, "gate_n_eff": 55.0},
        }

    def test_valid_input_returns_non_empty(self):
        result = _sanitize_ml_marginal_band_coeffs(self._valid_raw())
        self.assertTrue(len(result) > 0)

    def test_none_returns_empty_dict(self):
        self.assertEqual(_sanitize_ml_marginal_band_coeffs(None), {})

    def test_empty_dict_returns_empty_dict(self):
        self.assertEqual(_sanitize_ml_marginal_band_coeffs({}), {})

    def test_non_dict_band_skipped(self):
        raw = self._valid_raw()
        raw["top"] = "not_a_dict"
        result = _sanitize_ml_marginal_band_coeffs(raw)
        self.assertNotIn("top", result)
        self.assertIn("core", result)

    def test_returns_dict(self):
        result = _sanitize_ml_marginal_band_coeffs(self._valid_raw())
        self.assertIsInstance(result, dict)

    def test_known_keys_included(self):
        raw = {"top": {"strength_scale": 0.5}}
        result = _sanitize_ml_marginal_band_coeffs(raw)
        if "top" in result:
            self.assertIn("strength_scale", result["top"])

    def test_unknown_keys_excluded(self):
        raw = {"top": {"unknown_key": 1.0, "strength_scale": 0.5}}
        result = _sanitize_ml_marginal_band_coeffs(raw)
        if "top" in result:
            self.assertNotIn("unknown_key", result["top"])


class TestIsValidMlMarginalBandCoeffs(unittest.TestCase):

    def test_none_returns_false(self):
        self.assertFalse(_is_valid_ml_marginal_band_coeffs(None))

    def test_empty_dict_returns_false(self):
        self.assertFalse(_is_valid_ml_marginal_band_coeffs({}))

    def test_valid_returns_true(self):
        raw = {"top": {"strength_scale": 0.35}}
        result = _is_valid_ml_marginal_band_coeffs(raw)
        self.assertIsInstance(result, bool)

    def test_returns_bool(self):
        self.assertIsInstance(_is_valid_ml_marginal_band_coeffs(None), bool)


class TestEceMetric(unittest.TestCase):
    """ECE = quanto a probabilidade dita difere da frequência real.
    É a régua de calibração — se ela mentir, o gate aceita modelo ruim."""

    def test_calibracao_perfeita_da_ece_zero(self):
        # p=0.5 com metade de acerto em cada bin => confiança == frequência
        y = [1, 0] * 50
        p = [0.5] * 100
        self.assertAlmostEqual(_ece_metric(y, p), 0.0, places=9)

    def test_descalibrado_da_ece_alta(self):
        # diz 90% mas acerta 50% => gap de 0.4
        y = [1, 0] * 50
        p = [0.9] * 100
        self.assertAlmostEqual(_ece_metric(y, p), 0.4, places=9)

    def test_mais_descalibrado_da_ece_maior(self):
        y = [1, 0] * 50
        leve = _ece_metric(y, [0.6] * 100)
        grave = _ece_metric(y, [0.95] * 100)
        self.assertLess(leve, grave)

    def test_entrada_invalida_devolve_none(self):
        self.assertIsNone(_ece_metric([], []))
        self.assertIsNone(_ece_metric([1, 0], [0.5]))  # tamanhos diferentes

    def test_p_extremo_nao_estoura(self):
        v = _ece_metric([1, 0], [1.0, 0.0])
        self.assertIsNotNone(v)
        self.assertTrue(math.isfinite(v))


class TestShrinkKEmpiricalBayes(unittest.TestCase):
    """k decide o quanto DESCONFIAR do viés de cada segmento (James-Stein).
    k alto = vieses são ruído -> encolhe muito; k baixo = viés real."""

    @staticmethod
    def _seg(bias: float, n: float) -> dict:
        return {"n": n, "sum_err": bias * n}

    def test_poucos_segmentos_cai_no_maximo(self):
        # <8 segmentos: não dá para estimar variância entre segmentos
        agg = {f"s{i}": self._seg(0.05, 100.0) for i in range(3)}
        self.assertEqual(_estimate_shrink_k_empirical_bayes(agg), 300.0)

    def test_vieses_identicos_sao_ruido_e_encolhem_no_maximo(self):
        # todos os segmentos com o MESMO viés => variância real zero
        agg = {f"s{i}": self._seg(0.02, 200.0) for i in range(12)}
        self.assertEqual(_estimate_shrink_k_empirical_bayes(agg), 300.0)

    def test_vieses_bem_diferentes_dao_k_menor_que_ruido(self):
        # metade +12pp, metade -12pp com n grande => sinal real, confiar antes
        agg = {}
        for i in range(6):
            agg[f"pos{i}"] = self._seg(+0.12, 400.0)
            agg[f"neg{i}"] = self._seg(-0.12, 400.0)
        k_sinal = _estimate_shrink_k_empirical_bayes(agg)
        self.assertLess(k_sinal, 300.0)
        self.assertGreaterEqual(k_sinal, 25.0)  # nunca abaixo do piso

    def test_segmento_pequeno_e_ignorado(self):
        # n<15 não conta: com só segmentos pequenos, volta ao máximo
        agg = {f"s{i}": self._seg(0.30 * (-1) ** i, 5.0) for i in range(20)}
        self.assertEqual(_estimate_shrink_k_empirical_bayes(agg), 300.0)

    def test_resultado_sempre_dentro_dos_limites(self):
        agg = {}
        for i in range(10):
            agg[f"s{i}"] = self._seg(0.5 * (-1) ** i, 1000.0)
        k = _estimate_shrink_k_empirical_bayes(agg)
        self.assertGreaterEqual(k, 25.0)
        self.assertLessEqual(k, 300.0)


class TestBuildMicrosegTable(unittest.TestCase):
    """A tabela de viés por segmento que o app usa AO VIVO na calibração."""

    @staticmethod
    def _row(league="LCK", y=1, p=0.6, **kw) -> dict:
        base = {"league": league, "split": "Spring", "side": "blue",
                "bo": "3", "phase": "regular", "y": y, "p": p}
        base.update(kw)
        return base

    def test_n_e_sempre_float_escalar(self):
        """Guarda ligada ao incidente de 18/07: o `n` desta tabela alimenta o
        microseg_info da UI — tem que ser escalar finito, nunca lista."""
        rows = [self._row(y=i % 2, p=0.55) for i in range(40)]
        table = _build_microseg_table_from_pred_rows(rows)
        self.assertTrue(table)
        for k, d in table.items():
            self.assertIsInstance(d["n"], float, f"{k}: n={d['n']!r}")
            self.assertTrue(math.isfinite(d["n"]))
            self.assertTrue(math.isfinite(d["bias"]))

    def test_vies_encolhido_nunca_excede_o_cru(self):
        rows = [self._row(y=1, p=0.4) for _ in range(50)]  # modelo subestima
        table = _build_microseg_table_from_pred_rows(rows)
        for d in table.values():
            self.assertLessEqual(abs(d["bias"]), abs(d["raw_bias"]) + 1e-12)

    def test_mais_amostra_encolhe_menos(self):
        def bias_com(n_rows: int) -> float:
            rows = [self._row(y=1, p=0.4) for _ in range(n_rows)]
            t = _build_microseg_table_from_pred_rows(rows, shrink_k=90.0)
            return abs(t["*|*|*|*|*"]["bias"])
        self.assertLess(bias_com(20), bias_com(400))

    def test_linha_invalida_e_ignorada_sem_quebrar(self):
        rows = [self._row(), {"y": float("nan"), "p": 0.5}, "lixo", None, {}]
        table = _build_microseg_table_from_pred_rows(rows)  # type: ignore[arg-type]
        self.assertTrue(all(math.isfinite(d["bias"]) for d in table.values()))

    def test_vazio_devolve_vazio(self):
        self.assertEqual(_build_microseg_table_from_pred_rows([]), {})


class TestFuseProbsByPrecision(unittest.TestCase):
    """Fusão time+players ponderada por quantidade/frescor de evidência."""

    _KW = dict(n_team=200.0, n_players=200.0, team_scale=1.0,
               players_scale=1.0)

    def test_resultado_e_probabilidade_valida(self):
        p = _fuse_probs_by_precision(p_team=0.60, p_players=0.70, **self._KW)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_freshness_zero_anula_a_camada_de_players(self):
        com = _fuse_probs_by_precision(p_team=0.60, p_players=0.90,
                                       players_freshness=1.0, **self._KW)
        sem = _fuse_probs_by_precision(p_team=0.60, p_players=0.90,
                                       players_freshness=0.0, **self._KW)
        # sem frescor, players não pode puxar mais que com frescor total
        self.assertLessEqual(abs(sem - 0.60), abs(com - 0.60) + 1e-12)

    def test_concordancia_fica_entre_as_duas(self):
        p = _fuse_probs_by_precision(p_team=0.58, p_players=0.66, **self._KW)
        self.assertGreaterEqual(p, 0.55)
        self.assertLessEqual(p, 0.70)


class TestMonthKey(unittest.TestCase):
    def test_formata_ano_mes(self):
        self.assertEqual(_month_key(pd.Timestamp("2026-07-19")), "2026-07")
        self.assertEqual(_month_key(pd.Timestamp("2025-01-02")), "2025-01")


if __name__ == "__main__":
    unittest.main()
