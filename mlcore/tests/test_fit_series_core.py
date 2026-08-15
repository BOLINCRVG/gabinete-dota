"""Tests for mlcore/fit_series_core.py — series probability helper.

Ampliado em 19/07 (campanha de cobertura): o módulo estava com 6% de linhas
cobertas — `_series_prob_with_gamma` já tinha testes, mas os PARÂMETROS que
alimentam o pricing de série (defaults por BO, grade de busca, merge) não
tinham nenhum. Erro nesses valores vira preço errado em BO3/BO5.
"""
import math
import unittest

from mlcore.fit_series_core import (
    _merge_series_v5_params,
    _patch_status_from_seen,
    _series_dplite_candidates,
    _series_prob_with_gamma,
    _series_v5_default_extras,
    _series_v5_extra_grid,
)


class TestSeriesProbWithGamma(unittest.TestCase):

    def test_returns_float(self):
        result = _series_prob_with_gamma(0.6, 3, 1.0)
        self.assertIsInstance(result, float)

    def test_result_strictly_between_0_and_1(self):
        result = _series_prob_with_gamma(0.6, 3, 1.0)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_fifty_fifty_stays_near_half(self):
        result = _series_prob_with_gamma(0.5, 3, 1.0)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_higher_p_map_gives_higher_series_prob(self):
        r1 = _series_prob_with_gamma(0.6, 3, 1.0)
        r2 = _series_prob_with_gamma(0.4, 3, 1.0)
        self.assertGreater(r1, r2)

    def test_gamma_gt_1_amplifies_advantage(self):
        r_base = _series_prob_with_gamma(0.6, 3, 1.0)
        r_amp = _series_prob_with_gamma(0.6, 3, 2.0)
        self.assertGreater(r_amp, r_base)

    def test_gamma_lt_1_shrinks_advantage(self):
        r_base = _series_prob_with_gamma(0.6, 3, 1.0)
        r_shrunk = _series_prob_with_gamma(0.6, 3, 0.5)
        self.assertLess(r_shrunk, r_base)

    def test_bo5_higher_series_prob_than_bo3_for_favorite(self):
        r3 = _series_prob_with_gamma(0.6, 3, 1.0)
        r5 = _series_prob_with_gamma(0.6, 5, 1.0)
        self.assertGreater(r5, r3)

    def test_result_is_finite(self):
        result = _series_prob_with_gamma(0.7, 5, 1.0)
        self.assertTrue(math.isfinite(result))

    def test_extreme_p_map_clipped(self):
        result = _series_prob_with_gamma(0.999, 3, 1.0)
        self.assertLess(result, 1.0)
        self.assertGreater(result, 0.9)

    def test_low_p_map_clipped(self):
        result = _series_prob_with_gamma(0.001, 3, 1.0)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.1)

    def test_very_low_gamma_clamped_to_min(self):
        # gamma is clamped to max(0.25, ...) so very low values stay bounded
        result = _series_prob_with_gamma(0.6, 3, 0.01)
        self.assertTrue(math.isfinite(result))
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_symmetry_at_half(self):
        # p(0.5) should stay ≈0.5 regardless of gamma
        result = _series_prob_with_gamma(0.5, 5, 2.0)
        self.assertAlmostEqual(result, 0.5, places=4)


class TestPatchStatusFromSeen(unittest.TestCase):
    """Semáforo de amostra do patch: decide se refita, avisa ou está ok."""

    def test_faixas_de_decisao(self):
        self.assertEqual(_patch_status_from_seen(0), "REFIT")
        self.assertEqual(_patch_status_from_seen(19), "REFIT")
        self.assertEqual(_patch_status_from_seen(20), "WARN")
        self.assertEqual(_patch_status_from_seen(59), "WARN")
        self.assertEqual(_patch_status_from_seen(60), "OK")
        self.assertEqual(_patch_status_from_seen(5000), "OK")

    def test_negativo_tratado_como_zero(self):
        self.assertEqual(_patch_status_from_seen(-99), "REFIT")


class TestParametrosSerieV5(unittest.TestCase):
    """Defaults/grades que alimentam o pricing de série por BO."""

    def test_bo5_corrige_mais_que_bo3(self):
        # BO5 tem mais mapas => mais reversão de lado no decider e mais
        # correlação entre mapas do que BO3.
        b3 = _series_v5_default_extras(3)
        b5 = _series_v5_default_extras(5)
        self.assertGreater(b5["decider_side_revert"], b3["decider_side_revert"])
        self.assertGreater(b5["series_corr_sigma"], b3["series_corr_sigma"])

    def test_defaults_sao_positivos_e_finitos(self):
        for bo in (1, 3, 5):
            for k, v in _series_v5_default_extras(bo).items():
                self.assertTrue(math.isfinite(v), f"{k} não finito em bo{bo}")
                self.assertGreater(v, 0.0, f"{k} deveria ser > 0")

    def test_grade_de_busca_nao_e_vazia_e_tem_as_chaves(self):
        for bo in (3, 5):
            for speed in ("fast", "full"):
                grid = _series_v5_extra_grid(bo, speed)
                self.assertTrue(grid, f"grade vazia em bo{bo}/{speed}")
                for item in grid:
                    self.assertIn("decider_side_revert", item)
                    self.assertIn("series_corr_sigma", item)

    def test_grade_fast_nao_e_maior_que_full(self):
        """'fast' existe para custar menos: nunca pode ter mais candidatos."""
        for bo in (3, 5):
            self.assertLessEqual(len(_series_v5_extra_grid(bo, "fast")),
                                 len(_series_v5_extra_grid(bo, "full")))
            self.assertLessEqual(len(_series_dplite_candidates(bo, "fast")),
                                 len(_series_dplite_candidates(bo, "full")))

    def test_candidatos_dplite_tem_valores_validos(self):
        for bo in (3, 5):
            cands = _series_dplite_candidates(bo, "fast")
            self.assertTrue(cands)
            for c in cands:
                for k, v in c.items():
                    self.assertTrue(math.isfinite(float(v)), f"{k}={v} não finito")

    def test_merge_parte_dos_defaults_do_bo(self):
        out = _merge_series_v5_params(5, None, None)
        d5 = _series_v5_default_extras(5)
        self.assertEqual(out["decider_side_revert"], d5["decider_side_revert"])

    def test_merge_extra_sobrescreve_base(self):
        out = _merge_series_v5_params(3, {"decider_side_revert": 0.10},
                                      {"decider_side_revert": 0.30})
        self.assertAlmostEqual(out["decider_side_revert"], 0.30)

    def test_merge_ignora_entrada_invalida(self):
        # base/extra não-dict não podem quebrar o pricing
        out = _merge_series_v5_params(3, "lixo", 42)  # type: ignore[arg-type]
        self.assertIn("decider_side_revert", out)

    def test_merge_nao_muta_os_dicts_de_entrada(self):
        base = {"decider_side_revert": 0.10}
        extra = {"series_corr_sigma": 0.20}
        _merge_series_v5_params(3, base, extra)
        self.assertEqual(base, {"decider_side_revert": 0.10})
        self.assertEqual(extra, {"series_corr_sigma": 0.20})


if __name__ == "__main__":
    unittest.main()
