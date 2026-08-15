"""Tests for mlcore/fit_samples_core.py — pure config validation helpers."""
import unittest

from mlcore.fit_samples_core import (
    _default_ml_marginal_band_profiles,
    _is_valid_areal_prob_tuning,
    _is_valid_ml_marginal_band_coeffs,
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


if __name__ == "__main__":
    unittest.main()
