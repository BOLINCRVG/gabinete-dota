import unittest

from app_core.resumo_guardrail_core import build_resumo_guardrail_cfg, extract_resumo_operational_confidence


class TestExtractResumoOperationalConfidence(unittest.TestCase):

    def test_handles_mixed_scales(self):
        score, marginal, layer = extract_resumo_operational_confidence(
            {
                "ml_trace": {
                    "marginal_trust_guard": {"confidence_score": 66.0},
                    "layer_uncertainty_live": {"confidence_score": 0.42},
                }
            }
        )
        self.assertEqual(marginal, 66.0)
        self.assertEqual(layer, 42.0)
        self.assertEqual(score, 54.0)

    def test_none_input_returns_nones(self):
        score, marginal, layer = extract_resumo_operational_confidence(None)
        self.assertIsNone(score)
        self.assertIsNone(marginal)
        self.assertIsNone(layer)

    def test_empty_trace_returns_nones(self):
        score, marginal, layer = extract_resumo_operational_confidence({})
        self.assertIsNone(score)
        self.assertIsNone(marginal)
        self.assertIsNone(layer)

    def test_only_marginal_present(self):
        score, marginal, layer = extract_resumo_operational_confidence(
            {"ml_trace": {"marginal_trust_guard": {"confidence_score": 70.0}}}
        )
        self.assertEqual(marginal, 70.0)
        self.assertIsNone(layer)
        self.assertEqual(score, 70.0)

    def test_only_layer_present_rescaled(self):
        score, marginal, layer = extract_resumo_operational_confidence(
            {"ml_trace": {"layer_uncertainty_live": {"confidence_score": 0.80}}}
        )
        self.assertIsNone(marginal)
        self.assertEqual(layer, 80.0)
        self.assertEqual(score, 80.0)

    def test_layer_already_in_100_scale_kept(self):
        _score, _marginal, layer = extract_resumo_operational_confidence(
            {"ml_trace": {"layer_uncertainty_live": {"confidence_score": 80.0}}}
        )
        self.assertEqual(layer, 80.0)

    def test_returns_tuple_of_three(self):
        result = extract_resumo_operational_confidence({})
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)


class TestBuildResumoGuardrailCfg(unittest.TestCase):

    def test_defaults_used_for_empty_state(self):
        cfg = build_resumo_guardrail_cfg({}, ml_trace_last=None)
        self.assertEqual(cfg["profile"], "Equilibrado")
        self.assertEqual(cfg["strength_major"], 1.0)
        self.assertEqual(cfg["operational_confidence_weight"], 0.25)
        self.assertIsNone(cfg["operational_confidence_score"])

    def test_state_values_used(self):
        state = {
            "resumo_guardrail_profile": "Conservador",
            "resumo_guardrail_strength_major": 0.8,
            "resumo_operational_confidence_weight": 0.5,
        }
        cfg = build_resumo_guardrail_cfg(state, ml_trace_last=None)
        self.assertEqual(cfg["profile"], "Conservador")
        self.assertAlmostEqual(cfg["strength_major"], 0.8)
        self.assertAlmostEqual(cfg["operational_confidence_weight"], 0.5)

    def test_confidence_populated_from_trace(self):
        trace = {
            "ml_trace": {
                "marginal_trust_guard": {"confidence_score": 75.0},
                "layer_uncertainty_live": {"confidence_score": 0.65},
            }
        }
        cfg = build_resumo_guardrail_cfg({}, ml_trace_last=trace)
        self.assertIsNotNone(cfg["operational_confidence_score"])
        self.assertAlmostEqual(cfg["operational_confidence_marginal"], 75.0)
        self.assertAlmostEqual(cfg["operational_confidence_layer"], 65.0)

    def test_returns_dict(self):
        cfg = build_resumo_guardrail_cfg({}, ml_trace_last=None)
        self.assertIsInstance(cfg, dict)

    def test_required_keys_present(self):
        cfg = build_resumo_guardrail_cfg({}, ml_trace_last=None)
        for key in ("profile", "strength_major", "strength_mid", "strength_minor",
                    "target_major", "target_mid", "target_minor",
                    "operational_confidence_score", "operational_confidence_weight"):
            self.assertIn(key, cfg)


if __name__ == "__main__":
    unittest.main()
