import unittest

from app_core.ml_settings_runtime_core import (
    build_runtime_param_defaults,
    local_runtime_defaults,
    shared_ml_defaults,
)


class TestSharedMlDefaults(unittest.TestCase):

    def test_players_context_disabled_by_default(self):
        shared = shared_ml_defaults()
        self.assertEqual(shared.get("ml_players_context_enabled"), False)

    def test_marginal_trust_strength(self):
        shared = shared_ml_defaults()
        self.assertEqual(shared.get("ml_marginal_trust_strength"), 0.70)

    def test_fused_post_cal_enabled(self):
        shared = shared_ml_defaults()
        self.assertTrue(shared.get("ml_fused_post_cal_use_fitted"))

    def test_returns_dict(self):
        self.assertIsInstance(shared_ml_defaults(), dict)

    def test_marginal_trust_disabled_by_default(self):
        # Disabled after ablation study (tools/run_ablation_confidence_sweep.py)
        # 0/21 configurations beat baseline on 1313-sample holdout.
        shared = shared_ml_defaults()
        self.assertFalse(shared.get("ml_marginal_trust_enabled"))

    def test_uncertainty_disabled_by_default(self):
        # Disabled after ablation study (tools/run_ablation_confidence_sweep.py)
        # 0/7 configurations beat baseline on 1313-sample holdout.
        shared = shared_ml_defaults()
        self.assertFalse(shared.get("ml_uncertainty_enabled"))

    def test_postfusion_cal_enabled_by_default(self):
        # Validated by Phase 1 gate (tools/run_phase1_validation.py) on 2827-sample
        # holdout: delta_logloss = +0.218pp [+0.034, +0.381], p=0.008, delta_ECE = +0.677pp.
        # Gate decision: ACCEPT.
        shared = shared_ml_defaults()
        self.assertTrue(shared.get("ml_postfusion_cal_enabled"))

    def test_live_state_model_enabled_by_default(self):
        # Validated by Phase 6 gate (tools/run_phase6_live_gate.py) on 5000+ samples per snapshot:
        #   @10min: +9.87pp, @15min: +14.85pp, @20min: +20.94pp, @25min: +29.28pp (all p=0.0000).
        # Models trained and saved under live_models/. Integrated into
        # build_live_bet_live_assessment(). Gate decision: ACCEPT MASSIVE.
        shared = shared_ml_defaults()
        self.assertTrue(shared.get("ml_live_state_model_enabled"))

    def test_live_state_models_dir_default(self):
        shared = shared_ml_defaults()
        self.assertEqual(shared.get("ml_live_state_models_dir"), "live_models")


class TestBuildRuntimeParamDefaults(unittest.TestCase):

    def test_forces_players_context_false(self):
        merged = build_runtime_param_defaults({"ml_players_context_enabled": True})
        self.assertEqual(merged.get("ml_players_context_enabled"), False)

    def test_preserves_existing_keys(self):
        merged = build_runtime_param_defaults({"cfg_profile": "Padrão"})
        self.assertEqual(merged.get("cfg_profile"), "Padrão")

    def test_ff_ml_state_areal_live_not_overwritten(self):
        merged = build_runtime_param_defaults({"ff_ml_state_areal_live": False})
        self.assertEqual(merged.get("ff_ml_state_areal_live"), False)

    def test_confidence_governor_added(self):
        merged = build_runtime_param_defaults({})
        self.assertIn("ml_confidence_governor_enabled", merged)

    def test_returns_dict(self):
        self.assertIsInstance(build_runtime_param_defaults(None), dict)

    def test_none_base_defaults_returns_defaults(self):
        merged = build_runtime_param_defaults(None)
        self.assertEqual(merged.get("ml_players_context_enabled"), False)


class TestLocalRuntimeDefaults(unittest.TestCase):

    def test_microseg_enabled(self):
        local = local_runtime_defaults()
        self.assertTrue(local.get("ml_microseg_enabled"))

    def test_players_context_disabled(self):
        local = local_runtime_defaults()
        self.assertEqual(local.get("ml_players_context_enabled"), False)

    def test_conservador_in_filter_profiles(self):
        local = local_runtime_defaults()
        self.assertIn("Conservador", local.get("resumo_filter_profiles", {}))

    def test_balanceado_in_filter_profiles(self):
        local = local_runtime_defaults()
        self.assertIn("Balanceado", local.get("resumo_filter_profiles", {}))

    def test_returns_dict(self):
        self.assertIsInstance(local_runtime_defaults(), dict)


if __name__ == "__main__":
    unittest.main()
