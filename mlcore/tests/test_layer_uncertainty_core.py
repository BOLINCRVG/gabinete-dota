import unittest

from app_core.layer_uncertainty_core import apply_layer_uncertainty_live


class TestLayerUncertaintyCore(unittest.TestCase):
    def test_outputs_contract_fields(self):
        out = apply_layer_uncertainty_live(
            p_map_in=0.61,
            p_series_in=0.70,
            bo=5,
            interregional=False,
            playoffs=True,
            patch_status="ok",
            n_team_eff=48,
            n_players_eff=22,
            state_areal_live={"applied": True, "diff_pp": 1.2, "method": "areal"},
            style_b_live={"applied": True, "diff_pp": 0.8, "method": "style"},
            rank_sos_live={"applied": True, "diff_pp": 1.0, "method": "sos"},
            meta_c_live={"applied": True, "diff_pp": 2.2, "method": "meta"},
        )
        self.assertEqual(str(out.get("method")), "layer_uncertainty_v2")
        self.assertIn(str(out.get("actionability_zone")), {"low", "medium", "high"})
        self.assertTrue(isinstance(out.get("layer_scores"), dict))
        self.assertTrue(isinstance(out.get("dominant_risk_layers"), list))
        self.assertIsInstance(out.get("applied_layers"), int)
        self.assertIsInstance(out.get("missing_layers"), int)
        self.assertTrue(0.0 <= float(out.get("uncertainty_score")) <= 1.0)
        self.assertTrue(0.0 <= float(out.get("confidence_score")) <= 1.0)

    def test_low_evidence_pushes_lower_actionability(self):
        out_low = apply_layer_uncertainty_live(
            p_map_in=0.58,
            p_series_in=0.64,
            bo=3,
            patch_status="warn",
            n_team_eff=3,
            n_players_eff=1,
            state_areal_live={"applied": False, "reason": "missing"},
            state_alite_live={"applied": False, "reason": "missing"},
            style_b_live={"applied": False, "reason": "missing"},
            rank_sos_live={"applied": False, "reason": "missing"},
            meta_c_live={"applied": False, "reason": "missing"},
        )
        out_hi = apply_layer_uncertainty_live(
            p_map_in=0.58,
            p_series_in=0.64,
            bo=3,
            patch_status="ok",
            n_team_eff=55,
            n_players_eff=30,
            state_areal_live={"applied": True, "diff_pp": 0.3},
            state_alite_live={"applied": True, "diff_pp": 0.2},
            style_b_live={"applied": True, "diff_pp": 0.4},
            rank_sos_live={"applied": True, "diff_pp": 0.2},
            meta_c_live={"applied": True, "diff_pp": 0.6},
        )
        self.assertGreaterEqual(float(out_low.get("uncertainty_score")), float(out_hi.get("uncertainty_score")))
        self.assertLessEqual(float(out_low.get("confidence_score")), float(out_hi.get("confidence_score")))

    def test_bo1_no_series_explosion(self):
        out = apply_layer_uncertainty_live(
            p_map_in=0.54,
            p_series_in=0.54,
            bo=1,
            n_team_eff=30,
            n_players_eff=12,
            state_areal_live={"applied": True, "diff_pp": 1.0},
        )
        self.assertTrue(abs(float(out.get("diff_pp") or 0.0)) <= float(out.get("cap_pp_map") or 99.0) + 1e-9)

    def test_strong_cross_tier_interleague_reduces_uncertainty_shrink(self):
        weak = apply_layer_uncertainty_live(
            p_map_in=0.55,
            p_series_in=0.60,
            bo=5,
            interregional=True,
            playoffs=True,
            patch_status="ok",
            n_team_eff=0,
            n_players_eff=0,
            state_areal_live={"applied": True, "diff_pp": 3.5, "method": "areal"},
            style_b_live={"applied": True, "diff_pp": 0.8, "method": "style"},
            hier_calibration_live={"applied": True, "interleague": {"inter_ev": 0.40, "strength_gap": 0.05, "pair_scope_kind": "macro_pair", "strength_band_a": "major", "strength_band_b": "major"}},
        )
        strong = apply_layer_uncertainty_live(
            p_map_in=0.55,
            p_series_in=0.60,
            bo=5,
            interregional=True,
            playoffs=True,
            patch_status="ok",
            n_team_eff=0,
            n_players_eff=0,
            state_areal_live={"applied": True, "diff_pp": 3.5, "method": "areal"},
            style_b_live={"applied": True, "diff_pp": 0.8, "method": "style"},
            hier_calibration_live={"applied": True, "interleague": {"inter_ev": 0.95, "strength_gap": 0.19, "pair_scope_kind": "blend", "strength_band_a": "elite", "strength_band_b": "upper"}},
        )
        self.assertGreater(float(weak.get("shrink_map") or 0.0), float(strong.get("shrink_map") or 0.0))
        self.assertGreater(float(strong.get("evidence_score") or 0.0), float(weak.get("evidence_score") or 0.0))
        self.assertAlmostEqual(float(strong.get("cap_pp_map") or 0.0), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
