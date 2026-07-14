import unittest

from app_core.meta_blend_core import (
    META_BLEND_C_MODEL_FEATURES,
    apply_meta_blend_c_lite,
    apply_meta_blend_c_model,
    should_apply_meta_blend_c_model_gate,
)


class TestMetaBlendCore(unittest.TestCase):
    def test_gates_large_delta_on_low_evidence(self):
        low = apply_meta_blend_c_lite(
            p_anchor=0.60,
            p_adjusted=0.80,
            n_team_eff=2,
            n_players_eff=1,
            recent_evidence=0.2,
            rd_conf=0.3,
            interregional=False,
        )
        high = apply_meta_blend_c_lite(
            p_anchor=0.60,
            p_adjusted=0.80,
            n_team_eff=20,
            n_players_eff=10,
            recent_evidence=0.9,
            rd_conf=0.9,
            interregional=False,
        )
        self.assertTrue(low.get("applied"))
        self.assertTrue(high.get("applied"))
        p_low = float(low["p_after"])
        p_high = float(high["p_after"])
        self.assertGreater(p_high, p_low)
        self.assertLess(p_low, 0.80)

    def test_interregional_more_conservative(self):
        same = apply_meta_blend_c_lite(
            p_anchor=0.55,
            p_adjusted=0.70,
            n_team_eff=15,
            n_players_eff=4,
            recent_evidence=0.8,
            rd_conf=0.8,
            interregional=False,
            unknown_macro=False,
        )
        inter = apply_meta_blend_c_lite(
            p_anchor=0.55,
            p_adjusted=0.70,
            n_team_eff=15,
            n_players_eff=4,
            recent_evidence=0.8,
            rd_conf=0.8,
            interregional=True,
            unknown_macro=False,
        )
        self.assertTrue(same.get("applied"))
        self.assertTrue(inter.get("applied"))
        self.assertGreater(float(same["p_after"]), float(inter["p_after"]))

    def test_cross_strength_interregional_protects_stronger_favorite(self):
        protected = apply_meta_blend_c_lite(
            p_anchor=0.55,
            p_adjusted=0.70,
            n_team_eff=15,
            n_players_eff=4,
            recent_evidence=0.8,
            rd_conf=0.8,
            interregional=True,
            league_a="LCK",
            league_b="CBLOL",
        )
        plain = apply_meta_blend_c_lite(
            p_anchor=0.55,
            p_adjusted=0.70,
            n_team_eff=15,
            n_players_eff=4,
            recent_evidence=0.8,
            rd_conf=0.8,
            interregional=True,
            league_a="LCP",
            league_b="PCS",
        )
        self.assertTrue(protected.get("applied"))
        self.assertTrue(plain.get("applied"))
        self.assertGreater(float(protected["p_after"]), float(plain["p_after"]))

    def test_cross_strength_interregional_caps_adverse_move(self):
        out = apply_meta_blend_c_lite(
            p_anchor=0.55,
            p_adjusted=0.70,
            n_team_eff=18,
            n_players_eff=5,
            recent_evidence=0.8,
            rd_conf=0.8,
            interregional=True,
            league_a="LPL",
            league_b="CBLOL",
        )
        self.assertTrue(out.get("applied"))
        adverse_pp = (0.70 - float(out["p_after"])) * 100.0
        self.assertLessEqual(adverse_pp, 1.400001)

    def test_gate_blocks_interregional_low_team_evidence_even_in_bo5(self):
        out = should_apply_meta_blend_c_model_gate(
            bo=5,
            recent_evidence=0.76,
            n_team_eff=4.0,
            n_players_eff=0.0,
            is_playoffs=False,
            patch_status="ok",
            interregional=True,
        )
        self.assertFalse(bool(out.get("allow", True)))
        self.assertEqual(str(out.get("reason")), "blocked_interregional_low_team_evidence_v3")

    def test_no_delta_passthrough(self):
        out = apply_meta_blend_c_lite(p_anchor=0.61, p_adjusted=0.61)
        self.assertTrue(out.get("applied"))
        self.assertEqual(str(out.get("reason")), "no_delta")
        self.assertAlmostEqual(float(out.get("p_after")), 0.61, places=6)

    def test_trained_model_apply(self):
        model = {
            "enabled": True,
            "version": "meta_blend_c_model_v1",
            "features": list(META_BLEND_C_MODEL_FEATURES),
            "intercept": 0.0,
            "coef": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_mean": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_std": [1.0] * len(META_BLEND_C_MODEL_FEATURES),
            "n_total": 999,
            "oos_logloss": 0.61,
        }
        # Positive weight on z_adjusted and negative on z_anchor should move above 50% in this setup.
        i_za = META_BLEND_C_MODEL_FEATURES.index("z_anchor")
        i_zb = META_BLEND_C_MODEL_FEATURES.index("z_adjusted")
        model["coef"][i_za] = -0.2
        model["coef"][i_zb] = 0.8
        out = apply_meta_blend_c_model(
            model_cfg=model,
            p_anchor=0.55,
            p_adjusted=0.70,
            n_team_eff=20,
            n_players_eff=5,
            recent_evidence=0.8,
            rd_conf=0.9,
            hybrid_agree=0.6,
            bo=5,
            is_playoffs=True,
        )
        self.assertTrue(out.get("applied"))
        self.assertEqual(str(out.get("reason")), "meta_blend_c_model")
        self.assertGreater(float(out.get("p_after")), 0.55)
        self.assertLess(float(out.get("p_after")), 0.95)

    def test_trained_model_schema_mismatch_fails_clean(self):
        out = apply_meta_blend_c_model(
            model_cfg={"features": ["bad"], "coef": [1.0], "intercept": 0.0},
            p_anchor=0.5,
            p_adjusted=0.6,
        )
        self.assertFalse(out.get("applied"))
        self.assertEqual(str(out.get("reason")), "feature_schema_mismatch")

    def test_trained_model_runtime_stability_caps_small_anchor_delta(self):
        model = {
            "enabled": True,
            "version": "meta_blend_c_model_v3_test",
            "features": list(META_BLEND_C_MODEL_FEATURES),
            "intercept": -4.0,
            "coef": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_mean": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_std": [1.0] * len(META_BLEND_C_MODEL_FEATURES),
            "n_total": 1400,
            "oos_logloss": 0.60,
            "improvement_vs_adjusted": 0.01,
            "improvement_vs_anchor": 0.01,
        }
        out = apply_meta_blend_c_model(
            model_cfg=model,
            p_anchor=0.50,
            p_adjusted=0.51,
            n_team_eff=140.0,
            n_players_eff=12.0,
            recent_evidence=0.95,
            rd_conf=0.95,
            hybrid_agree=0.90,
            bo=5,
            is_playoffs=True,
            interregional=False,
            patch_status="ok",
        )
        self.assertTrue(out.get("applied"))
        rs = (((out.get("model_meta") or {}).get("runtime_stability")) or {})
        self.assertTrue(bool(rs.get("enabled", False)))
        self.assertTrue(bool(rs.get("capped", False)))
        self.assertLessEqual(abs(float(out.get("diff_pp"))), float(rs.get("cap_pp")) + 1e-9)
        # small anchor-adjusted delta should keep cap in conservative regime
        self.assertLessEqual(abs(float(out.get("diff_pp"))), 3.3)

    def test_runtime_stability_can_be_disabled(self):
        model = {
            "enabled": True,
            "version": "meta_blend_c_model_v3_test",
            "features": list(META_BLEND_C_MODEL_FEATURES),
            "intercept": -4.0,
            "coef": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_mean": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_std": [1.0] * len(META_BLEND_C_MODEL_FEATURES),
            "n_total": 1400,
            "oos_logloss": 0.60,
            "improvement_vs_adjusted": 0.01,
            "improvement_vs_anchor": 0.01,
            "runtime_stability": {"enabled": False},
        }
        out = apply_meta_blend_c_model(
            model_cfg=model,
            p_anchor=0.50,
            p_adjusted=0.51,
            n_team_eff=140.0,
            n_players_eff=12.0,
            recent_evidence=0.95,
            rd_conf=0.95,
            hybrid_agree=0.90,
            bo=5,
            is_playoffs=True,
            interregional=False,
            patch_status="ok",
        )
        self.assertTrue(out.get("applied"))
        rs = (((out.get("model_meta") or {}).get("runtime_stability")) or {})
        self.assertFalse(bool(rs.get("enabled", True)))
        self.assertFalse(bool(rs.get("capped", False)))
        self.assertGreater(abs(float(out.get("diff_pp"))), 10.0)

    def test_runtime_stability_map_selects_context_key(self):
        model = {
            "enabled": True,
            "version": "meta_blend_c_model_v3_test",
            "features": list(META_BLEND_C_MODEL_FEATURES),
            "intercept": -4.0,
            "coef": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_mean": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_std": [1.0] * len(META_BLEND_C_MODEL_FEATURES),
            "n_total": 1400,
            "oos_logloss": 0.60,
            "improvement_vs_adjusted": 0.01,
            "improvement_vs_anchor": 0.01,
            "context_models": {
                "global|patch:stable|phase:late": {
                    "version": "meta_blend_c_model_v3_ctx",
                    "features": list(META_BLEND_C_MODEL_FEATURES),
                    "intercept": -4.0,
                    "coef": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
                    "x_mean": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
                    "x_std": [1.0] * len(META_BLEND_C_MODEL_FEATURES),
                    "n_total": 420,
                    "oos_logloss": 0.59,
                    "improvement_vs_adjusted": 0.01,
                    "improvement_vs_anchor": 0.01,
                }
            },
            "runtime_stability": {
                "enabled": True,
                "cap_pp_hi_bo5": 8.0,
                "map": {
                    "global|patch:stable|phase:late": {
                        "cap_pp_lo": 1.0,
                        "cap_pp_hi_bo5": 1.0,
                        "cap_pp_small_delta": 1.0,
                        "cap_pp_tiny_delta": 1.0,
                    }
                },
            },
        }
        out = apply_meta_blend_c_model(
            model_cfg=model,
            p_anchor=0.50,
            p_adjusted=0.51,
            n_team_eff=180.0,
            n_players_eff=14.0,
            recent_evidence=0.95,
            rd_conf=0.95,
            hybrid_agree=0.90,
            bo=5,
            is_playoffs=True,
            interregional=False,
            patch_status="ok",
        )
        self.assertTrue(out.get("applied"))
        rs = (((out.get("model_meta") or {}).get("runtime_stability")) or {})
        self.assertEqual(str(rs.get("selected_key")), "global|patch:stable|phase:late")
        self.assertIn("global|patch:stable|phase:late", list(rs.get("applied_keys") or []))
        self.assertTrue(bool(rs.get("capped", False)))
        self.assertLessEqual(abs(float(out.get("diff_pp"))), 1.01)

    def test_trained_model_runtime_stability_caps_cross_strength_interregional(self):
        model = {
            "enabled": True,
            "version": "meta_blend_c_model_v3_test",
            "features": list(META_BLEND_C_MODEL_FEATURES),
            "intercept": -4.0,
            "coef": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_mean": [0.0] * len(META_BLEND_C_MODEL_FEATURES),
            "x_std": [1.0] * len(META_BLEND_C_MODEL_FEATURES),
            "n_total": 1400,
            "oos_logloss": 0.60,
            "improvement_vs_adjusted": 0.01,
            "improvement_vs_anchor": 0.01,
        }
        out = apply_meta_blend_c_model(
            model_cfg=model,
            p_anchor=0.55,
            p_adjusted=0.70,
            n_team_eff=180.0,
            n_players_eff=14.0,
            recent_evidence=0.95,
            rd_conf=0.95,
            hybrid_agree=0.90,
            bo=5,
            is_playoffs=True,
            interregional=True,
            league_a="LPL",
            league_b="CBLOL",
            patch_status="ok",
        )
        self.assertTrue(out.get("applied"))
        adverse_pp = (0.70 - float(out.get("p_after"))) * 100.0
        self.assertLessEqual(adverse_pp, 1.400001)


if __name__ == "__main__":
    unittest.main()
