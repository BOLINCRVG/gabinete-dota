import unittest

from mlcore.elo import EloModel, EloParams


class EloCoreTest(unittest.TestCase):
    def test_strength_caps_applied_in_inference(self):
        m = EloModel(
            EloParams(
                k=0.0,
                k_league=0.0,
                k_tier=0.0,
                tier_prior_weight=0.0,
                league_prior_weight=0.0,
                league_strength_cap_abs=10.0,
                league_strength_cap_mult_t1=1.0,
                league_strength_cap_mult_t2=1.0,
                league_strength_cap_mult_erl=1.0,
                league_strength_cap_mult_minor=1.0,
                league_strength_cap_mult_other=1.0,
                tier_strength_cap_abs=5.0,
            )
        )
        m.league_strength["A"] = 99.0
        m.tier_strength["T1"] = 99.0
        s = float(m.base_strength("TeamA", league="A", tier="T1"))
        self.assertAlmostEqual(s, 15.0, places=6)

    def test_caps_hold_after_update(self):
        m = EloModel(
            EloParams(
                k=0.0,
                k_league=250.0,
                k_tier=180.0,
                tier_prior_weight=0.0,
                league_prior_weight=0.0,
                league_strength_cap_abs=12.0,
                league_strength_cap_mult_t1=1.0,
                league_strength_cap_mult_t2=1.0,
                league_strength_cap_mult_erl=1.0,
                league_strength_cap_mult_minor=1.0,
                league_strength_cap_mult_other=1.0,
                tier_strength_cap_abs=8.0,
            )
        )
        m.update(
            "Blue",
            "Red",
            1,
            blue_league="LCS",
            red_league="LCK",
            blue_tier="T2",
            red_tier="T1",
        )
        self.assertLessEqual(abs(float(m.league_strength.get("LCS", 0.0))), 12.0 + 1e-9)
        self.assertLessEqual(abs(float(m.league_strength.get("LCK", 0.0))), 12.0 + 1e-9)
        self.assertLessEqual(abs(float(m.tier_strength.get("T2", 0.0))), 8.0 + 1e-9)
        self.assertLessEqual(abs(float(m.tier_strength.get("T1", 0.0))), 8.0 + 1e-9)

    def test_canonical_league_is_used(self):
        m = EloModel(
            EloParams(
                k=0.0,
                k_league=0.0,
                k_tier=0.0,
                tier_prior_weight=0.0,
                league_prior_weight=0.0,
                league_strength_cap_mult_t1=1.0,
                league_strength_cap_mult_t2=1.0,
                league_strength_cap_mult_erl=1.0,
                league_strength_cap_mult_minor=1.0,
                league_strength_cap_mult_other=1.0,
            )
        )
        m.league_strength["LCS"] = 5.0
        s = float(m.base_strength("TeamA", league="LTA North", tier="T2"))
        self.assertAlmostEqual(s, 5.0, places=6)

    def test_dynamic_k_feature_flag_increases_update_with_high_uncertainty(self):
        off = EloModel(
            EloParams(
                k=24.0,
                k_league=0.0,
                k_tier=0.0,
                k_side=0.0,
                tier_prior_weight=0.0,
                league_prior_weight=0.0,
                dynamic_k_enabled=False,
            )
        )
        on = EloModel(
            EloParams(
                k=24.0,
                k_league=0.0,
                k_tier=0.0,
                k_side=0.0,
                tier_prior_weight=0.0,
                league_prior_weight=0.0,
                dynamic_k_enabled=True,
                dynamic_k_games_prior=40.0,
                dynamic_k_uncertainty_mult_max=1.8,
                dynamic_k_volatility_ref=0.25,
                dynamic_k_volatility_mult_min=0.8,
                dynamic_k_volatility_mult_max=1.35,
                volatility_ewma_beta=0.10,
            )
        )

        # Force high volatility context for ON model.
        on.team_volatility["Blue"] = 0.50
        on.team_volatility["Red"] = 0.50

        off.update("Blue", "Red", 1, blue_league="LCS", red_league="LCS", blue_tier="T2", red_tier="T2")
        on.update("Blue", "Red", 1, blue_league="LCS", red_league="LCS", blue_tier="T2", red_tier="T2")

        self.assertGreater(abs(float(on.ratings.get("Blue", 0.0))), abs(float(off.ratings.get("Blue", 0.0))))
        self.assertIn("Blue", on.team_volatility)
        self.assertIn("Red", on.team_volatility)

    def test_dynamic_k_state_roundtrip_serialization(self):
        m = EloModel(EloParams(dynamic_k_enabled=True, dynamic_k_volatility_ref=0.33, volatility_ewma_beta=0.12))
        m.team_volatility["TeamA"] = 0.41
        d = m.to_dict()
        r = EloModel.from_dict(d)
        self.assertTrue(bool(r.params.dynamic_k_enabled))
        self.assertAlmostEqual(float(r.params.dynamic_k_volatility_ref), 0.33, places=6)
        self.assertAlmostEqual(float(r.params.volatility_ewma_beta), 0.12, places=6)
        self.assertAlmostEqual(float(r.team_volatility.get("TeamA", 0.0)), 0.41, places=6)


if __name__ == "__main__":
    unittest.main()
