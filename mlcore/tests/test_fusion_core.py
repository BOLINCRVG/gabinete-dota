import unittest
import pandas as pd

from app_core.fusion_core import fuse_with_coherence_guard, season_form_blend, current_season_override


class TestFusionCore(unittest.TestCase):
    def test_low_coverage_divergence_is_pulled_back(self):
        out = fuse_with_coherence_guard(
            p_team=0.55,
            p_players=0.90,
            n_team=30,
            n_players=1.0,
            coverage=0.05,
            season_phase="mid",
            divergence_pp_cap=20.0,
            divergence_low_coverage=0.30,
            divergence_shrink=0.7,
        )
        self.assertTrue(out.get("coherence_guard", {}).get("applied"))
        self.assertGreater(out["p_fused"], 0.55)
        self.assertLess(out["p_fused"], 0.90)

    def test_high_coverage_keeps_players_weight(self):
        out = fuse_with_coherence_guard(
            p_team=0.55,
            p_players=0.80,
            n_team=20,
            n_players=20,
            coverage=1.0,
            season_phase="early",
            divergence_pp_cap=20.0,
            divergence_low_coverage=0.30,
            divergence_shrink=0.7,
        )
        self.assertFalse(out.get("coherence_guard", {}).get("applied"))
        self.assertGreater(float(out.get("w_players", 0.0)), 0.20)

    def test_invalid_players_fallback(self):
        out = fuse_with_coherence_guard(
            p_team=0.61,
            p_players=float("nan"),
            n_team=10,
            n_players=2,
        )
        self.assertAlmostEqual(float(out.get("p_fused")), 0.61, places=6)
        self.assertEqual(str(out.get("reason")), "invalid_p_players_fallback_team")

    def test_scales_do_not_inflate_coverage_when_raw_evidence_is_thinner(self):
        out = fuse_with_coherence_guard(
            p_team=0.60,
            p_players=0.65,
            n_team=40,
            n_players=30,
            team_scale=0.25,
            players_scale=3.0,
            coverage=0.90,
            season_phase="auto",
        )
        self.assertAlmostEqual(float(out.get("coverage_inferred_raw", 0.0)), 30.0 / 70.0, places=9)
        self.assertAlmostEqual(float(out.get("coverage_used", 0.0)), 30.0 / 70.0, places=9)
        self.assertAlmostEqual(float(out.get("team_scale_eff", 0.0)), 1.0 + ((0.25 - 1.0) * (30.0 / 70.0)), places=9)
        self.assertAlmostEqual(float(out.get("players_scale_eff", 0.0)), 1.0 + ((3.0 - 1.0) * (30.0 / 70.0)), places=9)
        self.assertLess(float(out.get("w_players", 0.0)), 0.20)

    def test_season_form_blend_strengthens_clear_gap(self):
        df_a = pd.DataFrame({"win": [1] * 12 + [0] * 2})
        df_b = pd.DataFrame({"win": [1] * 1 + [0] * 12})
        out = season_form_blend(
            p_model=0.50,
            df_team_a=df_a,
            df_team_b=df_b,
            enabled=True,
            max_weight=0.85,
            knee_games=5.0,
            beta_prior=2.0,
            signal_power=1.0,
        )
        self.assertTrue(out.get("applied"))
        self.assertGreater(float(out.get("p_form")), 0.80)
        self.assertGreater(float(out.get("p_blended")), 0.65)

    def test_season_form_blend_disabled(self):
        df = pd.DataFrame({"win": [1, 0, 1, 0]})
        out = season_form_blend(
            p_model=0.57,
            df_team_a=df,
            df_team_b=df,
            enabled=False,
        )
        self.assertFalse(out.get("applied"))
        self.assertEqual(str(out.get("reason")), "disabled")

    def test_season_form_blend_elite_same_league_shrinks(self):
        # Two strong teams in same league: blend should avoid extreme confidence.
        df_a = pd.DataFrame({"win": [1] * 11 + [0], "league": ["LCK"] * 12})
        df_b = pd.DataFrame({"win": [1] * 9 + [0] * 3, "league": ["LCK"] * 12})
        out = season_form_blend(
            p_model=0.80,
            df_team_a=df_a,
            df_team_b=df_b,
            enabled=True,
            max_weight=0.85,
            knee_games=5.0,
            beta_prior=2.0,
            signal_power=1.0,
            elite_same_league_parity_strength=0.40,
        )
        self.assertTrue(out.get("applied"))
        self.assertTrue(out.get("same_league"))
        # Without shrink this would be closer to very high favorite; keep under 0.80
        self.assertLess(float(out.get("p_form")), 0.80)

    def test_current_season_override_applies_with_sample_and_divergence(self):
        out = current_season_override(
            p_model=0.58,
            p_form=0.74,
            w_form=0.20,
            n_a=14,
            n_b=13,
            min_games_gate=8,
            divergence_gate_pp=8.0,
            max_weight=0.88,
        )
        self.assertTrue(out.get("applied"))
        self.assertGreater(float(out.get("p_out")), 0.62)
        self.assertGreater(float(out.get("w_used")), 0.20)

    def test_current_season_override_not_applied_on_low_sample(self):
        out = current_season_override(
            p_model=0.55,
            p_form=0.80,
            w_form=0.25,
            n_a=3,
            n_b=4,
            min_games_gate=8,
            divergence_gate_pp=8.0,
            max_weight=0.88,
        )
        self.assertFalse(out.get("applied"))
        self.assertEqual(str(out.get("reason")), "low_sample")


if __name__ == "__main__":
    unittest.main()
