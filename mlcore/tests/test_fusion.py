import unittest

from mlcore.ml_fusion import fuse_team_players_probs, FusionConfig


class FusionTest(unittest.TestCase):
    def test_invalid_players_fallbacks_team(self):
        out = fuse_team_players_probs(p_team=0.62, p_players=float("nan"), coverage=0.9)
        self.assertAlmostEqual(out["p_fused"], 0.62, places=10)
        self.assertEqual(out["reason"], "invalid_p_players_fallback_team")

    def test_low_coverage_keeps_team_close(self):
        out = fuse_team_players_probs(p_team=0.60, p_players=0.80, coverage=0.05)
        self.assertTrue(0.60 <= out["p_fused"] < 0.62)
        self.assertTrue(out["w_players"] < 0.05)

    def test_high_coverage_uses_players_more(self):
        out = fuse_team_players_probs(
            p_team=0.60,
            p_players=0.80,
            coverage=1.0,
            season_phase="mid",
            cfg=FusionConfig(team_weight=0.6, team_weight_mid=0.6, coverage_power=1.0),
        )
        self.assertTrue(out["p_fused"] > 0.65)
        self.assertTrue(out["w_players"] >= 0.35)

    def test_early_season_boosts_players_weight(self):
        out_early = fuse_team_players_probs(
            p_team=0.60,
            p_players=0.70,
            coverage=1.0,
            team_games_min=3,
            cfg=FusionConfig(team_weight=0.75, early_season_team_weight=0.50, season_knee_games=20),
        )
        out_late = fuse_team_players_probs(
            p_team=0.60,
            p_players=0.70,
            coverage=1.0,
            team_games_min=40,
            cfg=FusionConfig(team_weight=0.75, early_season_team_weight=0.50, season_knee_games=20),
        )
        self.assertGreater(out_early["w_players"], out_late["w_players"])

    def test_transfer_signal_increases_players_weight(self):
        out_lo = fuse_team_players_probs(
            p_team=0.60,
            p_players=0.72,
            coverage=0.8,
            transfer_signal=0.0,
            cfg=FusionConfig(team_weight=0.7, transfer_boost=0.5),
        )
        out_hi = fuse_team_players_probs(
            p_team=0.60,
            p_players=0.72,
            coverage=0.8,
            transfer_signal=1.0,
            cfg=FusionConfig(team_weight=0.7, transfer_boost=0.5),
        )
        self.assertGreater(out_hi["w_players"], out_lo["w_players"])


if __name__ == "__main__":
    unittest.main()
