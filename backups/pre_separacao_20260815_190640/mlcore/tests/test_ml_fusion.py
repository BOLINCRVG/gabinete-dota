"""Tests for mlcore/ml_fusion.py — team+players probability fusion."""
import math
import unittest

from mlcore.ml_fusion import FusionConfig, fuse_team_players_probs


def _fuse(**kw):
    defaults = dict(p_team=0.60, p_players=0.65, coverage=0.8)
    defaults.update(kw)
    return fuse_team_players_probs(**defaults)


# ---------------------------------------------------------------------------
# FusionConfig defaults
# ---------------------------------------------------------------------------

class TestFusionConfigDefaults(unittest.TestCase):

    def test_team_weight_between_zero_and_one(self):
        cfg = FusionConfig()
        self.assertGreater(cfg.team_weight, 0.0)
        self.assertLess(cfg.team_weight, 1.0)

    def test_playoffs_weight_higher_than_mid(self):
        cfg = FusionConfig()
        self.assertGreater(cfg.team_weight_playoffs, cfg.team_weight_mid)

    def test_early_weight_lower_than_mid(self):
        cfg = FusionConfig()
        self.assertLess(cfg.team_weight_early, cfg.team_weight_mid)


# ---------------------------------------------------------------------------
# fuse_team_players_probs — basic behavior
# ---------------------------------------------------------------------------

class TestFuseTeamPlayersProbs(unittest.TestCase):

    def test_output_has_required_keys(self):
        out = _fuse()
        for key in ("p_fused", "w_team", "w_players", "coverage_eff", "reason"):
            self.assertIn(key, out)

    def test_reason_ok_for_valid_inputs(self):
        out = _fuse()
        self.assertEqual(out["reason"], "ok")

    def test_p_fused_in_unit_interval(self):
        out = _fuse()
        self.assertGreater(out["p_fused"], 0.0)
        self.assertLess(out["p_fused"], 1.0)

    def test_weights_sum_to_one(self):
        out = _fuse()
        self.assertAlmostEqual(out["w_team"] + out["w_players"], 1.0, places=8)

    def test_invalid_p_team_returns_nan(self):
        out = fuse_team_players_probs(p_team=float("nan"), p_players=0.6, coverage=0.8)
        self.assertTrue(math.isnan(out["p_fused"]))
        self.assertEqual(out["reason"], "invalid_p_team")

    def test_invalid_p_players_falls_back_to_team(self):
        out = fuse_team_players_probs(p_team=0.65, p_players=float("nan"), coverage=0.8)
        self.assertAlmostEqual(out["p_fused"], 0.65, places=4)
        self.assertEqual(out["reason"], "invalid_p_players_fallback_team")

    def test_zero_coverage_gives_team_model_weight(self):
        # zero coverage → players contribute nothing → w_team = 1.0
        out = fuse_team_players_probs(p_team=0.65, p_players=0.80, coverage=0.0)
        self.assertAlmostEqual(out["w_team"], 1.0, places=4)

    def test_full_coverage_increases_players_weight(self):
        out_low = fuse_team_players_probs(p_team=0.6, p_players=0.7, coverage=0.0)
        out_high = fuse_team_players_probs(p_team=0.6, p_players=0.7, coverage=1.0)
        self.assertGreater(out_high["w_players"], out_low["w_players"])

    def test_equal_probs_gives_same_fused(self):
        p = 0.65
        out = fuse_team_players_probs(p_team=p, p_players=p, coverage=0.8)
        self.assertAlmostEqual(out["p_fused"], p, places=5)

    def test_p_fused_between_team_and_players(self):
        out = fuse_team_players_probs(p_team=0.55, p_players=0.75, coverage=0.8)
        self.assertGreater(out["p_fused"], 0.55)
        self.assertLess(out["p_fused"], 0.75)


# ---------------------------------------------------------------------------
# Season phase behavior
# ---------------------------------------------------------------------------

class TestFuseSeasonPhase(unittest.TestCase):

    def test_playoffs_phase_increases_team_weight_vs_mid(self):
        out_mid = _fuse(season_phase="mid")
        out_po = _fuse(season_phase="playoffs")
        self.assertGreater(out_po["w_team"], out_mid["w_team"])

    def test_early_phase_decreases_team_weight_vs_mid(self):
        out_mid = _fuse(season_phase="mid")
        out_early = _fuse(season_phase="early")
        self.assertLess(out_early["w_team"], out_mid["w_team"])

    def test_auto_phase_with_many_games_resolves_mid(self):
        out = fuse_team_players_probs(
            p_team=0.6, p_players=0.7, coverage=0.8,
            season_phase="auto", team_games_min=100.0
        )
        self.assertEqual(out["season_phase"], "mid")

    def test_auto_phase_with_few_games_resolves_early(self):
        cfg = FusionConfig(season_knee_games=18.0)
        out = fuse_team_players_probs(
            p_team=0.6, p_players=0.7, coverage=0.8,
            season_phase="auto", team_games_min=1.0,
            cfg=cfg
        )
        self.assertEqual(out["season_phase"], "early")


# ---------------------------------------------------------------------------
# Transfer signal and players weight multiplier
# ---------------------------------------------------------------------------

class TestFuseTransferAndMultiplier(unittest.TestCase):

    def test_transfer_signal_increases_players_weight(self):
        out_no = _fuse(transfer_signal=0.0)
        out_ts = _fuse(transfer_signal=1.0)
        self.assertGreater(out_ts["w_players"], out_no["w_players"])

    def test_higher_players_mult_increases_players_weight(self):
        out_low = _fuse(players_weight_mult=0.5)
        out_high = _fuse(players_weight_mult=2.0)
        self.assertGreater(out_high["w_players"], out_low["w_players"])

    def test_zero_players_mult_gives_team_model(self):
        out = _fuse(players_weight_mult=0.0)
        self.assertAlmostEqual(out["w_team"], 1.0, places=4)

    def test_nan_transfer_signal_treated_as_zero(self):
        out_nan = _fuse(transfer_signal=float("nan"))
        out_zero = _fuse(transfer_signal=0.0)
        self.assertAlmostEqual(out_nan["p_fused"], out_zero["p_fused"], places=6)

    def test_nan_coverage_treated_as_zero(self):
        out_nan = fuse_team_players_probs(p_team=0.6, p_players=0.7, coverage=float("nan"))
        self.assertAlmostEqual(out_nan["w_team"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
