"""Tests for app_core/ml_probability_adjustments_core.py — probability adjustment fns."""
import math
import unittest

from mlcore.utils import clip_prob
from app_core.ml_probability_adjustments_core import (
    apply_ml_microseg_calibration,
    apply_ml_marginal_trust_adjust,
    apply_ml_uncertainty_adjust,
    apply_series_gamma,
    effective_games_from_ml_info,
    safe_logit,
    safe_sigmoid,
    seg_key,
)


# ---------------------------------------------------------------------------
# safe_logit / safe_sigmoid
# ---------------------------------------------------------------------------

class TestSafeLogit(unittest.TestCase):

    def test_half_gives_zero(self):
        self.assertAlmostEqual(safe_logit(0.5), 0.0, places=6)

    def test_three_quarters(self):
        # logit(0.75) = log(3) ≈ 1.0986
        self.assertAlmostEqual(safe_logit(0.75), math.log(3), places=5)

    def test_extreme_zero_clipped(self):
        # clip_prob prevents log(0): result finite
        result = safe_logit(0.0)
        self.assertTrue(math.isfinite(result))
        self.assertLess(result, 0.0)

    def test_extreme_one_clipped(self):
        result = safe_logit(1.0)
        self.assertTrue(math.isfinite(result))
        self.assertGreater(result, 0.0)


class TestSafeSigmoid(unittest.TestCase):

    def test_zero_gives_half(self):
        self.assertAlmostEqual(safe_sigmoid(0.0), 0.5, places=6)

    def test_large_positive_approaches_one(self):
        self.assertGreater(safe_sigmoid(50.0), 0.999)

    def test_large_negative_approaches_zero(self):
        self.assertLess(safe_sigmoid(-50.0), 0.001)

    def test_round_trip_with_logit(self):
        p = 0.65
        z = safe_logit(p)
        p_back = safe_sigmoid(z)
        self.assertAlmostEqual(p_back, p, places=5)


# ---------------------------------------------------------------------------
# effective_games_from_ml_info
# ---------------------------------------------------------------------------

class TestEffectiveGamesFromMlInfo(unittest.TestCase):

    def _ml_info(self, n_b, n_r, sf_b=1.0, sf_r=1.0):
        return {
            "blue": {"games_played": n_b, "shrink_factor": sf_b},
            "red": {"games_played": n_r, "shrink_factor": sf_r},
        }

    def test_none_returns_all_nan(self):
        out = effective_games_from_ml_info(None)
        for k in ("n_blue", "n_red", "n_blue_eff", "n_red_eff", "n_eff"):
            self.assertTrue(math.isnan(out[k]), f"{k} should be NaN")

    def test_non_dict_returns_all_nan(self):
        out = effective_games_from_ml_info("invalid")
        self.assertTrue(math.isnan(out["n_eff"]))

    def test_symmetric_harmonic_mean(self):
        # n_b=20, n_r=20, sf=1 → n_eff = harmonic(20,20) = 20
        out = effective_games_from_ml_info(self._ml_info(20, 20))
        self.assertAlmostEqual(out["n_eff"], 20.0, places=4)

    def test_asymmetric_harmonic_mean(self):
        # harmonic(10, 40) = 2*10*40/(10+40) = 16
        out = effective_games_from_ml_info(self._ml_info(10, 40))
        self.assertAlmostEqual(out["n_eff"], 16.0, places=4)

    def test_one_side_zero_uses_max(self):
        # n_b=0, n_r=30 → n_eff = max(0, 30) = 30
        out = effective_games_from_ml_info(self._ml_info(0, 30))
        self.assertAlmostEqual(out["n_eff"], 30.0, places=4)

    def test_shrink_factor_applied(self):
        # n_b=100, sf_b=0.5 → n_b_eff=50; n_r=100, sf_r=0.5 → n_r_eff=50 → n_eff=50
        out = effective_games_from_ml_info(self._ml_info(100, 100, sf_b=0.5, sf_r=0.5))
        self.assertAlmostEqual(out["n_blue_eff"], 50.0, places=4)
        self.assertAlmostEqual(out["n_eff"], 50.0, places=4)

    def test_missing_shrink_factor_defaults_to_1(self):
        # no shrink_factor key → treated as 1.0 since n_b > 0
        info = {"blue": {"games_played": 20}, "red": {"games_played": 20}}
        out = effective_games_from_ml_info(info)
        self.assertAlmostEqual(out["n_blue_eff"], 20.0, places=4)

    def test_games_zero_no_shrink_factor_defaults_to_0(self):
        # n_b=0, no shrink_factor → treated as 0 (since n_b=0)
        info = {"blue": {"games_played": 0}, "red": {"games_played": 10}}
        out = effective_games_from_ml_info(info)
        self.assertAlmostEqual(out["n_blue_eff"], 0.0, places=4)

    def test_raw_counts_preserved(self):
        out = effective_games_from_ml_info(self._ml_info(15, 25))
        self.assertAlmostEqual(out["n_blue"], 15.0, places=4)
        self.assertAlmostEqual(out["n_red"], 25.0, places=4)


# ---------------------------------------------------------------------------
# apply_ml_uncertainty_adjust
# ---------------------------------------------------------------------------

class TestApplyMlUncertaintyAdjust(unittest.TestCase):

    def _ml_info(self, n_b=20, n_r=20, sf=1.0):
        return {
            "blue": {"games_played": n_b, "shrink_factor": sf},
            "red": {"games_played": n_r, "shrink_factor": sf},
        }

    def test_disabled_returns_p_unchanged(self):
        p, info = apply_ml_uncertainty_adjust(
            0.65, self._ml_info(), clip_prob_fn=clip_prob, enabled=False
        )
        self.assertAlmostEqual(p, 0.65, places=6)
        self.assertFalse(info["applied"])

    def test_p_half_returns_half(self):
        # z0=0, shrink doesn't change p=0.5
        p, _ = apply_ml_uncertainty_adjust(
            0.5, self._ml_info(), clip_prob_fn=clip_prob
        )
        self.assertAlmostEqual(p, 0.5, places=4)

    def test_high_n_eff_less_shrink_toward_half(self):
        # Many games → confident → p stays close to p_in
        p_high, _ = apply_ml_uncertainty_adjust(
            0.70, self._ml_info(200, 200), clip_prob_fn=clip_prob
        )
        p_low, _ = apply_ml_uncertainty_adjust(
            0.70, self._ml_info(2, 2), clip_prob_fn=clip_prob
        )
        # More games = less shrink = result stays closer to 0.70
        self.assertGreater(p_high, p_low)

    def test_result_closer_to_half_when_few_games(self):
        # Few games → shrink toward 0.5
        p, _info = apply_ml_uncertainty_adjust(
            0.70, self._ml_info(1, 1), clip_prob_fn=clip_prob
        )
        self.assertLess(p, 0.70)
        self.assertGreater(p, 0.50)

    def test_info_has_required_keys(self):
        _, info = apply_ml_uncertainty_adjust(
            0.60, self._ml_info(), clip_prob_fn=clip_prob
        )
        for key in ("enabled", "applied", "shrink", "w_keep", "p_in", "p_out"):
            self.assertIn(key, info)

    def test_applied_flag_true_when_shrink_changes_p(self):
        _p, info = apply_ml_uncertainty_adjust(
            0.70, self._ml_info(1, 1), clip_prob_fn=clip_prob
        )
        self.assertTrue(info["applied"])

    def test_none_ml_info_uses_zero_n_eff(self):
        # No info → n_eff=0 → maximum shrink
        p_none, _ = apply_ml_uncertainty_adjust(
            0.70, None, clip_prob_fn=clip_prob
        )
        p_some, _ = apply_ml_uncertainty_adjust(
            0.70, self._ml_info(30, 30), clip_prob_fn=clip_prob
        )
        # No info → heavier shrink toward 0.5 → lower p
        self.assertLess(p_none, p_some)

    def test_output_in_unit_interval(self):
        p, _ = apply_ml_uncertainty_adjust(
            0.90, self._ml_info(5, 5), clip_prob_fn=clip_prob
        )
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)


# ---------------------------------------------------------------------------
# seg_key
# ---------------------------------------------------------------------------

class TestSegKey(unittest.TestCase):

    def test_basic_format(self):
        k = seg_key("LCK", "spring", "blue", "bo1", "early")
        self.assertEqual(k, "LCK|spring|blue|bo1|early")

    def test_side_lowercased(self):
        k = seg_key("LCK", "spring", "BLUE", "bo1", "early")
        self.assertIn("|blue|", k)

    def test_phase_lowercased(self):
        k = seg_key("LCK", "spring", "blue", "bo1", "EARLY")
        self.assertTrue(k.endswith("|early"))

    def test_none_league_normalised(self):
        k = seg_key(None, "spring", "blue", "bo1", "early")
        self.assertTrue(k.startswith("|"))

    def test_wildcard_parts(self):
        k = seg_key("*", "*", "*", "*", "*")
        self.assertEqual(k, "*|*|*|*|*")


# ---------------------------------------------------------------------------
# apply_ml_microseg_calibration
# ---------------------------------------------------------------------------

class TestApplyMlMicrosegCalibration(unittest.TestCase):

    def _call(self, p, table, *, strength=1.0, min_n=5, cap_pp=10.0,
              hierarchical=True, **kwargs):
        return apply_ml_microseg_calibration(
            p,
            clip_prob_fn=clip_prob,
            league="LCK", split="spring", side="blue", bo="bo1", phase="early",
            table=table,
            min_n=min_n,
            strength=strength,
            cap_pp=cap_pp,
            hierarchical=hierarchical,
            **kwargs,
        )

    def _key(self, league="LCK", split="spring", side="blue", bo="bo1", phase="early"):
        return seg_key(league, split, side, bo, phase)

    def test_empty_table_returns_p_unchanged(self):
        p, info = self._call(0.60, {})
        self.assertAlmostEqual(p, 0.60, places=4)
        self.assertFalse(info["applied"])

    def test_matching_key_applies_positive_bias(self):
        k = self._key()
        table = {k: {"n": 100, "bias": 0.05}}
        p, info = self._call(0.55, table)
        self.assertTrue(info["applied"])
        self.assertGreater(p, 0.55)

    def test_matching_key_applies_negative_bias(self):
        k = self._key()
        table = {k: {"n": 100, "bias": -0.05}}
        p, info = self._call(0.55, table)
        self.assertTrue(info["applied"])
        self.assertLess(p, 0.55)

    def test_zero_bias_no_change(self):
        k = self._key()
        table = {k: {"n": 100, "bias": 0.0}}
        p, _info = self._call(0.55, table)
        self.assertAlmostEqual(p, 0.55, places=4)

    def test_below_min_n_skipped_non_hierarchical(self):
        # In non-hierarchical mode, n < min_n → entry is skipped hard
        k = self._key()
        table = {k: {"n": 2, "bias": 0.10}}
        p, info = self._call(0.55, table, min_n=5, hierarchical=False)
        self.assertAlmostEqual(p, 0.55, places=4)
        self.assertFalse(info["applied"])

    def test_cap_pp_limits_bias(self):
        k = self._key()
        table = {k: {"n": 100, "bias": 0.30}}
        p_capped, _ = self._call(0.55, table, cap_pp=5.0)
        p_uncapped, _ = self._call(0.55, table, cap_pp=30.0)
        # Smaller cap → less adjustment
        self.assertLess(abs(p_capped - 0.55), abs(p_uncapped - 0.55))

    def test_hierarchical_falls_back_to_wildcard(self):
        # No exact key; wildcard key exists
        table = {seg_key("*", "*", "*", "*", "*"): {"n": 50, "bias": 0.03}}
        _p, info = self._call(0.55, table, hierarchical=True)
        self.assertTrue(info["applied"])

    def test_non_hierarchical_first_key_wins(self):
        exact = self._key()
        wild = seg_key("LCK", "*", "*", "*", "*")
        table = {
            exact: {"n": 50, "bias": 0.04},
            wild: {"n": 50, "bias": -0.04},
        }
        p, info = self._call(0.55, table, hierarchical=False)
        self.assertTrue(info["applied"])
        # First key (exact) has positive bias → p > 0.55
        self.assertGreater(p, 0.55)

    def test_result_in_unit_interval(self):
        k = self._key()
        table = {k: {"n": 200, "bias": 0.40}}
        p, _ = self._call(0.95, table, cap_pp=99.0)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)


# ---------------------------------------------------------------------------
# apply_series_gamma
# ---------------------------------------------------------------------------

class TestApplySeriesGamma(unittest.TestCase):

    def test_gamma_one_identity(self):
        # gamma=1 → sigmoid(logit(p)*1) = p
        p = 0.65
        result = apply_series_gamma(p, gamma=1.0)
        self.assertAlmostEqual(result, p, places=5)

    def test_gamma_greater_than_one_amplifies(self):
        # gamma>1 → favourite becomes more extreme
        p_in = 0.70
        result = apply_series_gamma(p_in, gamma=2.0)
        self.assertGreater(result, p_in)

    def test_gamma_less_than_one_shrinks(self):
        # gamma<1 → pulls toward 0.5
        p_in = 0.70
        result = apply_series_gamma(p_in, gamma=0.5)
        self.assertLess(result, p_in)
        self.assertGreater(result, 0.5)

    def test_symmetry_around_half(self):
        # p=0.3 → γ>1 pushes further below 0.5
        p_in = 0.30
        result = apply_series_gamma(p_in, gamma=2.0)
        self.assertLess(result, p_in)

    def test_invalid_input_returns_nan(self):
        result = apply_series_gamma("abc", gamma=1.0)
        self.assertTrue(math.isnan(result))

    def test_result_in_unit_interval(self):
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            result = apply_series_gamma(p, gamma=1.5)
            self.assertGreater(result, 0.0)
            self.assertLess(result, 1.0)

    def test_gamma_clipped_to_min_0_25(self):
        # gamma < 0.25 is clipped to 0.25 → still returns finite value
        result = apply_series_gamma(0.70, gamma=0.01)
        self.assertTrue(math.isfinite(result))


# ---------------------------------------------------------------------------
# apply_ml_marginal_trust_adjust — callback stubs
# ---------------------------------------------------------------------------

def _stub_callbacks():
    return dict(
        clip_prob_fn=clip_prob,
        norm_league_key_fn=lambda lg: lg.upper(),
        tier_from_league_fn=lambda lg: "T2",
        ml_league_conf_band_fn=lambda lg, tier: "mid",
        ml_marginal_band_profile_fn=lambda band, **kw: {
            "gate_score": 85.0, "gate_n_eff": 40.0,
            "strength_scale": 1.0, "prior_scale": 1.0,
            "max_shrink_scale": 1.0, "cap_shift": 0.0,
            "soft_shift": 0.0, "hard_shift": 0.0,
        },
        league_is_excluded_fn=lambda lg, excl: False,
        is_marginal_league_for_ml_fn=lambda lg: False,
        sanitize_ml_marginal_band_coeffs_fn=lambda x: {},
    )


def _ml_info(n_b=20, n_r=20):
    return {
        "blue": {"games_played": n_b, "shrink_factor": 1.0},
        "red": {"games_played": n_r, "shrink_factor": 1.0},
    }


class TestApplyMlMarginalTrustAdjust(unittest.TestCase):

    def _call(self, p, *, ml_info=None, **overrides):
        kwargs = _stub_callbacks()
        kwargs.update(overrides)
        return apply_ml_marginal_trust_adjust(p, ml_info or _ml_info(), **kwargs)

    def test_disabled_returns_p_unchanged(self):
        p, info = self._call(0.65, enabled=False)
        self.assertAlmostEqual(p, 0.65, places=6)
        self.assertFalse(info["applied"])
        self.assertEqual(info.get("reason"), "disabled")

    def test_excluded_league_returns_p_unchanged(self):
        p, info = self._call(
            0.65,
            league_is_excluded_fn=lambda lg, excl: True,
        )
        self.assertAlmostEqual(p, 0.65, places=6)
        self.assertEqual(info.get("reason"), "excluded_league")

    def test_strong_band_high_score_high_n_gates(self):
        # band="top", cs>=85, n_eff>=40 → gate_strong_band → no adjustment
        p, info = self._call(
            0.65,
            ml_info=_ml_info(50, 50),
            ml_league_conf_band_fn=lambda lg, tier: "top",
            confidence_score=90.0,
        )
        self.assertAlmostEqual(p, 0.65, places=4)
        self.assertEqual(info.get("reason"), "gate_strong_band")

    def test_weak_confidence_applies_shrink(self):
        # cs << score_soft → heavy conf_weak → shrink applies
        p, info = self._call(
            0.70,
            ml_info=_ml_info(2, 2),
            confidence_score=30.0,
        )
        self.assertLess(p, 0.70)
        self.assertEqual(info.get("reason"), "applied")

    def test_info_contains_required_keys(self):
        _, info = self._call(0.65, confidence_score=50.0)
        for k in ("enabled", "applied", "band", "tier", "p_in", "p_out"):
            self.assertIn(k, info)

    def test_output_in_unit_interval(self):
        p, _ = self._call(0.85, ml_info=_ml_info(5, 5), confidence_score=40.0)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_nan_confidence_treated_as_50(self):
        # NaN confidence_score → defaults to 50 → moderate shrink
        p_nan, _ = self._call(0.70, ml_info=_ml_info(5, 5), confidence_score=float("nan"))
        p_50, _ = self._call(0.70, ml_info=_ml_info(5, 5), confidence_score=50.0)
        self.assertAlmostEqual(p_nan, p_50, places=4)


if __name__ == "__main__":
    unittest.main()
