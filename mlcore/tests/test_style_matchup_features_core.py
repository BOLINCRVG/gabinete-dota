"""Tests for mlcore/style_matchup_features_core.py."""
import math
import unittest
from types import SimpleNamespace
from typing import Dict

from mlcore.style_matchup_features_core import (
    STYLE_MATCHUP_B_MODEL_FEATURES,
    STYLE_MATCHUP_TERM_WEIGHTS,
    build_style_matchup_b_model_features,
    should_apply_style_matchup_b_model_gate,
    _combine_mean,
    _safe_ratio,
    _phase_bucket_from_expected,
    _patch_bucket_from_status,
    _tier_rank,
    _score_terms_to_style_score,
)


# ---------------------------------------------------------------------------
# _combine_mean
# ---------------------------------------------------------------------------

class TestCombineMean(unittest.TestCase):

    def test_two_valid_values(self):
        self.assertAlmostEqual(_combine_mean(0.4, 0.6), 0.5)

    def test_symmetric(self):
        self.assertAlmostEqual(_combine_mean(0.3, 0.7), _combine_mean(0.7, 0.3))

    def test_nan_a_returns_nan(self):
        self.assertTrue(math.isnan(_combine_mean(float("nan"), 0.5)))

    def test_nan_b_returns_nan(self):
        self.assertTrue(math.isnan(_combine_mean(0.5, float("nan"))))

    def test_string_numbers_parsed(self):
        self.assertAlmostEqual(_combine_mean("0.4", "0.6"), 0.5)


# ---------------------------------------------------------------------------
# _safe_ratio
# ---------------------------------------------------------------------------

class TestSafeRatio(unittest.TestCase):

    def test_normal_division(self):
        self.assertAlmostEqual(_safe_ratio(10.0, 4.0), 2.5)

    def test_zero_denominator_returns_nan(self):
        self.assertTrue(math.isnan(_safe_ratio(5.0, 0.0)))

    def test_negative_denominator_returns_default(self):
        self.assertTrue(math.isnan(_safe_ratio(5.0, -1.0)))

    def test_nan_numerator_returns_default(self):
        self.assertTrue(math.isnan(_safe_ratio(float("nan"), 4.0)))

    def test_custom_default(self):
        result = _safe_ratio(5.0, 0.0, default=-1.0)
        self.assertAlmostEqual(result, -1.0)


# ---------------------------------------------------------------------------
# _phase_bucket_from_expected
# ---------------------------------------------------------------------------

class TestPhaseBucketFromExpected(unittest.TestCase):

    def test_short_game_fast_pace_is_early(self):
        bucket = _phase_bucket_from_expected(exp_time_min=28.0, exp_total_kills=28.0)
        self.assertEqual(bucket, "early")

    def test_long_game_is_late_by_time(self):
        bucket = _phase_bucket_from_expected(exp_time_min=35.0, exp_total_kills=25.0)
        self.assertEqual(bucket, "late")

    def test_low_kpm_is_late(self):
        # kpm = 15/35 ≈ 0.43 ≤ 0.72
        bucket = _phase_bucket_from_expected(exp_time_min=35.0, exp_total_kills=15.0)
        self.assertEqual(bucket, "late")

    def test_moderate_values_mid(self):
        # t=32, kpm=25/32≈0.78 → not early (t>30.5), not late (t<33.5, kpm>0.72)
        bucket = _phase_bucket_from_expected(exp_time_min=32.0, exp_total_kills=25.0)
        self.assertEqual(bucket, "mid")

    def test_invalid_time_returns_mid(self):
        bucket = _phase_bucket_from_expected(exp_time_min=float("nan"), exp_total_kills=20.0)
        self.assertEqual(bucket, "mid")

    def test_zero_time_returns_mid(self):
        bucket = _phase_bucket_from_expected(exp_time_min=0.0, exp_total_kills=20.0)
        self.assertEqual(bucket, "mid")


# ---------------------------------------------------------------------------
# _patch_bucket_from_status
# ---------------------------------------------------------------------------

class TestPatchBucketFromStatus(unittest.TestCase):

    def test_refit_aliases(self):
        for s in ("refit", "refit_needed", "changed"):
            self.assertEqual(_patch_bucket_from_status(s), "refit")

    def test_stable_for_others(self):
        for s in ("stable", "", "ok", "no_change"):
            self.assertEqual(_patch_bucket_from_status(s), "stable")

    def test_case_insensitive(self):
        self.assertEqual(_patch_bucket_from_status("REFIT"), "refit")


# ---------------------------------------------------------------------------
# _tier_rank
# ---------------------------------------------------------------------------

class TestTierRank(unittest.TestCase):

    def test_t1_is_strongest(self):
        self.assertEqual(_tier_rank("LCK"), 0)

    def test_t2_rank(self):
        self.assertEqual(_tier_rank("LCS"), 1)

    def test_empty_league_is_4(self):
        # Empty string → tier_for_league → "OTHER" → rank 4
        self.assertEqual(_tier_rank(""), 4)

    def test_tier_ordering(self):
        # T1 < T2 < ERL (by rank number, lower = stronger)
        self.assertLess(_tier_rank("LCK"), _tier_rank("LCS"))


# ---------------------------------------------------------------------------
# _score_terms_to_style_score
# ---------------------------------------------------------------------------

class TestScoreTermsToStyleScore(unittest.TestCase):

    def test_empty_weight_is_zero(self):
        _score, weight = _score_terms_to_style_score({})
        self.assertAlmostEqual(weight, 0.0)

    def test_empty_score_is_nan(self):
        score, _weight = _score_terms_to_style_score({})
        self.assertTrue(math.isnan(score))

    def test_single_known_term_weight(self):
        # kills_edge has weight 0.17
        _score, weight = _score_terms_to_style_score({"kills_edge": 1.0})
        self.assertAlmostEqual(weight, 0.17, places=5)

    def test_single_known_term_score(self):
        # kills_edge=1.0 → score = (1.0*0.17)/0.17 = 1.0
        score, _weight = _score_terms_to_style_score({"kills_edge": 1.0})
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_two_terms_weighted_average(self):
        # kills_edge=1.0 (w=0.17), towers_edge=0.0 (w=0.15)
        score, weight = _score_terms_to_style_score({"kills_edge": 1.0, "towers_edge": 0.0})
        total_w = 0.17 + 0.15
        expected_score = (1.0 * 0.17 + 0.0 * 0.15) / total_w
        self.assertAlmostEqual(weight, total_w, places=5)
        self.assertAlmostEqual(score, expected_score, places=5)

    def test_unknown_term_weight_zero(self):
        # Terms not in STYLE_MATCHUP_TERM_WEIGHTS → weight 0 → not counted
        _score, weight = _score_terms_to_style_score({"UNKNOWN_TERM": 1.0})
        self.assertAlmostEqual(weight, 0.0)


# ---------------------------------------------------------------------------
# should_apply_style_matchup_b_model_gate — 8 gate branches
# ---------------------------------------------------------------------------

def _gate(**kwargs) -> Dict:
    return should_apply_style_matchup_b_model_gate(**kwargs)


class TestStyleMatchupGate(unittest.TestCase):

    # BO5 always allowed
    def test_bo5_always_allow(self):
        out = _gate(bo=5)
        self.assertTrue(out["allow"])
        self.assertEqual(out["reason"], "bo5")

    def test_bo7_treated_as_bo5(self):
        out = _gate(bo=7)
        self.assertTrue(out["allow"])

    # BO3 with enough profile
    def test_bo3_with_profile_ok_allow(self):
        prof = SimpleNamespace(n_games=10)
        out = _gate(bo=3, pA=prof, pB=prof)
        self.assertTrue(out["allow"])
        self.assertEqual(out["reason"], "series_profile_ok")

    def test_bo3_low_profile_blocked(self):
        prof = SimpleNamespace(n_games=2)  # < 4.0
        out = _gate(bo=3, pA=prof, pB=prof)
        self.assertFalse(out["allow"])
        self.assertEqual(out["reason"], "low_profile_sample")

    # Low profile sample
    def test_low_profile_sample_blocked(self):
        prof = SimpleNamespace(n_games=2)
        out = _gate(bo=1, pA=prof, pB=prof)
        self.assertFalse(out["allow"])
        self.assertEqual(out["reason"], "low_profile_sample")

    # BO1 interregional + low evidence
    def test_bo1_interregional_low_evidence_blocked(self):
        prof = SimpleNamespace(n_games=5)
        out = _gate(bo=1, pA=prof, pB=prof, interregional=True, n_team_eff=10.0, n_players_eff=3.0)
        self.assertFalse(out["allow"])
        self.assertEqual(out["reason"], "bo1_interregional_low_evidence")

    def test_bo1_interregional_sufficient_evidence_not_blocked(self):
        prof = SimpleNamespace(n_games=5)
        out = _gate(bo=1, pA=prof, pB=prof, interregional=True, n_team_eff=20.0, n_players_eff=8.0)
        # Not blocked by interregional_low_evidence — may proceed to default_allow
        self.assertNotEqual(out["reason"], "bo1_interregional_low_evidence")

    # BO1 interregional + patch + roster change
    def test_bo1_interregional_patch_roster_blocked(self):
        prof = SimpleNamespace(n_games=5)
        out = _gate(
            bo=1, pA=prof, pB=prof,
            interregional=True, n_team_eff=25.0, n_players_eff=10.0,
            roster_changed=True, patch_status="refit",
        )
        self.assertFalse(out["allow"])
        self.assertEqual(out["reason"], "bo1_interregional_patch_roster")

    # BO1 low evidence
    def test_bo1_low_evidence_blocked(self):
        prof = SimpleNamespace(n_games=3)  # n_min_prof=3 >= 3 so passes low_profile
        out = _gate(bo=1, pA=prof, pB=prof, n_team_eff=8.0)
        self.assertFalse(out["allow"])
        self.assertEqual(out["reason"], "bo1_low_evidence")

    # Default allow
    def test_bo1_sufficient_evidence_allow(self):
        prof = SimpleNamespace(n_games=8)
        out = _gate(bo=1, pA=prof, pB=prof, n_team_eff=15.0, n_players_eff=6.0)
        self.assertTrue(out["allow"])

    # Output contract
    def test_output_always_has_allow_and_reason(self):
        for bo in (1, 3, 5):
            out = _gate(bo=bo)
            self.assertIn("allow", out)
            self.assertIn("reason", out)

    def test_bo_normalised_to_1_3_5(self):
        # bo=2 → bo_i=1
        prof = SimpleNamespace(n_games=0)
        out = _gate(bo=2, pA=prof, pB=prof)
        self.assertEqual(out["bo"], 1)

    def test_none_bo_defaults_to_1(self):
        prof = SimpleNamespace(n_games=0)
        out = _gate(bo=None, pA=prof, pB=prof)
        self.assertEqual(out["bo"], 1)


# ---------------------------------------------------------------------------
# build_style_matchup_b_model_features — output contract
# ---------------------------------------------------------------------------

def _make_profile(n_games=15, **kwargs) -> SimpleNamespace:
    base = dict(
        n_games=n_games,
        kills_for=10.0, kills_against=8.0,
        towers_for=7.0, towers_against=5.0,
        dragons_for=2.5, dragons_against=2.0,
        barons_for=1.0, barons_against=0.8,
        inhib_for=0.5, inhib_against=0.3,
        time_mean=32.0, kills_mean=20.0,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _base_input(**overrides):
    """Minimal valid input for build_style_matchup_b_model_features."""
    base = dict(
        p_map_in=0.55,
        pA=_make_profile(),
        pB=_make_profile(kills_for=9.0, kills_against=9.0),
        bo=1,
        league_a="LCK",
        league_b="LCK",
        patch_status="stable",
        roster_changed=False,
        is_playoffs=False,
    )
    base.update(overrides)
    return base


class TestBuildStyleMatchupBModelFeatures(unittest.TestCase):

    def _feats(self, **kwargs) -> dict:
        """Return out["features"] sub-dict."""
        out = build_style_matchup_b_model_features(**_base_input(**kwargs))
        return out.get("features", out)

    def test_output_is_dict(self):
        out = build_style_matchup_b_model_features(**_base_input())
        self.assertIsInstance(out, dict)

    def test_output_has_features_key(self):
        out = build_style_matchup_b_model_features(**_base_input())
        self.assertIn("features", out)

    def test_all_feature_keys_present(self):
        feats = self._feats()
        for key in STYLE_MATCHUP_B_MODEL_FEATURES:
            self.assertIn(key, feats, f"Missing feature key: {key}")

    def test_all_values_finite(self):
        feats = self._feats()
        for k in STYLE_MATCHUP_B_MODEL_FEATURES:
            v = feats.get(k)
            if isinstance(v, float):
                self.assertTrue(math.isfinite(v), f"Non-finite value for {k}: {v}")

    def test_bo_flags_correct(self):
        feats = self._feats(bo=1)
        self.assertAlmostEqual(feats["bo_is_1"], 1.0)
        self.assertAlmostEqual(feats["bo_is_3"], 0.0)
        self.assertAlmostEqual(feats["bo_is_5"], 0.0)

    def test_bo3_flags(self):
        feats = self._feats(bo=3)
        self.assertAlmostEqual(feats["bo_is_3"], 1.0)
        self.assertAlmostEqual(feats["bo_is_1"], 0.0)

    def test_same_league_flag_true_when_same(self):
        feats = self._feats(league_a="LCK", league_b="LCK")
        self.assertAlmostEqual(feats["same_league"], 1.0)

    def test_same_league_flag_false_when_different(self):
        feats = self._feats(league_a="LCK", league_b="LCS")
        self.assertAlmostEqual(feats["same_league"], 0.0)

    def test_playoffs_flag(self):
        feats = self._feats(is_playoffs=True)
        self.assertAlmostEqual(feats["is_playoffs"], 1.0)

    def test_patch_refit_flag(self):
        self.assertAlmostEqual(self._feats(patch_status="refit")["patch_refit"], 1.0)
        self.assertAlmostEqual(self._feats(patch_status="stable")["patch_refit"], 0.0)

    def test_roster_changed_flag(self):
        feats = self._feats(roster_changed=True)
        self.assertAlmostEqual(feats["roster_changed"], 1.0)

    def test_does_not_crash_with_nan_profile(self):
        inp = _base_input()
        inp["pA"] = _make_profile(
            n_games=0,
            kills_for=float("nan"), kills_against=float("nan"),
            towers_for=float("nan"), towers_against=float("nan"),
            dragons_for=float("nan"), dragons_against=float("nan"),
            barons_for=float("nan"), barons_against=float("nan"),
            inhib_for=float("nan"), inhib_against=float("nan"),
            time_mean=float("nan"), kills_mean=float("nan"),
        )
        out = build_style_matchup_b_model_features(**inp)
        self.assertIsInstance(out, dict)


if __name__ == "__main__":
    unittest.main()
