"""Tests for app_core/market.py — parsing and probability helpers."""
import math
import unittest

from app_core.market import (
    combine_ml_totals_probs,
    matchup_key,
    odd_from_prob,
    parse_lines_text,
    parse_time_lines_text,
    prob_from_odd,
)


# ---------------------------------------------------------------------------
# parse_lines_text
# ---------------------------------------------------------------------------

class TestParseLinesText(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(parse_lines_text(""), [])

    def test_none_returns_empty(self):
        self.assertEqual(parse_lines_text(None), [])

    def test_single_float(self):
        self.assertEqual(parse_lines_text("1.85"), [1.85])

    def test_comma_separated(self):
        result = parse_lines_text("1.85, 2.10, 1.90")
        self.assertEqual(result, [1.85, 2.10, 1.90])

    def test_space_separated(self):
        result = parse_lines_text("1.85 2.10 1.90")
        self.assertEqual(len(result), 3)

    def test_semicolon_separated(self):
        result = parse_lines_text("1.85;2.10")
        self.assertEqual(result, [1.85, 2.10])

    def test_skips_non_numeric(self):
        result = parse_lines_text("1.85 abc 2.10")
        self.assertEqual(result, [1.85, 2.10])

    def test_integers_parsed(self):
        result = parse_lines_text("2 3")
        self.assertEqual(result, [2.0, 3.0])


# ---------------------------------------------------------------------------
# parse_time_lines_text
# ---------------------------------------------------------------------------

class TestParseTimeLinesText(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(parse_time_lines_text(""), [])

    def test_mmss_format_30(self):
        # "28:30" → 28 + 30/60 = 28.5
        result = parse_time_lines_text("28:30")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 28.5, places=4)

    def test_mmss_format_00(self):
        # "28:00" → 28.0
        result = parse_time_lines_text("28:00")
        self.assertAlmostEqual(result[0], 28.0, places=4)

    def test_integer_minutes(self):
        result = parse_time_lines_text("32")
        self.assertEqual(result, [32.0])

    def test_decimal_half(self):
        # "28.5" → 28.5
        result = parse_time_lines_text("28.5")
        self.assertAlmostEqual(result[0], 28.5, places=4)

    def test_decimal_zero(self):
        result = parse_time_lines_text("30.0")
        self.assertAlmostEqual(result[0], 30.0, places=4)

    def test_invalid_format_skipped(self):
        # "abc" and "28.3" (invalid decimal) not matched → skipped
        result = parse_time_lines_text("abc 28.3")
        self.assertEqual(result, [])

    def test_multiple_values(self):
        result = parse_time_lines_text("28:00 30 31.5")
        self.assertEqual(len(result), 3)


# ---------------------------------------------------------------------------
# matchup_key
# ---------------------------------------------------------------------------

class TestMatchupKey(unittest.TestCase):

    def test_sorted_alphabetically(self):
        k1 = matchup_key("S2025", "T1", "G2")
        k2 = matchup_key("S2025", "G2", "T1")
        self.assertEqual(k1, k2)

    def test_includes_season(self):
        k = matchup_key("S2025", "T1", "G2")
        self.assertIn("S2025", k)

    def test_no_season_omits_prefix(self):
        k = matchup_key("", "T1", "G2")
        self.assertNotIn("::", k)

    def test_none_season_omits_prefix(self):
        k = matchup_key(None, "T1", "G2")
        self.assertNotIn("::", k)

    def test_teams_pipe_separated(self):
        k = matchup_key("", "T1", "G2")
        self.assertIn("|", k)

    def test_strips_whitespace(self):
        k1 = matchup_key("S25", " T1 ", "G2 ")
        k2 = matchup_key("S25", "T1", "G2")
        self.assertEqual(k1, k2)


# ---------------------------------------------------------------------------
# odd_from_prob / prob_from_odd
# ---------------------------------------------------------------------------

class TestOddFromProb(unittest.TestCase):

    def test_half_gives_2(self):
        result = odd_from_prob(0.5)
        self.assertAlmostEqual(result, 2.0, places=4)

    def test_high_prob_gives_low_odd(self):
        result = odd_from_prob(0.80)
        self.assertLess(result, 1.30)

    def test_low_prob_gives_high_odd(self):
        result = odd_from_prob(0.10)
        self.assertGreater(result, 8.0)

    def test_none_returns_none(self):
        self.assertIsNone(odd_from_prob(None))

    def test_nan_returns_none(self):
        self.assertIsNone(odd_from_prob(float("nan")))

    def test_result_is_finite(self):
        result = odd_from_prob(0.65)
        self.assertTrue(math.isfinite(result))


class TestProbFromOdd(unittest.TestCase):

    def test_2_gives_half(self):
        result = prob_from_odd(2.0)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_odd_1_returns_none(self):
        # odds <= 1.0 → invalid (decimal_odds_to_prob returns NaN)
        self.assertIsNone(prob_from_odd(1.0))

    def test_high_odd_gives_low_prob(self):
        result = prob_from_odd(10.0)
        self.assertLess(result, 0.15)

    def test_round_trip(self):
        p = 0.65
        odd = odd_from_prob(p)
        p_back = prob_from_odd(odd)
        self.assertAlmostEqual(p_back, p, places=4)

    def test_result_in_unit_interval(self):
        for odd in (1.5, 2.0, 3.0, 5.0, 10.0):
            p = prob_from_odd(odd)
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)


# ---------------------------------------------------------------------------
# combine_ml_totals_probs
# ---------------------------------------------------------------------------

class TestCombineMlTotalsProbs(unittest.TestCase):

    def test_independence_baseline(self):
        # No conditional input, dep=0 → P(ml&over) ≈ p_ml * p_over
        p_ml, p_over = 0.60, 0.50
        joint, under = combine_ml_totals_probs(p_ml, p_over)
        self.assertAlmostEqual(joint, 0.30, places=4)
        self.assertAlmostEqual(under, 0.30, places=4)

    def test_joint_plus_under_equals_p_ml(self):
        p_ml, p_over = 0.65, 0.55
        joint, under = combine_ml_totals_probs(p_ml, p_over)
        self.assertAlmostEqual(joint + under, p_ml, places=4)

    def test_with_conditional_overrides_baseline(self):
        # P(Over|ML)=0.70 → P(ML & Over) ≈ 0.60 * 0.70 = 0.42
        joint, _ = combine_ml_totals_probs(0.60, 0.50, p_over_given_ml=0.70)
        self.assertGreater(joint, 0.30)  # stronger than independence

    def test_positive_dependence_increases_joint(self):
        joint_dep, _ = combine_ml_totals_probs(0.60, 0.50, dependence=0.5)
        joint_ind, _ = combine_ml_totals_probs(0.60, 0.50, dependence=0.0)
        self.assertGreater(joint_dep, joint_ind)

    def test_negative_dependence_decreases_joint(self):
        joint_dep, _ = combine_ml_totals_probs(0.60, 0.50, dependence=-0.5)
        joint_ind, _ = combine_ml_totals_probs(0.60, 0.50, dependence=0.0)
        self.assertLess(joint_dep, joint_ind)

    def test_joint_within_frechet_bounds(self):
        p_ml, p_over = 0.70, 0.60
        joint, _ = combine_ml_totals_probs(p_ml, p_over)
        low = max(0.0, p_ml + p_over - 1.0)
        high = min(p_ml, p_over)
        self.assertGreaterEqual(joint, low - 1e-9)
        self.assertLessEqual(joint, high + 1e-9)

    def test_nan_input_returns_nan(self):
        joint, under = combine_ml_totals_probs(float("nan"), 0.5)
        self.assertTrue(math.isnan(joint))
        self.assertTrue(math.isnan(under))

    def test_result_non_negative(self):
        joint, under = combine_ml_totals_probs(0.40, 0.30)
        self.assertGreaterEqual(joint, 0.0)
        self.assertGreaterEqual(under, 0.0)


if __name__ == "__main__":
    unittest.main()
