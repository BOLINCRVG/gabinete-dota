"""Tests for app_core/market.py — pure helper functions."""
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


class TestParseLinesText(unittest.TestCase):

    def test_comma_separated(self):
        self.assertEqual(parse_lines_text("1.5,2.5,3.5"), [1.5, 2.5, 3.5])

    def test_space_separated(self):
        self.assertEqual(parse_lines_text("1.5 2.5 3.5"), [1.5, 2.5, 3.5])

    def test_semicolon_separated(self):
        self.assertEqual(parse_lines_text("1.5;2.5"), [1.5, 2.5])

    def test_empty_string_returns_empty(self):
        self.assertEqual(parse_lines_text(""), [])

    def test_none_string_returns_empty(self):
        self.assertEqual(parse_lines_text(None), [])

    def test_invalid_tokens_skipped(self):
        result = parse_lines_text("1.5, abc, 2.5")
        self.assertEqual(result, [1.5, 2.5])

    def test_returns_list_of_floats(self):
        result = parse_lines_text("10, 20")
        self.assertIsInstance(result[0], float)

    def test_single_value(self):
        self.assertEqual(parse_lines_text("5.0"), [5.0])

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(parse_lines_text("   "), [])

    def test_decimal_comma_single_value(self):
        # Brazilian decimal: "5,5" means 5.5, NOT [5, 5].
        self.assertEqual(parse_lines_text("5,5"), [5.5])
        self.assertEqual(parse_lines_text("1,5"), [1.5])
        self.assertEqual(parse_lines_text("27,5"), [27.5])

    def test_decimal_comma_multiple_with_space(self):
        # Comma+space separates; decimal comma kept within each token.
        self.assertEqual(parse_lines_text("27,5, 30,5"), [27.5, 30.5])

    def test_dot_decimals_with_comma_separator_no_space(self):
        # Dots are decimals → the comma must be a separator.
        self.assertEqual(parse_lines_text("11.5,12.5"), [11.5, 12.5])

    def test_comma_space_forces_separation_of_integers(self):
        self.assertEqual(parse_lines_text("1, 5"), [1.0, 5.0])

    def test_time_decimal_comma(self):
        # "32,5" minutes means 32.5, not 32 and 5.
        self.assertEqual(parse_time_lines_text("32,5"), [32.5])
        self.assertEqual(parse_time_lines_text("32,5, 33"), [32.5, 33.0])


class TestParseTimeLinesText(unittest.TestCase):

    def test_mmss_format_00(self):
        result = parse_time_lines_text("30:00")
        self.assertAlmostEqual(result[0], 30.0)

    def test_mmss_format_30(self):
        result = parse_time_lines_text("32:30")
        self.assertAlmostEqual(result[0], 32.5)

    def test_integer_string(self):
        self.assertEqual(parse_time_lines_text("33"), [33.0])

    def test_decimal_with_0(self):
        self.assertEqual(parse_time_lines_text("30.0"), [30.0])

    def test_decimal_with_5(self):
        self.assertEqual(parse_time_lines_text("30.5"), [30.5])

    def test_empty_string_returns_empty(self):
        self.assertEqual(parse_time_lines_text(""), [])

    def test_invalid_mm_ss_skipped(self):
        result = parse_time_lines_text("30:15")
        self.assertEqual(result, [])

    def test_multiple_entries(self):
        result = parse_time_lines_text("30:00 32:30")
        self.assertEqual(len(result), 2)


class TestMatchupKey(unittest.TestCase):

    def test_basic_key_contains_teams(self):
        key = matchup_key("2025", "T1", "GEN")
        self.assertIn("T1", key)
        self.assertIn("GEN", key)
        self.assertIn("2025", key)

    def test_sorted_teams(self):
        key_ab = matchup_key("", "A", "B")
        key_ba = matchup_key("", "B", "A")
        self.assertEqual(key_ab, key_ba)

    def test_empty_season_no_double_colon(self):
        key = matchup_key("", "T1", "GEN")
        self.assertNotIn("::", key)

    def test_returns_string(self):
        self.assertIsInstance(matchup_key("2025", "T1", "GEN"), str)


class TestOddFromProb(unittest.TestCase):

    def test_half_probability_gives_two(self):
        result = odd_from_prob(0.5)
        self.assertAlmostEqual(result, 2.0, places=4)

    def test_nan_returns_none(self):
        self.assertIsNone(odd_from_prob(float("nan")))

    def test_inf_returns_none(self):
        self.assertIsNone(odd_from_prob(float("inf")))

    def test_none_returns_none(self):
        self.assertIsNone(odd_from_prob(None))

    def test_returns_float(self):
        result = odd_from_prob(0.6)
        self.assertIsInstance(result, float)

    def test_higher_prob_lower_odds(self):
        self.assertLess(odd_from_prob(0.7), odd_from_prob(0.3))


class TestProbFromOdd(unittest.TestCase):

    def test_two_gives_half(self):
        result = prob_from_odd(2.0)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_nan_returns_none(self):
        self.assertIsNone(prob_from_odd(float("nan")))

    def test_returns_float(self):
        result = prob_from_odd(1.85)
        self.assertIsInstance(result, float)

    def test_result_between_0_and_1(self):
        result = prob_from_odd(1.85)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)


class TestCombineMlTotalsProbs(unittest.TestCase):

    def test_independence_baseline(self):
        p_over_joint, p_under_joint = combine_ml_totals_probs(0.60, 0.55)
        self.assertAlmostEqual(float(p_over_joint), 0.33, places=6)
        self.assertAlmostEqual(float(p_under_joint), 0.27, places=6)

    def test_with_conditional(self):
        p_over_joint, p_under_joint = combine_ml_totals_probs(
            0.62, 0.53, p_over_given_ml=0.70,
        )
        self.assertAlmostEqual(float(p_over_joint + p_under_joint), 0.62, places=6)

    def test_clamps_frechet_bounds(self):
        p_over_joint, p_under_joint = combine_ml_totals_probs(0.8, 0.8, p_over_given_ml=0.0)
        self.assertAlmostEqual(float(p_over_joint), 0.6, places=6)
        self.assertAlmostEqual(float(p_under_joint), 0.2, places=6)

    def test_dependence_extremes(self):
        p_over_pos, _ = combine_ml_totals_probs(0.40, 0.70, dependence=1.0)
        p_over_neg, _ = combine_ml_totals_probs(0.40, 0.70, dependence=-1.0)
        self.assertAlmostEqual(float(p_over_pos), 0.40, places=6)
        self.assertAlmostEqual(float(p_over_neg), 0.10, places=6)

    def test_returns_tuple_of_two(self):
        result = combine_ml_totals_probs(0.6, 0.5)
        self.assertEqual(len(result), 2)

    def test_nan_input_returns_nan(self):
        a, b = combine_ml_totals_probs(float("nan"), 0.5)
        self.assertTrue(math.isnan(a))
        self.assertTrue(math.isnan(b))

    def test_results_sum_to_p_ml(self):
        p_ml = 0.65
        a, b = combine_ml_totals_probs(p_ml, 0.55, dependence=0.0)
        self.assertAlmostEqual(a + b, p_ml, places=5)

    def test_positive_dependence_increases_joint(self):
        base, _ = combine_ml_totals_probs(0.6, 0.5, dependence=0.0)
        dep, _ = combine_ml_totals_probs(0.6, 0.5, dependence=0.9)
        self.assertGreaterEqual(dep, base)

    def test_results_are_finite(self):
        a, b = combine_ml_totals_probs(0.6, 0.4)
        self.assertTrue(math.isfinite(a))
        self.assertTrue(math.isfinite(b))

    def test_results_between_0_and_1(self):
        a, b = combine_ml_totals_probs(0.6, 0.4)
        self.assertGreaterEqual(a, 0.0)
        self.assertLessEqual(a, 1.0)
        self.assertGreaterEqual(b, 0.0)
        self.assertLessEqual(b, 1.0)


if __name__ == "__main__":
    unittest.main()
