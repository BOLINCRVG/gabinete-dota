"""Tests for app_core/lineup_state.py — pure helper functions."""
import math
import unittest

from app_core.lineup_state import (
    lineup_key,
    norm_pos,
    norm_pos_ml,
    parse_pct,
    safe_key,
)


class TestSafeKey(unittest.TestCase):

    def test_alphanumeric_unchanged(self):
        self.assertEqual(safe_key("hello123"), "hello123")

    def test_spaces_replaced_with_underscore(self):
        result = safe_key("hello world")
        self.assertNotIn(" ", result)
        self.assertIn("_", result)

    def test_special_chars_replaced(self):
        result = safe_key("a!b@c#d")
        self.assertNotIn("!", result)
        self.assertNotIn("@", result)
        self.assertNotIn("#", result)

    def test_empty_string_returns_x(self):
        self.assertEqual(safe_key(""), "x")

    def test_none_returns_x(self):
        self.assertEqual(safe_key(None), "x")

    def test_multiple_specials_collapsed(self):
        result = safe_key("a---b")
        self.assertNotIn("--", result)

    def test_returns_string(self):
        self.assertIsInstance(safe_key("test"), str)


class TestNormPos(unittest.TestCase):

    def test_top(self):
        self.assertEqual(norm_pos("top"), "Top")

    def test_jungle(self):
        self.assertEqual(norm_pos("jungle"), "Jungle")

    def test_jg_alias(self):
        self.assertEqual(norm_pos("jg"), "Jungle")

    def test_mid(self):
        self.assertEqual(norm_pos("mid"), "Mid")

    def test_adc(self):
        self.assertEqual(norm_pos("adc"), "ADC")

    def test_bot_is_adc(self):
        self.assertEqual(norm_pos("bot"), "ADC")

    def test_support(self):
        self.assertEqual(norm_pos("support"), "Support")

    def test_sup_alias(self):
        self.assertEqual(norm_pos("sup"), "Support")

    def test_empty_returns_empty(self):
        self.assertEqual(norm_pos(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(norm_pos(None), "")

    def test_case_insensitive(self):
        self.assertEqual(norm_pos("TOP"), "Top")

    def test_unknown_returns_titlecased(self):
        result = norm_pos("carry")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


class TestNormPosMl(unittest.TestCase):

    def test_bot_is_bot_not_adc(self):
        self.assertEqual(norm_pos_ml("bot"), "Bot")

    def test_adc_is_bot(self):
        self.assertEqual(norm_pos_ml("adc"), "Bot")

    def test_top(self):
        self.assertEqual(norm_pos_ml("top"), "Top")

    def test_jungle(self):
        self.assertEqual(norm_pos_ml("jungle"), "Jungle")

    def test_mid(self):
        self.assertEqual(norm_pos_ml("mid"), "Mid")

    def test_support(self):
        self.assertEqual(norm_pos_ml("support"), "Support")

    def test_empty_returns_empty(self):
        self.assertEqual(norm_pos_ml(""), "")

    def test_differs_from_norm_pos_for_adc(self):
        self.assertNotEqual(norm_pos("adc"), norm_pos_ml("adc"))


class TestParsePct(unittest.TestCase):

    def test_percent_string(self):
        self.assertAlmostEqual(parse_pct("64.5%"), 0.645, places=5)

    def test_decimal_string(self):
        self.assertAlmostEqual(parse_pct("0.75"), 0.75, places=5)

    def test_empty_returns_nan(self):
        self.assertTrue(math.isnan(parse_pct("")))

    def test_none_returns_nan(self):
        self.assertTrue(math.isnan(parse_pct(None)))

    def test_invalid_returns_nan(self):
        self.assertTrue(math.isnan(parse_pct("abc")))

    def test_zero_percent(self):
        self.assertAlmostEqual(parse_pct("0%"), 0.0, places=5)

    def test_hundred_percent(self):
        self.assertAlmostEqual(parse_pct("100%"), 1.0, places=5)

    def test_returns_float(self):
        self.assertIsInstance(parse_pct("50%"), float)


class TestLineupKey(unittest.TestCase):

    def test_contains_season_and_teams(self):
        key = lineup_key("2025", "T1", "GEN")
        self.assertIn("2025", key)
        self.assertIn("T1", key)
        self.assertIn("GEN", key)

    def test_sorted_teams(self):
        key_ab = lineup_key("", "A", "B")
        key_ba = lineup_key("", "B", "A")
        self.assertEqual(key_ab, key_ba)

    def test_empty_season_no_double_colon(self):
        key = lineup_key("", "T1", "GEN")
        self.assertNotIn("::", key)

    def test_returns_string(self):
        self.assertIsInstance(lineup_key("2025", "T1", "GEN"), str)


if __name__ == "__main__":
    unittest.main()
