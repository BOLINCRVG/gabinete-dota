"""Tests for app_core/lines_parsing_core.py — pure bet-line parsing helpers."""
import unittest

from app_core.lines_parsing_core import (
    parse_quick_lines_kv,
    parse_single_float,
    parse_single_time_min,
)


# ---------------------------------------------------------------------------
# parse_quick_lines_kv
# ---------------------------------------------------------------------------

class TestParseQuickLinesKv(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(parse_quick_lines_kv(""), {})

    def test_none_returns_empty(self):
        self.assertEqual(parse_quick_lines_kv(None), {})

    def test_basic_canonical_key(self):
        result = parse_quick_lines_kv("kills=29.5")
        self.assertEqual(result["kills"], "29.5")

    def test_multiple_keys(self):
        result = parse_quick_lines_kv("kills=29.5 towers=10.5")
        self.assertEqual(result["kills"], "29.5")
        self.assertEqual(result["towers"], "10.5")

    def test_alias_k_maps_to_kills(self):
        result = parse_quick_lines_kv("k=25.5")
        self.assertIn("kills", result)
        self.assertEqual(result["kills"], "25.5")

    def test_alias_t_maps_to_towers(self):
        result = parse_quick_lines_kv("t=11.5")
        self.assertIn("towers", result)

    def test_alias_drg_maps_to_dragons(self):
        result = parse_quick_lines_kv("drg=4.5")
        self.assertIn("dragons", result)

    def test_alias_bar_maps_to_barons(self):
        result = parse_quick_lines_kv("bar=1.5")
        self.assertIn("barons", result)

    def test_alias_inib_maps_to_inhibitors(self):
        result = parse_quick_lines_kv("inib=1.5")
        self.assertIn("inhibitors", result)

    def test_alias_tm_maps_to_time(self):
        result = parse_quick_lines_kv("tm=33:00")
        self.assertIn("time", result)
        self.assertEqual(result["time"], "33:00")

    def test_alias_tempo_maps_to_time(self):
        result = parse_quick_lines_kv("tempo=30:30")
        self.assertIn("time", result)

    def test_unknown_key_ignored(self):
        result = parse_quick_lines_kv("xyz=10.0 kills=25.5")
        self.assertNotIn("xyz", result)
        self.assertIn("kills", result)

    def test_no_equals_token_ignored(self):
        result = parse_quick_lines_kv("kills 25.5")
        self.assertEqual(result, {})

    def test_comma_separated_works(self):
        result = parse_quick_lines_kv("kills=29.5,towers=10.5")
        self.assertIn("kills", result)
        self.assertIn("towers", result)

    def test_full_line(self):
        result = parse_quick_lines_kv("kills=29.5 towers=10.5 drg=4.5 bar=1.5 inib=1.5 tm=33:00")
        self.assertEqual(len(result), 6)
        self.assertIn("kills", result)
        self.assertIn("time", result)


# ---------------------------------------------------------------------------
# parse_single_float
# ---------------------------------------------------------------------------

class TestParseSingleFloat(unittest.TestCase):

    def test_empty_returns_none(self):
        self.assertIsNone(parse_single_float(""))

    def test_none_returns_none(self):
        self.assertIsNone(parse_single_float(None))

    def test_single_value(self):
        self.assertAlmostEqual(parse_single_float("1.85"), 1.85)

    def test_returns_first_of_multiple(self):
        result = parse_single_float("1.85 2.10 3.00")
        self.assertAlmostEqual(result, 1.85)

    def test_invalid_text_returns_none(self):
        self.assertIsNone(parse_single_float("abc"))

    def test_integer_parsed(self):
        self.assertAlmostEqual(parse_single_float("2"), 2.0)


# ---------------------------------------------------------------------------
# parse_single_time_min
# ---------------------------------------------------------------------------

class TestParseSingleTimeMin(unittest.TestCase):

    def test_empty_returns_none(self):
        self.assertIsNone(parse_single_time_min(""))

    def test_none_returns_none(self):
        self.assertIsNone(parse_single_time_min(None))

    def test_mmss_30(self):
        result = parse_single_time_min("28:30")
        self.assertAlmostEqual(result, 28.5, places=4)

    def test_mmss_00(self):
        result = parse_single_time_min("33:00")
        self.assertAlmostEqual(result, 33.0, places=4)

    def test_integer_minutes(self):
        result = parse_single_time_min("32")
        self.assertAlmostEqual(result, 32.0, places=4)

    def test_decimal_half(self):
        result = parse_single_time_min("30.5")
        self.assertAlmostEqual(result, 30.5, places=4)

    def test_returns_first_of_multiple(self):
        result = parse_single_time_min("28:00 30:00")
        self.assertAlmostEqual(result, 28.0, places=4)

    def test_invalid_format_returns_none(self):
        self.assertIsNone(parse_single_time_min("abc"))


if __name__ == "__main__":
    unittest.main()
