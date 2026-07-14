"""Tests for app_core/champions_tab_core.py — pure name parser helper."""
import unittest

from app_core.champions_tab_core import _parse_names


class TestParseNames(unittest.TestCase):

    def test_newline_separated(self):
        result = _parse_names("Jinx\nAhri\nLux")
        self.assertEqual(result, ["Jinx", "Ahri", "Lux"])

    def test_comma_separated(self):
        result = _parse_names("Jinx,Ahri,Lux")
        self.assertEqual(result, ["Jinx", "Ahri", "Lux"])

    def test_semicolon_separated(self):
        result = _parse_names("Jinx;Ahri;Lux")
        self.assertEqual(result, ["Jinx", "Ahri", "Lux"])

    def test_tab_separated(self):
        result = _parse_names("Jinx\tAhri\tLux")
        self.assertEqual(result, ["Jinx", "Ahri", "Lux"])

    def test_strips_whitespace(self):
        result = _parse_names("  Jinx  ,  Ahri  ")
        self.assertEqual(result, ["Jinx", "Ahri"])

    def test_empty_string_returns_empty(self):
        self.assertEqual(_parse_names(""), [])

    def test_none_returns_empty(self):
        self.assertEqual(_parse_names(None), [])

    def test_single_name(self):
        result = _parse_names("Faker")
        self.assertEqual(result, ["Faker"])

    def test_mixed_delimiters(self):
        result = _parse_names("Jinx\nAhri,Lux;Thresh")
        self.assertEqual(result, ["Jinx", "Ahri", "Lux", "Thresh"])

    def test_empty_segments_skipped(self):
        result = _parse_names("Jinx,,Ahri")
        self.assertEqual(result, ["Jinx", "Ahri"])

    def test_returns_list(self):
        self.assertIsInstance(_parse_names("Jinx"), list)

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(_parse_names("   "), [])

    def test_multiple_newlines_skipped(self):
        result = _parse_names("Jinx\n\nAhri")
        self.assertEqual(result, ["Jinx", "Ahri"])


if __name__ == "__main__":
    unittest.main()
