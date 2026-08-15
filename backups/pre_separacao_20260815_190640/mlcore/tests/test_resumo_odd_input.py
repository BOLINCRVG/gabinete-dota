"""Tests for _parse_odd_input (comma-decimal odd parsing in Resumo)."""
from __future__ import annotations

import unittest

from app_core.resumo_tab_core import _parse_odd_input


class TestParseOddInput(unittest.TestCase):

    def test_comma_decimal(self):
        self.assertEqual(_parse_odd_input("1,83"), 1.83)
        self.assertEqual(_parse_odd_input("1,50"), 1.50)

    def test_dot_decimal(self):
        self.assertEqual(_parse_odd_input("1.83"), 1.83)

    def test_whitespace_stripped(self):
        self.assertEqual(_parse_odd_input("  2,00 "), 2.0)

    def test_empty_and_placeholder(self):
        self.assertIsNone(_parse_odd_input(""))
        self.assertIsNone(_parse_odd_input("-"))
        self.assertIsNone(_parse_odd_input(None))

    def test_unparseable(self):
        self.assertIsNone(_parse_odd_input("abc"))

    def test_numeric_passthrough(self):
        self.assertEqual(_parse_odd_input(1.52), 1.52)


if __name__ == "__main__":
    unittest.main()
