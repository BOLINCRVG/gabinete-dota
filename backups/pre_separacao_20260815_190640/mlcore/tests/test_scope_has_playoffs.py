"""Tests for scope_has_playoffs (conditional Playoffs selector)."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.tab_filters_core import scope_has_playoffs


def _df():
    return pd.DataFrame([
        {"league": "LPL", "year": 2026, "split": "Split 1", "playoffs": 0},
        {"league": "LPL", "year": 2026, "split": "Split 1", "playoffs": 1},
        {"league": "LPL", "year": 2026, "split": "Split 2", "playoffs": 0},
        {"league": "LCK", "year": 2026, "split": "Rounds 1-2", "playoffs": 0},
    ])


class TestScopeHasPlayoffs(unittest.TestCase):

    def test_true_when_split_has_playoffs(self):
        self.assertTrue(scope_has_playoffs(_df(), league="LPL", year=2026, split="Split 1"))

    def test_false_when_split_has_no_playoffs(self):
        self.assertFalse(scope_has_playoffs(_df(), league="LPL", year=2026, split="Split 2"))
        self.assertFalse(scope_has_playoffs(_df(), league="LCK", year=2026, split="Rounds 1-2"))

    def test_false_when_scope_empty(self):
        self.assertFalse(scope_has_playoffs(_df(), league="NONEXISTENT"))

    def test_blank_filters_ignored(self):
        # No filters → any playoff game anywhere makes it True.
        self.assertTrue(scope_has_playoffs(_df()))

    def test_missing_column_returns_false(self):
        self.assertFalse(scope_has_playoffs(pd.DataFrame([{"league": "LPL"}]), league="LPL"))

    def test_none_df_returns_false(self):
        self.assertFalse(scope_has_playoffs(None, league="LPL"))


if __name__ == "__main__":
    unittest.main()
