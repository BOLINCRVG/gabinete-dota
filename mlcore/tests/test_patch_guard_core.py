"""Tests for app_core/patch_guard_core.py — patch_changed_leagues_set."""
import unittest

from app_core.patch_guard_core import patch_changed_leagues_set


class TestPatchChangedLeaguesSet(unittest.TestCase):

    def test_none_returns_empty_set(self):
        result = patch_changed_leagues_set(None)
        self.assertEqual(result, set())

    def test_non_list_returns_empty_set(self):
        result = patch_changed_leagues_set("LCK")
        self.assertEqual(result, set())
        result = patch_changed_leagues_set({"league": "LCK"})
        self.assertEqual(result, set())

    def test_empty_list_returns_empty_set(self):
        result = patch_changed_leagues_set([])
        self.assertEqual(result, set())

    def test_single_league_row(self):
        result = patch_changed_leagues_set([{"league": "LCK"}])
        self.assertIn("LCK", result)

    def test_multiple_leagues(self):
        rows = [{"league": "LCK"}, {"league": "LPL"}, {"league": "LEC"}]
        result = patch_changed_leagues_set(rows)
        self.assertEqual(result, {"LCK", "LPL", "LEC"})

    def test_deduplicates(self):
        rows = [{"league": "LCK"}, {"league": "LCK"}]
        result = patch_changed_leagues_set(rows)
        self.assertEqual(result, {"LCK"})

    def test_uppercase_normalized(self):
        rows = [{"league": "lck"}, {"league": "Lpl"}]
        result = patch_changed_leagues_set(rows)
        self.assertIn("LCK", result)
        self.assertIn("LPL", result)

    def test_strips_whitespace(self):
        rows = [{"league": "  LCK  "}]
        result = patch_changed_leagues_set(rows)
        self.assertIn("LCK", result)

    def test_league_key_alias(self):
        # 'League' (capital L) is also accepted
        rows = [{"League": "LCK"}]
        result = patch_changed_leagues_set(rows)
        self.assertIn("LCK", result)

    def test_empty_league_excluded(self):
        rows = [{"league": ""}, {"league": "LCK"}]
        result = patch_changed_leagues_set(rows)
        self.assertEqual(result, {"LCK"})

    def test_non_dict_rows_skipped(self):
        rows = ["LCK", {"league": "LPL"}, 42]
        result = patch_changed_leagues_set(rows)
        self.assertEqual(result, {"LPL"})

    def test_returns_set_type(self):
        result = patch_changed_leagues_set([{"league": "LCK"}])
        self.assertIsInstance(result, set)


if __name__ == "__main__":
    unittest.main()
