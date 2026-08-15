"""Tests for app_core/lines_state.py — load/save helpers and defaults."""
import unittest

from app_core.lines_state import (
    LINE_FIELDS_DEFAULTS,
    load_lines_for_matchup,
    save_lines_for_matchup,
)


class TestLineFieldsDefaults(unittest.TestCase):

    def test_is_dict(self):
        self.assertIsInstance(LINE_FIELDS_DEFAULTS, dict)

    def test_has_kills_key(self):
        self.assertIn("lines_kills", LINE_FIELDS_DEFAULTS)

    def test_has_time_key(self):
        self.assertIn("lines_time", LINE_FIELDS_DEFAULTS)

    def test_all_values_are_strings(self):
        for v in LINE_FIELDS_DEFAULTS.values():
            self.assertIsInstance(v, str)

    def test_has_expected_count(self):
        self.assertGreater(len(LINE_FIELDS_DEFAULTS), 5)


class TestLoadLinesForMatchup(unittest.TestCase):

    def test_loads_stored_values(self):
        session_state = {}
        store_all = {"match1": {"lines_kills": "10.5,11.5"}}
        load_lines_for_matchup(session_state, store_all, "match1")
        self.assertEqual(session_state["lines_kills"], "10.5,11.5")

    def test_missing_key_uses_defaults(self):
        session_state = {}
        store_all = {}
        load_lines_for_matchup(session_state, store_all, "nonexistent")
        self.assertEqual(session_state["lines_kills"], "")

    def test_empty_store_uses_defaults_key(self):
        session_state = {}
        store_all = {"__defaults__": {"lines_kills": "9.5"}}
        load_lines_for_matchup(session_state, store_all, "other")
        self.assertEqual(session_state["lines_kills"], "9.5")

    def test_all_default_fields_set(self):
        session_state = {}
        store_all = {}
        load_lines_for_matchup(session_state, store_all, "none")
        for k in LINE_FIELDS_DEFAULTS:
            self.assertIn(k, session_state)

    def test_values_coerced_to_string(self):
        session_state = {}
        store_all = {"m": {"lines_kills": 10}}
        load_lines_for_matchup(session_state, store_all, "m")
        self.assertIsInstance(session_state["lines_kills"], str)

    def test_non_dict_store_falls_back_to_defaults(self):
        session_state = {}
        store_all = {"m": "invalid"}
        load_lines_for_matchup(session_state, store_all, "m")
        self.assertEqual(session_state["lines_kills"], "")


class TestSaveLinesForMatchup(unittest.TestCase):

    def test_saves_all_fields(self):
        session_state = {k: f"val_{k}" for k in LINE_FIELDS_DEFAULTS}
        store_all = {}
        save_lines_for_matchup(session_state, store_all, "match1")
        self.assertIn("match1", store_all)
        for k in LINE_FIELDS_DEFAULTS:
            self.assertIn(k, store_all["match1"])

    def test_saved_values_match_session(self):
        session_state = {"lines_kills": "10.5,11.5"}
        store_all = {}
        save_lines_for_matchup(session_state, store_all, "m1")
        self.assertEqual(store_all["m1"]["lines_kills"], "10.5,11.5")

    def test_missing_session_key_defaults_empty(self):
        session_state = {}
        store_all = {}
        save_lines_for_matchup(session_state, store_all, "m1")
        self.assertEqual(store_all["m1"]["lines_kills"], "")

    def test_overwrites_existing_entry(self):
        session_state = {"lines_kills": "new"}
        store_all = {"m1": {"lines_kills": "old"}}
        save_lines_for_matchup(session_state, store_all, "m1")
        self.assertEqual(store_all["m1"]["lines_kills"], "new")


if __name__ == "__main__":
    unittest.main()
