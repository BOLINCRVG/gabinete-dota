"""Tests for app_core/tab_state_core.py — session-state helper functions."""
import unittest

from app_core.tab_state_core import (
    blank_space_label,
    ensure_blank_keys,
    normalize_select_key,
    reset_keys,
    select_placeholder,
)


class TestEnsureBlankKeys(unittest.TestCase):

    def test_missing_key_set_to_empty(self):
        ss = {}
        ensure_blank_keys(ss, ["league"])
        self.assertEqual(ss["league"], "")

    def test_all_invalid_value_replaced(self):
        state = {"a": "All", "b": None, "c": "ok"}
        ensure_blank_keys(state, ["a", "b", "c", "d"])
        self.assertEqual(state["a"], "")
        self.assertEqual(state["b"], "")
        self.assertEqual(state["c"], "ok")
        self.assertEqual(state["d"], "")

    def test_none_replaced(self):
        ss = {"league": None}
        ensure_blank_keys(ss, ["league"])
        self.assertEqual(ss["league"], "")

    def test_valid_value_unchanged(self):
        ss = {"league": "LCK"}
        ensure_blank_keys(ss, ["league"])
        self.assertEqual(ss["league"], "LCK")

    def test_multiple_keys(self):
        ss = {"a": None, "b": "All"}
        ensure_blank_keys(ss, ["a", "b"])
        self.assertEqual(ss["a"], "")
        self.assertEqual(ss["b"], "")

    def test_empty_keys_list_no_change(self):
        ss = {"x": "keep"}
        ensure_blank_keys(ss, [])
        self.assertEqual(ss["x"], "keep")

    def test_none_value_in_invalid_values(self):
        ss = {"k": "None"}
        ensure_blank_keys(ss, ["k"])
        self.assertEqual(ss["k"], "")


class TestResetKeys(unittest.TestCase):

    def test_resets_to_empty_by_default(self):
        ss = {"league": "LCK", "year": "2025"}
        reset_keys(ss, ["league", "year"])
        self.assertEqual(ss["league"], "")
        self.assertEqual(ss["year"], "")

    def test_resets_to_custom_value(self):
        state = {"c": "old", "d": "old"}
        reset_keys(state, ["c", "d"], value="x")
        self.assertEqual(state["c"], "x")
        self.assertEqual(state["d"], "x")

    def test_empty_keys_list_no_change(self):
        ss = {"a": "keep"}
        reset_keys(ss, [])
        self.assertEqual(ss["a"], "keep")

    def test_creates_key_if_missing(self):
        ss = {}
        reset_keys(ss, ["new_key"])
        self.assertEqual(ss["new_key"], "")


class TestNormalizeSelectKey(unittest.TestCase):

    def test_valid_value_returned(self):
        state = {"league": "LCK"}
        self.assertEqual(normalize_select_key(state, "league", ["", "LCK", "LPL"]), "LCK")

    def test_invalid_value_replaced_by_fallback(self):
        state = {"league": "LCK"}
        self.assertEqual(normalize_select_key(state, "league", ["", "LPL"]), "")
        self.assertEqual(state["league"], "")

    def test_missing_key_returns_fallback(self):
        ss = {}
        result = normalize_select_key(ss, "league", ["LCK"])
        self.assertEqual(result, "")

    def test_custom_fallback(self):
        ss = {"league": "BAD"}
        result = normalize_select_key(ss, "league", ["LCK"], fallback="LCK")
        self.assertEqual(result, "LCK")

    def test_returns_string(self):
        ss = {"x": "A"}
        result = normalize_select_key(ss, "x", ["A", "B"])
        self.assertIsInstance(result, str)


class TestSelectPlaceholder(unittest.TestCase):

    def test_empty_returns_default_label(self):
        self.assertEqual(select_placeholder(""), "Selecione ...")

    def test_non_empty_returns_value(self):
        self.assertEqual(select_placeholder("LCK"), "LCK")

    def test_custom_blank_label(self):
        self.assertEqual(select_placeholder("", blank_label="Choose..."), "Choose...")

    def test_returns_string(self):
        self.assertIsInstance(select_placeholder("X"), str)


class TestBlankSpaceLabel(unittest.TestCase):

    def test_empty_returns_space(self):
        self.assertEqual(blank_space_label(""), " ")

    def test_non_empty_returned_as_is(self):
        self.assertEqual(blank_space_label("LCK"), "LCK")

    def test_lpl_returned_unchanged(self):
        self.assertEqual(blank_space_label("LPL"), "LPL")

    def test_returns_string(self):
        self.assertIsInstance(blank_space_label(""), str)


if __name__ == "__main__":
    unittest.main()
