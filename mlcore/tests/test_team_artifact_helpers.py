"""Tests for mlcore/team_artifact_normalize.py and mlcore/team_artifact_validate.py."""
import math
import unittest

from mlcore.team_artifact_normalize import _norm_team_key, _norm_upper_key, _patch_key
from mlcore.team_artifact_validate import _coerce_float_map, _param_map_sig


class TestNormTeamKey(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(_norm_team_key(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(_norm_team_key(None), "")

    def test_lowercase_alphanumeric(self):
        self.assertEqual(_norm_team_key("T1"), "t1")

    def test_spaces_removed(self):
        self.assertEqual(_norm_team_key("Gen G"), "geng")

    def test_special_chars_removed(self):
        result = _norm_team_key("T1 Esports!")
        self.assertEqual(result, "t1esports")

    def test_accented_chars_normalized(self):
        result = _norm_team_key("Équipe")
        self.assertNotIn("é", result)
        self.assertNotIn("É", result)

    def test_strips_whitespace(self):
        self.assertEqual(_norm_team_key("  T1  "), "t1")

    def test_returns_string(self):
        self.assertIsInstance(_norm_team_key("LCK"), str)


class TestPatchKey(unittest.TestCase):

    def test_extracts_major_minor(self):
        self.assertEqual(_patch_key("Patch 14.5"), "14.5")

    def test_plain_version(self):
        self.assertEqual(_patch_key("14.1"), "14.1")

    def test_empty_returns_empty(self):
        self.assertEqual(_patch_key(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(_patch_key(None), "")

    def test_no_digits_returns_as_is(self):
        result = _patch_key("unknown")
        self.assertIsInstance(result, str)


class TestNormUpperKey(unittest.TestCase):

    def test_lowercases_to_upper(self):
        self.assertEqual(_norm_upper_key("lck"), "LCK")

    def test_strips_whitespace(self):
        self.assertEqual(_norm_upper_key("  lck  "), "LCK")

    def test_none_returns_empty(self):
        self.assertEqual(_norm_upper_key(None), "")

    def test_empty_returns_empty(self):
        self.assertEqual(_norm_upper_key(""), "")

    def test_returns_string(self):
        self.assertIsInstance(_norm_upper_key("x"), str)


class TestCoerceFloatMap(unittest.TestCase):

    def test_basic_conversion(self):
        out = _coerce_float_map({"lck": "0.5", "lpl": 0.7})
        self.assertAlmostEqual(out.get("LCK", 0.0), 0.5, places=6)
        self.assertAlmostEqual(out.get("LPL", 0.0), 0.7, places=6)

    def test_non_finite_skipped(self):
        out = _coerce_float_map({"lck": float("nan"), "lpl": 0.5})
        self.assertNotIn("LCK", out)
        self.assertIn("LPL", out)

    def test_non_dict_returns_empty(self):
        self.assertEqual(_coerce_float_map(None), {})
        self.assertEqual(_coerce_float_map([]), {})

    def test_empty_key_skipped(self):
        out = _coerce_float_map({"": 0.5})
        self.assertEqual(out, {})

    def test_returns_dict(self):
        self.assertIsInstance(_coerce_float_map({}), dict)

    def test_keys_normalized_to_upper(self):
        out = _coerce_float_map({"lck": 0.5})
        self.assertIn("LCK", out)
        self.assertNotIn("lck", out)


class TestParamMapSig(unittest.TestCase):

    def test_sorted_tuple(self):
        sig = _param_map_sig({"lpl": 0.8, "lck": 0.5})
        keys = [k for k, _ in sig]
        self.assertEqual(keys, sorted(keys))

    def test_empty_dict_returns_empty_tuple(self):
        self.assertEqual(_param_map_sig({}), ())

    def test_none_returns_empty_tuple(self):
        self.assertEqual(_param_map_sig(None), ())

    def test_same_content_same_sig(self):
        s1 = _param_map_sig({"lck": 0.5, "lpl": 0.8})
        s2 = _param_map_sig({"lpl": 0.8, "lck": 0.5})
        self.assertEqual(s1, s2)

    def test_returns_tuple(self):
        self.assertIsInstance(_param_map_sig({"lck": 0.5}), tuple)


if __name__ == "__main__":
    unittest.main()
