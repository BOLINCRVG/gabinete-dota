import unittest

from mlcore.team_artifact_validate import _coerce_float_map, _param_map_sig


class TestCoerceFloatMap(unittest.TestCase):
    def test_basic_coercion(self):
        result = _coerce_float_map({"win_rate": "0.65", "kills": 12})
        self.assertAlmostEqual(result["WIN_RATE"], 0.65)
        self.assertAlmostEqual(result["KILLS"], 12.0)

    def test_keys_uppercased(self):
        result = _coerce_float_map({"alpha": 1.0})
        self.assertIn("ALPHA", result)
        self.assertNotIn("alpha", result)

    def test_non_finite_excluded(self):
        result = _coerce_float_map({"a": float("nan"), "b": float("inf"), "c": 1.0})
        self.assertNotIn("A", result)
        self.assertNotIn("B", result)
        self.assertIn("C", result)

    def test_non_numeric_excluded(self):
        result = _coerce_float_map({"x": "notanumber", "y": 2.0})
        self.assertNotIn("X", result)
        self.assertIn("Y", result)

    def test_empty_key_excluded(self):
        result = _coerce_float_map({"": 1.0, "  ": 2.0, "valid": 3.0})
        self.assertEqual(list(result.keys()), ["VALID"])

    def test_non_dict_input_returns_empty(self):
        self.assertEqual(_coerce_float_map(None), {})
        self.assertEqual(_coerce_float_map([1, 2, 3]), {})
        self.assertEqual(_coerce_float_map("string"), {})

    def test_empty_dict(self):
        self.assertEqual(_coerce_float_map({}), {})


class TestParamMapSig(unittest.TestCase):
    def test_returns_sorted_tuple(self):
        sig = _param_map_sig({"b": 2.0, "a": 1.0})
        self.assertEqual(sig, (("A", 1.0), ("B", 2.0)))

    def test_same_content_same_sig(self):
        sig1 = _param_map_sig({"x": 0.5, "y": 0.3})
        sig2 = _param_map_sig({"y": 0.3, "x": 0.5})
        self.assertEqual(sig1, sig2)

    def test_different_content_different_sig(self):
        sig1 = _param_map_sig({"x": 0.5})
        sig2 = _param_map_sig({"x": 0.6})
        self.assertNotEqual(sig1, sig2)

    def test_empty_returns_empty_tuple(self):
        self.assertEqual(_param_map_sig({}), ())
        self.assertEqual(_param_map_sig(None), ())

    def test_non_finite_excluded_from_sig(self):
        sig = _param_map_sig({"a": float("nan"), "b": 1.0})
        keys = [k for k, _ in sig]
        self.assertNotIn("A", keys)
        self.assertIn("B", keys)


if __name__ == "__main__":
    unittest.main()
