"""Tests for app_core/ml_output_models.py — output model helpers."""
import math
import unittest

from app_core.ml_output_models import (
    MlPrePipelineOutput,
    _b,
    _d,
    _l,
    _s,
)
from app_core.ml_pre_pipeline_models import _require_callable


# ---------------------------------------------------------------------------
# _b (safe bool)
# ---------------------------------------------------------------------------

class TestSafeBool(unittest.TestCase):

    def test_true(self):
        self.assertTrue(_b(True))

    def test_false(self):
        self.assertFalse(_b(False))

    def test_truthy_int(self):
        self.assertTrue(_b(1))

    def test_falsy_int(self):
        self.assertFalse(_b(0))

    def test_none_returns_false(self):
        # bool(None) is False; default only used on exception, not for None
        self.assertFalse(_b(None, False))
        self.assertFalse(_b(None, True))

    def test_string_true(self):
        self.assertTrue(_b("hello"))

    def test_empty_string_false(self):
        self.assertFalse(_b(""))


# ---------------------------------------------------------------------------
# _s (safe str)
# ---------------------------------------------------------------------------

class TestSafeStr(unittest.TestCase):

    def test_string(self):
        self.assertEqual(_s("hello"), "hello")

    def test_int_to_str(self):
        self.assertEqual(_s(42), "42")

    def test_none_returns_default(self):
        self.assertEqual(_s(None, "default"), "default")

    def test_empty_default(self):
        self.assertEqual(_s(None, ""), "")

    def test_bool_to_str(self):
        self.assertEqual(_s(True), "True")


# ---------------------------------------------------------------------------
# _d (safe dict)
# ---------------------------------------------------------------------------

class TestSafeDict(unittest.TestCase):

    def test_dict_returns_copy(self):
        d = {"a": 1}
        result = _d(d)
        self.assertEqual(result, {"a": 1})

    def test_none_returns_empty(self):
        self.assertEqual(_d(None), {})

    def test_list_returns_empty(self):
        self.assertEqual(_d([1, 2]), {})

    def test_string_returns_empty(self):
        self.assertEqual(_d("abc"), {})

    def test_mapping_returns_dict(self):
        from types import MappingProxyType
        m = MappingProxyType({"x": 10})
        result = _d(m)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["x"], 10)


# ---------------------------------------------------------------------------
# _l (safe list)
# ---------------------------------------------------------------------------

class TestSafeList(unittest.TestCase):

    def test_list_returns_copy(self):
        result = _l([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_none_returns_empty(self):
        self.assertEqual(_l(None), [])

    def test_dict_returns_empty(self):
        self.assertEqual(_l({"a": 1}), [])

    def test_tuple_returns_empty(self):
        # tuples are not list → returns []
        self.assertEqual(_l((1, 2)), [])

    def test_empty_list(self):
        self.assertEqual(_l([]), [])


# ---------------------------------------------------------------------------
# _require_callable
# ---------------------------------------------------------------------------

class TestRequireCallable(unittest.TestCase):

    def test_callable_found(self):
        fn = lambda: None
        result = _require_callable({"my_fn": fn}, "my_fn")
        self.assertIs(result, fn)

    def test_missing_required_raises(self):
        with self.assertRaises(ValueError):
            _require_callable({}, "missing_fn", required=True)

    def test_missing_optional_returns_none(self):
        result = _require_callable({}, "missing_fn", required=False)
        self.assertIsNone(result)

    def test_non_callable_required_raises(self):
        with self.assertRaises(ValueError):
            _require_callable({"fn": "not_callable"}, "fn", required=True)

    def test_non_callable_optional_returns_none(self):
        result = _require_callable({"fn": 42}, "fn", required=False)
        self.assertIsNone(result)

    def test_none_callbacks_required_raises(self):
        with self.assertRaises(ValueError):
            _require_callable(None, "fn", required=True)

    def test_none_callbacks_optional_returns_none(self):
        result = _require_callable(None, "fn", required=False)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# MlPrePipelineOutput.from_mapping
# ---------------------------------------------------------------------------

class TestMlPrePipelineOutputFromMapping(unittest.TestCase):

    def test_from_none_mapping(self):
        out = MlPrePipelineOutput.from_mapping(None)
        self.assertIsInstance(out, MlPrePipelineOutput)
        self.assertTrue(math.isnan(out.p_map_used))

    def test_from_empty_mapping(self):
        out = MlPrePipelineOutput.from_mapping({})
        self.assertIsInstance(out, MlPrePipelineOutput)

    def test_float_fields_parsed(self):
        out = MlPrePipelineOutput.from_mapping({"p_map_used": 0.65, "delta_players": 0.03})
        self.assertAlmostEqual(out.p_map_used, 0.65)
        self.assertAlmostEqual(out.delta_players, 0.03)

    def test_dict_fields_parsed(self):
        out = MlPrePipelineOutput.from_mapping({"trace_fusion_info": {"w_team": 0.8}})
        self.assertEqual(out.trace_fusion_info["w_team"], 0.8)

    def test_string_field_parsed(self):
        out = MlPrePipelineOutput.from_mapping({"lineup_mode": "partial"})
        self.assertEqual(out.lineup_mode, "partial")

    def test_extra_keys_go_to_extras(self):
        out = MlPrePipelineOutput.from_mapping({"unknown_key": "some_value"})
        self.assertIn("unknown_key", out.extras)

    def test_delta_players_default_zero(self):
        out = MlPrePipelineOutput.from_mapping({})
        self.assertAlmostEqual(out.delta_players, 0.0)


if __name__ == "__main__":
    unittest.main()
