"""Tests for mlcore/fit_stages_core.py — stage order, normalization, and index helpers."""
import unittest

from mlcore.fit_stages_core import (
    FIT_STAGE_ORDER,
    _normalize_fit_stage,
    _fit_stage_index,
)


class TestFitStageOrder(unittest.TestCase):

    def test_is_list(self):
        self.assertIsInstance(FIT_STAGE_ORDER, list)

    def test_has_nine_stages(self):
        self.assertEqual(len(FIT_STAGE_ORDER), 9)

    def test_first_stage_is_load_inputs(self):
        self.assertEqual(FIT_STAGE_ORDER[0], "load_inputs")

    def test_last_stage_is_finalize(self):
        self.assertEqual(FIT_STAGE_ORDER[-1], "finalize")

    def test_contains_expected_stages(self):
        expected = {
            "load_inputs", "player_weights", "global_scales", "form_combined",
            "segmented_form", "series_models", "context_models", "snapshots", "finalize",
        }
        self.assertEqual(set(FIT_STAGE_ORDER), expected)

    def test_all_stages_are_strings(self):
        for s in FIT_STAGE_ORDER:
            self.assertIsInstance(s, str)


class TestNormalizeFitStage(unittest.TestCase):

    def test_valid_stage_passthrough(self):
        self.assertEqual(_normalize_fit_stage("load_inputs"), "load_inputs")

    def test_uppercase_lowercased(self):
        self.assertEqual(_normalize_fit_stage("FINALIZE"), "finalize")

    def test_mixed_case(self):
        self.assertEqual(_normalize_fit_stage("Global_Scales"), "global_scales")

    def test_strips_whitespace(self):
        self.assertEqual(_normalize_fit_stage("  snapshots  "), "snapshots")

    def test_none_defaults_to_load_inputs(self):
        self.assertEqual(_normalize_fit_stage(None), "load_inputs")

    def test_empty_string_defaults_to_load_inputs(self):
        self.assertEqual(_normalize_fit_stage(""), "load_inputs")

    def test_invalid_stage_raises(self):
        with self.assertRaises(ValueError):
            _normalize_fit_stage("nonexistent_stage")

    def test_invalid_stage_message(self):
        with self.assertRaises(ValueError) as ctx:
            _normalize_fit_stage("bad_stage")
        self.assertIn("bad_stage", str(ctx.exception))

    def test_finalize_allowed_by_default(self):
        self.assertEqual(_normalize_fit_stage("finalize"), "finalize")

    def test_finalize_blocked_when_not_allowed(self):
        with self.assertRaises(ValueError):
            _normalize_fit_stage("finalize", allow_finalize=False)

    def test_finalize_blocked_message(self):
        with self.assertRaises(ValueError) as ctx:
            _normalize_fit_stage("finalize", allow_finalize=False)
        self.assertIn("finalize", str(ctx.exception))

    def test_other_stages_not_blocked_by_allow_finalize_false(self):
        self.assertEqual(_normalize_fit_stage("snapshots", allow_finalize=False), "snapshots")

    def test_returns_string(self):
        result = _normalize_fit_stage("player_weights")
        self.assertIsInstance(result, str)


class TestFitStageIndex(unittest.TestCase):

    def test_load_inputs_is_zero(self):
        self.assertEqual(_fit_stage_index("load_inputs"), 0)

    def test_player_weights_is_one(self):
        self.assertEqual(_fit_stage_index("player_weights"), 1)

    def test_finalize_is_eight(self):
        self.assertEqual(_fit_stage_index("finalize"), 8)

    def test_snapshots_is_seven(self):
        self.assertEqual(_fit_stage_index("snapshots"), 7)

    def test_index_order_is_consistent_with_list(self):
        for i, stage in enumerate(FIT_STAGE_ORDER):
            self.assertEqual(_fit_stage_index(stage), i)

    def test_returns_int(self):
        self.assertIsInstance(_fit_stage_index("load_inputs"), int)

    def test_invalid_stage_raises(self):
        with self.assertRaises(ValueError):
            _fit_stage_index("invalid_stage")

    def test_uppercase_normalized(self):
        self.assertEqual(_fit_stage_index("LOAD_INPUTS"), 0)

    def test_all_indices_unique(self):
        indices = [_fit_stage_index(s) for s in FIT_STAGE_ORDER]
        self.assertEqual(len(indices), len(set(indices)))


if __name__ == "__main__":
    unittest.main()
