"""Tests for app_core/pipeline_trace_core.py — append_pipeline_step."""
import unittest

from app_core.pipeline_trace_core import append_pipeline_step, _opt_float


# ---------------------------------------------------------------------------
# _opt_float
# ---------------------------------------------------------------------------

class TestOptFloat(unittest.TestCase):

    def test_int_returns_float(self):
        self.assertAlmostEqual(_opt_float(5), 5.0)
        self.assertIsInstance(_opt_float(5), float)

    def test_float_returns_float(self):
        self.assertAlmostEqual(_opt_float(3.14), 3.14)

    def test_string_numeric_returns_float(self):
        self.assertAlmostEqual(_opt_float("2.5"), 2.5)

    def test_nan_returns_none(self):
        self.assertIsNone(_opt_float(float("nan")))

    def test_inf_returns_none(self):
        self.assertIsNone(_opt_float(float("inf")))
        self.assertIsNone(_opt_float(float("-inf")))

    def test_none_returns_none(self):
        self.assertIsNone(_opt_float(None))

    def test_non_numeric_string_returns_none(self):
        self.assertIsNone(_opt_float("abc"))

    def test_zero_returns_zero(self):
        self.assertEqual(_opt_float(0), 0.0)


# ---------------------------------------------------------------------------
# append_pipeline_step
# ---------------------------------------------------------------------------

class TestAppendPipelineStep(unittest.TestCase):

    def test_appends_to_list(self):
        steps = []
        append_pipeline_step(steps=steps, stage="test_stage")
        self.assertEqual(len(steps), 1)

    def test_stage_stored(self):
        steps = []
        append_pipeline_step(steps=steps, stage="microseg")
        self.assertEqual(steps[0]["stage"], "microseg")

    def test_before_and_after_stored(self):
        steps = []
        append_pipeline_step(steps=steps, stage="x", before=0.50, after=0.55)
        self.assertAlmostEqual(steps[0]["before"], 0.50)
        self.assertAlmostEqual(steps[0]["after"], 0.55)

    def test_nan_before_stored_as_none(self):
        steps = []
        append_pipeline_step(steps=steps, stage="x", before=float("nan"))
        self.assertIsNone(steps[0]["before"])

    def test_applied_defaults_false(self):
        steps = []
        append_pipeline_step(steps=steps, stage="x")
        self.assertFalse(steps[0]["applied"])

    def test_applied_true(self):
        steps = []
        append_pipeline_step(steps=steps, stage="x", applied=True)
        self.assertTrue(steps[0]["applied"])

    def test_extras_stored(self):
        steps = []
        append_pipeline_step(steps=steps, stage="x", shrink=0.3, note="test")
        self.assertAlmostEqual(steps[0]["shrink"], 0.3)
        self.assertEqual(steps[0]["note"], "test")

    def test_non_list_does_nothing(self):
        append_pipeline_step(steps=None, stage="x")  # should not raise
        append_pipeline_step(steps="not_list", stage="x")

    def test_multiple_appends(self):
        steps = []
        append_pipeline_step(steps=steps, stage="a")
        append_pipeline_step(steps=steps, stage="b")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[1]["stage"], "b")

    def test_empty_stage_stored(self):
        steps = []
        append_pipeline_step(steps=steps, stage="")
        self.assertEqual(steps[0]["stage"], "")


if __name__ == "__main__":
    unittest.main()
