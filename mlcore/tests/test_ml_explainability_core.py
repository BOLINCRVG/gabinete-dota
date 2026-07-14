import unittest

from app_core.ml_explainability_core import (
    _build_baseline_line,
    _build_pipeline_line,
    _fmt_num,
    build_confidence_summary,
    build_pipeline_explainability,
    build_pipeline_step_rows,
)


class TestFmtNum(unittest.TestCase):

    def test_float_formatted(self):
        self.assertEqual(_fmt_num(3.14159, 2), "3.14")

    def test_int_as_float(self):
        self.assertEqual(_fmt_num(5, 1), "5.0")

    def test_invalid_returns_nan(self):
        self.assertEqual(_fmt_num("abc", 1), "nan")

    def test_zero_decimals(self):
        self.assertEqual(_fmt_num(3.7, 0), "4")

    def test_returns_string(self):
        self.assertIsInstance(_fmt_num(1.0, 2), str)


class TestBuildBaselineLine(unittest.TestCase):

    def test_empty_source_returns_empty(self):
        self.assertEqual(_build_baseline_line({}), "")

    def test_with_source_and_band(self):
        result = _build_baseline_line({"baseline_source": "fitted", "baseline_band": "fav", "baseline_adj": -1.0})
        self.assertIn("fitted", result)
        self.assertIn("band=fav", result)

    def test_with_source_no_band(self):
        result = _build_baseline_line({"baseline_source": "fitted", "baseline_adj": 2.0})
        self.assertIn("fitted", result)
        self.assertNotIn("band=", result)

    def test_returns_string(self):
        self.assertIsInstance(_build_baseline_line({}), str)


class TestBuildPipelineLine(unittest.TestCase):

    def test_empty_summary_returns_empty(self):
        self.assertEqual(_build_pipeline_line({}), "")

    def test_with_summary_and_method(self):
        result = _build_pipeline_line({"pipeline_summary": "A-real | B-real", "pipeline_meta_method": "trained"})
        self.assertIn("A-real", result)
        self.assertIn("C=trained", result)

    def test_with_summary_no_method(self):
        result = _build_pipeline_line({"pipeline_summary": "A-real"})
        self.assertIn("A-real", result)
        self.assertNotIn("C=", result)

    def test_returns_string(self):
        self.assertIsInstance(_build_pipeline_line({}), str)


class TestBuildConfidenceSummary(unittest.TestCase):

    def test_full_payload(self):
        got = build_confidence_summary(
            {
                "grade": "A",
                "score": 81.4,
                "source": "global",
                "n_total": 10,
                "logloss": 0.51,
                "gain_vs_team_logloss": 0.02,
                "pipeline_adj": 3.0,
                "baseline_adj": -1.0,
                "baseline_source": "fitted",
                "baseline_band": "fav",
                "pipeline_summary": "A-real | B-real",
                "pipeline_meta_method": "trained",
            }
        )
        self.assertEqual(got["grade"], "A")
        self.assertIn("Semaforo: n=10", got["summary_line"])
        self.assertIn("band=fav", got["baseline_line"])
        self.assertIn("C=trained", got["pipeline_line"])

    def test_none_payload(self):
        got = build_confidence_summary(None)
        self.assertIn("grade", got)

    def test_returns_dict(self):
        self.assertIsInstance(build_confidence_summary({}), dict)


class TestBuildPipelineStepRows(unittest.TestCase):

    def test_two_steps_returned(self):
        rows = build_pipeline_step_rows(
            {
                "pipeline": {
                    "steps": [
                        {"stage": "A", "applied": True, "before": 0.5, "after": 0.55, "source": "x"},
                        {"stage": "B", "applied": False, "before": 0.55, "after": 0.55, "source": "y"},
                    ]
                }
            }
        )
        self.assertEqual(len(rows), 2)

    def test_none_input_returns_empty(self):
        self.assertEqual(build_pipeline_step_rows(None), [])

    def test_delta_pp_computed(self):
        rows = build_pipeline_step_rows(
            {"pipeline": {"steps": [{"stage": "X", "applied": True, "before": 0.5, "after": 0.6, "source": "z"}]}}
        )
        self.assertAlmostEqual(rows[0]["Delta (pp)"], 10.0, places=5)

    def test_applied_field_is_bool(self):
        rows = build_pipeline_step_rows(
            {"pipeline": {"steps": [{"stage": "X", "applied": True, "before": 0.5, "after": 0.5, "source": "z"}]}}
        )
        self.assertIsInstance(rows[0]["Aplicou"], bool)


class TestBuildPipelineExplainability(unittest.TestCase):

    def test_only_applied_rows_included(self):
        rows = build_pipeline_step_rows(
            {
                "pipeline": {
                    "steps": [
                        {"stage": "A", "applied": True, "before": 0.5, "after": 0.55, "source": "x"},
                        {"stage": "B", "applied": False, "before": 0.55, "after": 0.55, "source": "y"},
                    ]
                }
            }
        )
        expl = build_pipeline_explainability(rows, {"pipeline_adj": 2.0, "baseline_adj": -1.0})
        self.assertEqual(len(expl["rows"]), 1)

    def test_explainability_keys(self):
        expl = build_pipeline_explainability([], {"pipeline_adj": 2.0, "baseline_adj": -1.0})
        self.assertIn("rows", expl)
        self.assertIn("moves_line", expl)
        self.assertIn("why_line", expl)

    def test_moves_line_content(self):
        rows = build_pipeline_step_rows(
            {"pipeline": {"steps": [{"stage": "A", "applied": True, "before": 0.5, "after": 0.55, "source": "x"}]}}
        )
        expl = build_pipeline_explainability(rows, {"pipeline_adj": 2.0, "baseline_adj": -1.0})
        self.assertIn("Quem puxou a odd", expl["moves_line"])


if __name__ == "__main__":
    unittest.main()
