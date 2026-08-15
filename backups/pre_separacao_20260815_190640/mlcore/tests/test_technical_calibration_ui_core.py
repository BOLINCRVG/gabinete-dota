import unittest

from app_core.technical_calibration_ui_core import (
    build_confidence_baseline_caption,
    build_state_artifact_caption,
    build_wf_correction_caption,
    infer_same_league_hint,
)


class TestInferSameLeagueHint(unittest.TestCase):

    def test_same_league_returns_league(self):
        ml_info = {"blue": {"league": "LCK"}, "red": {"league": "LCK"}}
        self.assertEqual(infer_same_league_hint(ml_info), "LCK")

    def test_different_leagues_returns_empty(self):
        ml_info = {"blue": {"league": "LCK"}, "red": {"league": "LPL"}}
        self.assertEqual(infer_same_league_hint(ml_info), "")

    def test_none_input_returns_empty(self):
        self.assertEqual(infer_same_league_hint(None), "")

    def test_empty_dict_returns_empty(self):
        self.assertEqual(infer_same_league_hint({}), "")

    def test_returns_string(self):
        self.assertIsInstance(infer_same_league_hint({}), str)


class TestBuildWfCorrectionCaption(unittest.TestCase):

    def test_applied_formats_values(self):
        caption = build_wf_correction_caption({"applied": True, "p_in": 0.55, "p_out": 0.57, "shrink": 0.125})
        self.assertIn("0.5500 -> 0.5700", caption)
        self.assertIn("0.125", caption)

    def test_not_applied_returns_empty(self):
        caption = build_wf_correction_caption({"applied": False, "p_in": 0.55, "p_out": 0.57, "shrink": 0.0})
        self.assertEqual(caption, "")

    def test_none_input_returns_empty(self):
        self.assertEqual(build_wf_correction_caption(None), "")

    def test_empty_dict_returns_empty(self):
        self.assertEqual(build_wf_correction_caption({}), "")

    def test_returns_string(self):
        self.assertIsInstance(build_wf_correction_caption(None), str)


class TestBuildConfidenceBaselineCaption(unittest.TestCase):

    def test_live_payload_formatted(self):
        caption = build_confidence_baseline_caption(
            {
                "generated_at": "2026-03-10T12:00:00",
                "by_league": {"LCK": {}, "LPL": {}},
                "global": {"score_bias": 1.25},
            },
            baseline_name="ml_confidence_baseline.json",
            format_num_fn=lambda value, decimals: f"{value:.{decimals}f}",
        )
        self.assertIn("ligas=2", caption)
        self.assertIn("bias_global=1.2", caption)

    def test_error_in_payload_returns_error_message(self):
        caption = build_confidence_baseline_caption(
            {"error": True},
            baseline_name="ml_confidence_baseline.json",
            format_num_fn=lambda value, decimals: str(value),
        )
        self.assertIn("ainda nao gerado", caption)

    def test_none_input_returns_error_message(self):
        caption = build_confidence_baseline_caption(
            None,
            baseline_name="ml_confidence_baseline.json",
            format_num_fn=lambda value, decimals: str(value),
        )
        self.assertIn("ainda nao gerado", caption)

    def test_returns_string(self):
        result = build_confidence_baseline_caption(
            None,
            baseline_name="ml_confidence_baseline.json",
            format_num_fn=lambda v, d: str(v),
        )
        self.assertIsInstance(result, str)


class TestBuildStateArtifactCaption(unittest.TestCase):

    def test_live_payload_formatted(self):
        caption = build_state_artifact_caption(
            {
                "generated_at": "2026-03-10T12:00:00",
                "meta": {"teams": 10, "games_used": 200, "update_mode": "full"},
            },
            artifact_name="team_state_artifact.json",
        )
        self.assertIn("times=10", caption)
        self.assertIn("jogos=200", caption)
        self.assertIn("mode=full", caption)

    def test_error_returns_error_message(self):
        caption = build_state_artifact_caption({"error": True}, artifact_name="team_state_artifact.json")
        self.assertIn("ainda nao gerado", caption)

    def test_none_input_returns_error_message(self):
        caption = build_state_artifact_caption(None, artifact_name="team_state_artifact.json")
        self.assertIn("ainda nao gerado", caption)

    def test_returns_string(self):
        self.assertIsInstance(build_state_artifact_caption(None, artifact_name="x.json"), str)


if __name__ == "__main__":
    unittest.main()
