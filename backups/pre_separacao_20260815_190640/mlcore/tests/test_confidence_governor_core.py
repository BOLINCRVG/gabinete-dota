import unittest

from app_core.confidence_governor_core import apply_confidence_governor


class TestConfidenceGovernorCore(unittest.TestCase):
    def test_reset_initializes_state(self):
        ss = {}
        out = apply_confidence_governor(
            stage="reset",
            p_before=0.60,
            p_candidate=0.55,
            session_state=ss,
            enabled=True,
            max_total_shrink_pp=8.0,
            reset=True,
        )
        self.assertTrue(bool(out.get("reset")))
        self.assertAlmostEqual(float(out.get("remaining_shrink_pp")), 8.0, places=6)
        self.assertAlmostEqual(float(out.get("p_out")), 0.55, places=6)
        state = ss.get("_ml_confidence_governor") if isinstance(ss.get("_ml_confidence_governor"), dict) else {}
        self.assertEqual(float(state.get("used_shrink_pp", -1.0)), 0.0)
        self.assertEqual(float(state.get("max_total_shrink_pp", -1.0)), 8.0)
        self.assertEqual(list(state.get("stages") or []), [])

    def test_move_away_from_50_does_not_consume_budget(self):
        ss = {}
        apply_confidence_governor(
            stage="reset",
            p_before=0.55,
            p_candidate=0.55,
            session_state=ss,
            enabled=True,
            max_total_shrink_pp=8.0,
            reset=True,
        )
        out = apply_confidence_governor(
            stage="uncertainty",
            p_before=0.55,
            p_candidate=0.60,
            session_state=ss,
            enabled=True,
            max_total_shrink_pp=8.0,
        )
        self.assertFalse(bool(out.get("limited", True)))
        self.assertAlmostEqual(float(out.get("requested_shrink_pp")), 0.0, places=6)
        self.assertAlmostEqual(float(out.get("applied_shrink_pp")), 0.0, places=6)
        self.assertAlmostEqual(float(out.get("used_shrink_pp")), 0.0, places=6)
        self.assertAlmostEqual(float(out.get("p_out")), 0.60, places=6)

    def test_limited_when_budget_is_insufficient(self):
        ss = {}
        apply_confidence_governor(
            stage="reset",
            p_before=0.70,
            p_candidate=0.70,
            session_state=ss,
            enabled=True,
            max_total_shrink_pp=3.0,
            reset=True,
        )
        out = apply_confidence_governor(
            stage="uncertainty",
            p_before=0.70,
            p_candidate=0.60,
            session_state=ss,
            enabled=True,
            max_total_shrink_pp=3.0,
        )
        self.assertTrue(bool(out.get("limited", False)))
        self.assertAlmostEqual(float(out.get("requested_shrink_pp")), 10.0, places=6)
        self.assertAlmostEqual(float(out.get("applied_shrink_pp")), 3.0, places=6)
        self.assertAlmostEqual(float(out.get("used_shrink_pp")), 3.0, places=6)
        self.assertAlmostEqual(float(out.get("remaining_shrink_pp")), 0.0, places=6)
        self.assertAlmostEqual(float(out.get("p_out")), 0.67, places=6)

    def test_budget_is_tracked_across_stages(self):
        ss = {}
        apply_confidence_governor(
            stage="reset",
            p_before=0.70,
            p_candidate=0.70,
            session_state=ss,
            enabled=True,
            max_total_shrink_pp=8.0,
            reset=True,
        )
        out_1 = apply_confidence_governor(
            stage="uncertainty",
            p_before=0.70,
            p_candidate=0.65,
            session_state=ss,
            enabled=True,
            max_total_shrink_pp=8.0,
        )
        out_2 = apply_confidence_governor(
            stage="patch",
            p_before=0.65,
            p_candidate=0.55,
            session_state=ss,
            enabled=True,
            max_total_shrink_pp=8.0,
        )
        self.assertFalse(bool(out_1.get("limited", True)))
        self.assertAlmostEqual(float(out_1.get("applied_shrink_pp")), 5.0, places=6)
        self.assertTrue(bool(out_2.get("limited", False)))
        self.assertAlmostEqual(float(out_2.get("requested_shrink_pp")), 10.0, places=6)
        self.assertAlmostEqual(float(out_2.get("applied_shrink_pp")), 3.0, places=6)
        self.assertAlmostEqual(float(out_2.get("used_shrink_pp")), 8.0, places=6)
        self.assertAlmostEqual(float(out_2.get("remaining_shrink_pp")), 0.0, places=6)
        self.assertAlmostEqual(float(out_2.get("p_out")), 0.62, places=6)
        state = ss.get("_ml_confidence_governor") if isinstance(ss.get("_ml_confidence_governor"), dict) else {}
        stages = list(state.get("stages") or [])
        self.assertEqual(len(stages), 2)
        self.assertEqual(str(stages[0].get("stage")), "uncertainty")
        self.assertEqual(str(stages[1].get("stage")), "patch")

    def test_disabled_mode_never_limits_candidate(self):
        ss = {}
        out = apply_confidence_governor(
            stage="uncertainty",
            p_before=0.70,
            p_candidate=0.60,
            session_state=ss,
            enabled=False,
            max_total_shrink_pp=8.0,
        )
        self.assertFalse(bool(out.get("limited", True)))
        self.assertAlmostEqual(float(out.get("requested_shrink_pp")), 10.0, places=6)
        self.assertAlmostEqual(float(out.get("applied_shrink_pp")), 10.0, places=6)
        self.assertAlmostEqual(float(out.get("used_shrink_pp")), 0.0, places=6)
        self.assertIsNone(out.get("remaining_shrink_pp"))
        self.assertAlmostEqual(float(out.get("p_out")), 0.60, places=6)

    def test_invalid_input_returns_reason(self):
        out = apply_confidence_governor(
            stage="uncertainty",
            p_before="invalid",
            p_candidate=0.60,
            session_state={},
            enabled=True,
            max_total_shrink_pp=8.0,
        )
        self.assertEqual(str(out.get("reason")), "invalid_inputs")
        self.assertIsNone(out.get("requested_shrink_pp"))
        self.assertAlmostEqual(float(out.get("p_out")), 0.60, places=6)


if __name__ == "__main__":
    unittest.main()
