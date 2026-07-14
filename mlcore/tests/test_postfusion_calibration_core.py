"""Tests for app_core/postfusion_calibration_core.py."""
import math
import unittest

from app_core.postfusion_calibration_core import apply_postfusion_calibration


def _mock_microseg(p_in, **kw):
    """Returns p_in + 0.02 (capped at 0.95) and applied=True."""
    p_out = min(0.95, float(p_in) + 0.02)
    return float(p_out), {"applied": True, "bias": 0.02, "league": kw.get("league", "")}


def _mock_postcal(p_in, **kw):
    """Returns p_in + 0.01 (capped) and applied=True."""
    p_out = min(0.95, float(p_in) + 0.01)
    return float(p_out), {"applied": True, "source": "global"}


def _failing_microseg(p_in, **kw):
    raise ValueError("simulated microseg failure")


class TestDisabledPassthrough(unittest.TestCase):

    def test_disabled_returns_p_unchanged(self):
        p, info = apply_postfusion_calibration(0.65, enabled=False, league_hint="LCK", bo=3)
        self.assertAlmostEqual(p, 0.65)
        self.assertFalse(info["enabled"])
        self.assertFalse(info["applied"])

    def test_disabled_reason(self):
        _, info = apply_postfusion_calibration(0.65, enabled=False, league_hint="LCK", bo=3)
        self.assertEqual(info["reason"], "disabled")

    def test_disabled_with_callbacks_still_passthrough(self):
        p, info = apply_postfusion_calibration(
            0.65, enabled=False, league_hint="LCK", bo=3,
            microseg_table={"x": {}},
            apply_microseg_fn=_mock_microseg,
            apply_fused_postcal_fn=_mock_postcal,
        )
        self.assertAlmostEqual(p, 0.65)
        self.assertFalse(info["microseg"]["applied"])


class TestInvalidInputs(unittest.TestCase):

    def test_nan_input(self):
        p, info = apply_postfusion_calibration(float("nan"), enabled=True, league_hint="LCK", bo=3)
        self.assertTrue(math.isnan(p))
        self.assertEqual(info["reason"], "invalid_p_in")

    def test_zero_input(self):
        p, info = apply_postfusion_calibration(0.0, enabled=True, league_hint="LCK", bo=3)
        self.assertAlmostEqual(p, 0.0)
        self.assertEqual(info["reason"], "invalid_p_in")

    def test_one_input(self):
        p, info = apply_postfusion_calibration(1.0, enabled=True, league_hint="LCK", bo=3)
        self.assertAlmostEqual(p, 1.0)
        self.assertEqual(info["reason"], "invalid_p_in")


class TestEnabledWithoutCallbacks(unittest.TestCase):

    def test_no_callbacks_passthrough(self):
        p, info = apply_postfusion_calibration(0.65, enabled=True, league_hint="LCK", bo=3)
        self.assertAlmostEqual(p, 0.65)
        self.assertTrue(info["enabled"])
        self.assertFalse(info["applied"])

    def test_microseg_only(self):
        p, info = apply_postfusion_calibration(
            0.65, enabled=True, league_hint="LCK", bo=3,
            microseg_table={"x": {}},
            apply_microseg_fn=_mock_microseg,
        )
        self.assertAlmostEqual(p, 0.67)
        self.assertTrue(info["microseg"]["applied"])
        self.assertFalse(info["postcal"]["applied"])

    def test_postcal_only(self):
        p, info = apply_postfusion_calibration(
            0.65, enabled=True, league_hint="LCK", bo=3,
            apply_fused_postcal_fn=_mock_postcal,
        )
        self.assertAlmostEqual(p, 0.66)
        self.assertFalse(info["microseg"]["applied"])
        self.assertTrue(info["postcal"]["applied"])


class TestBothCallbacks(unittest.TestCase):

    def test_both_applied_sequentially(self):
        p, info = apply_postfusion_calibration(
            0.65, enabled=True, league_hint="LCK", bo=3,
            microseg_table={"x": {}},
            apply_microseg_fn=_mock_microseg,
            apply_fused_postcal_fn=_mock_postcal,
        )
        self.assertAlmostEqual(p, 0.68)  # 0.65 + 0.02 + 0.01
        self.assertTrue(info["applied"])
        self.assertAlmostEqual(info["diff_pp"], 3.0, places=4)

    def test_p_in_and_p_out_recorded(self):
        _, info = apply_postfusion_calibration(
            0.65, enabled=True, league_hint="LCK", bo=3,
            microseg_table={"x": {}},
            apply_microseg_fn=_mock_microseg,
            apply_fused_postcal_fn=_mock_postcal,
        )
        self.assertAlmostEqual(info["p_in"], 0.65)
        self.assertAlmostEqual(info["p_out"], 0.68)


class TestErrorRecovery(unittest.TestCase):

    def test_failing_microseg_does_not_crash(self):
        p, info = apply_postfusion_calibration(
            0.65, enabled=True, league_hint="LCK", bo=3,
            microseg_table={"x": {}},
            apply_microseg_fn=_failing_microseg,
            apply_fused_postcal_fn=_mock_postcal,
        )
        # Microseg failed, but postcal should still apply
        self.assertAlmostEqual(p, 0.66)
        self.assertIn("error", info["microseg"])

    def test_microseg_skipped_without_table(self):
        # No microseg_table -> microseg not called even if callback provided
        p, info = apply_postfusion_calibration(
            0.65, enabled=True, league_hint="LCK", bo=3,
            microseg_table=None,
            apply_microseg_fn=_mock_microseg,
            apply_fused_postcal_fn=_mock_postcal,
        )
        self.assertAlmostEqual(p, 0.66)  # only postcal applied
        self.assertFalse(info["microseg"]["applied"])


class TestBoHandling(unittest.TestCase):

    def test_bo3_passed_to_microseg(self):
        captured = {}
        def _capture_ms(p_in, **kw):
            captured.update(kw)
            return float(p_in), {"applied": True}
        apply_postfusion_calibration(
            0.65, enabled=True, league_hint="LCK", bo=3,
            microseg_table={"x": {}}, apply_microseg_fn=_capture_ms,
        )
        self.assertEqual(captured.get("bo"), "3")

    def test_bo_unknown_defaults_to_1(self):
        captured = {}
        def _capture_ms(p_in, **kw):
            captured.update(kw)
            return float(p_in), {"applied": True}
        apply_postfusion_calibration(
            0.65, enabled=True, league_hint="LCK", bo=7,  # invalid BO
            microseg_table={"x": {}}, apply_microseg_fn=_capture_ms,
        )
        self.assertEqual(captured.get("bo"), "1")


if __name__ == "__main__":
    unittest.main()
