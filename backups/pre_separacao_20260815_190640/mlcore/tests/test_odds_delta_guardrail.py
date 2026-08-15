import unittest

from tools.odds_delta_guardrail import _case_tolerance_pct, _decide, _evaluate_case


class TestCaseTolerancePct(unittest.TestCase):

    def test_explicit_tolerance_returned(self):
        self.assertEqual(_case_tolerance_pct({"tolerance_pct": 0.25}, 0.1), 0.25)

    def test_negative_clamped_to_zero(self):
        self.assertEqual(_case_tolerance_pct({"tolerance_pct": -1}, 0.1), 0.0)

    def test_above_one_clamped_to_one(self):
        self.assertEqual(_case_tolerance_pct({"tolerance_pct": 9}, 0.1), 1.0)

    def test_missing_key_uses_default(self):
        self.assertEqual(_case_tolerance_pct({}, 0.1), 0.1)

    def test_invalid_string_uses_default(self):
        self.assertEqual(_case_tolerance_pct({"tolerance_pct": "abc"}, 0.15), 0.15)

    def test_zero_tolerance_valid(self):
        self.assertEqual(_case_tolerance_pct({"tolerance_pct": 0.0}, 0.1), 0.0)

    def test_returns_float(self):
        self.assertIsInstance(_case_tolerance_pct({}, 0.1), float)


class TestDecide(unittest.TestCase):

    def test_zero_total_is_shadow_only(self):
        self.assertEqual(_decide(0, 0, 0), "shadow_only")

    def test_all_pass_is_promote(self):
        self.assertEqual(_decide(3, 0, 3), "promote")

    def test_one_of_five_fail_is_shadow_only(self):
        self.assertEqual(_decide(4, 1, 5), "shadow_only")

    def test_majority_fail_is_reject(self):
        self.assertEqual(_decide(2, 3, 5), "reject")

    def test_all_fail_is_reject(self):
        self.assertEqual(_decide(0, 5, 5), "reject")

    def test_returns_string(self):
        self.assertIsInstance(_decide(1, 0, 1), str)


class TestEvaluateCase(unittest.TestCase):

    def test_within_tolerance_is_ok(self):
        c = {"baseline_odd": 2.0, "tolerance_pct": 0.10}
        result = _evaluate_case(c, 2.17, default_tol_pct=0.10, min_abs_delta=0.03)
        self.assertTrue(result["ok"])

    def test_max_allowed_pct_of_baseline(self):
        c = {"baseline_odd": 2.0, "tolerance_pct": 0.10}
        result = _evaluate_case(c, 2.17, default_tol_pct=0.10, min_abs_delta=0.03)
        self.assertAlmostEqual(result["max_allowed_delta_abs"], 0.2, delta=1e-12)

    def test_exceeds_tolerance_is_fail(self):
        c = {"baseline_odd": 2.0, "tolerance_pct": 0.10}
        result = _evaluate_case(c, 2.25, default_tol_pct=0.10, min_abs_delta=0.03)
        self.assertFalse(result["ok"])

    def test_min_abs_delta_floor_applies(self):
        c_low = {"baseline_odd": 1.05, "tolerance_pct": 0.01}
        result = _evaluate_case(c_low, 1.07, default_tol_pct=0.01, min_abs_delta=0.03)
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(result["max_allowed_delta_abs"], 0.03, delta=1e-12)

    def test_delta_abs_computed(self):
        c = {"baseline_odd": 2.0, "tolerance_pct": 0.10}
        result = _evaluate_case(c, 2.3, default_tol_pct=0.10, min_abs_delta=0.03)
        self.assertAlmostEqual(result["delta_abs"], 0.3, delta=1e-12)

    def test_returns_dict(self):
        c = {"baseline_odd": 2.0}
        self.assertIsInstance(_evaluate_case(c, 2.0, default_tol_pct=0.1, min_abs_delta=0.03), dict)

    def test_required_keys_present(self):
        c = {"baseline_odd": 2.0}
        result = _evaluate_case(c, 2.0, default_tol_pct=0.1, min_abs_delta=0.03)
        for key in ("ok", "delta_abs", "max_allowed_delta_abs", "tolerance_pct"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
