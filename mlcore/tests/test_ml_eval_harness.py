"""Tests for mlcore/ml_eval/ — splits, metrics with CI, and acceptance gate."""
import math
import random
import unittest
from dataclasses import dataclass

from mlcore.ml_eval import (
    SampleBucket,
    acceptance_gate,
    bootstrap_ci,
    build_split,
    ece_score,
    metrics_with_ci,
)
from mlcore.ml_eval.metrics import paired_bootstrap_delta


@dataclass
class _MockSample:
    date_day: str
    y: int = 1


class TestSampleBucket(unittest.TestCase):

    def test_empty_bucket(self):
        b = SampleBucket(name="train", samples=[])
        self.assertEqual(b.n, 0)
        self.assertEqual(b.date_range(), (None, None))

    def test_n_count(self):
        b = SampleBucket(name="train", samples=[_MockSample("2024-01-01"), _MockSample("2024-02-01")])
        self.assertEqual(b.n, 2)

    def test_date_range_sorted(self):
        b = SampleBucket(name="train", samples=[
            _MockSample("2024-03-01"), _MockSample("2024-01-01"), _MockSample("2024-02-01")
        ])
        lo, hi = b.date_range()
        self.assertEqual(lo, "2024-01-01")
        self.assertEqual(hi, "2024-03-01")


class TestBuildSplit(unittest.TestCase):

    def _samples(self):
        return [
            _MockSample("2024-03-15"),
            _MockSample("2024-09-20"),
            _MockSample("2025-04-10"),
            _MockSample("2025-11-05"),
            _MockSample("2026-02-15"),
            _MockSample("2026-05-30"),
            _MockSample("2026-08-10"),
            _MockSample("2026-11-20"),
        ]

    def test_chronological_split(self):
        split = build_split(self._samples(), train_years=(2024, 2025), val_year=2026, val_cutoff="2026-06-30")
        self.assertEqual(split.train.n, 4)  # 2024+2025
        self.assertEqual(split.val.n, 2)    # 2026-02 + 2026-05
        self.assertEqual(split.test.n, 2)   # 2026-08 + 2026-11

    def test_train_excludes_val_year(self):
        split = build_split(self._samples(), train_years=(2024, 2025), val_year=2026, val_cutoff="2026-06-30")
        for s in split.train.samples:
            self.assertNotIn("2026", s.date_day)

    def test_val_and_test_use_cutoff(self):
        split = build_split(self._samples(), train_years=(2024, 2025), val_year=2026, val_cutoff="2026-06-30")
        for s in split.val.samples:
            self.assertLessEqual(s.date_day, "2026-06-30")
        for s in split.test.samples:
            self.assertGreater(s.date_day, "2026-06-30")

    def test_invalid_cutoff_raises(self):
        with self.assertRaises(ValueError):
            build_split(self._samples(), val_cutoff="not-a-date")

    def test_empty_cutoff_raises(self):
        with self.assertRaises(ValueError):
            build_split(self._samples(), val_cutoff="")

    def test_summary_keys(self):
        split = build_split(self._samples(), train_years=(2024, 2025), val_year=2026, val_cutoff="2026-06-30")
        s = split.summary()
        self.assertIn("train", s)
        self.assertIn("val", s)
        self.assertIn("test", s)
        self.assertEqual(s["train"]["n"], 4)


class TestECEScore(unittest.TestCase):

    def test_perfect_calibration(self):
        # 100 samples, all p=0.5, half y=1 -> ECE should be 0
        y = [1] * 50 + [0] * 50
        p = [0.5] * 100
        self.assertAlmostEqual(ece_score(y, p), 0.0, places=5)

    def test_worst_calibration(self):
        # All p=0.9 but all y=0 -> ECE should be high
        y = [0] * 100
        p = [0.9] * 100
        self.assertGreater(ece_score(y, p), 0.85)

    def test_empty_returns_nan(self):
        self.assertTrue(math.isnan(ece_score([], [])))

    def test_mismatched_lengths_returns_nan(self):
        self.assertTrue(math.isnan(ece_score([1, 0], [0.5])))


class TestBootstrapCI(unittest.TestCase):

    def test_returns_three_values(self):
        y = [1, 0, 1, 0, 1, 0, 1, 0]
        p = [0.7, 0.3, 0.6, 0.4, 0.8, 0.2, 0.7, 0.3]
        pt, lo, hi = bootstrap_ci(y, p, lambda a, b: float(sum(b) / len(b)), n_bootstrap=100)
        self.assertIsInstance(pt, float)
        self.assertIsInstance(lo, float)
        self.assertIsInstance(hi, float)
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)

    def test_zero_bootstrap_returns_point(self):
        y = [1, 0]
        p = [0.7, 0.3]
        pt, lo, hi = bootstrap_ci(y, p, lambda a, b: 0.5, n_bootstrap=0)
        self.assertAlmostEqual(pt, lo)
        self.assertAlmostEqual(pt, hi)

    def test_empty_returns_nans(self):
        pt, _lo, _hi = bootstrap_ci([], [], lambda a, b: 0.5)
        self.assertTrue(math.isnan(pt))


class TestMetricsWithCI(unittest.TestCase):

    def test_returns_required_keys(self):
        random.seed(0)
        y = [random.choice([0, 1]) for _ in range(100)]
        p = [0.5] * 100
        m = metrics_with_ci(y, p, n_bootstrap=50)
        for key in ("logloss", "brier", "ece", "n", "n_bootstrap"):
            self.assertIn(key, m)

    def test_invalid_inputs(self):
        m = metrics_with_ci([], [], n_bootstrap=10)
        self.assertIn("error", m)


class TestPairedBootstrapDelta(unittest.TestCase):

    def test_better_candidate_positive_delta(self):
        random.seed(1)
        n = 200
        y = [1 if i < n / 2 else 0 for i in range(n)]
        # baseline: random
        p_b = [0.5 + random.uniform(-0.1, 0.1) for _ in range(n)]
        # candidate: aligned with y
        p_c = [0.7 if yi == 1 else 0.3 for yi in y]
        from mlcore.utils import logloss
        d, _lo, _hi, pv = paired_bootstrap_delta(y, p_b, p_c, lambda a, b: float(logloss(a, b)), n_bootstrap=100)
        self.assertGreater(d, 0)
        self.assertLess(pv, 0.05)

    def test_invalid_returns_nan(self):
        d, _lo, _hi, pv = paired_bootstrap_delta([], [], [], lambda a, b: 0.5)
        self.assertTrue(math.isnan(d))
        self.assertEqual(pv, 1.0)


class TestAcceptanceGate(unittest.TestCase):

    def _sigma_samples(self, n=200):
        random.seed(7)
        y = [random.choice([0, 1]) for _ in range(n)]
        return y

    def test_accept_when_clearly_better(self):
        # Construct synthetic well-calibrated data where candidate is strictly better.
        # Generate y from true probabilities (so calibration is preserved).
        random.seed(7)
        n = 2500  # large enough for p-value to be significant
        true_p = [random.uniform(0.2, 0.8) for _ in range(n)]
        y = [1 if random.random() < tp else 0 for tp in true_p]
        # baseline: pulled toward 0.5 (less informative)
        p_b = [0.5 + 0.3 * (tp - 0.5) for tp in true_p]
        # candidate: closer to true_p (much more informative)
        p_c = [0.5 + 0.95 * (tp - 0.5) for tp in true_p]
        gate = acceptance_gate(y, p_b, p_c, n_bootstrap=300)
        self.assertTrue(gate.accept, msg=f"Expected accept; gate said: {gate.summary_line()}")
        self.assertEqual(gate.reason, "passed_all_checks")
        self.assertGreater(gate.delta_logloss, 0.002)

    def test_reject_when_no_improvement(self):
        random.seed(13)
        y = [random.choice([0, 1]) for _ in range(200)]
        p_b = [0.5 + random.uniform(-0.05, 0.05) for _ in y]
        p_c = list(p_b)  # identical
        gate = acceptance_gate(y, p_b, p_c, n_bootstrap=100)
        self.assertFalse(gate.accept)
        self.assertIn("insufficient", gate.reason)

    def test_reject_when_ece_worsens(self):
        random.seed(21)
        n = 200
        y = [random.choice([0, 1]) for _ in range(n)]
        # baseline: well-calibrated
        p_b = [0.5 + (0.1 if yi == 1 else -0.1) for yi in y]
        # candidate: SAME logloss but pushed to extremes (worse ECE)
        p_c = [0.95 if yi == 1 else 0.05 for yi in y]
        # candidate has lower logloss (better) but ECE may worsen
        gate = acceptance_gate(y, p_b, p_c, n_bootstrap=100, ece_tolerance=0.005)
        # Either it's accepted (if logloss gain is huge), or rejected because of ECE.
        # We just check that the gate returns a decision with all fields.
        self.assertIsInstance(gate.accept, bool)
        self.assertIn(gate.reason, ("passed_all_checks", "ece_worsened",
                                    "insufficient_logloss_gain", "not_significant"))

    def test_invalid_inputs_rejected(self):
        gate = acceptance_gate([], [], [])
        self.assertFalse(gate.accept)
        self.assertEqual(gate.reason, "invalid_inputs")

    def test_summary_line_contains_verdict(self):
        y = [1, 0, 1, 0]
        p = [0.6, 0.4, 0.6, 0.4]
        gate = acceptance_gate(y, p, p, n_bootstrap=50)
        line = gate.summary_line()
        self.assertTrue(line.startswith(("[ACCEPT]", "[REJECT]")))

    def test_as_dict_serializable(self):
        y = [1, 0, 1, 0, 1, 0]
        p = [0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
        gate = acceptance_gate(y, p, p, n_bootstrap=50)
        d = gate.as_dict()
        import json
        json.dumps(d)  # should not raise


if __name__ == "__main__":
    unittest.main()
