import unittest

from app_core.calibration_core import calibrate_market_odds


def _odd_mix_passthrough(odd_model, odd_hist, _stats_pct):
    if odd_model is not None:
        return odd_model
    return odd_hist


class TestCalibrationCore(unittest.TestCase):
    def test_mix_preferred_when_valid(self):
        out = calibrate_market_odds(
            odd_model=1.80,
            odd_hist=1.90,
            stats_pct=60.0,
            max_odd=2.50,
            require_model=False,
            odd_mix_fn=lambda om, oh, sc: 1.75,
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["source"], "mix")
        self.assertEqual(out["reason"], "mix")
        self.assertAlmostEqual(float(out["odd_mix"]), 1.75, places=6)

    def test_fallback_model_when_mix_invalid(self):
        out = calibrate_market_odds(
            odd_model=1.60,
            odd_hist=1.70,
            stats_pct=55.0,
            max_odd=2.00,
            require_model=False,
            odd_mix_fn=lambda om, oh, sc: None,
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["source"], "model")
        self.assertEqual(out["reason"], "fallback_model")
        self.assertAlmostEqual(float(out["odd_mix"]), 1.60, places=6)

    def test_fallback_hist_when_model_unavailable(self):
        out = calibrate_market_odds(
            odd_model=None,
            odd_hist=1.88,
            stats_pct=50.0,
            max_odd=2.00,
            require_model=False,
            odd_mix_fn=lambda om, oh, sc: None,
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["source"], "hist")
        self.assertEqual(out["reason"], "fallback_hist")
        self.assertAlmostEqual(float(out["odd_mix"]), 1.88, places=6)

    def test_require_model_blocks_when_model_missing(self):
        out = calibrate_market_odds(
            odd_model=None,
            odd_hist=1.40,
            stats_pct=70.0,
            max_odd=2.00,
            require_model=True,
            odd_mix_fn=_odd_mix_passthrough,
        )
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "model_missing_or_above_max")

    def test_no_valid_odd_under_max(self):
        out = calibrate_market_odds(
            odd_model=3.0,
            odd_hist=2.5,
            stats_pct=10.0,
            max_odd=2.0,
            require_model=False,
            odd_mix_fn=lambda om, oh, sc: 2.2,
        )
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "no_valid_odd_under_max")


if __name__ == "__main__":
    unittest.main()
