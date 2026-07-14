import math
import unittest
from datetime import date

from mlcore.achievements import (
    DEFAULT_CONFIG,
    _days_between,
    _half_life_for_event,
    _parse_iso_date,
    boost_points,
    decay_factor,
    feature_key,
)


class AchievementsConfigTest(unittest.TestCase):
    def test_half_life_invalid_non_positive_falls_back_to_default(self):
        cfg = {
            "half_life_days": {
                "worlds": 0,
                "other": 120,
            }
        }
        self.assertEqual(_half_life_for_event("worlds", cfg), 120.0)

    def test_half_life_invalid_other_still_returns_safe_default(self):
        cfg = {"half_life_days": {"other": -5}}
        self.assertEqual(_half_life_for_event("regional", cfg), 120.0)

    def test_half_life_valid_value_is_kept(self):
        self.assertEqual(_half_life_for_event("msi", DEFAULT_CONFIG), 240.0)


class TestFeatureKey(unittest.TestCase):

    def test_basic_key(self):
        result = feature_key("worlds", "champion")
        self.assertEqual(result, "worlds:champion")

    def test_normalized_to_lowercase(self):
        result = feature_key("WORLDS", "Champion")
        self.assertEqual(result, "worlds:champion")

    def test_empty_event_defaults_to_other(self):
        result = feature_key("", "top4")
        self.assertIn("other", result)

    def test_empty_placement_defaults_to_playoffs(self):
        result = feature_key("regional", "")
        self.assertIn("playoffs", result)

    def test_returns_string(self):
        self.assertIsInstance(feature_key("msi", "runner_up"), str)


class TestParseIsoDate(unittest.TestCase):

    def test_valid_iso_string(self):
        result = _parse_iso_date("2025-06-15")
        self.assertEqual(result, date(2025, 6, 15))

    def test_date_object_returned_as_is(self):
        d = date(2025, 1, 1)
        self.assertEqual(_parse_iso_date(d), d)

    def test_none_returns_none(self):
        self.assertIsNone(_parse_iso_date(None))

    def test_invalid_string_returns_none(self):
        self.assertIsNone(_parse_iso_date("not-a-date"))

    def test_returns_date_type(self):
        result = _parse_iso_date("2025-03-10")
        self.assertIsInstance(result, date)


class TestDaysBetween(unittest.TestCase):

    def test_same_day_is_zero(self):
        d = date(2025, 1, 1)
        self.assertEqual(_days_between(d, d), 0)

    def test_one_day_apart(self):
        d1 = date(2025, 1, 1)
        d2 = date(2025, 1, 2)
        self.assertEqual(_days_between(d1, d2), 1)

    def test_one_year_apart(self):
        d1 = date(2025, 1, 1)
        d2 = date(2026, 1, 1)
        self.assertEqual(_days_between(d1, d2), 365)

    def test_returns_int(self):
        d1 = date(2025, 1, 1)
        d2 = date(2025, 6, 1)
        self.assertIsInstance(_days_between(d1, d2), int)


class TestDecayFactor(unittest.TestCase):

    def test_same_date_gives_one(self):
        d = date(2025, 6, 1)
        result = decay_factor(d, d, 180.0)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_future_as_of_gives_zero(self):
        end = date(2025, 6, 1)
        as_of = date(2025, 5, 1)
        result = decay_factor(end, as_of, 180.0)
        self.assertAlmostEqual(result, 0.0)

    def test_half_life_period_gives_half(self):
        end = date(2025, 1, 1)
        as_of = date(2025, 7, 1)
        result = decay_factor(end, as_of, 181.0)
        self.assertAlmostEqual(result, 0.5, delta=0.01)

    def test_zero_half_life_gives_zero(self):
        d = date(2025, 6, 1)
        result = decay_factor(d, d, 0.0)
        self.assertAlmostEqual(result, 0.0)

    def test_result_between_0_and_1(self):
        end = date(2025, 1, 1)
        as_of = date(2025, 6, 1)
        result = decay_factor(end, as_of, 180.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_returns_float(self):
        d = date(2025, 1, 1)
        self.assertIsInstance(decay_factor(d, d, 180.0), float)


class TestBoostPoints(unittest.TestCase):

    def test_empty_inputs_give_zero(self):
        self.assertAlmostEqual(boost_points({}, {}), 0.0)

    def test_single_feature(self):
        result = boost_points({"worlds:champion": 1.0}, {"worlds:champion": 5.0})
        self.assertAlmostEqual(result, 5.0)

    def test_missing_weight_treated_as_zero(self):
        result = boost_points({"worlds:champion": 1.0}, {})
        self.assertAlmostEqual(result, 0.0)

    def test_multiple_features(self):
        feat_vec = {"worlds:champion": 1.0, "msi:runner_up": 0.5}
        weights = {"worlds:champion": 10.0, "msi:runner_up": 4.0}
        result = boost_points(feat_vec, weights)
        self.assertAlmostEqual(result, 12.0, places=5)

    def test_returns_float(self):
        self.assertIsInstance(boost_points({}, {}), float)


if __name__ == "__main__":
    unittest.main()
