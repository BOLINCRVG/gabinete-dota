import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from app_core.calibration_schedule import build_patch_snapshot, recommend_recalibration
from mlcore.calibration_schedule import _patch_gt, _patch_key, _to_dt


class CalibrationScheduleTest(unittest.TestCase):
    def test_build_patch_snapshot(self):
        df = pd.DataFrame(
            [
                {"gameid": "1", "league": "LPL", "patch": "14.1", "date": "2026-02-01"},
                {"gameid": "2", "league": "LPL", "patch": "14.2", "date": "2026-02-10"},
                {"gameid": "3", "league": "LCK", "patch": "14.2", "date": "2026-02-09"},
            ]
        )
        snap = build_patch_snapshot(df)
        self.assertIn("LPL", snap)
        self.assertIn("LCK", snap)
        self.assertEqual("14.2", snap["LPL"]["latest_patch"])
        self.assertEqual("14.2", snap["LCK"]["latest_patch"])

    def test_recommend_recalibration_patch_change(self):
        cur = {
            "LPL": {"latest_patch": "14.2", "games_on_latest_patch": 35},
            "LCK": {"latest_patch": "14.2", "games_on_latest_patch": 8},
        }
        fit = {
            "LPL": {"latest_patch": "14.1"},
            "LCK": {"latest_patch": "14.2"},
        }
        now = datetime.now(timezone.utc).isoformat()
        rep = recommend_recalibration(
            current_snapshot=cur,
            fitted_snapshot=fit,
            fitted_generated_at=now,
            max_age_days=7,
            min_games_new_patch=20,
        )
        self.assertEqual("refit", rep["status"])
        self.assertGreaterEqual(len(rep.get("changed_leagues") or []), 1)

    def test_recommend_recalibration_age_only(self):
        cur = {"LPL": {"latest_patch": "14.2", "games_on_latest_patch": 10}}
        fit = {"LPL": {"latest_patch": "14.2"}}
        old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        rep = recommend_recalibration(
            current_snapshot=cur,
            fitted_snapshot=fit,
            fitted_generated_at=old,
            max_age_days=7,
            min_games_new_patch=20,
        )
        self.assertEqual("warn", rep["status"])


class TestPatchKey(unittest.TestCase):

    def test_normal_patch(self):
        self.assertEqual(_patch_key("14.5"), [14, 5])

    def test_three_part_patch(self):
        self.assertEqual(_patch_key("14.5.1"), [14, 5, 1])

    def test_empty_returns_empty(self):
        self.assertEqual(_patch_key(""), [])

    def test_none_returns_empty(self):
        self.assertEqual(_patch_key(None), [])

    def test_returns_list_of_ints(self):
        result = _patch_key("14.2")
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(x, int) for x in result))


class TestPatchGt(unittest.TestCase):

    def test_higher_minor_is_greater(self):
        self.assertTrue(_patch_gt("14.5", "14.2"))

    def test_lower_minor_is_not_greater(self):
        self.assertFalse(_patch_gt("14.2", "14.5"))

    def test_equal_patches_not_greater(self):
        self.assertFalse(_patch_gt("14.5", "14.5"))

    def test_higher_major_is_greater(self):
        self.assertTrue(_patch_gt("15.1", "14.9"))

    def test_empty_a_is_not_greater(self):
        self.assertFalse(_patch_gt("", "14.5"))

    def test_empty_b_a_is_greater(self):
        self.assertTrue(_patch_gt("14.5", ""))


class TestToDt(unittest.TestCase):

    def test_valid_iso_string(self):
        result = _to_dt("2026-01-15T10:00:00+00:00")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, datetime)

    def test_z_suffix_parsed(self):
        result = _to_dt("2026-01-15T10:00:00Z")
        self.assertIsNotNone(result)

    def test_none_returns_none(self):
        self.assertIsNone(_to_dt(None))

    def test_empty_returns_none(self):
        self.assertIsNone(_to_dt(""))

    def test_invalid_returns_none(self):
        self.assertIsNone(_to_dt("not-a-date"))

    def test_datetime_passthrough(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = _to_dt(dt)
        self.assertIs(result, dt)


class TestBuildPatchSnapshotEdgeCases(unittest.TestCase):

    def test_empty_df_returns_empty(self):
        self.assertEqual(build_patch_snapshot(pd.DataFrame()), {})

    def test_none_returns_empty(self):
        self.assertEqual(build_patch_snapshot(None), {})

    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame([{"league": "LCK", "patch": "14.1"}])
        self.assertEqual(build_patch_snapshot(df), {})

    def test_counts_games_on_latest_patch(self):
        df = pd.DataFrame([
            {"league": "LCK", "patch": "14.2", "date": "2026-01-01", "gameid": "g1"},
            {"league": "LCK", "patch": "14.2", "date": "2026-01-02", "gameid": "g2"},
            {"league": "LCK", "patch": "14.1", "date": "2025-12-01", "gameid": "g3"},
        ])
        snap = build_patch_snapshot(df)
        self.assertEqual(snap["LCK"]["games_on_latest_patch"], 2)


if __name__ == "__main__":
    unittest.main()
