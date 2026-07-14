"""Tests for mlcore/team_artifact_utils.py — pure utility helpers."""
import math
import unittest
from pathlib import Path

from mlcore.team_artifact_utils import (
    _days_between_iso,
    _file_sig_info,
    _league_k_multiplier,
    _league_macro_key,
    _league_patch_vol_multiplier,
    _running_avg,
    _safe_iso_date,
    _safe_nonneg_int,
    _same_source_path_order,
    _source_sig_changed,
)


class TestSafeNonnegInt(unittest.TestCase):

    def test_positive_int(self):
        self.assertEqual(_safe_nonneg_int(5), 5)

    def test_negative_clamped_to_zero(self):
        self.assertEqual(_safe_nonneg_int(-3), 0)

    def test_float_truncated(self):
        self.assertEqual(_safe_nonneg_int(3.9), 3)

    def test_string_int(self):
        self.assertEqual(_safe_nonneg_int("7"), 7)

    def test_nan_returns_default(self):
        self.assertEqual(_safe_nonneg_int(float("nan"), 99), 99)

    def test_none_returns_default(self):
        self.assertEqual(_safe_nonneg_int(None, 0), 0)

    def test_zero(self):
        self.assertEqual(_safe_nonneg_int(0), 0)


class TestSafeIsoDate(unittest.TestCase):

    def test_valid_date_string(self):
        result = _safe_iso_date("2026-03-15")
        self.assertEqual(result, "2026-03-15")

    def test_none_returns_none(self):
        self.assertIsNone(_safe_iso_date(None))

    def test_invalid_returns_none(self):
        self.assertIsNone(_safe_iso_date("not-a-date"))

    def test_datetime_object_works(self):
        import pandas as pd
        result = _safe_iso_date(pd.Timestamp("2026-01-01"))
        self.assertEqual(result, "2026-01-01")


class TestDaysBetweenIso(unittest.TestCase):

    def test_same_date_zero(self):
        self.assertEqual(_days_between_iso("2026-01-01", "2026-01-01"), 0)

    def test_positive_diff(self):
        self.assertEqual(_days_between_iso("2026-01-01", "2026-01-08"), 7)

    def test_negative_diff(self):
        self.assertEqual(_days_between_iso("2026-01-08", "2026-01-01"), -7)

    def test_none_a_returns_zero(self):
        self.assertEqual(_days_between_iso(None, "2026-01-01"), 0)

    def test_none_b_returns_zero(self):
        self.assertEqual(_days_between_iso("2026-01-01", None), 0)

    def test_invalid_returns_zero(self):
        self.assertEqual(_days_between_iso("bad", "2026-01-01"), 0)


class TestFileSigInfo(unittest.TestCase):

    def test_nonexistent_file(self):
        out = _file_sig_info("__nonexistent_file__.csv")
        self.assertFalse(out["exists"])
        self.assertIsNone(out["size"])

    def test_existing_file(self, tmp_path=None):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("hello")
            out = _file_sig_info(path)
            self.assertTrue(out["exists"])
            self.assertGreater(out["size"], 0)
            self.assertIsNotNone(out["mtime_ns"])
        finally:
            os.unlink(path)

    def test_returns_dict_with_keys(self):
        out = _file_sig_info("anything")
        for key in ("path", "exists", "size", "mtime_ns"):
            self.assertIn(key, out)


class TestSameSourcePathOrder(unittest.TestCase):

    def test_identical(self):
        a = [{"path": "/a"}, {"path": "/b"}]
        self.assertTrue(_same_source_path_order(a, a))

    def test_different_order(self):
        a = [{"path": "/a"}, {"path": "/b"}]
        b = [{"path": "/b"}, {"path": "/a"}]
        self.assertFalse(_same_source_path_order(a, b))

    def test_different_length(self):
        a = [{"path": "/a"}]
        b = [{"path": "/a"}, {"path": "/b"}]
        self.assertFalse(_same_source_path_order(a, b))

    def test_empty_both(self):
        self.assertTrue(_same_source_path_order([], []))

    def test_non_dict_entries(self):
        self.assertFalse(_same_source_path_order(["bad"], ["bad"]))


class TestSourceSigChanged(unittest.TestCase):

    def test_identical_sigs(self):
        sig = {"exists": True, "size": 100, "mtime_ns": 123456}
        self.assertFalse(_source_sig_changed(sig, sig))

    def test_size_changed(self):
        a = {"exists": True, "size": 100, "mtime_ns": 123}
        b = {"exists": True, "size": 200, "mtime_ns": 123}
        self.assertTrue(_source_sig_changed(a, b))

    def test_exists_changed(self):
        a = {"exists": True, "size": 100, "mtime_ns": 123}
        b = {"exists": False, "size": 100, "mtime_ns": 123}
        self.assertTrue(_source_sig_changed(a, b))

    def test_mtime_changed(self):
        a = {"exists": True, "size": 100, "mtime_ns": 100}
        b = {"exists": True, "size": 100, "mtime_ns": 200}
        self.assertTrue(_source_sig_changed(a, b))


class TestRunningAvg(unittest.TestCase):

    def test_first_value(self):
        self.assertAlmostEqual(_running_avg(0.0, 0, 5.0), 5.0)

    def test_running_update(self):
        avg = _running_avg(0.0, 0, 10.0)
        avg = _running_avg(avg, 1, 20.0)
        self.assertAlmostEqual(avg, 15.0, places=6)

    def test_converges(self):
        avg = 0.0
        for i, v in enumerate([10.0, 10.0, 10.0]):
            avg = _running_avg(avg, i, v)
        self.assertAlmostEqual(avg, 10.0, places=6)

    def test_zero_n(self):
        self.assertAlmostEqual(_running_avg(99.0, 0, 5.0), 5.0)


class TestLeagueMultipliers(unittest.TestCase):

    def test_macro_key_lck(self):
        self.assertEqual(_league_macro_key("LCK"), "KR")

    def test_macro_key_lpl(self):
        self.assertEqual(_league_macro_key("LPL"), "CN")

    def test_macro_key_unknown(self):
        self.assertEqual(_league_macro_key("UNKNOWN_LEAGUE"), "OTHER")

    def test_k_multiplier_kr_higher(self):
        kr = _league_k_multiplier("LCK")
        other = _league_k_multiplier("UNKNOWN")
        self.assertGreater(kr, other)

    def test_k_multiplier_returns_float(self):
        self.assertIsInstance(_league_k_multiplier("LCK"), float)

    def test_patch_vol_multiplier_cn_higher(self):
        cn = _league_patch_vol_multiplier("LPL")
        other = _league_patch_vol_multiplier("UNKNOWN")
        self.assertGreater(cn, other)

    def test_patch_vol_multiplier_returns_float(self):
        self.assertIsInstance(_league_patch_vol_multiplier("LEC"), float)


if __name__ == "__main__":
    unittest.main()
