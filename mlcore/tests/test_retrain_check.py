"""Tests for mlcore/retrain_check.py — pure helpers."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from mlcore.retrain_check import CheckConfig, _parse_iso_dt, _resolve_csv_path


# ---------------------------------------------------------------------------
# CheckConfig defaults
# ---------------------------------------------------------------------------

class TestCheckConfig(unittest.TestCase):

    def test_default_max_days(self):
        cfg = CheckConfig()
        self.assertEqual(cfg.max_days, 14)

    def test_default_min_new_games(self):
        cfg = CheckConfig()
        self.assertEqual(cfg.min_new_games, 1200)

    def test_default_recent_n(self):
        cfg = CheckConfig()
        self.assertEqual(cfg.recent_n, 2000)

    def test_default_drift_logloss_delta(self):
        cfg = CheckConfig()
        self.assertAlmostEqual(cfg.drift_logloss_delta, 0.025)

    def test_default_min_recent_eval(self):
        cfg = CheckConfig()
        self.assertEqual(cfg.min_recent_eval, 250)

    def test_custom_values(self):
        cfg = CheckConfig(max_days=7, min_new_games=800)
        self.assertEqual(cfg.max_days, 7)
        self.assertEqual(cfg.min_new_games, 800)


# ---------------------------------------------------------------------------
# _parse_iso_dt
# ---------------------------------------------------------------------------

class TestParseIsoDt(unittest.TestCase):

    def test_iso_string(self):
        result = _parse_iso_dt("2026-01-15T12:00:00")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2026)
        self.assertEqual(result.month, 1)
        self.assertEqual(result.day, 15)

    def test_z_suffix(self):
        result = _parse_iso_dt("2026-03-10T08:30:00Z")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, datetime)

    def test_returns_datetime(self):
        result = _parse_iso_dt("2025-06-01T00:00:00")
        self.assertIsInstance(result, datetime)

    def test_invalid_string_returns_none(self):
        self.assertIsNone(_parse_iso_dt("not a date"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_parse_iso_dt(""))

    def test_none_coerced_returns_none(self):
        self.assertIsNone(_parse_iso_dt("None"))

    def test_partial_date(self):
        result = _parse_iso_dt("2026-01-15")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2026)


# ---------------------------------------------------------------------------
# _resolve_csv_path
# ---------------------------------------------------------------------------

class TestResolveCsvPath(unittest.TestCase):

    def test_file_path_returned_unchanged(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
        try:
            result = _resolve_csv_path(path)
            self.assertEqual(result, path)
        finally:
            path.unlink(missing_ok=True)

    def test_directory_with_csv_returns_newest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "a.csv"
            p2 = Path(tmpdir) / "b.csv"
            p1.write_text("data1")
            import time; time.sleep(0.01)
            p2.write_text("data2")
            result = _resolve_csv_path(Path(tmpdir))
            self.assertEqual(result.name, "b.csv")

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                _resolve_csv_path(Path(tmpdir))

    def test_nonexistent_path_returned_as_is(self):
        p = Path("/nonexistent/path.csv")
        result = _resolve_csv_path(p)
        self.assertEqual(result, p)


if __name__ == "__main__":
    unittest.main()
