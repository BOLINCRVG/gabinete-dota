"""Tests for mlcore/fit_io_core.py — IO and serialization helpers."""
import json
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mlcore.fit_io_core import (
    _json_default,
    _read_json_file_safe,
    _safe_int,
    _utc_now_iso,
    _write_json_file_safe,
)


# ---------------------------------------------------------------------------
# _safe_int
# ---------------------------------------------------------------------------

class TestSafeInt(unittest.TestCase):

    def test_valid_int(self):
        self.assertEqual(_safe_int(5), 5)

    def test_valid_float_truncates(self):
        self.assertEqual(_safe_int(3.7), 3)

    def test_valid_string(self):
        self.assertEqual(_safe_int("42"), 42)

    def test_nan_returns_default(self):
        self.assertEqual(_safe_int(float("nan"), 99), 99)

    def test_inf_returns_default(self):
        self.assertEqual(_safe_int(float("inf"), -1), -1)

    def test_invalid_string_returns_default(self):
        self.assertEqual(_safe_int("abc", 7), 7)

    def test_none_returns_default(self):
        self.assertEqual(_safe_int(None, 0), 0)

    def test_zero(self):
        self.assertEqual(_safe_int(0), 0)

    def test_negative(self):
        self.assertEqual(_safe_int(-3), -3)

    def test_default_is_zero_when_omitted(self):
        self.assertEqual(_safe_int(float("nan")), 0)


# ---------------------------------------------------------------------------
# _json_default
# ---------------------------------------------------------------------------

class TestJsonDefault(unittest.TestCase):

    def test_path_to_str(self):
        p = Path("some") / "path"
        result = _json_default(p)
        self.assertIsInstance(result, str)
        self.assertIn("path", result)

    def test_datetime_to_isoformat(self):
        dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _json_default(dt)
        self.assertIn("2026", result)
        self.assertIsInstance(result, str)

    def test_pd_timestamp_to_isoformat(self):
        ts = pd.Timestamp("2026-01-15 12:00:00")
        result = _json_default(ts)
        self.assertIn("2026", result)
        self.assertIsInstance(result, str)

    def test_numpy_integer(self):
        result = _json_default(np.int64(42))
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_numpy_float(self):
        result = _json_default(np.float64(3.14))
        self.assertAlmostEqual(result, 3.14, places=5)
        self.assertIsInstance(result, float)

    def test_numpy_bool_true(self):
        result = _json_default(np.bool_(True))
        self.assertEqual(result, True)
        self.assertIsInstance(result, bool)

    def test_numpy_bool_false(self):
        result = _json_default(np.bool_(False))
        self.assertEqual(result, False)
        self.assertIsInstance(result, bool)

    def test_unknown_type_raises(self):
        with self.assertRaises(TypeError):
            _json_default(object())

    def test_json_roundtrip_with_numpy(self):
        data = {"n": np.int64(10), "v": np.float64(1.5)}
        serialized = json.dumps(data, default=_json_default)
        loaded = json.loads(serialized)
        self.assertEqual(loaded["n"], 10)
        self.assertAlmostEqual(loaded["v"], 1.5)


# ---------------------------------------------------------------------------
# _utc_now_iso
# ---------------------------------------------------------------------------

class TestUtcNowIso(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(_utc_now_iso(), str)

    def test_parseable_as_iso(self):
        s = _utc_now_iso()
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        self.assertIsNotNone(dt)

    def test_recent_timestamp(self):
        s = _utc_now_iso()
        dt = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        delta = abs((now - dt).total_seconds())
        self.assertLess(delta, 5.0)


# ---------------------------------------------------------------------------
# _read_json_file_safe / _write_json_file_safe
# ---------------------------------------------------------------------------

class TestReadWriteJsonFileSafe(unittest.TestCase):

    def test_missing_file_returns_empty_dict(self):
        result = _read_json_file_safe(Path("/nonexistent/path/file.json"))
        self.assertEqual(result, {})

    def test_read_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"key": "value", "n": 42}, f)
            path = Path(f.name)
        try:
            result = _read_json_file_safe(path)
            self.assertEqual(result["key"], "value")
            self.assertEqual(result["n"], 42)
        finally:
            path.unlink(missing_ok=True)

    def test_read_invalid_json_returns_empty_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("not valid json {{{")
            path = Path(f.name)
        try:
            result = _read_json_file_safe(path)
            self.assertEqual(result, {})
        finally:
            path.unlink(missing_ok=True)

    def test_roundtrip_write_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            payload = {"a": 1, "b": [1, 2, 3], "c": "hello"}
            _write_json_file_safe(path, payload)
            result = _read_json_file_safe(path)
            self.assertEqual(result["a"], 1)
            self.assertEqual(result["b"], [1, 2, 3])
            self.assertEqual(result["c"], "hello")

    def test_write_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "test.json"
            _write_json_file_safe(path, {"x": 1})
            self.assertTrue(path.exists())

    def test_non_dict_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
            path = Path(f.name)
        try:
            result = _read_json_file_safe(path)
            self.assertEqual(result, {})
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
