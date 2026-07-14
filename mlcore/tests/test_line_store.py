"""Tests for app_core/line_store.py — JSON store helpers."""
import json
import tempfile
import unittest
from pathlib import Path

from app_core.line_store import (
    ensure_session_store,
    load_json_store,
    persist_json_store,
)


# ---------------------------------------------------------------------------
# load_json_store
# ---------------------------------------------------------------------------

class TestLoadJsonStore(unittest.TestCase):

    def test_missing_file_returns_empty_dict(self):
        result = load_json_store(Path("/nonexistent/path/store.json"))
        self.assertEqual(result, {})

    def test_valid_json_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"key": "value", "n": 42}, f)
            path = Path(f.name)
        try:
            result = load_json_store(path)
            self.assertEqual(result["key"], "value")
            self.assertEqual(result["n"], 42)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_json_returns_empty_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{invalid json}")
            path = Path(f.name)
        try:
            result = load_json_store(path)
            self.assertEqual(result, {})
        finally:
            path.unlink(missing_ok=True)

    def test_non_dict_json_returns_empty_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
            path = Path(f.name)
        try:
            result = load_json_store(path)
            self.assertEqual(result, {})
        finally:
            path.unlink(missing_ok=True)

    def test_returns_dict_type(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"a": 1}, f)
            path = Path(f.name)
        try:
            result = load_json_store(path)
            self.assertIsInstance(result, dict)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# persist_json_store
# ---------------------------------------------------------------------------

class TestPersistJsonStore(unittest.TestCase):

    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store.json"
            store = {"foo": "bar", "x": 123}
            persist_json_store(path, store)
            self.assertTrue(path.exists())
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["foo"], "bar")

    def test_non_dict_store_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store.json"
            persist_json_store(path, [1, 2, 3])
            self.assertFalse(path.exists())

    def test_roundtrip_with_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store.json"
            store = {"a": 1, "b": [2, 3], "c": {"nested": True}}
            persist_json_store(path, store)
            result = load_json_store(path)
            self.assertEqual(result["a"], 1)
            self.assertEqual(result["b"], [2, 3])
            self.assertTrue(result["c"]["nested"])

    def test_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store.json"
            persist_json_store(path, {"v": 1})
            persist_json_store(path, {"v": 2})
            result = load_json_store(path)
            self.assertEqual(result["v"], 2)

    def test_empty_dict_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store.json"
            persist_json_store(path, {})
            result = load_json_store(path)
            self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# ensure_session_store
# ---------------------------------------------------------------------------

class TestEnsureSessionStore(unittest.TestCase):

    def test_creates_store_when_missing(self):
        state = {}
        result = ensure_session_store(state, "my_key", {"default": 1})
        self.assertEqual(result["default"], 1)
        self.assertIn("my_key", state)

    def test_returns_existing_dict(self):
        existing = {"existing_key": "value"}
        state = {"my_key": existing}
        result = ensure_session_store(state, "my_key", {})
        self.assertIs(result, existing)

    def test_replaces_non_dict_value(self):
        state = {"my_key": "not a dict"}
        result = ensure_session_store(state, "my_key", {"fallback": True})
        self.assertIsInstance(result, dict)
        self.assertTrue(result["fallback"])

    def test_replaces_none_value(self):
        state = {"my_key": None}
        result = ensure_session_store(state, "my_key", {"x": 99})
        self.assertEqual(result["x"], 99)

    def test_empty_fallback_gives_empty_dict(self):
        state = {}
        result = ensure_session_store(state, "my_key", {})
        self.assertEqual(result, {})

    def test_state_updated_in_place(self):
        state = {}
        result = ensure_session_store(state, "k", {"v": 42})
        self.assertIs(state["k"], result)


if __name__ == "__main__":
    unittest.main()
