"""Tests for ui_state_persistence_core."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app_core.ui_state_persistence_core import (
    ALLOWED_KEYS,
    collect_session_snapshot,
    load_persisted_state,
    save_persisted_state,
    sync_session_with_persistence,
    validate_number_input,
    validate_text_required,
)


class TestPersistRoundtrip(unittest.TestCase):

    def test_save_and_load(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "ui_state.json"
            ok = save_persisted_state({"lb_team_a": "Gen.G", "lb_team_b": "T1"}, path=p)
            self.assertTrue(ok)
            loaded = load_persisted_state(p)
            self.assertEqual(loaded["lb_team_a"], "Gen.G")
            self.assertEqual(loaded["lb_team_b"], "T1")

    def test_missing_file_returns_empty(self):
        loaded = load_persisted_state("/nonexistent/path/file.json")
        self.assertEqual(loaded, {})

    def test_corrupt_file_returns_empty(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "ui_state.json"
            p.write_text("not valid json{{{", encoding="utf-8")
            self.assertEqual(load_persisted_state(p), {})

    def test_only_allowed_keys_saved(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "ui_state.json"
            save_persisted_state({
                "lb_team_a": "Gen.G",
                "DANGEROUS_KEY": "leaked",
                "_internal": "x",
            }, path=p)
            loaded = load_persisted_state(p)
            self.assertIn("lb_team_a", loaded)
            self.assertNotIn("DANGEROUS_KEY", loaded)
            self.assertNotIn("_internal", loaded)

    def test_non_serializable_skipped(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "ui_state.json"
            save_persisted_state({
                "lb_team_a": "ok",
                "lb_team_b": object(),  # non-JSON
            }, path=p)
            loaded = load_persisted_state(p)
            self.assertEqual(loaded.get("lb_team_a"), "ok")
            self.assertNotIn("lb_team_b", loaded)

    def test_merges_with_existing(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "ui_state.json"
            save_persisted_state({"lb_team_a": "A"}, path=p)
            save_persisted_state({"lb_team_b": "B"}, path=p)
            loaded = load_persisted_state(p)
            self.assertEqual(loaded["lb_team_a"], "A")
            self.assertEqual(loaded["lb_team_b"], "B")


class TestSyncSession(unittest.TestCase):

    def test_hydrates_empty_session(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "ui_state.json"
            save_persisted_state({"lb_team_a": "Gen.G"}, path=p)
            ss = {}
            applied = sync_session_with_persistence(ss, path=p)
            self.assertEqual(ss.get("lb_team_a"), "Gen.G")
            self.assertEqual(applied.get("lb_team_a"), "Gen.G")

    def test_existing_session_value_wins(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "ui_state.json"
            save_persisted_state({"lb_team_a": "PersistedTeam"}, path=p)
            ss = {"lb_team_a": "CurrentTeam"}
            sync_session_with_persistence(ss, path=p)
            self.assertEqual(ss["lb_team_a"], "CurrentTeam")


class TestCollectSnapshot(unittest.TestCase):

    def test_picks_only_allowed(self):
        ss = {"lb_team_a": "X", "DANGEROUS": "y", "lb_team_b": "Z"}
        snap = collect_session_snapshot(ss)
        self.assertIn("lb_team_a", snap)
        self.assertIn("lb_team_b", snap)
        self.assertNotIn("DANGEROUS", snap)


class TestValidateNumber(unittest.TestCase):

    def test_valid_number(self):
        v, err = validate_number_input(3.5, label="X", min_value=0, max_value=10)
        self.assertEqual(v, 3.5)
        self.assertIsNone(err)

    def test_below_min(self):
        _v, err = validate_number_input(-1, label="X", min_value=0)
        self.assertIn("≥ 0", err)

    def test_above_max(self):
        _v, err = validate_number_input(20, label="X", max_value=10)
        self.assertIn("≤ 10", err)

    def test_invalid_string(self):
        v, err = validate_number_input("abc", label="X")
        self.assertIsNone(v)
        self.assertIn("número", err)

    def test_nan_rejected(self):
        _v, err = validate_number_input(float("nan"), label="X")
        self.assertIn("NaN", err)


class TestValidateText(unittest.TestCase):

    def test_valid_text(self):
        v, err = validate_text_required("Hello", label="X")
        self.assertEqual(v, "Hello")
        self.assertIsNone(err)

    def test_empty_rejected(self):
        _v, err = validate_text_required("", label="X")
        self.assertIn("vazio", err)

    def test_min_length_rejected(self):
        _v, err = validate_text_required("ab", label="X", min_length=5)
        self.assertIn("5", err)


if __name__ == "__main__":
    unittest.main()
