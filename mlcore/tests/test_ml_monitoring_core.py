"""Tests for app_core/ml_monitoring_core.py — monitoring snapshot and history helpers."""
import os
import tempfile
import unittest
from datetime import datetime, timezone

from app_core.ml_monitoring_core import (
    summarize_monitor_window,
    update_monitor_history,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# utc_now_iso
# ---------------------------------------------------------------------------

class TestUtcNowIso(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(utc_now_iso(), str)

    def test_parseable_iso(self):
        s = utc_now_iso()
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        self.assertIsNotNone(dt)

    def test_contains_timezone(self):
        s = utc_now_iso()
        self.assertTrue("+" in s or "Z" in s)

    def test_recent_timestamp(self):
        s = utc_now_iso()
        dt = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        delta = abs((now - dt).total_seconds())
        self.assertLess(delta, 5.0)


# ---------------------------------------------------------------------------
# summarize_monitor_window
# ---------------------------------------------------------------------------

def _entry(ok: bool, ts: str = "2026-05-25T10:00:00+00:00") -> dict:
    return {
        "generated_at": ts,
        "health": {"overall_ok": ok, "blockers": [] if ok else ["test_blocker"]},
    }


class TestSummarizeMonitorWindow(unittest.TestCase):

    def test_empty_entries_returns_zeros(self):
        out = summarize_monitor_window([], now_iso="2026-05-25T12:00:00+00:00")
        self.assertEqual(out["entries"], 0)
        self.assertEqual(out["healthy_entries"], 0)
        self.assertAlmostEqual(out["healthy_ratio"], 0.0)

    def test_all_healthy(self):
        entries = [_entry(True, "2026-05-25T10:00:00+00:00")] * 3
        out = summarize_monitor_window(entries, now_iso="2026-05-25T12:00:00+00:00")
        self.assertEqual(out["entries"], 3)
        self.assertEqual(out["healthy_entries"], 3)
        self.assertAlmostEqual(out["healthy_ratio"], 1.0)

    def test_all_unhealthy(self):
        entries = [_entry(False, "2026-05-25T10:00:00+00:00")] * 2
        out = summarize_monitor_window(entries, now_iso="2026-05-25T12:00:00+00:00")
        self.assertEqual(out["unhealthy_entries"], 2)
        self.assertAlmostEqual(out["healthy_ratio"], 0.0)

    def test_mixed_healthy_unhealthy(self):
        entries = [
            _entry(True,  "2026-05-25T10:00:00+00:00"),
            _entry(False, "2026-05-25T10:30:00+00:00"),
            _entry(True,  "2026-05-25T11:00:00+00:00"),
        ]
        out = summarize_monitor_window(entries, now_iso="2026-05-25T12:00:00+00:00")
        self.assertEqual(out["healthy_entries"], 2)
        self.assertEqual(out["unhealthy_entries"], 1)

    def test_streak_all_healthy(self):
        entries = [_entry(True, "2026-05-25T10:00:00+00:00")] * 4
        out = summarize_monitor_window(entries, now_iso="2026-05-25T12:00:00+00:00")
        self.assertEqual(out["consecutive_healthy_streak"], 4)

    def test_streak_broken_by_unhealthy(self):
        entries = [
            _entry(False, "2026-05-25T09:00:00+00:00"),
            _entry(True,  "2026-05-25T10:00:00+00:00"),
            _entry(True,  "2026-05-25T11:00:00+00:00"),
        ]
        out = summarize_monitor_window(entries, now_iso="2026-05-25T12:00:00+00:00")
        self.assertEqual(out["consecutive_healthy_streak"], 2)

    def test_cutoff_filters_old_entries(self):
        entries = [
            _entry(True, "2026-05-10T10:00:00+00:00"),  # 15 days ago → outside 7d window
            _entry(True, "2026-05-24T10:00:00+00:00"),  # 1 day ago → inside
        ]
        out = summarize_monitor_window(entries, days=7, now_iso="2026-05-25T12:00:00+00:00")
        self.assertEqual(out["entries"], 1)

    def test_output_has_required_keys(self):
        out = summarize_monitor_window([])
        for key in ("window_days", "entries", "healthy_entries", "unhealthy_entries",
                    "healthy_ratio", "consecutive_healthy_streak",
                    "latest_overall_ok", "latest_blockers"):
            self.assertIn(key, out)

    def test_latest_ok_reflects_last_entry(self):
        entries = [
            _entry(True,  "2026-05-25T09:00:00+00:00"),
            _entry(False, "2026-05-25T11:00:00+00:00"),
        ]
        out = summarize_monitor_window(entries, now_iso="2026-05-25T12:00:00+00:00")
        self.assertFalse(out["latest_overall_ok"])
        self.assertGreater(len(out["latest_blockers"]), 0)

    def test_non_dict_entries_skipped(self):
        out = summarize_monitor_window(["not", "a", "dict"], now_iso="2026-05-25T12:00:00+00:00")
        self.assertEqual(out["entries"], 0)


# ---------------------------------------------------------------------------
# update_monitor_history
# ---------------------------------------------------------------------------

class TestUpdateMonitorHistory(unittest.TestCase):

    def _snap(self, ok: bool = True, ts: str = "2026-05-25T10:00:00+00:00") -> dict:
        return {
            "generated_at": ts,
            "health": {"overall_ok": ok, "blockers": []},
        }

    def test_none_history_with_snapshot(self):
        snap = self._snap()
        out = update_monitor_history(None, snap)
        self.assertIn("entries", out)
        self.assertEqual(len(out["entries"]), 1)

    def test_appends_to_existing(self):
        existing = {"entries": [self._snap(ts="2026-05-24T10:00:00+00:00")]}
        out = update_monitor_history(existing, self._snap(ts="2026-05-25T10:00:00+00:00"))
        self.assertEqual(len(out["entries"]), 2)

    def test_sorted_by_generated_at(self):
        existing = {"entries": [self._snap(ts="2026-05-25T10:00:00+00:00")]}
        out = update_monitor_history(existing, self._snap(ts="2026-05-24T10:00:00+00:00"))
        timestamps = [e["generated_at"] for e in out["entries"]]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_keeps_max_entries(self):
        snaps = [self._snap(ts=f"2026-05-{10+i:02d}T10:00:00+00:00") for i in range(5)]
        history = {"entries": snaps}
        out = update_monitor_history(history, self._snap(ts="2026-05-20T10:00:00+00:00"), keep_entries=3)
        self.assertLessEqual(len(out["entries"]), 3)

    def test_output_has_required_keys(self):
        out = update_monitor_history(None, self._snap())
        for key in ("generated_at", "entries", "summary"):
            self.assertIn(key, out)

    def test_summary_is_dict(self):
        out = update_monitor_history(None, self._snap())
        self.assertIsInstance(out["summary"], dict)

    def test_generated_at_is_string(self):
        out = update_monitor_history(None, self._snap())
        self.assertIsInstance(out["generated_at"], str)

    def test_non_dict_entries_in_history_skipped(self):
        history = {"entries": ["bad", 42, {"generated_at": "2026-05-24T10:00:00+00:00",
                                            "health": {"overall_ok": True, "blockers": []}}]}
        out = update_monitor_history(history, self._snap())
        # Only dict entries + new snap kept
        self.assertEqual(len(out["entries"]), 2)


if __name__ == "__main__":
    unittest.main()
