"""Tests for resumo_presets_core + resumo_calibration_core."""
from __future__ import annotations

import unittest

from app_core.picks_tracker_core import Pick, new_pick
from app_core.resumo_calibration_core import (
    calibration_in_range,
    calibration_summary,
)
from app_core.resumo_presets_core import (
    SOURCE_PRESETS,
    SOURCE_PRESET_LABELS,
    aggressiveness_label,
    apply_aggressiveness,
    apply_source_preset,
    resolve_aggressiveness,
)


class TestResolveAggressiveness(unittest.TestCase):

    def test_conservative_band(self):
        p = resolve_aggressiveness(0)
        self.assertEqual(p["resumo_max_odd"], 1.7)
        self.assertEqual(p["resumo_min_sample_h2h"], 5)

    def test_neutral_band(self):
        p = resolve_aggressiveness(5)
        self.assertEqual(p["resumo_max_odd"], 2.0)
        self.assertEqual(p["resumo_min_sample_liga"], 8)

    def test_aggressive_band(self):
        p = resolve_aggressiveness(10)
        self.assertEqual(p["resumo_max_odd"], 3.0)
        self.assertEqual(p["resumo_min_sample_h2h"], 1)

    def test_clamps_out_of_range(self):
        self.assertEqual(resolve_aggressiveness(-99)["resumo_max_odd"], 1.7)
        self.assertEqual(resolve_aggressiveness(999)["resumo_max_odd"], 3.0)

    def test_label_changes_with_band(self):
        self.assertIn("Conservador", aggressiveness_label(2))
        self.assertIn("Neutro", aggressiveness_label(5))
        self.assertIn("Agressivo", aggressiveness_label(9))


class TestApplySourcePreset(unittest.TestCase):

    def test_hist_first_writes_user_preferred_weights(self):
        ss = {}
        applied = apply_source_preset(ss, "hist_first")
        self.assertTrue(applied)
        self.assertEqual(ss["resumo_mix_w_laplace"], 70.0)
        self.assertEqual(ss["resumo_mix_w_model"], 20.0)

    def test_custom_is_noop(self):
        ss = {"resumo_mix_w_laplace": 33.3}
        applied = apply_source_preset(ss, "custom")
        self.assertFalse(applied)
        self.assertEqual(ss["resumo_mix_w_laplace"], 33.3)

    def test_unknown_preset_is_noop(self):
        ss = {"resumo_mix_w_laplace": 33.3}
        applied = apply_source_preset(ss, "qualquer_coisa")
        self.assertFalse(applied)
        self.assertEqual(ss["resumo_mix_w_laplace"], 33.3)

    def test_all_presets_sum_to_100(self):
        for name, weights in SOURCE_PRESETS.items():
            total = sum(weights.values())
            self.assertAlmostEqual(total, 100.0, msg=f"{name} sums to {total}")

    def test_labels_cover_all_presets_plus_custom(self):
        for name in SOURCE_PRESETS:
            self.assertIn(name, SOURCE_PRESET_LABELS)
        self.assertIn("custom", SOURCE_PRESET_LABELS)


class TestApplyAggressiveness(unittest.TestCase):

    def test_writes_session_state(self):
        ss = {}
        apply_aggressiveness(ss, 5)
        self.assertEqual(ss["resumo_max_odd"], 2.0)
        self.assertEqual(ss["resumo_min_sample_liga"], 8)


def _settled(status: str, p_hist: float, metric: str = "kills") -> Pick:
    base = new_pick(team_a="A", team_b="B", league="LCK",
                     metric_key=metric, side_key="Over", line=27.5,
                     odd_casa=2.0, stake_units=1.0, p_hist=p_hist,
                     ts=f"2026-05-{20 + len(metric)}T18:00:00")
    profit = 1.0 if status == "won" else -1.0
    return Pick(**{**base.to_dict(), "status": status, "profit_units": profit})


class TestCalibrationInRange(unittest.TestCase):

    def test_basic_in_window(self):
        picks = [
            _settled("won", 0.62),
            _settled("won", 0.66),
            _settled("lost", 0.68, metric="towers"),
        ]
        r = calibration_in_range(picks, p_center=0.65, half_window_pp=0.05)
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["wins"], 2)
        self.assertAlmostEqual(r["hit_pct"], 66.666666, places=3)

    def test_filter_by_metric(self):
        picks = [
            _settled("won", 0.62),
            _settled("won", 0.66),
            _settled("lost", 0.68, metric="towers"),
        ]
        r = calibration_in_range(picks, p_center=0.65, half_window_pp=0.05, metric_key="kills")
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["wins"], 2)
        self.assertEqual(r["hit_pct"], 100.0)

    def test_empty_returns_none_hit(self):
        r = calibration_in_range([], p_center=0.65)
        self.assertEqual(r["n"], 0)
        self.assertIsNone(r["hit_pct"])
        self.assertEqual(r["confidence"], "low")

    def test_void_skipped(self):
        p = _settled("won", 0.65)
        from dataclasses import replace
        void_pick = Pick(**{**p.to_dict(), "status": "void", "profit_units": 0.0})
        r = calibration_in_range([void_pick], p_center=0.65)
        self.assertEqual(r["n"], 0)

    def test_open_skipped(self):
        p = new_pick(team_a="A", team_b="B", league="LCK",
                      metric_key="kills", side_key="Over", line=27.5,
                      odd_casa=2.0, stake_units=1.0, p_hist=0.65)
        r = calibration_in_range([p], p_center=0.65)
        self.assertEqual(r["n"], 0)

    def test_confidence_buckets(self):
        # Build 20 picks all in same window: should be "high"
        many = [_settled("won", 0.65) for _ in range(20)]
        # Need distinct ts so ids differ — patch them
        for i, p in enumerate(many):
            p.ts = f"2026-05-{i+1:02d}T18:00:00"
            p.id = f"id{i}"
        r = calibration_in_range(many, p_center=0.65, half_window_pp=0.05)
        self.assertEqual(r["confidence"], "high")


class TestCalibrationSummary(unittest.TestCase):

    def test_bins_returned(self):
        picks = [_settled("won", 0.45), _settled("lost", 0.55)]
        out = calibration_summary(picks)
        # 5 bins: 0.40-0.50, 0.50-0.60, ..., 0.80-0.90
        self.assertEqual(len(out), 5)
        # 0.45 falls in [0.40, 0.50)
        self.assertEqual(out[0]["n"], 1)
        self.assertEqual(out[0]["wins"], 1)
        # 0.55 falls in [0.50, 0.60)
        self.assertEqual(out[1]["n"], 1)
        self.assertEqual(out[1]["wins"], 0)

    def test_empty_bins_have_none_hit(self):
        out = calibration_summary([])
        for b in out:
            self.assertEqual(b["n"], 0)
            self.assertIsNone(b["hit_pct"])


if __name__ == "__main__":
    unittest.main()
