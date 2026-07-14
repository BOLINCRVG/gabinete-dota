"""Tests for mlcore/roster_signal_core.py — roster continuity and transfer signal."""
import unittest

from mlcore.roster_signal_core import (
    lineup_continuity_from_details,
    transfer_signal_from_details,
)


def _tier_priority(tier: str) -> int:
    """Stub: higher tier → higher priority number (T1 best=3, T3 worst=1).
    Implementation counts 'good' when priority(last) < priority(cur), i.e. tier improved."""
    return {"T3": 1, "T2": 2, "T1": 3}.get(tier.upper(), 0)


def _detail(*, blue_last="T2", blue_cur="T1", red_last="T2", red_cur="T2", used=True) -> dict:
    return {
        "used": used,
        "blue_dbg": {"last_tier": blue_last, "cur_tier": blue_cur},
        "red_dbg": {"last_tier": red_last, "cur_tier": red_cur},
    }


# ---------------------------------------------------------------------------
# transfer_signal_from_details
# ---------------------------------------------------------------------------

class TestTransferSignalFromDetails(unittest.TestCase):

    def test_none_returns_zero(self):
        self.assertAlmostEqual(transfer_signal_from_details(None, tier_priority_fn=_tier_priority), 0.0)

    def test_empty_list_returns_zero(self):
        self.assertAlmostEqual(transfer_signal_from_details([], tier_priority_fn=_tier_priority), 0.0)

    def test_non_dict_entries_skipped(self):
        result = transfer_signal_from_details(["not", "a", "dict"], tier_priority_fn=_tier_priority)
        self.assertAlmostEqual(result, 0.0)

    def test_unused_entries_skipped(self):
        detail = _detail(used=False)
        result = transfer_signal_from_details([detail], tier_priority_fn=_tier_priority)
        self.assertAlmostEqual(result, 0.0)

    def test_upgrade_tier_counts_as_good(self):
        # blue moved from T2→T1 (priority 2→1, improvement), red same tier
        detail = _detail(blue_last="T2", blue_cur="T1", red_last="T3", red_cur="T3")
        result = transfer_signal_from_details([detail], tier_priority_fn=_tier_priority)
        # 1 good out of 2 sides
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_no_upgrades_gives_zero(self):
        detail = _detail(blue_last="T1", blue_cur="T2", red_last="T1", red_cur="T2")
        result = transfer_signal_from_details([detail], tier_priority_fn=_tier_priority)
        self.assertAlmostEqual(result, 0.0)

    def test_all_upgrades_gives_one(self):
        detail = _detail(blue_last="T2", blue_cur="T1", red_last="T3", red_cur="T1")
        result = transfer_signal_from_details([detail], tier_priority_fn=_tier_priority)
        self.assertAlmostEqual(result, 1.0)

    def test_result_clamped_to_unit_interval(self):
        details = [_detail(blue_last="T2", blue_cur="T1", red_last="T3", red_cur="T1")] * 5
        result = transfer_signal_from_details(details, tier_priority_fn=_tier_priority)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_missing_tier_keys_skipped(self):
        detail = {"used": True, "blue_dbg": {}, "red_dbg": {}}
        result = transfer_signal_from_details([detail], tier_priority_fn=_tier_priority)
        self.assertAlmostEqual(result, 0.0)

    def test_non_dict_dbg_skipped(self):
        detail = {"used": True, "blue_dbg": "bad", "red_dbg": None}
        result = transfer_signal_from_details([detail], tier_priority_fn=_tier_priority)
        self.assertAlmostEqual(result, 0.0)


# ---------------------------------------------------------------------------
# lineup_continuity_from_details
# ---------------------------------------------------------------------------

class TestLineupContinuityFromDetails(unittest.TestCase):

    def test_none_details_uses_fresh_cov(self):
        result = lineup_continuity_from_details(None, freshness_avg=0.8, cov_ratio=0.6)
        expected = 0.5 * 0.8 + 0.5 * 0.6
        self.assertAlmostEqual(result, expected, places=5)

    def test_empty_list_uses_fresh_cov(self):
        result = lineup_continuity_from_details([], freshness_avg=1.0, cov_ratio=1.0)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_output_clamped_to_unit_interval(self):
        detail = {"used": True, "blue_dbg": {"gp": 20.0, "transfer_scale": 1.0}, "red_dbg": {}}
        result = lineup_continuity_from_details([detail], freshness_avg=1.0, cov_ratio=1.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_high_gp_high_transfer_gives_high_continuity(self):
        detail = {"used": True,
                  "blue_dbg": {"gp": 24.0, "transfer_scale": 1.0},
                  "red_dbg": {"gp": 24.0, "transfer_scale": 1.0}}
        result = lineup_continuity_from_details([detail], freshness_avg=1.0, cov_ratio=1.0)
        self.assertGreater(result, 0.7)

    def test_zero_gp_low_transfer_lowers_continuity(self):
        detail = {"used": True,
                  "blue_dbg": {"gp": 0.0, "transfer_scale": 0.0},
                  "red_dbg": {"gp": 0.0, "transfer_scale": 0.0}}
        result = lineup_continuity_from_details([detail], freshness_avg=0.0, cov_ratio=0.0)
        self.assertLess(result, 0.5)

    def test_unused_entries_ignored(self):
        detail_used = {"used": True, "blue_dbg": {"gp": 12.0, "transfer_scale": 1.0}, "red_dbg": {}}
        detail_unused = {"used": False, "blue_dbg": {"gp": 0.0, "transfer_scale": 0.0}, "red_dbg": {}}
        result_used = lineup_continuity_from_details([detail_used], freshness_avg=0.5, cov_ratio=0.5)
        result_both = lineup_continuity_from_details([detail_used, detail_unused], freshness_avg=0.5, cov_ratio=0.5)
        self.assertAlmostEqual(result_used, result_both, places=8)

    def test_nan_freshness_treated_as_zero(self):
        result = lineup_continuity_from_details(None, freshness_avg=float("nan"), cov_ratio=0.0)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_non_dict_details_entries_skipped(self):
        result = lineup_continuity_from_details(["bad", 42], freshness_avg=0.5, cov_ratio=0.5)
        expected = 0.5 * 0.5 + 0.5 * 0.5
        self.assertAlmostEqual(result, expected, places=5)


if __name__ == "__main__":
    unittest.main()
