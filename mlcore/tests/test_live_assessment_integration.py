"""Tests for the Phase 6 live state-model integration in build_live_bet_live_assessment.

Verifies:
  - Backwards compatibility: existing callers (without new params) still work
  - live_p_win is None when feature flag is OFF
  - live_p_win is None when p_map_pre is missing or invalid
  - live_p_win is a valid probability when flag is ON and inputs are valid
  - live_p_win is None when no model file exists for the minute
"""
import unittest
from pathlib import Path

import pandas as pd

from app_core.live_bet_analysis_core import build_live_bet_live_assessment


_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "live_models"


def _base_kwargs() -> dict:
    """Minimal kwargs for build_live_bet_live_assessment."""
    return {
        "team_a": "T1",
        "team_b": "GEN",
        "df_scope": pd.DataFrame(),
        "minute": 15,
        "gold_diff": 2500.0,
        "kills_a": 5,
        "kills_b": 2,
        "towers_a": 4,
        "towers_b": 2,
        "dragons_a": 2,
        "dragons_b": 1,
        "barons_a": 0,
        "barons_b": 0,
        "inhibitors_a": 0,
        "inhibitors_b": 0,
        "comp_compare": {},
        "pregame_resolution": {},
        "baselines": None,
    }


class TestBackwardsCompat(unittest.TestCase):

    def test_existing_callers_still_work(self):
        """The function must accept the old call signature without breaking."""
        out = build_live_bet_live_assessment(**_base_kwargs())
        self.assertIsInstance(out, dict)
        self.assertIn("state_edge", out)
        self.assertIn("leader", out)

    def test_old_callers_get_live_p_win_none(self):
        """When session_state is None (old behavior), live_p_win is None."""
        out = build_live_bet_live_assessment(**_base_kwargs())
        self.assertIn("live_p_win", out)
        self.assertIsNone(out["live_p_win"])


class TestFeatureFlagOff(unittest.TestCase):

    def test_flag_off_returns_none(self):
        out = build_live_bet_live_assessment(
            session_state={"ml_live_state_model_enabled": False},
            p_map_pre=0.55, league="LCK", bo=5, is_playoffs=True,
            xp_diff=2000.0, cs_diff=40.0,
            deaths_a=2, deaths_b=5, assists_a=8, assists_b=3,
            **_base_kwargs(),
        )
        self.assertIsNone(out["live_p_win"])


class TestInvalidInputs(unittest.TestCase):

    def test_p_map_pre_none_returns_none(self):
        ss = {"ml_live_state_model_enabled": True}
        out = build_live_bet_live_assessment(
            session_state=ss, p_map_pre=None,
            **_base_kwargs(),
        )
        self.assertIsNone(out["live_p_win"])

    def test_p_map_pre_invalid_returns_none(self):
        ss = {"ml_live_state_model_enabled": True}
        out = build_live_bet_live_assessment(
            session_state=ss, p_map_pre=1.5,  # invalid (>= 1)
            **_base_kwargs(),
        )
        self.assertIsNone(out["live_p_win"])

    def test_minute_far_from_snapshot_returns_none(self):
        """Minutes too far from supported snapshots (10/15/20/25) return None."""
        ss = {"ml_live_state_model_enabled": True}
        kwargs = _base_kwargs()
        kwargs["minute"] = 35  # too far from 25 (delta > 3)
        out = build_live_bet_live_assessment(
            session_state=ss, p_map_pre=0.55, league="LCK",
            **kwargs,
        )
        self.assertIsNone(out["live_p_win"])


class TestWithModelLoaded(unittest.TestCase):

    def setUp(self) -> None:
        if not _MODELS_DIR.exists():
            self.skipTest("live_models/ not present (run Phase 6 to train)")

    def test_blue_lead_returns_high_prob(self):
        """Strong blue lead at minute 15 should give p_win > 0.5 for team A (blue)."""
        ss = {
            "ml_live_state_model_enabled": True,
            "ml_live_state_models_dir": str(_MODELS_DIR),
        }
        out = build_live_bet_live_assessment(
            session_state=ss, p_map_pre=0.50,
            league="LCK", bo=5, is_playoffs=True,
            xp_diff=2500.0, cs_diff=50.0,
            deaths_a=2, deaths_b=5, assists_a=8, assists_b=3,
            **_base_kwargs(),
        )
        self.assertIsNotNone(out["live_p_win"])
        self.assertGreater(out["live_p_win"], 0.5)

    def test_red_lead_returns_low_prob_for_blue(self):
        """When red has the lead, P(blue=team_a wins) < 0.5."""
        ss = {
            "ml_live_state_model_enabled": True,
            "ml_live_state_models_dir": str(_MODELS_DIR),
        }
        kwargs = _base_kwargs()
        kwargs["gold_diff"] = -2500.0  # red leads
        kwargs["kills_a"] = 2
        kwargs["kills_b"] = 5
        kwargs["towers_a"] = 2
        kwargs["towers_b"] = 4
        kwargs["dragons_a"] = 1
        kwargs["dragons_b"] = 2
        out = build_live_bet_live_assessment(
            session_state=ss, p_map_pre=0.50,
            league="LCK", bo=5, is_playoffs=True,
            xp_diff=-2500.0, cs_diff=-50.0,
            deaths_a=5, deaths_b=2, assists_a=3, assists_b=8,
            **kwargs,
        )
        self.assertIsNotNone(out["live_p_win"])
        self.assertLess(out["live_p_win"], 0.5)

    def test_minute_close_to_snapshot_snaps_to_nearest(self):
        """Minute 14 should snap to 15."""
        ss = {
            "ml_live_state_model_enabled": True,
            "ml_live_state_models_dir": str(_MODELS_DIR),
        }
        kwargs = _base_kwargs()
        kwargs["minute"] = 14
        out = build_live_bet_live_assessment(
            session_state=ss, p_map_pre=0.55,
            league="LCK", bo=5, **kwargs,
        )
        self.assertIsNotNone(out["live_p_win"])


if __name__ == "__main__":
    unittest.main()
