"""Tests for app_core/live_state_panel_ui_core.py.

We use a minimal FakeStreamlit double so the panel can be exercised without
launching Streamlit. The double captures inputs and metrics so we can assert
behavior.
"""
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from app_core.live_state_panel_ui_core import (
    _init_state,
    _safe_float,
    _state_is_filled,
    _state_key,
    render_live_state_panel,
)


_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "live_models"


class _FakeColumn:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeContainer:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeSt:
    """Minimal Streamlit double for headless testing."""

    def __init__(self):
        self.metrics = []
        self.markdowns = []
        self.captions = []
        self.errors = []
        self.infos = []
        self.successes = []
        self.warnings = []
        self.buttons = []
        self.codes = []

    def container(self, *args, **kwargs):
        return _FakeContainer()

    def expander(self, *args, **kwargs):
        return _FakeContainer()

    def markdown(self, text, **kwargs):
        self.markdowns.append(str(text))

    def caption(self, text, **kwargs):
        self.captions.append(str(text))

    def error(self, text, **kwargs):
        self.errors.append(str(text))

    def info(self, text, **kwargs):
        self.infos.append(str(text))

    def success(self, text, **kwargs):
        self.successes.append(str(text))

    def warning(self, text, **kwargs):
        self.warnings.append(str(text))

    def button(self, text, **kwargs):
        self.buttons.append(str(text))
        return False

    def code(self, text, **kwargs):
        self.codes.append(str(text))

    def columns(self, spec, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return tuple(_FakeColumn() for _ in range(n))

    def number_input(self, label, *, min_value=None, max_value=None,
                     step=None, value=0, key=None, help=None):
        return value

    def metric(self, label, value, *, delta=None, help=None, **kwargs):
        self.metrics.append({"label": label, "value": value, "delta": delta})


class TestStateKey(unittest.TestCase):

    def test_key_includes_field_and_teams(self):
        k = _state_key("T1", "GEN", "minute")
        self.assertIn("minute", k)
        self.assertIn("T1", k)
        self.assertIn("GEN", k)

    def test_key_truncates_long_team_names(self):
        long_name = "X" * 200
        k = _state_key(long_name, "GEN", "minute")
        # Should not blow up; field still in key
        self.assertIn("minute", k)


class TestSafeFloat(unittest.TestCase):

    def test_valid_value(self):
        self.assertAlmostEqual(_safe_float(3.14), 3.14)

    def test_string_number(self):
        self.assertAlmostEqual(_safe_float("2.5"), 2.5)

    def test_nan_returns_default(self):
        self.assertAlmostEqual(_safe_float(float("nan"), default=7.0), 7.0)

    def test_invalid_returns_default(self):
        self.assertAlmostEqual(_safe_float("bad", default=1.0), 1.0)


class TestInitState(unittest.TestCase):

    def test_initial_keys_set(self):
        ss = {}
        _init_state(ss, "T1", "GEN")
        self.assertIn(_state_key("T1", "GEN", "minute"), ss)
        self.assertIn(_state_key("T1", "GEN", "gold_diff"), ss)

    def test_existing_values_preserved(self):
        ss = {_state_key("T1", "GEN", "minute"): 22.0}
        _init_state(ss, "T1", "GEN")
        self.assertEqual(ss[_state_key("T1", "GEN", "minute")], 22.0)


class TestRenderPanel(unittest.TestCase):

    def test_empty_team_shows_info(self):
        fake = _FakeSt()
        out = render_live_state_panel(
            st=fake, session_state={}, team_a="", team_b="",
        )
        self.assertEqual(out, {})
        self.assertTrue(any("Selecione" in msg for msg in fake.infos))

    def test_basic_render_returns_assessment(self):
        fake = _FakeSt()
        ss = {}
        # Pre-populate inputs to simulate a non-trivial state
        _init_state(ss, "T1", "GEN")
        ss[_state_key("T1", "GEN", "minute")] = 15.0
        ss[_state_key("T1", "GEN", "gold_diff")] = 2500.0
        ss[_state_key("T1", "GEN", "kills_a")] = 5
        ss[_state_key("T1", "GEN", "kills_b")] = 2

        out = render_live_state_panel(
            st=fake, session_state=ss,
            team_a="T1", team_b="GEN",
            p_map_pre=0.55, league="LCK", bo=5, is_playoffs=True,
        )
        # Assessment dict returned
        self.assertIsInstance(out, dict)
        # When the model exists, live_p_win should be present (possibly None)
        self.assertIn("live_p_win", out)
        # State_edge always present (heuristic)
        self.assertIn("state_edge", out)
        # The metric calls captured the live prob (or N/A)
        self.assertTrue(any("vence" in m["label"].lower() or "edge" in m["label"].lower()
                            for m in fake.metrics))

    def test_render_with_models_returns_calibrated_p(self):
        if not _MODELS_DIR.exists():
            self.skipTest("live_models/ not present")
        fake = _FakeSt()
        ss = {
            "ml_live_state_model_enabled": True,
            "ml_live_state_models_dir": str(_MODELS_DIR),
        }
        _init_state(ss, "T1", "GEN")
        ss[_state_key("T1", "GEN", "minute")] = 15.0
        ss[_state_key("T1", "GEN", "gold_diff")] = 3000.0
        ss[_state_key("T1", "GEN", "xp_diff")] = 2500.0
        ss[_state_key("T1", "GEN", "cs_diff")] = 50.0
        ss[_state_key("T1", "GEN", "kills_a")] = 6
        ss[_state_key("T1", "GEN", "kills_b")] = 2

        out = render_live_state_panel(
            st=fake, session_state=ss,
            team_a="T1", team_b="GEN",
            p_map_pre=0.55, league="LCK", bo=5, is_playoffs=True,
        )
        # With models active and strong blue lead, live_p_win should be > 0.5
        self.assertIsNotNone(out.get("live_p_win"))
        self.assertGreater(out["live_p_win"], 0.5)
        # Metric value should be percentage string
        live_metric = [m for m in fake.metrics if "modelo" in m["label"].lower()]
        self.assertTrue(len(live_metric) >= 1)

    def test_panel_handles_invalid_p_map_pre(self):
        fake = _FakeSt()
        ss = {}
        _init_state(ss, "T1", "GEN")
        # p_map_pre out of bounds should default to 0.5, not crash
        out = render_live_state_panel(
            st=fake, session_state=ss,
            team_a="T1", team_b="GEN",
            p_map_pre=1.5,  # invalid
        )
        self.assertIsInstance(out, dict)

    def test_default_mode_shows_pregame_p_without_state_inputs(self):
        """Default mode: only teams + league + p_map_pre, no state required."""
        fake = _FakeSt()
        ss = {}
        out = render_live_state_panel(
            st=fake, session_state=ss,
            team_a="T1", team_b="GEN",
            p_map_pre=0.62, league="LCK", bo=5, is_playoffs=True,
        )
        # No state was filled — assessment is empty, but the headline metric
        # is still rendered using p_map_pre.
        self.assertEqual(out, {})
        pre_metric = [m for m in fake.metrics
                      if "vence" in m["label"].lower() and "pré-jogo" in m["label"].lower()]
        self.assertTrue(len(pre_metric) >= 1)
        # Value should reflect p_map_pre = 0.62 → "62.0%"
        self.assertIn("62.0%", str(pre_metric[0]["value"]))

    def test_default_mode_with_no_p_map_pre_shows_na(self):
        """No p_map_pre and no state → display N/A with helpful caption."""
        fake = _FakeSt()
        ss = {}
        out = render_live_state_panel(
            st=fake, session_state=ss,
            team_a="T1", team_b="GEN",
            p_map_pre=None, league="LCK",
        )
        self.assertEqual(out, {})
        na_metric = [m for m in fake.metrics
                     if m["value"] == "N/A" and "vence" in m["label"].lower()]
        self.assertTrue(len(na_metric) >= 1)


class TestStateIsFilled(unittest.TestCase):

    def test_no_state_returns_false(self):
        ss = {}
        _init_state(ss, "T1", "GEN")
        self.assertFalse(_state_is_filled(ss, "T1", "GEN"))

    def test_gold_diff_set_returns_true(self):
        ss = {}
        _init_state(ss, "T1", "GEN")
        ss[_state_key("T1", "GEN", "gold_diff")] = 1200.0
        self.assertTrue(_state_is_filled(ss, "T1", "GEN"))

    def test_kills_set_returns_true(self):
        ss = {}
        _init_state(ss, "T1", "GEN")
        ss[_state_key("T1", "GEN", "kills_a")] = 3
        self.assertTrue(_state_is_filled(ss, "T1", "GEN"))

    def test_minute_non_default_returns_true(self):
        ss = {}
        _init_state(ss, "T1", "GEN")
        ss[_state_key("T1", "GEN", "minute")] = 20.0
        self.assertTrue(_state_is_filled(ss, "T1", "GEN"))

    def test_minute_at_default_alone_returns_false(self):
        ss = {}
        _init_state(ss, "T1", "GEN")
        # minute stays at default 15.0; nothing else touched
        self.assertFalse(_state_is_filled(ss, "T1", "GEN"))


if __name__ == "__main__":
    unittest.main()
