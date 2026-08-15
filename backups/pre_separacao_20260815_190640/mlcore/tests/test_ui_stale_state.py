"""Tests for ui_stale_state_core."""
from __future__ import annotations

import unittest

from app_core.ui_stale_state_core import (
    context_signature,
    is_stale,
    mark_panel_refreshed,
    render_stale_banner_if_needed,
)


class _FakeSt:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.captions = []
        self.errors = []
        self.successes = []
    def info(self, m, **k): self.infos.append(str(m))
    def warning(self, m, **k): self.warnings.append(str(m))
    def caption(self, m, **k): self.captions.append(str(m))
    def error(self, m, **k): self.errors.append(str(m))
    def success(self, m, **k): self.successes.append(str(m))


class TestSignature(unittest.TestCase):

    def test_same_ctx_same_signature(self):
        sig1 = context_signature({"team_a": "A", "team_b": "B"})
        sig2 = context_signature({"team_b": "B", "team_a": "A"})  # reordered
        self.assertEqual(sig1, sig2)

    def test_different_ctx_different_signature(self):
        sig1 = context_signature({"team_a": "A"})
        sig2 = context_signature({"team_a": "B"})
        self.assertNotEqual(sig1, sig2)

    def test_signature_length(self):
        sig = context_signature({"x": 1})
        self.assertEqual(len(sig), 12)


class TestStaleness(unittest.TestCase):

    def test_first_render_not_stale(self):
        ss = {}
        ctx = {"team_a": "A"}
        self.assertFalse(is_stale(ss, "panel", ctx))

    def test_after_mark_same_ctx_not_stale(self):
        ss = {}
        ctx = {"team_a": "A"}
        mark_panel_refreshed(ss, "panel", ctx)
        self.assertFalse(is_stale(ss, "panel", ctx))

    def test_after_mark_different_ctx_stale(self):
        ss = {}
        mark_panel_refreshed(ss, "panel", {"team_a": "A"})
        self.assertTrue(is_stale(ss, "panel", {"team_a": "B"}))

    def test_per_panel_isolation(self):
        ss = {}
        mark_panel_refreshed(ss, "panel_a", {"x": 1})
        # panel_b has no signature → not stale (first render)
        self.assertFalse(is_stale(ss, "panel_b", {"x": 2}))


class TestRenderBanner(unittest.TestCase):

    def test_no_banner_when_not_stale(self):
        fake = _FakeSt()
        ss = {}
        ctx = {"x": 1}
        rendered = render_stale_banner_if_needed(
            st=fake, session_state=ss, panel_key="p", context=ctx,
        )
        self.assertFalse(rendered)
        self.assertEqual(len(fake.warnings), 0)

    def test_banner_when_stale(self):
        fake = _FakeSt()
        ss = {}
        mark_panel_refreshed(ss, "p", {"x": 1})
        rendered = render_stale_banner_if_needed(
            st=fake, session_state=ss, panel_key="p", context={"x": 2},
        )
        self.assertTrue(rendered)
        self.assertGreaterEqual(len(fake.warnings), 1)


if __name__ == "__main__":
    unittest.main()
