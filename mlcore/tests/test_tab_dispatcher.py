"""Tests for app_core/tab_dispatcher.py — tab label constants and dispatch helpers."""
import unittest

from app_core.tab_dispatcher import (
    ALL_TABS,
    SECONDARY_TAB_ORDER,
    TAB_CAMPEOES,
    TAB_CONS,
    TAB_IDENTIDADE,
    TAB_JOGOS,
    TAB_LINE_ANALYSIS,
    TAB_PARAMS,
    TAB_PLAYERS,
    TAB_RANKINGS,
    TAB_SERIES,
    TAB_STATSCAMPS,
    TAB_VISAO,
    dispatch_secondary_tabs,
)


class TestTabConstants(unittest.TestCase):

    def test_all_constants_are_strings(self):
        tabs = [TAB_PARAMS, TAB_VISAO, TAB_CONS, TAB_RANKINGS, TAB_JOGOS,
                TAB_SERIES, TAB_STATSCAMPS, TAB_CAMPEOES, TAB_PLAYERS,
                TAB_IDENTIDADE, TAB_LINE_ANALYSIS]
        for t in tabs:
            self.assertIsInstance(t, str, msg=f"{t!r} is not a string")

    def test_all_constants_non_empty(self):
        tabs = [TAB_PARAMS, TAB_VISAO, TAB_CONS, TAB_RANKINGS, TAB_JOGOS,
                TAB_SERIES, TAB_STATSCAMPS, TAB_CAMPEOES, TAB_PLAYERS,
                TAB_IDENTIDADE, TAB_LINE_ANALYSIS]
        for t in tabs:
            self.assertTrue(t.strip(), msg=f"Empty tab label: {t!r}")

    def test_no_duplicates_in_secondary_tab_order(self):
        self.assertEqual(len(SECONDARY_TAB_ORDER), len(set(SECONDARY_TAB_ORDER)))

    def test_no_duplicates_in_all_tabs(self):
        self.assertEqual(len(ALL_TABS), len(set(ALL_TABS)))

    def test_secondary_order_is_subset_of_all_tabs(self):
        for tab in SECONDARY_TAB_ORDER:
            self.assertIn(tab, ALL_TABS)

    def test_primary_tabs_in_all_tabs(self):
        for tab in (TAB_PARAMS, TAB_VISAO, TAB_CONS):
            self.assertIn(tab, ALL_TABS)

    def test_secondary_tab_order_is_tuple(self):
        self.assertIsInstance(SECONDARY_TAB_ORDER, tuple)

    def test_all_tabs_is_tuple(self):
        self.assertIsInstance(ALL_TABS, tuple)


class TestDispatchSecondaryTabs(unittest.TestCase):

    class _FakeTab:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *_):
            pass

    class _FakeSt:
        pass

    def test_calls_renderer_for_present_tabs(self):
        called = []
        tab_a = self._FakeTab()
        tab_b = self._FakeTab()

        dispatch_secondary_tabs(
            self._FakeSt(),
            tab_refs={"TabA": tab_a, "TabB": tab_b},
            renderers={"TabA": lambda: called.append("A"), "TabB": lambda: called.append("B")},
            order=("TabA", "TabB"),
        )
        self.assertIn("A", called)
        self.assertIn("B", called)

    def test_skips_missing_tab_ref(self):
        called = []
        tab_a = self._FakeTab()

        dispatch_secondary_tabs(
            self._FakeSt(),
            tab_refs={"TabA": tab_a},
            renderers={"TabA": lambda: called.append("A"), "TabB": lambda: called.append("B")},
            order=("TabA", "TabB"),
        )
        self.assertIn("A", called)
        self.assertNotIn("B", called)

    def test_skips_missing_renderer(self):
        called = []
        tab_a = self._FakeTab()
        tab_b = self._FakeTab()

        dispatch_secondary_tabs(
            self._FakeSt(),
            tab_refs={"TabA": tab_a, "TabB": tab_b},
            renderers={"TabA": lambda: called.append("A")},
            order=("TabA", "TabB"),
        )
        self.assertIn("A", called)
        self.assertNotIn("B", called)

    def test_empty_order_nothing_called(self):
        called = []
        tab_a = self._FakeTab()

        dispatch_secondary_tabs(
            self._FakeSt(),
            tab_refs={"TabA": tab_a},
            renderers={"TabA": lambda: called.append("A")},
            order=(),
        )
        self.assertEqual(called, [])

    def test_tab_context_manager_entered(self):
        tab_a = self._FakeTab()

        dispatch_secondary_tabs(
            self._FakeSt(),
            tab_refs={"TabA": tab_a},
            renderers={"TabA": lambda: None},
            order=("TabA",),
        )
        self.assertTrue(tab_a.entered)


if __name__ == "__main__":
    unittest.main()
