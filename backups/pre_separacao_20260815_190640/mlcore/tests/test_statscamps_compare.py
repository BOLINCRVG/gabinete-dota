"""Tests for the v3 StatsCamps comparison helpers (synthetic data)."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.statscamps_tab_core import _sc_team_row, _SC_COMPARE_METRICS


def _raw_df():
    return pd.DataFrame([
        {"Team": "T1", "# Games": 20, "Win Rate": 65.0, "Kills": 15.0,
         "Deaths": 11.0, "Towers": 8.0, "C. Kills": 27.0, "C. Towers": 11.5,
         "C. Dragons": 4.5, "C. Nashors": 1.2, "FB%": 60.0, "FT%": 55.0},
        {"Team": "GenG", "# Games": 18, "Win Rate": 58.0, "Kills": 13.0,
         "Deaths": 12.0, "Towers": 7.5, "C. Kills": 25.0, "C. Towers": 11.0,
         "C. Dragons": 4.2, "C. Nashors": 1.0, "FB%": 50.0, "FT%": 52.0},
    ])


class TestScTeamRow(unittest.TestCase):

    def test_finds_exact(self):
        row = _sc_team_row(_raw_df(), "T1")
        self.assertIsNotNone(row)
        self.assertEqual(row["Win Rate"], 65.0)

    def test_mojibake_tolerant(self):
        df = pd.DataFrame([{"Team": "L�S", "# Games": 10, "Win Rate": 50.0}])
        row = _sc_team_row(df, "LØS")
        self.assertIsNotNone(row)

    def test_missing_team_none(self):
        self.assertIsNone(_sc_team_row(_raw_df(), "FNC"))

    def test_empty_df_none(self):
        self.assertIsNone(_sc_team_row(pd.DataFrame(), "T1"))

    def test_case_insensitive(self):
        self.assertIsNotNone(_sc_team_row(_raw_df(), "t1"))


class TestCompareMetricsConfig(unittest.TestCase):

    def test_all_metrics_have_three_fields(self):
        for entry in _SC_COMPARE_METRICS:
            self.assertEqual(len(entry), 3)
            label, col, higher = entry
            self.assertIsInstance(label, str)
            self.assertIsInstance(col, str)
            self.assertIsInstance(higher, bool)

    def test_deaths_is_lower_better(self):
        deaths = next((e for e in _SC_COMPARE_METRICS if e[1] == "Deaths"), None)
        self.assertIsNotNone(deaths)
        self.assertFalse(deaths[2])  # lower is better for deaths

    def test_winrate_higher_better(self):
        wr = next((e for e in _SC_COMPARE_METRICS if e[1] == "Win Rate"), None)
        self.assertIsNotNone(wr)
        self.assertTrue(wr[2])


class TestRenderCompareCards(unittest.TestCase):
    """Render path with a fake st — verifies no crash + returns True/False right."""

    class _FakeCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _FakeSt:
        def __init__(self): self.calls = []
        def markdown(self, *a, **k): self.calls.append("markdown")
        def container(self, *a, **k): return TestRenderCompareCards._FakeCtx()

    def test_renders_when_both_present(self):
        from app_core.statscamps_tab_core import render_statscamps_compare_cards
        st = self._FakeSt()
        ok = render_statscamps_compare_cards(
            st=st, df_tab_raw=_raw_df(), team_a="T1", team_b="GenG",
            min_to_mmss_fn=lambda x: str(x), format_num_fn=lambda v, d: f"{v:.{d}f}",
        )
        self.assertTrue(ok)
        self.assertGreater(len(st.calls), 0)

    def test_skips_when_team_missing(self):
        from app_core.statscamps_tab_core import render_statscamps_compare_cards
        st = self._FakeSt()
        ok = render_statscamps_compare_cards(
            st=st, df_tab_raw=_raw_df(), team_a="T1", team_b="FNC",
            min_to_mmss_fn=lambda x: str(x), format_num_fn=lambda v, d: f"{v:.{d}f}",
        )
        self.assertFalse(ok)

    def test_skips_when_blank_teams(self):
        from app_core.statscamps_tab_core import render_statscamps_compare_cards
        st = self._FakeSt()
        ok = render_statscamps_compare_cards(
            st=st, df_tab_raw=_raw_df(), team_a="", team_b="",
            min_to_mmss_fn=lambda x: str(x), format_num_fn=lambda v, d: f"{v:.{d}f}",
        )
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
