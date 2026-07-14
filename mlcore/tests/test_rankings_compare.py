"""Tests da comparação de rating entre ligas (aba Rankings)."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.rankings_tab_core import _rating_row, _render_rating_comparison

_DF = pd.DataFrame([
    {"rank": 1, "team": "Karmine Corp", "league": "LEC", "tier": "T1", "total_strength": 120.5},
    {"rank": 691, "team": "Magaza Esports", "league": "EBL", "tier": "ERL", "total_strength": -103.3},
])


class _Col:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeST:
    def __init__(self):
        self.md = []
        self.caps = []
        self.infos = []

    def columns(self, n):
        k = n if isinstance(n, int) else len(n)
        return [_Col() for _ in range(k)]

    def markdown(self, s, **k):
        self.md.append(str(s))

    def caption(self, s, **k):
        self.caps.append(str(s))

    def info(self, s, **k):
        self.infos.append(str(s))


def _fmt(v, n):
    return f"{float(v):.{int(n)}f}"


class TestRatingRow(unittest.TestCase):
    def test_found(self):
        r = _rating_row(_DF, "Magaza Esports")
        self.assertIsNotNone(r)
        self.assertEqual(int(r["rank"]), 691)

    def test_missing(self):
        self.assertIsNone(_rating_row(_DF, "ZZZ Inexistente"))

    def test_empty_df(self):
        self.assertIsNone(_rating_row(pd.DataFrame(), "x"))


class TestComparison(unittest.TestCase):
    def test_verdict_picks_stronger_and_diff(self):
        st = _FakeST()
        _render_rating_comparison(st, _DF, "Magaza Esports", "Karmine Corp", format_num_fn=_fmt)
        self.assertEqual(len(st.caps), 1)
        v = st.caps[0]
        self.assertIn("Karmine Corp", v)         # mais forte
        self.assertIn("223.8", v)                # 120.5 - (-103.3)
        # dois cards renderizados (um por time)
        self.assertEqual(len(st.md), 2)
        self.assertTrue(any("Magaza" in m for m in st.md))

    def test_missing_team_shows_info_no_verdict(self):
        st = _FakeST()
        _render_rating_comparison(st, _DF, "ZZZ", "Karmine Corp", format_num_fn=_fmt)
        self.assertEqual(st.caps, [])
        self.assertTrue(st.infos)

    def test_tie(self):
        df = pd.DataFrame([
            {"rank": 1, "team": "A", "league": "LCK", "tier": "T1", "total_strength": 50.0},
            {"rank": 2, "team": "B", "league": "LPL", "tier": "T1", "total_strength": 50.0},
        ])
        st = _FakeST()
        _render_rating_comparison(st, df, "A", "B", format_num_fn=_fmt)
        self.assertIn("Empate", st.caps[0])


if __name__ == "__main__":
    unittest.main()
