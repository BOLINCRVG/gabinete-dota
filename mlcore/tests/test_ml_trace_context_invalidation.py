"""Regression test for the ML trace context-invalidation fix.

Bug (2026-05-30): toggling "Janela (Histórico)" between Ano todo / Últimos 10 /
Últimos 5 in the Visão Geral tab produced different odds for the same window
on revisit. Root cause: the post-layer ML step read `_ml_trace_last` from
session_state without checking it matched the current scope's context, so a
trace computed for one window leaked into another window's render.

Fix: at the top of the Visão Geral tab render, compute a context fingerprint
from (matchup, filters, league mode, window, modes). If the stored
`_ml_trace_last_ctx_fp` differs from the current one, drop the stale trace.

This test simulates the fingerprint mismatch logic in isolation, since the
full render path requires Streamlit + CSVs.
"""
from __future__ import annotations

import unittest
from typing import Any, MutableMapping


def _compute_ctx_fingerprint(
    *,
    season_label: str,
    teamA: str,
    teamB: str,
    bo: int,
    year_opt: str,
    split_opt: str,
    playoffs_opt: str,
    league_mode: str,
    fixed_league: str,
    max_games: int | None,
    map_mode_ui: str,
    totals_mode: str,
    ml_mode: str,
    weight_mode: str,
    half_life_days: int | None,
) -> tuple:
    """Mirror the fingerprint computed in visao_tab_core."""
    return (
        str(season_label or ""), str(teamA or ""), str(teamB or ""), int(bo or 1),
        str(year_opt or ""), str(split_opt or ""), str(playoffs_opt or ""),
        str(league_mode or ""), str(fixed_league or ""),
        int(max_games) if max_games else 0,
        str(map_mode_ui or ""), str(totals_mode or ""), str(ml_mode or ""),
        str(weight_mode or ""), int(half_life_days) if half_life_days is not None else -1,
    )


def _maybe_invalidate_trace(session_state: MutableMapping[str, Any], current_fp: tuple) -> bool:
    """Replicate the invalidation logic from visao_tab_core. Returns True if cleared."""
    if session_state.get("_ml_trace_last_ctx_fp") != current_fp:
        session_state.pop("_ml_trace_last", None)
        session_state["_ml_trace_last_ctx_fp"] = current_fp
        return True
    return False


_BASE_CTX = dict(
    season_label="2026 Spring",
    teamA="Gen.G", teamB="T1", bo=3,
    year_opt="2026", split_opt="Spring", playoffs_opt="0",
    league_mode="auto", fixed_league="",
    max_games=None,
    map_mode_ui="avg", totals_mode="map", ml_mode="map",
    weight_mode="Recencia (half-life)", half_life_days=180,
)


class TestContextInvalidation(unittest.TestCase):

    def test_first_render_clears_and_sets(self):
        """Empty session: invalidate sets fingerprint, no trace to clear."""
        ss: dict = {}
        fp = _compute_ctx_fingerprint(**_BASE_CTX)
        cleared = _maybe_invalidate_trace(ss, fp)
        self.assertTrue(cleared)
        self.assertEqual(ss["_ml_trace_last_ctx_fp"], fp)
        self.assertNotIn("_ml_trace_last", ss)

    def test_same_context_keeps_trace(self):
        """Same scope between renders: do not clear the trace."""
        ss: dict = {}
        fp = _compute_ctx_fingerprint(**_BASE_CTX)
        _maybe_invalidate_trace(ss, fp)
        # Simulate analysis having stored a trace
        ss["_ml_trace_last"] = {"some": "trace_data"}
        # Second render, same context
        cleared = _maybe_invalidate_trace(ss, fp)
        self.assertFalse(cleared)
        self.assertEqual(ss["_ml_trace_last"], {"some": "trace_data"})

    def test_window_change_clears_trace(self):
        """Switching from 'Ano todo' (max_games=None) to 'Últimos 5' clears trace."""
        ss: dict = {}
        fp_all = _compute_ctx_fingerprint(**{**_BASE_CTX, "max_games": None})
        _maybe_invalidate_trace(ss, fp_all)
        ss["_ml_trace_last"] = {"scope": "all"}
        # User clicks "Últimos 5"
        fp_last5 = _compute_ctx_fingerprint(**{**_BASE_CTX, "max_games": 5})
        cleared = _maybe_invalidate_trace(ss, fp_last5)
        self.assertTrue(cleared)
        self.assertNotIn("_ml_trace_last", ss)
        self.assertEqual(ss["_ml_trace_last_ctx_fp"], fp_last5)

    def test_round_trip_consistency(self):
        """Bug regression: Ano todo -> Últimos 5 -> Ano todo, the second visit
        to Ano todo should clear the stale trace from Últimos 5."""
        ss: dict = {}
        fp_all = _compute_ctx_fingerprint(**{**_BASE_CTX, "max_games": None})
        fp_last5 = _compute_ctx_fingerprint(**{**_BASE_CTX, "max_games": 5})

        # Visit 1: Ano todo
        _maybe_invalidate_trace(ss, fp_all)
        ss["_ml_trace_last"] = {"computed_under": "Ano todo"}

        # Visit 2: Últimos 5 — must clear the Ano todo trace
        cleared = _maybe_invalidate_trace(ss, fp_last5)
        self.assertTrue(cleared)
        ss["_ml_trace_last"] = {"computed_under": "Últimos 5"}

        # Visit 3: back to Ano todo — must clear the Últimos 5 trace
        cleared = _maybe_invalidate_trace(ss, fp_all)
        self.assertTrue(cleared)
        # Trace from Últimos 5 has been dropped; downstream layers will compute fresh
        self.assertNotIn("_ml_trace_last", ss)

    def test_teamA_change_clears_trace(self):
        ss: dict = {}
        fp_a = _compute_ctx_fingerprint(**_BASE_CTX)
        _maybe_invalidate_trace(ss, fp_a)
        ss["_ml_trace_last"] = {"trace": "for_A"}
        # Switch teamA
        fp_b = _compute_ctx_fingerprint(**{**_BASE_CTX, "teamA": "T1", "teamB": "DK"})
        cleared = _maybe_invalidate_trace(ss, fp_b)
        self.assertTrue(cleared)
        self.assertNotIn("_ml_trace_last", ss)

    def test_split_change_clears_trace(self):
        ss: dict = {}
        fp_spring = _compute_ctx_fingerprint(**_BASE_CTX)
        _maybe_invalidate_trace(ss, fp_spring)
        ss["_ml_trace_last"] = {"trace": "spring"}
        fp_summer = _compute_ctx_fingerprint(**{**_BASE_CTX, "split_opt": "Summer"})
        cleared = _maybe_invalidate_trace(ss, fp_summer)
        self.assertTrue(cleared)


if __name__ == "__main__":
    unittest.main()
