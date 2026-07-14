from __future__ import annotations

from dataclasses import dataclass
import unittest

import pandas as pd

from app_core.resumo_runtime_core import (
    SC_METRIC_COL,
    build_market_trace_rows,
    build_statscamps_scope_frame,
    compute_statscamps_pct,
    ensure_metric_lines,
    format_statscamps_badge,
    infer_matchup_league,
)


@dataclass
class DummyFilters:
    year: int | None = None
    league: str | None = None
    split: str | None = None
    playoffs: bool | None = None


def _apply_filters(df: pd.DataFrame, filters: DummyFilters) -> pd.DataFrame:
    out = df.copy()
    if filters.year is not None and "year" in out.columns:
        out = out[out["year"] == filters.year]
    if filters.league is not None and "league" in out.columns:
        out = out[out["league"].astype(str) == str(filters.league)]
    if filters.split is not None and "split" in out.columns:
        out = out[out["split"].astype(str) == str(filters.split)]
    if filters.playoffs is not None and "playoffs" in out.columns:
        out = out[out["playoffs"] == filters.playoffs]
    return out


class ResumoRuntimeCoreTest(unittest.TestCase):
    def test_build_statscamps_scope_frame_uses_explicit_recorte(self):
        df = pd.DataFrame(
            [
                {"year": 2026, "league": "LCK", "split": "Spring", "playoffs": False, "map_number": 2, "team": "T1", "gameid": "g1", "total_kills": 27},
                {"year": 2026, "league": "LCK", "split": "Spring", "playoffs": False, "map_number": 2, "team": "GEN", "gameid": "g1", "total_kills": 25},
                {"year": 2026, "league": "LCK", "split": "Spring", "playoffs": False, "map_number": 1, "team": "T1", "gameid": "g2", "total_kills": 20},
            ]
        )
        scoped = build_statscamps_scope_frame(
            enabled=True,
            session_state={"sc_year": "2026", "sc_league": "LCK", "sc_split": "Spring", "sc_map": "Map 2"},
            team_games=df,
            filters_cls=DummyFilters,
            active_filters=DummyFilters(),
            apply_filters_fn=_apply_filters,
            ml_info=None,
            map_mode_ui=None,
            team_a="T1",
            team_b="GEN",
        )
        self.assertIsNotNone(scoped)
        self.assertEqual(int(scoped.shape[0]), 1)
        self.assertEqual(str(scoped.iloc[0]["team"]), "GEN")

    def test_build_statscamps_scope_frame_fallbacks_to_match_league(self):
        df = pd.DataFrame(
            [
                {"year": 2026, "league": "LPL", "split": "Spring", "playoffs": False, "map_number": 3, "team": "BLG", "gameId": "g9"},
                {"year": 2026, "league": "LPL", "split": "Spring", "playoffs": False, "map_number": 3, "team": "TES", "gameId": "g9"},
            ]
        )
        scoped = build_statscamps_scope_frame(
            enabled=True,
            session_state={},
            team_games=df,
            filters_cls=DummyFilters,
            active_filters=DummyFilters(year=2026, split="Spring"),
            apply_filters_fn=_apply_filters,
            ml_info={"blue": {"league": "LPL"}, "red": {"league": "LPL"}},
            map_mode_ui="map3",
            team_a="BLG",
            team_b="TES",
        )
        self.assertIsNotNone(scoped)
        self.assertEqual(int(scoped.shape[0]), 1)

    def test_compute_statscamps_pct_and_badge(self):
        df = pd.DataFrame({"total_kills": [20, 30, 32]})
        pct = compute_statscamps_pct(
            enabled=True,
            df_sc_ab=df,
            metric_key="kills",
            side_key="over",
            line_val=25.5,
            metric_col_map=SC_METRIC_COL,
        )
        self.assertAlmostEqual(float(pct), 66.6666666, places=3)
        self.assertEqual(format_statscamps_badge(pct, threshold=60.0), "OK 67%")

    def test_ensure_metric_lines_parses_multiple_shapes(self):
        self.assertEqual(ensure_metric_lines("kills", "25.5, 26.5", parse_lines_fn=lambda s: [25.5, 26.5], parse_time_lines_fn=lambda s: []), [25.5, 26.5])
        self.assertEqual(ensure_metric_lines("time", "31:30", parse_lines_fn=lambda s: [], parse_time_lines_fn=lambda s: [31.5]), [31.5])
        self.assertEqual(ensure_metric_lines("kills", [1, "2.5", "x"], parse_lines_fn=lambda s: [], parse_time_lines_fn=lambda s: []), [1.0, 2.5])
        self.assertEqual(ensure_metric_lines("kills", 3.0, parse_lines_fn=lambda s: [], parse_time_lines_fn=lambda s: []), [3.0])

    def test_infer_matchup_league_prefers_dataframe_then_trace(self):
        df = pd.DataFrame({"league": ["LCK", "LCK", "LPL"]})
        self.assertEqual(infer_matchup_league(df_league_year=df, ml_info={"blue": {"league": "LPL"}}), "LCK")
        self.assertEqual(infer_matchup_league(df_league_year=pd.DataFrame(), ml_info={"blue": {"league": "LTA"}, "red": {"league": "LTA"}}), "LTA")

    def test_build_market_trace_rows_serializes_supported_fields(self):
        rows = [{"Mercado": "Vencedor", "Lado": "T1", "Linha": "-", "Odd (Modelo)": 1.8, "Obs Mix": "ok"}]
        out = build_market_trace_rows(rows)
        self.assertEqual(out[0]["mercado"], "Vencedor")
        self.assertEqual(out[0]["odd_model"], 1.8)
        self.assertEqual(out[0]["obs_mix"], "ok")


if __name__ == "__main__":
    unittest.main()
