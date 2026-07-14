from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app_core.players_elo_core import (
    compute_player_elo_global,
    compute_team_elo_season,
    discover_player_csv_paths,
    elo_expected,
    load_player_rows_multi,
    player_key_from_row,
)


class TestPlayersEloCore(unittest.TestCase):
    def test_elo_expected(self) -> None:
        self.assertAlmostEqual(elo_expected(1500, 1500), 0.5, places=6)
        self.assertGreater(elo_expected(1600, 1500), 0.5)

    def test_discover_player_csv_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            daily = root / "CSV_DIARIO"
            hist = root / "CSV_HISTORICO"
            daily.mkdir(parents=True, exist_ok=True)
            hist.mkdir(parents=True, exist_ok=True)
            p_now = daily / "2026_test.csv"
            p_2024 = hist / "2024_test.csv"
            p_2023 = hist / "2023_test.csv"
            p_now.write_text("a,b\n1,2\n", encoding="utf-8")
            p_2024.write_text("a,b\n1,2\n", encoding="utf-8")
            p_2023.write_text("a,b\n1,2\n", encoding="utf-8")

            out = discover_player_csv_paths(str(p_now))
            self.assertIn(str(p_now), out)
            self.assertIn(str(p_2024), out)
            self.assertNotIn(str(p_2023), out)

    def test_load_player_rows_multi(self) -> None:
        rows = {
            ("a.csv", 1, 1): pd.DataFrame([{"year": 2026, "x": 1}]),
            ("b.csv", 1, 1): pd.DataFrame([{"year": 2025, "x": 2}]),
        }

        def _loader(sig):
            return rows.get(sig, pd.DataFrame())

        out = load_player_rows_multi(sig_list=tuple(rows.keys()), load_player_rows_fn=_loader)
        self.assertEqual(int(out.shape[0]), 2)
        self.assertTrue("year" in out.columns)

    def test_player_key_from_row(self) -> None:
        r1 = SimpleNamespace(playerid="pid1", playername="Name")
        r2 = SimpleNamespace(playerid="", playername="Name")
        self.assertEqual(player_key_from_row(r1), "pid1")
        self.assertEqual(player_key_from_row(r2), "Name")

    def test_compute_player_elo_global_and_team_elo(self) -> None:
        df_players = pd.DataFrame(
            [
                {"gameid": "g1", "date": "2026-01-01", "teamname": "A", "result": 1, "playername": "A1", "playerid": "a1", "year": 2026},
                {"gameid": "g1", "date": "2026-01-01", "teamname": "B", "result": 0, "playername": "B1", "playerid": "b1", "year": 2026},
                {"gameid": "g2", "date": "2026-01-02", "teamname": "A", "result": 0, "playername": "A1", "playerid": "a1", "year": 2026},
                {"gameid": "g2", "date": "2026-01-02", "teamname": "B", "result": 1, "playername": "B1", "playerid": "b1", "year": 2026},
            ]
        )

        out_p = compute_player_elo_global(
            sig_list=(("x", 1, 1),),
            load_player_rows_multi_fn=lambda _sig: df_players,
            half_life_days=180.0,
            k_player=24.0,
            shrink_m=5.0,
            base_rating=1500.0,
            scale=400.0,
        )
        self.assertTrue("rating" in out_p)
        self.assertTrue("a1" in out_p["rating"])
        self.assertTrue("b1" in out_p["rating"])

        team_games = pd.DataFrame(
            [
                {"gameid": "g1", "date": "2026-01-01", "team": "A", "opponent": "B", "win": 1, "year": 2026},
                {"gameid": "g1", "date": "2026-01-01", "team": "B", "opponent": "A", "win": 0, "year": 2026},
                {"gameid": "g2", "date": "2026-01-02", "team": "A", "opponent": "B", "win": 0, "year": 2026},
                {"gameid": "g2", "date": "2026-01-02", "team": "B", "opponent": "A", "win": 1, "year": 2026},
            ]
        )
        filters = SimpleNamespace(year=None, league=None, split=None, playoffs=None)
        out_t = compute_team_elo_season(
            team_games=team_games,
            filters=filters,
            apply_filters_fn=lambda df, _f: df,
            half_life_days=180.0,
            k_team=24.0,
            shrink_m=5.0,
            base_rating=1500.0,
            scale=400.0,
        )
        self.assertTrue("rating" in out_t)
        self.assertTrue("A" in out_t["rating"])
        self.assertTrue("B" in out_t["rating"])


if __name__ == "__main__":
    unittest.main()

