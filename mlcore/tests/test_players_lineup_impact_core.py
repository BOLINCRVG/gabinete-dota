from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app_core.players_lineup_impact_core import (
    apply_shrink_m,
    best_match,
    latest_lineup_for_team,
    latest_lineup_names,
    load_players_impact,
    parse_pct,
)


def _norm_pos(v: str) -> str:
    s = str(v or "").strip().lower()
    if s in ("top",):
        return "top"
    if s in ("jg", "jng", "jungle"):
        return "jng"
    if s in ("mid",):
        return "mid"
    if s in ("bot", "adc"):
        return "bot"
    if s in ("sup", "support"):
        return "sup"
    return ""


def _norm_pos_ml(v: str) -> str:
    s = str(v or "").strip().lower()
    if s in ("top",):
        return "Top"
    if s in ("jg", "jng", "jungle"):
        return "Jungle"
    if s in ("mid",):
        return "Mid"
    if s in ("bot", "adc"):
        return "Bot"
    if s in ("sup", "support"):
        return "Support"
    return ""


class TestPlayersLineupImpactCore(unittest.TestCase):
    def test_parse_pct(self) -> None:
        self.assertAlmostEqual(parse_pct("51.5%"), 0.515, places=6)
        self.assertTrue(pd.isna(parse_pct("")))

    def test_apply_shrink_m(self) -> None:
        df = pd.DataFrame([{"player": "A", "gp": 10, "score": 2.0}])
        out = apply_shrink_m(df, 10)
        self.assertAlmostEqual(float(out.iloc[0]["score"]), 1.0, places=6)

    def test_best_match(self) -> None:
        self.assertEqual(best_match("faker", ["Faker", "Chovy"]), "Faker")
        self.assertEqual(best_match("Fakr", ["Faker", "Chovy"], cutoff=0.4), "Faker")

    def test_latest_lineup_helpers(self) -> None:
        df = pd.DataFrame(
            [
                {"gameid": "g1", "date": "2026-01-01", "team": "TA", "league": "LCK", "year": 2026, "split": "Spring", "playoffs": 0, "pos": "Top", "player": "T1", "pos_ml": "Top", "player_key": "pid_t1"},
                {"gameid": "g1", "date": "2026-01-01", "team": "TA", "league": "LCK", "year": 2026, "split": "Spring", "playoffs": 0, "pos": "Jungle", "player": "J1", "pos_ml": "Jungle", "player_key": "pid_j1"},
                {"gameid": "g2", "date": "2026-01-05", "team": "TA", "league": "LCK", "year": 2026, "split": "Spring", "playoffs": 0, "pos": "Top", "player": "T2", "pos_ml": "Top", "player_key": "pid_t2"},
                {"gameid": "g2", "date": "2026-01-05", "team": "TA", "league": "LCK", "year": 2026, "split": "Spring", "playoffs": 0, "pos": "Jungle", "player": "J2", "pos_ml": "Jungle", "player_key": "pid_j2"},
            ]
        )
        df["date"] = pd.to_datetime(df["date"])

        names = latest_lineup_names(df_in=df, team_name="TA", norm_pos_fn=_norm_pos)
        self.assertEqual(names, ["T2", "J2"])

        # Regressão: autofill "não pegava nenhum" quando a grafia do time diverge
        # entre os CSVs. Match normalizado deve resolver (TA == "t.a." == "T A").
        self.assertEqual(latest_lineup_names(df_in=df, team_name="t.a.", norm_pos_fn=_norm_pos), ["T2", "J2"])
        self.assertEqual(latest_lineup_names(df_in=df, team_name="T A", norm_pos_fn=_norm_pos), ["T2", "J2"])
        # Time inexistente continua retornando vazio (sem falso-positivo).
        self.assertEqual(latest_lineup_names(df_in=df, team_name="ZZZ", norm_pos_fn=_norm_pos), [])

        filters = SimpleNamespace(year=2026, split="Spring", playoffs=False)
        out = latest_lineup_for_team(
            df_player_rows=df,
            team="TA",
            filters=filters,
            league_fixed="LCK",
            pos_col="pos",
            player_col="player",
            pos_list=["top", "jng"],
            norm_pos_fn=_norm_pos,
            norm_pos_ml_fn=_norm_pos_ml,
        )
        self.assertEqual(out.get("top"), "T2")
        self.assertEqual(out.get("jng"), "J2")

        out_ml = latest_lineup_for_team(
            df_player_rows=df,
            team="TA",
            filters=filters,
            league_fixed="LCK",
            pos_col="pos_ml",
            player_col="player_key",
            pos_list=["Top", "Jungle"],
            norm_pos_fn=_norm_pos,
            norm_pos_ml_fn=_norm_pos_ml,
        )
        self.assertEqual(out_ml.get("Top"), "pid_t2")
        self.assertEqual(out_ml.get("Jungle"), "pid_j2")

    def test_load_players_impact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "impact.csv"
            pd.DataFrame(
                [
                    {"Player": "A", "Team": "TA", "Pos": "Top", "GP": 10, "GD10": 100, "XPD10": 50, "CSD10": 3, "DMG%": "25%", "DTH%": "18%"},
                    {"Player": "B", "Team": "TB", "Pos": "Top", "GP": 8, "GD10": -20, "XPD10": -30, "CSD10": -1, "DMG%": "19%", "DTH%": "24%"},
                ]
            ).to_csv(csv_path, index=False)

            out = load_players_impact(csv_path=str(csv_path), norm_pos_fn=_norm_pos, parse_pct_fn=parse_pct)
            self.assertFalse(out.empty)
            self.assertEqual(set(["team", "pos", "player", "gp", "score"]).issubset(set(out.columns)), True)


if __name__ == "__main__":
    unittest.main()

