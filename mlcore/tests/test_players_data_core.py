from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app_core.players_data_core import (
    apply_filters_players,
    champion_vocab,
    fuzzy_match_one,
    load_player_rows,
)


class TestPlayersDataCore(unittest.TestCase):
    def test_apply_filters_players(self) -> None:
        df = pd.DataFrame(
            [
                {"year": 2026, "league": "LCK", "split": "Spring", "playoffs": 1, "player": "A"},
                {"year": 2026, "league": "LPL", "split": "Spring", "playoffs": 0, "player": "B"},
                {"year": 2025, "league": "LCK", "split": "Summer", "playoffs": 1, "player": "C"},
            ]
        )
        filters = SimpleNamespace(year=2026, league="LCK", split="Spring", playoffs=True)
        out = apply_filters_players(df, filters)
        self.assertEqual(int(out.shape[0]), 1)
        self.assertEqual(str(out.iloc[0]["player"]), "A")

    def test_champion_vocab_and_fuzzy(self) -> None:
        df = pd.DataFrame([{"champ": "Ahri"}, {"champ": "Orianna"}, {"champ": "Ahri"}, {"champ": ""}, {"champ": "Wukong"}])
        vocab = champion_vocab(df)
        self.assertEqual(vocab, ["Ahri", "Orianna", "Wukong"])

        match, suggestions = fuzzy_match_one("Oriana", ["Orianna", "Ahri"], cutoff=0.4)
        self.assertEqual(match, "Orianna")
        self.assertTrue(len(suggestions) >= 1)

        alias_match, alias_suggestions = fuzzy_match_one("MonkeyKing", ["Wukong", "Ahri"], cutoff=0.4)
        self.assertEqual(alias_match, "Wukong")
        self.assertEqual(alias_suggestions, [])

    def test_load_player_rows_basic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "players.csv"
            df = pd.DataFrame(
                [
                    {
                        "gameid": "g1",
                        "date": "2026-01-01",
                        "league": "lck",
                        "year": 2026,
                        "split": "Spring",
                        "playoffs": 0,
                        "game": 1,
                        "patch": "14.1",
                        "participantid": 1,
                        "side": "Blue",
                        "position": "Top",
                        "playerid": "pid_top",
                        "playername": "TopA",
                        "teamname": "TeamA",
                        "champion": "Gnar",
                        "result": 1,
                        "kills": 2,
                        "deaths": 1,
                        "assists": 5,
                        "gamelength": 1800,
                        "teamkills": 10,
                        "teamdeaths": 5,
                        "ckpm": 0.7,
                        "total cs": 250,
                        "cspm": 8.3,
                        "damagetochampions": 12000,
                        "visionscore": 20,
                    },
                    {
                        "gameid": "g1",
                        "date": "2026-01-01",
                        "league": "lck",
                        "year": 2026,
                        "split": "Spring",
                        "playoffs": 0,
                        "game": 1,
                        "patch": "14.1",
                        "participantid": 11,
                        "side": "Blue",
                        "position": "Top",
                        "playerid": "bad",
                        "playername": "ShouldDrop",
                        "teamname": "TeamA",
                        "champion": "Gnar",
                        "result": 1,
                        "kills": 0,
                        "deaths": 0,
                        "assists": 0,
                        "gamelength": 1800,
                        "teamkills": 10,
                        "teamdeaths": 5,
                        "ckpm": 0.7,
                        "total cs": 250,
                        "cspm": 8.3,
                        "damagetochampions": 12000,
                        "visionscore": 20,
                    },
                ]
            )
            df.to_csv(csv_path, index=False)

            out = load_player_rows(
                csv_path=str(csv_path),
                canon_league_display_label_fn=lambda x: str(x).upper(),
                norm_pos_ml_fn=lambda x: "TOP" if str(x).lower() == "top" else "",
            )
            self.assertEqual(int(out.shape[0]), 1)
            self.assertEqual(str(out.iloc[0]["league"]), "LCK")
            self.assertEqual(str(out.iloc[0]["team"]), "TeamA")
            self.assertEqual(str(out.iloc[0]["player"]), "TopA")
            self.assertEqual(str(out.iloc[0]["champ"]), "Gnar")
            self.assertEqual(str(out.iloc[0]["pos_ml"]), "TOP")
            self.assertEqual(str(out.iloc[0]["player_key"]), "pid_top")


if __name__ == "__main__":
    unittest.main()
