import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app_core.ml_league_inference_core import (
    infer_match_league_from_csv,
    infer_team_league_from_csv,
)
from app_core.ml_league_runtime_core import MAJOR_LEAGUES, norm_league_key


class TestMlLeagueInferenceCore(unittest.TestCase):
    def test_infer_team_league_prefers_major_with_sufficient_volume(self):
        rows = []
        for i in range(8):
            rows.append({"teamname": "Alpha", "league": "LCK", "year": 2026, "participantid": 100, "gameid": i})
        for i in range(10, 16):
            rows.append({"teamname": "Alpha", "league": "Minor Cup", "year": 2026, "participantid": 100, "gameid": i})
        df = pd.DataFrame(rows)
        with TemporaryDirectory() as td:
            csv_path = Path(td) / "sample.csv"
            df.to_csv(csv_path, index=False)
            out = infer_team_league_from_csv(
                (str(csv_path), csv_path.stat().st_size, int(csv_path.stat().st_mtime)),
                "Alpha",
                2026,
                norm_league_key_fn=norm_league_key,
                major_leagues=MAJOR_LEAGUES,
            )
        self.assertEqual(out, "LCK")

    def test_infer_team_league_falls_back_to_top_frequency(self):
        rows = []
        for i in range(6):
            rows.append({"teamname": "Beta", "league": "Minor Cup", "year": 2026, "participantid": 100, "gameid": i})
        for i in range(10, 13):
            rows.append({"teamname": "Beta", "league": "LCK", "year": 2026, "participantid": 100, "gameid": i})
        df = pd.DataFrame(rows)
        with TemporaryDirectory() as td:
            csv_path = Path(td) / "sample.csv"
            df.to_csv(csv_path, index=False)
            out = infer_team_league_from_csv(
                (str(csv_path), csv_path.stat().st_size, int(csv_path.stat().st_mtime)),
                "Beta",
                2026,
                norm_league_key_fn=norm_league_key,
                major_leagues=MAJOR_LEAGUES,
            )
        self.assertEqual(out, "Minor Cup")

    def test_infer_match_league_uses_h2h_when_available(self):
        team_games = pd.DataFrame(
            [
                {"team": "A", "opponent": "B", "league": "CBLOL", "year": 2026},
                {"team": "B", "opponent": "A", "league": "CBLOL", "year": 2026},
                {"team": "A", "opponent": "B", "league": "LCK", "year": 2026},
            ]
        )

        out = infer_match_league_from_csv(
            ("dummy.csv", 0, 0),
            "A",
            "B",
            2026,
            norm_league_key_fn=norm_league_key,
            major_leagues=MAJOR_LEAGUES,
            load_team_games_fn=lambda _sig: team_games,
            infer_team_league_fn=lambda *_args: "",
        )
        self.assertEqual(out, "CBLOL")

    def test_infer_match_league_falls_back_to_team_inference(self):
        out = infer_match_league_from_csv(
            ("dummy.csv", 0, 0),
            "A",
            "B",
            2026,
            norm_league_key_fn=norm_league_key,
            major_leagues=MAJOR_LEAGUES,
            load_team_games_fn=lambda _sig: pd.DataFrame(),
            infer_team_league_fn=lambda _sig, team, _year: "LCK" if team in {"A", "B"} else "",
        )
        self.assertEqual(out, "LCK")

    def test_infer_match_league_major_tie_break_is_stable(self):
        team_games = pd.DataFrame(
            [
                {"team": "A", "opponent": "B", "league": "Minor Cup", "year": 2026},
                {"team": "B", "opponent": "A", "league": "LCK", "year": 2026},
            ]
        )
        out = infer_match_league_from_csv(
            ("dummy.csv", 0, 0),
            "A",
            "B",
            2026,
            norm_league_key_fn=norm_league_key,
            major_leagues=MAJOR_LEAGUES,
            load_team_games_fn=lambda _sig: team_games,
            infer_team_league_fn=lambda *_args: "",
        )
        self.assertEqual(out, "LCK")

    def test_infer_team_league_uses_aliases_and_latest_year_fallback(self):
        rows = [
            {"teamname": "Witchcraft", "league": "EM", "year": 2026, "participantid": 100, "gameid": 1},
            {"teamname": "PSG Talon", "league": "LCP", "year": 2025, "participantid": 100, "gameid": 2},
        ]
        df = pd.DataFrame(rows)
        with TemporaryDirectory() as td:
            csv_path = Path(td) / "sample.csv"
            df.to_csv(csv_path, index=False)
            sig = (str(csv_path), csv_path.stat().st_size, int(csv_path.stat().st_mtime))
            out_alias = infer_team_league_from_csv(
                sig,
                "Los Ratones",
                2026,
                norm_league_key_fn=norm_league_key,
                major_leagues=MAJOR_LEAGUES,
                team_aliases=["Witchcraft"],
                fallback_any_year=True,
            )
            out_fallback = infer_team_league_from_csv(
                sig,
                "PSG Talon",
                2026,
                norm_league_key_fn=norm_league_key,
                major_leagues=MAJOR_LEAGUES,
                fallback_any_year=True,
            )
        self.assertEqual(out_alias, "EM")
        self.assertEqual(out_fallback, "LCP")

    def test_infer_match_league_uses_alias_candidates(self):
        team_games = pd.DataFrame(
            [
                {"team": "Witchcraft", "opponent": "Partizan Sangal", "league": "EM", "year": 2026},
                {"team": "Partizan Sangal", "opponent": "Witchcraft", "league": "EM", "year": 2026},
            ]
        )
        out = infer_match_league_from_csv(
            ("dummy.csv", 0, 0),
            "Los Ratones",
            "Partizan Sangal",
            2026,
            norm_league_key_fn=norm_league_key,
            major_leagues=MAJOR_LEAGUES,
            load_team_games_fn=lambda _sig: team_games,
            infer_team_league_fn=lambda *_args: "",
            team_aliases_a=["Witchcraft"],
            team_aliases_b=[],
        )
        self.assertEqual(out, "EM")


if __name__ == "__main__":
    unittest.main()
