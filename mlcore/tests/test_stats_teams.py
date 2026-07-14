import unittest

import pandas as pd

from mlcore.stats_teams import build_team_games


class TestStatsTeams(unittest.TestCase):
    def test_build_team_games_derives_total_elementaldrakes(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "gameid": "g1",
                    "date": "2026-01-01",
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": 0,
                    "league": "LCK",
                    "game": 1,
                    "participantid": 100,
                    "side": "Blue",
                    "teamname": "T1",
                    "gamelength": 1800,
                    "result": 1,
                    "kills": 16,
                    "deaths": 8,
                    "assists": 30,
                    "totalgold": 62000,
                    "towers": 9,
                    "dragons": 4,
                    "barons": 2,
                    "inhibitors": 2,
                    "elementaldrakes": 4,
                },
                {
                    "gameid": "g1",
                    "date": "2026-01-01",
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": 0,
                    "league": "LCK",
                    "game": 1,
                    "participantid": 200,
                    "side": "Red",
                    "teamname": "GEN",
                    "gamelength": 1800,
                    "result": 0,
                    "kills": 8,
                    "deaths": 16,
                    "assists": 14,
                    "totalgold": 54000,
                    "towers": 3,
                    "dragons": 1,
                    "barons": 0,
                    "inhibitors": 0,
                    "elementaldrakes": 1,
                },
            ]
        )

        out = build_team_games(rows)
        self.assertIn("elementaldrakes_for", out.columns)
        self.assertIn("elementaldrakes_against", out.columns)
        self.assertIn("total_elementaldrakes", out.columns)
        self.assertEqual(float(out.iloc[0]["total_elementaldrakes"]), 5.0)


if __name__ == "__main__":
    unittest.main()
