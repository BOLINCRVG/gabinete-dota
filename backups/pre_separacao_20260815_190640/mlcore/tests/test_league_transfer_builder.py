import unittest

import pandas as pd

from mlcore.league_transfer_builder import build_league_transfer_features_from_team_rows


class TestLeagueTransferBuilder(unittest.TestCase):
    def _rows(self) -> pd.DataFrame:
        rows = [
            {"gameid": "g1", "league": "LCK", "year": 2026, "split": "Spring", "playoffs": 0, "date": "2026-02-01", "patch": "16.1", "participantid": 100, "position": "team", "teamname": "T1", "result": 1, "gamelength": 1800, "kills": 16, "deaths": 8, "assists": 30, "ckpm": 0.8, "totalgold": 62000, "earnedgold": 58000, "towers": 9, "opp_towers": 3, "dragons": 4, "opp_dragons": 1, "barons": 2, "opp_barons": 0, "inhibitors": 2, "opp_inhibitors": 0, "firstblood": 1, "firstdragon": 1, "firstherald": 1, "firstbaron": 1, "firsttower": 1, "firsttothreetowers": 1, "goldat15": 28000, "golddiffat15": 2200, "xpdiffat15": 1200, "csdiffat15": 25},
            {"gameid": "g1", "league": "LCK", "year": 2026, "split": "Spring", "playoffs": 0, "date": "2026-02-01", "patch": "16.1", "participantid": 200, "position": "team", "teamname": "DRX", "result": 0, "gamelength": 1800, "kills": 8, "deaths": 16, "assists": 14, "ckpm": 0.8, "totalgold": 54000, "earnedgold": 50000, "towers": 3, "opp_towers": 9, "dragons": 1, "opp_dragons": 4, "barons": 0, "opp_barons": 2, "inhibitors": 0, "opp_inhibitors": 2, "firstblood": 0, "firstdragon": 0, "firstherald": 0, "firstbaron": 0, "firsttower": 0, "firsttothreetowers": 0, "goldat15": 25000, "golddiffat15": -2200, "xpdiffat15": -1200, "csdiffat15": -25},
            {"gameid": "g2", "league": "PRM", "year": 2026, "split": "Spring", "playoffs": 0, "date": "2026-02-02", "patch": "16.1", "participantid": 100, "position": "team", "teamname": "UOL Sexy Edition", "result": 1, "gamelength": 1860, "kills": 14, "deaths": 9, "assists": 28, "ckpm": 0.74, "totalgold": 60000, "earnedgold": 55500, "towers": 8, "opp_towers": 4, "dragons": 3, "opp_dragons": 1, "barons": 1, "opp_barons": 0, "inhibitors": 1, "opp_inhibitors": 0, "firstblood": 1, "firstdragon": 1, "firstherald": 0, "firstbaron": 1, "firsttower": 1, "firsttothreetowers": 1, "goldat15": 27000, "golddiffat15": 1400, "xpdiffat15": 700, "csdiffat15": 11},
            {"gameid": "g2", "league": "PRM", "year": 2026, "split": "Spring", "playoffs": 0, "date": "2026-02-02", "patch": "16.1", "participantid": 200, "position": "team", "teamname": "Eintracht Spandau", "result": 0, "gamelength": 1860, "kills": 9, "deaths": 14, "assists": 19, "ckpm": 0.74, "totalgold": 54800, "earnedgold": 51200, "towers": 4, "opp_towers": 8, "dragons": 1, "opp_dragons": 3, "barons": 0, "opp_barons": 1, "inhibitors": 0, "opp_inhibitors": 1, "firstblood": 0, "firstdragon": 0, "firstherald": 1, "firstbaron": 0, "firsttower": 0, "firsttothreetowers": 0, "goldat15": 25500, "golddiffat15": -1400, "xpdiffat15": -700, "csdiffat15": -11},
            {"gameid": "g3", "league": "WORLDS", "year": 2026, "split": "Spring", "playoffs": 1, "date": "2026-03-01", "patch": "16.2", "participantid": 100, "position": "team", "teamname": "T1", "result": 1, "gamelength": 1900, "kills": 17, "deaths": 10, "assists": 33, "ckpm": 0.85, "totalgold": 64000, "earnedgold": 59500, "towers": 10, "opp_towers": 4, "dragons": 4, "opp_dragons": 2, "barons": 2, "opp_barons": 0, "inhibitors": 2, "opp_inhibitors": 0, "firstblood": 0, "firstdragon": 1, "firstherald": 1, "firstbaron": 1, "firsttower": 1, "firsttothreetowers": 1, "goldat15": 28200, "golddiffat15": 900, "xpdiffat15": 450, "csdiffat15": 8},
            {"gameid": "g3", "league": "WORLDS", "year": 2026, "split": "Spring", "playoffs": 1, "date": "2026-03-01", "patch": "16.2", "participantid": 200, "position": "team", "teamname": "UOL Sexy Edition", "result": 0, "gamelength": 1900, "kills": 10, "deaths": 17, "assists": 21, "ckpm": 0.85, "totalgold": 56300, "earnedgold": 52400, "towers": 4, "opp_towers": 10, "dragons": 2, "opp_dragons": 4, "barons": 0, "opp_barons": 2, "inhibitors": 0, "opp_inhibitors": 2, "firstblood": 1, "firstdragon": 0, "firstherald": 0, "firstbaron": 0, "firsttower": 0, "firsttothreetowers": 0, "goldat15": 27200, "golddiffat15": -900, "xpdiffat15": -450, "csdiffat15": -8},
        ]
        return pd.DataFrame(rows)

    def _player_rows(self) -> pd.DataFrame:
        rows = []
        lineup_t1 = {"Top": "Zeus", "Jungle": "Oner", "Mid": "Faker", "Bot": "Gumayusi", "Support": "Keria"}
        lineup_drx = {"Top": "Rich", "Jungle": "Sponge", "Mid": "Ucal", "Bot": "Teddy", "Support": "Pleata"}
        lineup_uol = {"Top": "BOSS", "Jungle": "AHaHaCiK", "Mid": "Nomanz", "Bot": "Argonavt", "Support": "SaNTaS"}
        lineup_es = {"Top": "Sola", "Jungle": "Obvious", "Mid": "Xico", "Bot": "Pyrka", "Support": "Tolerant"}
        mapping = {
            "g1": ("LCK", 2026, "T1", lineup_t1, "DRX", lineup_drx, 1, 2200.0),
            "g2": ("PRM", 2026, "UOL Sexy Edition", lineup_uol, "Eintracht Spandau", lineup_es, 1, 1400.0),
            "g3": ("WORLDS", 2026, "T1", lineup_t1, "UOL Sexy Edition", lineup_uol, 1, 900.0),
        }
        for gameid, (league, year, blue_team, blue_lineup, red_team, red_lineup, blue_win, gd15) in mapping.items():
            for side, team, lineup in [("blue", blue_team, blue_lineup), ("red", red_team, red_lineup)]:
                pid_base = 1 if side == "blue" else 6
                result = 1 if (side == "blue" and blue_win == 1) or (side == "red" and blue_win == 0) else 0
                signed_gd = gd15 if side == "blue" else -gd15
                for idx, (pos, player) in enumerate(lineup.items()):
                    rows.append(
                        {
                            "gameid": gameid,
                            "league": league,
                            "year": year,
                            "split": "Spring",
                            "playoffs": 1 if league == "WORLDS" else 0,
                            "date": "2026-03-01",
                            "patch": "16.2",
                            "participantid": pid_base + idx,
                            "position": pos,
                            "playername": player,
                            "playerid": f"{player.lower()}_id",
                            "teamname": team,
                            "result": result,
                            "gamelength": 1900,
                            "kills": 5 + idx,
                            "deaths": 2 + (0 if result else 2),
                            "assists": 8,
                            "damagetochampions": 12000 + (idx * 500),
                            "dpm": 420 + (idx * 30) + (20 if result else -10),
                            "damageshare": 0.18 + (idx * 0.01),
                            "visionscore": 24 + (idx * 4),
                            "vspm": 1.0 + (idx * 0.06),
                            "wpm": 0.60,
                            "wcpm": 0.18,
                            "controlwardsbought": 2 + idx,
                            "cspm": 6.3 + (idx * 0.2),
                            "earnedgoldshare": 0.18 + (idx * 0.01),
                            "golddiffat15": signed_gd,
                            "xpdiffat15": signed_gd / 2.5,
                            "csdiffat15": signed_gd / 100.0,
                            "firstblood": 1 if result else 0,
                            "firsttower": 1 if result else 0,
                        }
                    )
        return pd.DataFrame(rows)

    def test_builder_returns_league_and_team_profiles(self):
        payload = build_league_transfer_features_from_team_rows(
            self._rows(),
            player_rows=self._player_rows(),
            identity_registry=None,
            source_labels=["synthetic"],
        )
        self.assertIn("by_league", payload)
        self.assertIn("by_team", payload)
        self.assertIn("by_player", payload)
        self.assertIn("by_lineup", payload)
        self.assertIn("by_pair", payload)
        self.assertIn("LCK", payload["by_league"])
        self.assertIn("PRM", payload["by_league"])
        self.assertTrue(any(row.get("current_name") == "T1" for row in payload["by_team"].values()))
        self.assertTrue(any(float(row.get("team_transfer_power", 0.0)) > 0.0 for row in payload["by_team"].values()))
        self.assertTrue(any(float(row.get("player_portability", 0.0)) >= 0.0 for row in payload["by_player"].values()))
        self.assertTrue(any(float(row.get("lineup_transfer_power", 0.0)) >= 0.0 for row in payload["by_lineup"].values()))
        self.assertTrue(any(str(key).startswith("league_pair:") for key in payload["by_pair"].keys()))


if __name__ == "__main__":
    unittest.main()
