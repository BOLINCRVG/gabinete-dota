"""Tests for mlcore/league_transfer_loader_core.py — CSV loaders and inference helpers."""
import io
import os
import tempfile
import unittest

import pandas as pd

from mlcore.league_transfer_loader_core import (
    ROLE_CANON,
    TEAM_REQUIRED_COLUMNS,
    PLAYER_REQUIRED_COLUMNS,
    infer_team_domestic_league_by_year,
    infer_team_meta_latest,
    load_oracle_player_rows,
    load_oracle_team_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _team_csv_df(**row_overrides) -> pd.DataFrame:
    """Minimal team CSV DataFrame — default is a single 'team' position row."""
    base = {
        "gameid": "g1",
        "league": "LCK",
        "year": 2025,
        "split": "spring",
        "playoffs": 0,
        "date": "2025-03-01",
        "patch": "15.1",
        "participantid": 100,
        "position": "team",
        "teamname": "T1",
        "result": 1,
        "gamelength": 1800,
        "kills": 15,
        "deaths": 5,
        "assists": 30,
        "ckpm": 0.83,
        "totalgold": 60000,
        "earnedgold": 45000,
        "towers": 8,
        "opp_towers": 3,
        "dragons": 3,
        "opp_dragons": 1,
        "barons": 1,
        "opp_barons": 0,
        "inhibitors": 2,
        "opp_inhibitors": 0,
        "firstblood": 1,
        "firstdragon": 1,
        "firstherald": 1,
        "firstbaron": 1,
        "firsttower": 1,
        "firsttothreetowers": 1,
        "goldat15": 30000,
        "golddiffat15": 2000,
        "xpdiffat15": 1500,
        "csdiffat15": 10,
    }
    base.update(row_overrides)
    return pd.DataFrame([base])


def _player_csv_df(**row_overrides) -> pd.DataFrame:
    """Minimal player CSV DataFrame — default is a single bot-lane row."""
    base = {
        "gameid": "g1",
        "league": "LCK",
        "year": 2025,
        "split": "spring",
        "playoffs": 0,
        "date": "2025-03-01",
        "patch": "15.1",
        "participantid": 1,
        "position": "bot",
        "playername": "Gumayushi",
        "playerid": "p1",
        "teamname": "T1",
        "result": 1,
        "gamelength": 1800,
        "kills": 5,
        "deaths": 1,
        "assists": 3,
        "damagetochampions": 20000,
        "dpm": 667,
        "damageshare": 0.30,
        "visionscore": 20,
        "vspm": 0.67,
        "wpm": 0.2,
        "wcpm": 0.1,
        "controlwardsbought": 3,
        "cspm": 8.5,
        "earnedgoldshare": 0.28,
        "golddiffat15": 200,
        "xpdiffat15": 100,
        "csdiffat15": 5,
        "firstblood": 0,
        "firsttower": 0,
    }
    base.update(row_overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# load_oracle_team_rows
# ---------------------------------------------------------------------------

class TestLoadOracleTeamRows(unittest.TestCase):

    def test_empty_list_returns_empty_df(self):
        df = load_oracle_team_rows([])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_empty_df_has_required_columns(self):
        df = load_oracle_team_rows([])
        for col in TEAM_REQUIRED_COLUMNS:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_nonexistent_file_returns_empty(self):
        df = load_oracle_team_rows(["/nonexistent/path/file.csv"])
        self.assertTrue(df.empty)

    def test_loads_team_position_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            _write_csv(f.name, _team_csv_df(position="team"))
            path = f.name
        try:
            df = load_oracle_team_rows([path])
            self.assertEqual(len(df), 1)
        finally:
            os.unlink(path)

    def test_filters_out_player_position_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            rows = pd.concat([
                _team_csv_df(position="team", participantid=100),
                _team_csv_df(position="top", participantid=1),
                _team_csv_df(position="mid", participantid=2),
            ], ignore_index=True)
            _write_csv(f.name, rows)
            path = f.name
        try:
            df = load_oracle_team_rows([path])
            # Only position="team" row should survive
            self.assertEqual(len(df), 1)
        finally:
            os.unlink(path)

    def test_participantid_100_200_also_kept(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            rows = pd.concat([
                _team_csv_df(position="", participantid=100),
                _team_csv_df(position="", participantid=200),
                _team_csv_df(position="", participantid=5),
            ], ignore_index=True)
            _write_csv(f.name, rows)
            path = f.name
        try:
            df = load_oracle_team_rows([path])
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(path)

    def test_league_canonicalized(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            _write_csv(f.name, _team_csv_df(league="LTA"))
            path = f.name
        try:
            df = load_oracle_team_rows([path])
            # LTA → LCS
            self.assertEqual(df["league"].iloc[0], "LCS")
        finally:
            os.unlink(path)

    def test_multiple_files_concatenated(self):
        paths = []
        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
                    _write_csv(f.name, _team_csv_df(gameid=f"g{i}"))
                    paths.append(f.name)
            df = load_oracle_team_rows(paths)
            self.assertEqual(len(df), 2)
        finally:
            for p in paths:
                os.unlink(p)


# ---------------------------------------------------------------------------
# load_oracle_player_rows
# ---------------------------------------------------------------------------

class TestLoadOraclePlayerRows(unittest.TestCase):

    def test_empty_list_returns_empty_df(self):
        df = load_oracle_player_rows([])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_empty_df_has_required_columns(self):
        df = load_oracle_player_rows([])
        for col in PLAYER_REQUIRED_COLUMNS:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_nonexistent_file_returns_empty(self):
        df = load_oracle_player_rows(["/nonexistent/path.csv"])
        self.assertTrue(df.empty)

    def test_loads_player_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            _write_csv(f.name, _player_csv_df(participantid=1, position="bot"))
            path = f.name
        try:
            df = load_oracle_player_rows([path])
            self.assertEqual(len(df), 1)
        finally:
            os.unlink(path)

    def test_filters_team_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            rows = pd.concat([
                _player_csv_df(participantid=100, position="team"),
                _player_csv_df(participantid=1, position="top"),
                _player_csv_df(participantid=2, position="mid"),
            ], ignore_index=True)
            _write_csv(f.name, rows)
            path = f.name
        try:
            df = load_oracle_player_rows([path])
            # Only participantid in [1..10] with valid positions should survive
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(path)

    def test_pos_can_column_added(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            _write_csv(f.name, _player_csv_df(participantid=1, position="bot"))
            path = f.name
        try:
            df = load_oracle_player_rows([path])
            self.assertIn("pos_can", df.columns)
            self.assertEqual(df["pos_can"].iloc[0], "Bot")
        finally:
            os.unlink(path)

    def test_all_five_roles_kept(self):
        # Filter uses lowercase ROLE_CANON names ("jungle", not "jng") + alternates ("sup", "adc")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            rows = pd.concat([
                _player_csv_df(participantid=1, position="top"),
                _player_csv_df(participantid=2, position="jungle"),
                _player_csv_df(participantid=3, position="mid"),
                _player_csv_df(participantid=4, position="bot"),
                _player_csv_df(participantid=5, position="sup"),
            ], ignore_index=True)
            _write_csv(f.name, rows)
            path = f.name
        try:
            df = load_oracle_player_rows([path])
            self.assertEqual(len(df), 5)
            self.assertEqual(set(df["pos_can"]), set(ROLE_CANON))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# infer_team_domestic_league_by_year
# ---------------------------------------------------------------------------

class TestInferTeamDomesticLeagueByYear(unittest.TestCase):

    def _df(self, rows):
        return pd.DataFrame(rows)

    def test_empty_df_returns_empty(self):
        result = infer_team_domestic_league_by_year(pd.DataFrame())
        self.assertEqual(result, {})

    def test_none_returns_empty(self):
        result = infer_team_domestic_league_by_year(None)
        self.assertEqual(result, {})

    def test_single_row(self):
        df = self._df([{"teamname": "T1", "league": "LCK", "year": 2025}])
        result = infer_team_domestic_league_by_year(df)
        self.assertEqual(result["T1"][2025], "LCK")

    def test_most_frequent_league_wins(self):
        # 3 rows LCK, 1 row LCS for T1 in 2025 → LCK wins
        rows = (
            [{"teamname": "T1", "league": "LCK", "year": 2025}] * 3
            + [{"teamname": "T1", "league": "LCS", "year": 2025}]
        )
        df = self._df(rows)
        result = infer_team_domestic_league_by_year(df)
        self.assertEqual(result["T1"][2025], "LCK")

    def test_each_year_tracked_separately(self):
        rows = [
            {"teamname": "T1", "league": "LCK", "year": 2024},
            {"teamname": "T1", "league": "LCK", "year": 2025},
            {"teamname": "T1", "league": "LCS", "year": 2023},
        ]
        df = self._df(rows)
        result = infer_team_domestic_league_by_year(df)
        self.assertEqual(result["T1"][2024], "LCK")
        self.assertEqual(result["T1"][2025], "LCK")
        self.assertEqual(result["T1"][2023], "LCS")

    def test_international_leagues_excluded(self):
        # WORLDS is international → should be filtered out
        rows = [
            {"teamname": "T1", "league": "WORLDS", "year": 2025},
            {"teamname": "T1", "league": "LCK", "year": 2025},
        ]
        df = self._df(rows)
        result = infer_team_domestic_league_by_year(df)
        # WORLDS is excluded; LCK should be the result
        self.assertEqual(result["T1"][2025], "LCK")

    def test_multiple_teams(self):
        rows = [
            {"teamname": "T1", "league": "LCK", "year": 2025},
            {"teamname": "G2", "league": "LEC", "year": 2025},
        ]
        df = self._df(rows)
        result = infer_team_domestic_league_by_year(df)
        self.assertIn("T1", result)
        self.assertIn("G2", result)
        self.assertEqual(result["T1"][2025], "LCK")
        self.assertEqual(result["G2"][2025], "LEC")

    def test_empty_teamname_excluded(self):
        rows = [
            {"teamname": "", "league": "LCK", "year": 2025},
            {"teamname": "T1", "league": "LCK", "year": 2025},
        ]
        df = self._df(rows)
        result = infer_team_domestic_league_by_year(df)
        self.assertNotIn("", result)

    def test_nan_year_excluded(self):
        rows = [
            {"teamname": "T1", "league": "LCK", "year": float("nan")},
            {"teamname": "T1", "league": "LCK", "year": 2025},
        ]
        df = self._df(rows)
        result = infer_team_domestic_league_by_year(df)
        self.assertIn(2025, result.get("T1", {}))
        self.assertNotIn(float("nan"), result.get("T1", {}))


# ---------------------------------------------------------------------------
# infer_team_meta_latest
# ---------------------------------------------------------------------------

class TestInferTeamMetaLatest(unittest.TestCase):

    def test_empty_dict_returns_empty(self):
        result = infer_team_meta_latest({})
        self.assertEqual(result, {})

    def test_none_returns_empty(self):
        result = infer_team_meta_latest(None)
        self.assertEqual(result, {})

    def test_single_team_single_year(self):
        result = infer_team_meta_latest({"T1": {2025: "LCK"}})
        self.assertEqual(result["T1"]["league"], "LCK")
        self.assertEqual(result["T1"]["year"], 2025)

    def test_uses_latest_year(self):
        result = infer_team_meta_latest({"T1": {2023: "LCS", 2025: "LCK", 2024: "LEC"}})
        self.assertEqual(result["T1"]["year"], 2025)
        self.assertEqual(result["T1"]["league"], "LCK")

    def test_multiple_teams(self):
        data = {
            "T1": {2025: "LCK"},
            "G2": {2025: "LEC"},
        }
        result = infer_team_meta_latest(data)
        self.assertEqual(result["T1"]["league"], "LCK")
        self.assertEqual(result["G2"]["league"], "LEC")

    def test_empty_mapping_for_team_skipped(self):
        result = infer_team_meta_latest({"T1": {2025: "LCK"}, "G2": {}})
        self.assertIn("T1", result)
        self.assertNotIn("G2", result)

    def test_output_has_league_and_year(self):
        result = infer_team_meta_latest({"T1": {2025: "LCK"}})
        self.assertIn("league", result["T1"])
        self.assertIn("year", result["T1"])

    def test_none_league_defaults_to_other(self):
        result = infer_team_meta_latest({"T1": {2025: None}})
        self.assertEqual(result["T1"]["league"], "OTHER")


if __name__ == "__main__":
    unittest.main()
