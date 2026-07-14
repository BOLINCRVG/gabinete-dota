from __future__ import annotations

import unittest

import pandas as pd

from app_core.opponent_profile_core import (
    build_matchup_focus_table,
    build_opponent_reference_table,
    build_team_opponent_profile,
    describe_profile_signal,
    phase_options_for_market,
)


class OpponentProfileCoreTest(unittest.TestCase):
    def _sample_scope(self) -> pd.DataFrame:
        rows = [
            {"date": "2026-03-01", "league": "EMEA Masters", "team": "Alpha", "opponent": "Bravo", "win": 1, "total_kills": 35.0, "total_kills_at15": 10.0, "total_kills_15_20": 8.0, "total_kills_20_plus": 17.0},
            {"date": "2026-03-02", "league": "EMEA Masters", "team": "Alpha", "opponent": "Charlie", "win": 1, "total_kills": 33.0, "total_kills_at15": 11.0, "total_kills_15_20": 7.0, "total_kills_20_plus": 15.0},
            {"date": "2026-03-03", "league": "EMEA Masters", "team": "Alpha", "opponent": "Delta", "win": 0, "total_kills": 24.0, "total_kills_at15": 7.0, "total_kills_15_20": 5.0, "total_kills_20_plus": 12.0},
            {"date": "2026-03-04", "league": "EMEA Masters", "team": "Alpha", "opponent": "Echo", "win": 1, "total_kills": 22.0, "total_kills_at15": 6.0, "total_kills_15_20": 4.0, "total_kills_20_plus": 12.0},
            {"date": "2026-03-01", "league": "EMEA Masters", "team": "Bravo", "opponent": "Alpha", "win": 0, "total_kills": 35.0, "total_kills_at15": 10.0, "total_kills_15_20": 8.0, "total_kills_20_plus": 17.0},
            {"date": "2026-03-05", "league": "EMEA Masters", "team": "Bravo", "opponent": "Charlie", "win": 1, "total_kills": 38.0, "total_kills_at15": 14.0, "total_kills_15_20": 9.0, "total_kills_20_plus": 15.0},
            {"date": "2026-03-06", "league": "EMEA Masters", "team": "Bravo", "opponent": "Delta", "win": 1, "total_kills": 36.0, "total_kills_at15": 12.0, "total_kills_15_20": 8.0, "total_kills_20_plus": 16.0},
            {"date": "2026-03-02", "league": "EMEA Masters", "team": "Charlie", "opponent": "Alpha", "win": 0, "total_kills": 33.0, "total_kills_at15": 11.0, "total_kills_15_20": 7.0, "total_kills_20_plus": 15.0},
            {"date": "2026-03-05", "league": "EMEA Masters", "team": "Charlie", "opponent": "Bravo", "win": 0, "total_kills": 38.0, "total_kills_at15": 14.0, "total_kills_15_20": 9.0, "total_kills_20_plus": 15.0},
            {"date": "2026-03-07", "league": "EMEA Masters", "team": "Charlie", "opponent": "Delta", "win": 1, "total_kills": 31.0, "total_kills_at15": 9.0, "total_kills_15_20": 6.0, "total_kills_20_plus": 16.0},
            {"date": "2026-03-03", "league": "EMEA Masters", "team": "Delta", "opponent": "Alpha", "win": 1, "total_kills": 24.0, "total_kills_at15": 7.0, "total_kills_15_20": 5.0, "total_kills_20_plus": 12.0},
            {"date": "2026-03-06", "league": "EMEA Masters", "team": "Delta", "opponent": "Bravo", "win": 0, "total_kills": 36.0, "total_kills_at15": 12.0, "total_kills_15_20": 8.0, "total_kills_20_plus": 16.0},
            {"date": "2026-03-07", "league": "EMEA Masters", "team": "Delta", "opponent": "Charlie", "win": 0, "total_kills": 31.0, "total_kills_at15": 9.0, "total_kills_15_20": 6.0, "total_kills_20_plus": 16.0},
            {"date": "2026-03-04", "league": "EMEA Masters", "team": "Echo", "opponent": "Alpha", "win": 0, "total_kills": 22.0, "total_kills_at15": 6.0, "total_kills_15_20": 4.0, "total_kills_20_plus": 12.0},
            {"date": "2026-03-08", "league": "EMEA Masters", "team": "Echo", "opponent": "Delta", "win": 0, "total_kills": 21.0, "total_kills_at15": 6.0, "total_kills_15_20": 3.0, "total_kills_20_plus": 12.0},
        ]
        return pd.DataFrame(rows)

    def test_build_reference_table_assigns_buckets(self) -> None:
        ref = build_opponent_reference_table(self._sample_scope(), "Kills")
        self.assertFalse(ref.empty)
        self.assertIn("Bucket", ref.columns)
        bravo_bucket = str(ref.loc[ref["Team"] == "Bravo", "Bucket"].iloc[0])
        echo_bucket = str(ref.loc[ref["Team"] == "Echo", "Bucket"].iloc[0])
        self.assertIn(bravo_bucket, {"Over", "Muito Over"})
        self.assertIn(echo_bucket, {"Under", "Muito Under"})

    def test_phase_options_for_kills(self) -> None:
        opts = phase_options_for_market(self._sample_scope(), "Kills")
        self.assertEqual(opts, ["Final", "0-15", "15-20", "20+"])

    def test_build_team_profile_returns_focus_bucket_and_confidence(self) -> None:
        scope = self._sample_scope()
        ref = build_opponent_reference_table(scope, "Kills")
        payload = build_team_opponent_profile(
            scope,
            team="Alpha",
            opponent_team="Bravo",
            market_label="Kills",
            line_val=29.5,
            reference_table=ref,
        )
        self.assertEqual(payload["team"], "Alpha")
        self.assertEqual(payload["opponent_team"], "Bravo")
        self.assertIn(payload["opponent_bucket"], {"Over", "Muito Over"})
        rows = payload["rows"]
        self.assertFalse(rows.empty)
        over_row = rows[rows["Perfil do Oponente"].astype(str) == str(payload["opponent_bucket"])]
        self.assertFalse(over_row.empty)
        self.assertIn("Exemplos", rows.columns)
        self.assertIn("Confiança", rows.columns)
        self.assertIn("Jogos no bucket", rows.columns)
        self.assertIn("Delta vs Liga", rows.columns)
        self.assertIn("Risco", rows.columns)
        self.assertEqual(str(over_row.iloc[0]["Atual"]), "Atual")
        focus_games = payload["focus_games"]
        self.assertFalse(focus_games.empty)
        self.assertIn("Campeonato", focus_games.columns)
        self.assertIn("Valor", focus_games.columns)
        bucket_games = payload["bucket_games"]
        self.assertIsInstance(bucket_games, dict)
        self.assertIn(str(payload["opponent_bucket"]), bucket_games)
        self.assertIn("team_games_total", payload)
        self.assertIn("opponent_metric_mean", payload)
        self.assertIn("opponent_metric_z", payload)

    def test_build_team_profile_phase_payload(self) -> None:
        scope = self._sample_scope()
        ref = build_opponent_reference_table(scope, "Kills", phase_label="15-20")
        payload = build_team_opponent_profile(
            scope,
            team="Alpha",
            opponent_team="Bravo",
            market_label="Kills",
            reference_table=ref,
            phase_label="15-20",
        )
        self.assertEqual(str(payload["phase_label"]), "15-20")
        self.assertIsNone(payload["baseline_over"])
        self.assertIsNone(payload["baseline_under"])

    def test_build_matchup_focus_table(self) -> None:
        scope = self._sample_scope()
        ref = build_opponent_reference_table(scope, "Kills")
        payload_a = build_team_opponent_profile(scope, team="Alpha", opponent_team="Bravo", market_label="Kills", line_val=29.5, reference_table=ref)
        payload_b = build_team_opponent_profile(scope, team="Bravo", opponent_team="Alpha", market_label="Kills", line_val=29.5, reference_table=ref)
        focus = build_matchup_focus_table(payload_a, payload_b)
        self.assertEqual(list(focus["Time"]), ["Alpha", "Bravo"])
        self.assertIn("Perfil Atual", focus.columns)
        self.assertIn("Confiança", focus.columns)
        self.assertIn("Fase", focus.columns)
        self.assertIn("Jogos no bucket", focus.columns)
        self.assertIn("Jogos do time", focus.columns)
        self.assertIn("Score Adv.", focus.columns)
        self.assertIn("Delta vs Liga", focus.columns)

    def test_signal_description(self) -> None:
        self.assertEqual(describe_profile_signal(3.0, "kills", 2.0), "Acelera")
        self.assertEqual(describe_profile_signal(-3.0, "kills", 2.0), "Desacelera")
        self.assertEqual(describe_profile_signal(0.2, "kills", 2.0), "Mantém")


if __name__ == "__main__":
    unittest.main()
