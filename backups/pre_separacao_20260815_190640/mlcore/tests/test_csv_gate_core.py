import unittest

from app_core.csv_gate_core import (
    detect_series_gap_suspects,
    evaluate_csv_gate_metrics,
    fingerprint_inventory,
    rank_decision,
    summarize_league_changes,
)


class CsvGateCoreTest(unittest.TestCase):
    def test_rank_decision_orders_expected(self):
        self.assertLess(rank_decision("promote"), rank_decision("shadow_only"))
        self.assertLess(rank_decision("shadow_only"), rank_decision("reject"))

    def test_fingerprint_inventory_is_order_independent(self):
        rows_a = [
            {"path": "b.csv", "size": 2, "mtime_ns": 22},
            {"path": "a.csv", "size": 1, "mtime_ns": 11},
        ]
        rows_b = list(reversed(rows_a))
        self.assertEqual(fingerprint_inventory(rows_a), fingerprint_inventory(rows_b))

    def test_summarize_league_changes_detects_aliases_and_new_canonical(self):
        out = summarize_league_changes(
            ["LTA N", "LCK", "NEW LEAGUE"],
            known_raw_leagues=["LCK"],
            known_canonical_leagues=["LCS", "LCK"],
            canonicalize_fn=lambda x: {"LTA N": "LCS"}.get(x, x),
        )
        self.assertEqual(out["new_canonical_leagues"], ["NEW LEAGUE"])
        self.assertEqual(out["new_raw_mapped_to_known_canonical"][0]["canonical"], "LCS")
        self.assertEqual(out["alias_pairs"][0]["raw"], "LTA N")

    def test_summarize_league_changes_normalizes_known_canonical_alias_state(self):
        out = summarize_league_changes(
            ["WLDS"],
            known_raw_leagues=["WLDS"],
            known_canonical_leagues=["WLDS"],
            canonicalize_fn=lambda x: {"WLDS": "WORLDS"}.get(x, x),
        )
        self.assertEqual(out["canonical_leagues"], ["WORLDS"])
        self.assertEqual(out["new_canonical_leagues"], [])

    def test_evaluate_csv_gate_metrics_rejects_hard_failures(self):
        out = evaluate_csv_gate_metrics(
            {
                "phase1_decision": "promote",
                "row_key_dup_rate": 0.0,
                "date_parse_fail_rate": 0.01,
                "blank_team_rate": 0.0,
                "blank_league_rate": 0.0,
                "year_mismatch_rate": 0.0,
                "missing_required_cols_files": 0,
                "new_canonical_count": 0,
            }
        )
        self.assertEqual(out["decision"], "reject")
        self.assertIn("date_parse_fail_rate_high", out["reasons"])

    def test_evaluate_csv_gate_metrics_shadow_only_for_new_canonical(self):
        out = evaluate_csv_gate_metrics(
            {
                "phase1_decision": "promote",
                "row_key_dup_rate": 0.0,
                "date_parse_fail_rate": 0.0,
                "blank_team_rate": 0.0,
                "blank_league_rate": 0.0,
                "year_mismatch_rate": 0.0,
                "missing_required_cols_files": 0,
                "new_canonical_count": 2,
            }
        )
        self.assertEqual(out["decision"], "shadow_only")
        self.assertIn("new_canonical_leagues", out["reasons"])

    def test_evaluate_csv_gate_metrics_shadow_only_for_daily_series_gaps(self):
        out = evaluate_csv_gate_metrics(
            {
                "phase1_decision": "promote",
                "row_key_dup_rate": 0.0,
                "date_parse_fail_rate": 0.0,
                "blank_team_rate": 0.0,
                "blank_league_rate": 0.0,
                "year_mismatch_rate": 0.0,
                "missing_required_cols_files": 0,
                "new_canonical_count": 0,
                "daily_series_gap_suspect_groups": 1,
            }
        )
        self.assertEqual(out["decision"], "shadow_only")
        self.assertIn("daily_series_gap_suspects", out["reasons"])

    def test_evaluate_csv_gate_metrics_respects_phase1(self):
        out = evaluate_csv_gate_metrics(
            {
                "phase1_decision": "reject",
                "row_key_dup_rate": 0.0,
                "date_parse_fail_rate": 0.0,
                "blank_team_rate": 0.0,
                "blank_league_rate": 0.0,
                "year_mismatch_rate": 0.0,
                "missing_required_cols_files": 0,
                "new_canonical_count": 0,
            }
        )
        self.assertEqual(out["decision"], "reject")
        self.assertIn("phase1:reject", out["reasons"])

    def test_detect_series_gap_suspects_flags_missing_opening_game(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {"gameid": "g2", "participantid": 100, "teamname": "Apex Mission Impossible", "date": "2026-03-27 20:54:09", "league": "NACL", "year": 2026, "split": "Spring", "playoffs": 0, "game": 2},
                {"gameid": "g2", "participantid": 200, "teamname": "NRG", "date": "2026-03-27 20:54:09", "league": "NACL", "year": 2026, "split": "Spring", "playoffs": 0, "game": 2},
            ]
        )

        out = detect_series_gap_suspects(df, source_name="daily.csv")
        self.assertEqual(int(out["suspect_groups"]), 1)
        self.assertEqual(int(out["suspect_games"]), 1)
        self.assertEqual(out["examples"][0]["missing_game_numbers"], [1])
        self.assertEqual(out["examples"][0]["present_game_numbers"], [2])

    def test_detect_series_gap_suspects_accepts_complete_prefix(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {"gameid": "g1", "participantid": 100, "teamname": "A", "date": "2026-03-27 18:00:00", "league": "NACL", "year": 2026, "split": "Spring", "playoffs": 0, "game": 1},
                {"gameid": "g1", "participantid": 200, "teamname": "B", "date": "2026-03-27 18:00:00", "league": "NACL", "year": 2026, "split": "Spring", "playoffs": 0, "game": 1},
                {"gameid": "g2", "participantid": 100, "teamname": "A", "date": "2026-03-27 19:00:00", "league": "NACL", "year": 2026, "split": "Spring", "playoffs": 0, "game": 2},
                {"gameid": "g2", "participantid": 200, "teamname": "B", "date": "2026-03-27 19:00:00", "league": "NACL", "year": 2026, "split": "Spring", "playoffs": 0, "game": 2},
            ]
        )

        out = detect_series_gap_suspects(df, source_name="daily.csv")
        self.assertEqual(int(out["suspect_groups"]), 0)
        self.assertEqual(int(out["suspect_games"]), 0)
        self.assertEqual(out["examples"], [])


if __name__ == "__main__":
    unittest.main()
