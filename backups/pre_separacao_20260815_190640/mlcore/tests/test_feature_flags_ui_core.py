import unittest

from app_core.feature_flags_ui_core import (
    build_feature_trace_download_payload,
    build_feature_trace_events_table,
    build_feature_trace_summary_rows,
    normalize_feature_trace_rows,
)


class FeatureFlagsUICoreTests(unittest.TestCase):
    def test_normalize_feature_trace_rows_fills_missing_columns(self) -> None:
        rows = normalize_feature_trace_rows([{"feature": "players", "payload": {"rows_after": 3}}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["feature"], "players")
        self.assertEqual(rows[0]["event"], "-")
        self.assertEqual(rows[0]["generated_at"], "-")

    def test_build_feature_trace_summary_rows_aggregates_last_payload(self) -> None:
        rows = [
            {
                "feature": "players",
                "event": "panel_open",
                "generated_at": "2026-03-10T10:00:00",
                "payload": {"method": "shadow", "rows_after": 4, "rows_total": 7, "fair_rows_total": 2, "min_h2h_games": 1},
            },
            {
                "feature": "players",
                "event": "panel_open",
                "generated_at": "2026-03-10T10:10:00",
                "payload": {"method": "shadow_v2", "rows_after": 5, "rows_total": 8, "fair_rows_total": 3, "min_h2h_games": 2},
            },
        ]
        summary = build_feature_trace_summary_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["Feature"], "players")
        self.assertEqual(summary[0]["Evento"], "panel_open")
        self.assertEqual(summary[0]["Method"], "shadow_v2")
        self.assertEqual(summary[0]["Ocorrencias"], 2)
        self.assertEqual(summary[0]["Rows after"], 5.0)

    def test_build_feature_trace_events_table_can_include_payload(self) -> None:
        rows = [
            {
                "feature": "players",
                "event": "panel_open",
                "generated_at": "2026-03-10T10:10:00",
                "enabled": True,
                "dedupe_signature": "abc",
                "payload": {"method": "shadow_v2", "matchup": "A x B", "rows_after": 5},
            }
        ]
        df = build_feature_trace_events_table(rows, show_payload=True)
        self.assertEqual(list(df["Feature"]), ["players"])
        self.assertIn("Payload", df.columns)
        self.assertEqual(df.iloc[0]["Method"], "shadow_v2")
        self.assertEqual(df.iloc[0]["Rows after"], 5.0)

    def test_build_feature_trace_download_payload_normalizes_events(self) -> None:
        payload = build_feature_trace_download_payload([{"feature": "players"}], generated_at="2026-03-10T11:00:00")
        self.assertEqual(payload["generated_at"], "2026-03-10T11:00:00")
        self.assertEqual(payload["events"][0]["feature"], "players")
        self.assertEqual(payload["events"][0]["event"], "-")


if __name__ == "__main__":
    unittest.main()
