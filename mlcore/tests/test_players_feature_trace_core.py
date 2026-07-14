import json
import unittest

from app_core.players_feature_trace_core import (
    append_players_fair_panel_trace,
    append_players_markets_panel_trace,
    append_players_panels_bundle_trace,
    append_players_recent_stats_panel_trace,
)


class TestPlayersFeatureTraceCore(unittest.TestCase):
    def test_appends_all_players_panel_events(self):
        calls = []

        def _append_feature_trace(**kwargs):
            calls.append(dict(kwargs))

        self.assertTrue(
            append_players_recent_stats_panel_trace(
                append_feature_trace_fn=_append_feature_trace,
                enabled=True,
                teamA="A",
                teamB="B",
                team_filter="Todos",
                lane_filter="TOP",
                player_filter="P1",
                opp_filter="P2",
                rows_before=10,
                rows_after=4,
            )
        )
        self.assertTrue(
            append_players_fair_panel_trace(
                append_feature_trace_fn=_append_feature_trace,
                enabled=True,
                teamA="A",
                teamB="B",
                team_filter="A",
                lane_filter="TOP",
                player_filter="P1",
                rows_after=2,
                rows_total=11,
            )
        )
        self.assertTrue(
            append_players_markets_panel_trace(
                append_feature_trace_fn=_append_feature_trace,
                enabled=True,
                teamA="A",
                teamB="B",
                team_filter="A",
                lane_filter="TOP",
                market_filter="Kills",
                player_filter="P1",
                rows_after=3,
                rows_total=18,
                fair_rows_total=11,
                min_h2h_games=6,
            )
        )
        self.assertTrue(
            append_players_panels_bundle_trace(
                append_feature_trace_fn=_append_feature_trace,
                enabled=True,
                teamA="A",
                teamB="B",
                recent_rows_total=14,
                market_rows_total=18,
                fair_rows_total=11,
            )
        )

        self.assertEqual(len(calls), 4)
        self.assertEqual(
            [c.get("feature") for c in calls],
            [
                "players_recent_stats_panel_v1",
                "players_fair_panel_v1",
                "players_markets_panel_v1",
                "players_panels_bundle_v1",
            ],
        )
        self.assertEqual([c.get("event") for c in calls[:3]], ["shadow_render", "shadow_render", "shadow_render"])
        self.assertEqual(calls[3].get("event"), "persist_after_players_render")
        self.assertEqual(calls[3].get("persist_files"), False)

        d0 = json.loads(str(calls[0].get("dedupe_signature")))
        self.assertEqual(d0.get("ta"), "A")
        self.assertEqual(d0.get("tb"), "B")
        self.assertEqual(int(d0.get("n0")), 10)
        self.assertEqual(int(d0.get("n1")), 4)

    def test_disabled_emits_nothing(self):
        calls = []

        self.assertFalse(
            append_players_recent_stats_panel_trace(
                append_feature_trace_fn=lambda **kwargs: calls.append(dict(kwargs)),
                enabled=False,
                teamA="A",
                teamB="B",
                team_filter="Todos",
                lane_filter="TOP",
                player_filter="P1",
                opp_filter="P2",
                rows_before=10,
                rows_after=4,
            )
        )
        self.assertFalse(
            append_players_fair_panel_trace(
                append_feature_trace_fn=lambda **kwargs: calls.append(dict(kwargs)),
                enabled=False,
                teamA="A",
                teamB="B",
                team_filter="A",
                lane_filter="TOP",
                player_filter="P1",
                rows_after=2,
                rows_total=11,
            )
        )
        self.assertFalse(
            append_players_markets_panel_trace(
                append_feature_trace_fn=lambda **kwargs: calls.append(dict(kwargs)),
                enabled=False,
                teamA="A",
                teamB="B",
                team_filter="A",
                lane_filter="TOP",
                market_filter="Kills",
                player_filter="P1",
                rows_after=3,
                rows_total=18,
                fair_rows_total=11,
                min_h2h_games=6,
            )
        )
        self.assertFalse(
            append_players_panels_bundle_trace(
                append_feature_trace_fn=lambda **kwargs: calls.append(dict(kwargs)),
                enabled=False,
                teamA="A",
                teamB="B",
                recent_rows_total=14,
                market_rows_total=18,
                fair_rows_total=11,
            )
        )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
