import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from mlcore.players_layer import (
    PlayersArtifact,
    PlayerState,
    _lane_weight_for,
    _league_weight_for,
    _parse_date,
    compute_delta_and_confidence,
    safe_json_write,
)


class PlayersLayerWeightsTest(unittest.TestCase):
    def test_helper_guards_handle_invalid_inputs(self):
        self.assertEqual(_lane_weight_for("Top", {"Top": "bad"}), 1.0)
        self.assertEqual(_league_weight_for("LPL", {"LPL": object()}), 1.0)
        self.assertEqual(_parse_date("2026-01-02"), date(2026, 1, 2))
        self.assertIsNone(_parse_date(["2026-01-02"]))

    def test_safe_json_write_creates_backup(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "players_artifact.json"
            path.write_text("{}", encoding="utf-8")
            safe_json_write(path, {"version": "players_artifact_v1"})
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".json.bak").exists())
            self.assertIn('"version": "players_artifact_v1"', path.read_text(encoding="utf-8"))

    def _mk_artifact(self) -> PlayersArtifact:
        players = {}
        for pos in ["Top", "Jungle", "Mid", "Bot", "Support"]:
            players[f"b_{pos}|||{pos}"] = PlayerState(
                r=1.0, gp=200, last_date="2026-01-01", last_league="LPL", last_tier="T1"
            )
            players[f"r_{pos}|||{pos}"] = PlayerState(
                r=0.0, gp=200, last_date="2026-01-01", last_league="LPL", last_tier="T1"
            )
        return PlayersArtifact(
            shrink_m=1e-9,
            half_life_days=3650.0,
            players=players,
        )

    def test_lane_and_league_weights_scale_delta(self):
        art = self._mk_artifact()
        blue = {p: f"b_{p}" for p in ["Top", "Jungle", "Mid", "Bot", "Support"]}
        red = {p: f"r_{p}" for p in ["Top", "Jungle", "Mid", "Bot", "Support"]}

        base = compute_delta_and_confidence(
            art,
            blue_lineup=blue,
            red_lineup=red,
            as_of=date(2026, 2, 1),
            match_league="LPL",
            require_both_sides=True,
        )
        base_delta = float(base["delta"])
        self.assertTrue(4.8 <= base_delta <= 5.1)

        weighted = compute_delta_and_confidence(
            art,
            blue_lineup=blue,
            red_lineup=red,
            as_of=date(2026, 2, 1),
            match_league="LPL",
            require_both_sides=True,
            lane_weights={"Top": 2.0, "Jungle": 1.0, "Mid": 1.0, "Bot": 1.0, "Support": 1.0},
            league_weights={"LPL": 1.20},
        )
        # Expected scale vs base: (2+1+1+1+1)/5 * 1.2 = 1.44x
        self.assertAlmostEqual(float(weighted["delta"]), base_delta * 1.44, places=3)
        self.assertAlmostEqual(float(weighted.get("league_weight", 1.0)), 1.2, places=6)

    def test_interleague_transfer_scale_damps_cross_macro_effective_rating(self):
        art = self._mk_artifact()
        same_r, _, same_dbg = art.r_effective_at(
            "b_Top",
            "Top",
            as_of=date(2026, 2, 1),
            match_league="LPL",
        )
        cross_r, _, cross_dbg = art.r_effective_at(
            "b_Top",
            "Top",
            as_of=date(2026, 2, 1),
            match_league="LCS",
        )
        self.assertGreater(float(same_r), float(cross_r))
        self.assertAlmostEqual(float(same_dbg.get("interleague_scale", 1.0)), 1.0, places=6)
        self.assertLess(float(cross_dbg.get("interleague_scale", 1.0)), 1.0)

    def test_interleague_transfer_scale_respects_player_evidence(self):
        art = self._mk_artifact()
        art.players["rookie_top|||Top"] = PlayerState(
            r=1.0, gp=8, last_date="2026-01-01", last_league="LPL", last_tier="T1"
        )
        art.players["vet_top|||Top"] = PlayerState(
            r=1.0, gp=320, last_date="2026-01-01", last_league="LPL", last_tier="T1"
        )
        rookie_r, _, rookie_dbg = art.r_effective_at(
            "rookie_top",
            "Top",
            as_of=date(2026, 2, 1),
            match_league="LCS",
        )
        vet_r, _, vet_dbg = art.r_effective_at(
            "vet_top",
            "Top",
            as_of=date(2026, 2, 1),
            match_league="LCS",
        )
        self.assertLess(float(rookie_dbg.get("interleague_scale", 1.0)), float(vet_dbg.get("interleague_scale", 1.0)))
        self.assertLess(float(rookie_dbg.get("interleague_gp_support", 1.0)), float(vet_dbg.get("interleague_gp_support", 1.0)))
        self.assertLess(float(rookie_r), float(vet_r))

    def test_compute_delta_uses_side_specific_leagues(self):
        art = self._mk_artifact()
        blue = {p: f"b_{p}" for p in ["Top", "Jungle", "Mid", "Bot", "Support"]}
        red = {p: f"r_{p}" for p in ["Top", "Jungle", "Mid", "Bot", "Support"]}

        out = compute_delta_and_confidence(
            art,
            blue_lineup=blue,
            red_lineup=red,
            as_of=date(2026, 2, 1),
            match_league="",
            blue_league="LPL",
            red_league="LCS",
            require_both_sides=True,
            league_weights={"LPL": 1.20, "LCS": 0.80},
        )
        self.assertAlmostEqual(float(out.get("league_weight_blue", 0.0)), 1.20, places=6)
        self.assertAlmostEqual(float(out.get("league_weight_red", 0.0)), 0.80, places=6)
        self.assertAlmostEqual(float(out.get("league_weight", 0.0)), (1.20 * 0.80) ** 0.5, places=6)

    def test_compute_delta_includes_transfer_summary_when_payload_exists(self):
        art = self._mk_artifact()
        blue = {p: f"b_{p}" for p in ["Top", "Jungle", "Mid", "Bot", "Support"]}
        red = {p: f"r_{p}" for p in ["Top", "Jungle", "Mid", "Bot", "Support"]}
        payload = {
            "by_team": {
                "bluekings": {"entity_key": "bluekings"},
                "redhawks": {"entity_key": "redhawks"},
            },
            "team_alias_index": {"bluekings": "bluekings", "redhawks": "redhawks"},
            "by_player": {
                "b_top": {"entity_key": "b_top", "player_domestic_rating": 0.80, "player_portability": 0.78, "sample_games": 20},
                "b_jungle": {"entity_key": "b_jungle", "player_domestic_rating": 0.80, "player_portability": 0.78, "sample_games": 20},
                "b_mid": {"entity_key": "b_mid", "player_domestic_rating": 0.80, "player_portability": 0.78, "sample_games": 20},
                "b_bot": {"entity_key": "b_bot", "player_domestic_rating": 0.80, "player_portability": 0.78, "sample_games": 20},
                "b_support": {"entity_key": "b_support", "player_domestic_rating": 0.80, "player_portability": 0.78, "sample_games": 20},
                "r_top": {"entity_key": "r_top", "player_domestic_rating": 0.55, "player_portability": 0.52, "sample_games": 20},
                "r_jungle": {"entity_key": "r_jungle", "player_domestic_rating": 0.55, "player_portability": 0.52, "sample_games": 20},
                "r_mid": {"entity_key": "r_mid", "player_domestic_rating": 0.55, "player_portability": 0.52, "sample_games": 20},
                "r_bot": {"entity_key": "r_bot", "player_domestic_rating": 0.55, "player_portability": 0.52, "sample_games": 20},
                "r_support": {"entity_key": "r_support", "player_domestic_rating": 0.55, "player_portability": 0.52, "sample_games": 20},
            },
            "player_alias_index": {
                "btop": "b_top",
                "bjungle": "b_jungle",
                "bmid": "b_mid",
                "bbot": "b_bot",
                "bsupport": "b_support",
                "rtop": "r_top",
                "rjungle": "r_jungle",
                "rmid": "r_mid",
                "rbot": "r_bot",
                "rsupport": "r_support",
            },
            "by_lineup": {
                "bluekings": {
                    "entity_key": "bluekings",
                    "lineup_stability": 0.85,
                    "lineup_synergy": 0.82,
                    "player_entities_by_pos": {"Top": "b_top", "Jungle": "b_jungle", "Mid": "b_mid", "Bot": "b_bot", "Support": "b_support"},
                },
                "redhawks": {
                    "entity_key": "redhawks",
                    "lineup_stability": 0.70,
                    "lineup_synergy": 0.68,
                    "player_entities_by_pos": {"Top": "r_top", "Jungle": "r_jungle", "Mid": "r_mid", "Bot": "r_bot", "Support": "r_support"},
                },
            },
        }
        out = compute_delta_and_confidence(
            art,
            blue_lineup=blue,
            red_lineup=red,
            as_of=date(2026, 2, 1),
            match_league="",
            blue_league="LCK",
            red_league="PRM",
            require_both_sides=True,
            transfer_payload=payload,
            team_a="Blue Kings",
            team_b="Red Hawks",
        )
        summary = out.get("player_portability_summary") or {}
        self.assertTrue(summary.get("applied"))
        self.assertGreater(float(summary.get("lineup_transfer_power_a", 0.0)), float(summary.get("lineup_transfer_power_b", 0.0)))
        self.assertGreater(float(out.get("delta_transfer_adj", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
