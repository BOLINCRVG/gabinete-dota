import unittest

from app_core.league_transfer_features_core import (
    build_lineup_transfer_summary,
    build_interleague_transfer_adjustment,
    lookup_league_transfer_features,
    lookup_pair_transfer_features,
    lookup_player_transfer_features,
    lookup_team_transfer_features,
)


class TestLeagueTransferFeaturesCore(unittest.TestCase):
    def _payload(self):
        return {
            "version": "league_transfer_features_v1",
            "by_league": {
                "LCK": {
                    "league_reliability": 0.82,
                    "league_depth": 0.74,
                    "league_translation_score": 0.67,
                    "league_domestic_inflation": 0.22,
                },
                "PRM": {
                    "league_reliability": 0.48,
                    "league_depth": 0.44,
                    "league_translation_score": 0.39,
                    "league_domestic_inflation": 0.66,
                },
            },
            "by_team": {
                "t1": {
                    "entity_key": "t1",
                    "team_domestic_power": 0.82,
                    "team_transfer_power": 0.79,
                    "sample_games": 22,
                },
                "uol": {
                    "entity_key": "uol",
                    "team_domestic_power": 0.66,
                    "team_transfer_power": 0.54,
                    "sample_games": 18,
                },
            },
            "by_player": {
                "faker": {
                    "entity_key": "faker",
                    "player_domestic_rating": 0.82,
                    "player_portability": 0.80,
                    "sample_games": 30,
                },
                "oner": {
                    "entity_key": "oner",
                    "player_domestic_rating": 0.78,
                    "player_portability": 0.76,
                    "sample_games": 28,
                },
                "gumayusi": {
                    "entity_key": "gumayusi",
                    "player_domestic_rating": 0.77,
                    "player_portability": 0.74,
                    "sample_games": 28,
                },
                "keria": {
                    "entity_key": "keria",
                    "player_domestic_rating": 0.76,
                    "player_portability": 0.75,
                    "sample_games": 28,
                },
                "zeus": {
                    "entity_key": "zeus",
                    "player_domestic_rating": 0.79,
                    "player_portability": 0.77,
                    "sample_games": 28,
                },
                "boss": {
                    "entity_key": "boss",
                    "player_domestic_rating": 0.60,
                    "player_portability": 0.52,
                    "sample_games": 18,
                },
                "ahahacik": {
                    "entity_key": "ahahacik",
                    "player_domestic_rating": 0.58,
                    "player_portability": 0.50,
                    "sample_games": 18,
                },
                "nomanz": {
                    "entity_key": "nomanz",
                    "player_domestic_rating": 0.62,
                    "player_portability": 0.55,
                    "sample_games": 18,
                },
                "argonavt": {
                    "entity_key": "argonavt",
                    "player_domestic_rating": 0.59,
                    "player_portability": 0.51,
                    "sample_games": 18,
                },
                "santas": {
                    "entity_key": "santas",
                    "player_domestic_rating": 0.57,
                    "player_portability": 0.50,
                    "sample_games": 18,
                },
            },
            "by_lineup": {
                "t1": {
                    "entity_key": "t1",
                    "lineup_strength": 0.79,
                    "lineup_stability": 0.84,
                    "lineup_synergy": 0.80,
                    "lineup_transfer_power": 0.78,
                    "player_entities_by_pos": {
                        "Top": "zeus",
                        "Jungle": "oner",
                        "Mid": "faker",
                        "Bot": "gumayusi",
                        "Support": "keria",
                    },
                },
                "uol": {
                    "entity_key": "uol",
                    "lineup_strength": 0.59,
                    "lineup_stability": 0.74,
                    "lineup_synergy": 0.70,
                    "lineup_transfer_power": 0.54,
                    "player_entities_by_pos": {
                        "Top": "boss",
                        "Jungle": "ahahacik",
                        "Mid": "nomanz",
                        "Bot": "argonavt",
                        "Support": "santas",
                    },
                },
            },
            "by_pair": {
                "league_pair:LCK->PRM": {
                    "pair_confidence": 0.72,
                    "pair_portability_edge": 0.26,
                    "pair_result_edge": 0.18,
                    "pair_transfer_edge": 0.22,
                },
                "tier_pair:T1->ERL": {
                    "pair_confidence": 0.55,
                    "pair_portability_edge": 0.19,
                    "pair_result_edge": 0.10,
                    "pair_transfer_edge": 0.17,
                },
            },
            "team_alias_index": {
                "t1": "t1",
                "skt": "t1",
                "unicornsoflovesexyedition": "uol",
                "uolse": "uol",
            },
            "player_alias_index": {
                "faker": "faker",
                "oner": "oner",
                "gumayusi": "gumayusi",
                "keria": "keria",
                "zeus": "zeus",
                "boss": "boss",
                "ahahacik": "ahahacik",
                "nomanz": "nomanz",
                "argonavt": "argonavt",
                "santas": "santas",
            },
        }

    def test_lookup_team_uses_alias_index(self):
        row = lookup_team_transfer_features(self._payload(), "SKT", aliases=["T1"])
        self.assertEqual(str(row.get("entity_key")), "t1")

    def test_lookup_league_uses_league_aliases(self):
        row = lookup_league_transfer_features(self._payload(), "LCK")
        self.assertAlmostEqual(float(row.get("league_depth", 0.0)), 0.74, places=6)

    def test_lookup_player_uses_alias_index(self):
        row = lookup_player_transfer_features(self._payload(), "Faker")
        self.assertEqual(str(row.get("entity_key")), "faker")

    def test_lookup_pair_prefers_exact_league_pair(self):
        row = lookup_pair_transfer_features(self._payload(), league_a="LCK", league_b="PRM")
        self.assertEqual(str(row.get("lookup_scope")), "league_pair")
        self.assertAlmostEqual(float(row.get("pair_portability_edge", 0.0)), 0.26, places=6)

    def test_interleague_adjustment_prefers_portable_side(self):
        out = build_interleague_transfer_adjustment(
            payload=self._payload(),
            interregional=True,
            league_a="LCK",
            league_b="PRM",
            team_a="T1",
            team_b="Unicorns of Love Sexy Edition",
            team_aliases_a=["SKT"],
            team_aliases_b=["UOLSE"],
        )
        self.assertEqual(str(out.get("reason")), "ok")
        self.assertTrue(bool(out.get("applied")))
        self.assertTrue(float(out.get("dir_bonus", 0.0)) > 0.0)
        self.assertTrue(float(out.get("shrink_adj", 0.0)) < 0.0)
        self.assertEqual(str(out.get("pair_lookup_scope")), "league_pair")
        self.assertTrue(float(out.get("pair_confidence", 0.0)) > 0.0)

    def test_build_lineup_transfer_summary_prefers_stronger_portable_lineup(self):
        payload = self._payload()
        payload["by_team"]["t1"].update(
            {
                "relation_type": "sporting_successor",
                "identity_confidence": "high",
                "kept_players": ["Zeus", "Oner", "Faker"],
                "added_players": ["Gumayusi", "Keria"],
            }
        )
        out = build_lineup_transfer_summary(
            payload=payload,
            team_a="T1",
            team_b="Unicorns of Love Sexy Edition",
            blue_lineup={"Top": "Zeus", "Jungle": "Oner", "Mid": "Faker", "Bot": "Gumayusi", "Support": "Keria"},
            red_lineup={"Top": "BOSS", "Jungle": "AHaHaCiK", "Mid": "Nomanz", "Bot": "Argonavt", "Support": "SaNTaS"},
        )
        self.assertTrue(bool(out.get("applied")))
        self.assertGreater(float(out.get("lineup_transfer_power_a", 0.0)), float(out.get("lineup_transfer_power_b", 0.0)))
        self.assertGreater(float(out.get("delta_adj", 0.0)), 0.0)
        self.assertGreater(float(out.get("identity_successor_signal_a", 0.0)), 0.7)


if __name__ == "__main__":
    unittest.main()
