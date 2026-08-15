import json
import tempfile
import unittest
from pathlib import Path

from app_core.identity_registry_core import (
    empty_identity_registry,
    load_identity_registry,
    normalize_identity_key,
    player_identity_candidates,
    resolve_team_identity_record,
    resolve_player_identity_record,
    search_identity_records,
    team_identity_candidates,
    team_identity_league_candidates,
    upsert_player_identity,
    upsert_team_identity,
)


class IdentityRegistryCoreTests(unittest.TestCase):
    def test_normalize_identity_key(self) -> None:
        self.assertEqual(normalize_identity_key("Los Ratones"), "losratones")
        self.assertEqual(normalize_identity_key(" Witchcraft "), "witchcraft")

    def test_seed_registry_matches_predecessor_name(self) -> None:
        registry = load_identity_registry("identity_registry.json")
        matches = search_identity_records(registry, query="Los Ratones", entity_type="teams")
        self.assertTrue(matches)
        top = matches[0]["record"]
        self.assertEqual(top["canonical_name"], "Witchcraft")
        self.assertIn("Los Ratones", top["previous_names"])
        resolved = resolve_team_identity_record(registry, "LR")
        self.assertEqual(resolved["canonical_name"], "Witchcraft")
        aliases = team_identity_candidates(registry, "Los Ratones")
        self.assertIn("Witchcraft", aliases)
        self.assertIn("Los Ratones", aliases)
        leagues = team_identity_league_candidates(registry, "Los Ratones")
        self.assertEqual(leagues[0], "NLC")
        self.assertIn("EMEA MASTERS", leagues)

    def test_upsert_team_and_player_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identity_registry.json"
            path.write_text(json.dumps(empty_identity_registry()), encoding="utf-8")
            upsert_team_identity(
                path,
                {
                    "canonical_name": "My New Team",
                    "current_name": "My New Team",
                    "previous_names": ["Old Org"],
                    "aliases": ["MNT"],
                    "summary": "test",
                },
            )
            upsert_player_identity(
                path,
                {
                    "canonical_name": "Player Current",
                    "current_name": "Player Current",
                    "previous_names": ["Player Old"],
                    "aliases": ["OldNick"],
                    "summary": "test",
                },
            )
            registry = load_identity_registry(path)
            self.assertEqual(len(registry["teams"]), 1)
            self.assertEqual(len(registry["players"]), 1)
            team_hit = search_identity_records(registry, query="Old Org", entity_type="teams")[0]["record"]
            self.assertEqual(team_hit["canonical_name"], "My New Team")
            player_hit = search_identity_records(registry, query="OldNick", entity_type="players")[0]["record"]
            self.assertEqual(player_hit["canonical_name"], "Player Current")
            player_resolved = resolve_player_identity_record(registry, "Player Old")
            self.assertEqual(player_resolved["canonical_name"], "Player Current")
            player_aliases = player_identity_candidates(registry, "OldNick")
            self.assertIn("Player Current", player_aliases)
