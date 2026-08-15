import unittest

from app_core.identity_registry_audit_core import audit_identity_registry
from mlcore.tiers import canonical_league, is_international_league


class IdentityRegistryAuditCoreTests(unittest.TestCase):
    def test_warns_on_missing_sources_and_current_leagues(self) -> None:
        report = audit_identity_registry(
            {
                "teams": [
                    {
                        "entity_key": "foo",
                        "canonical_name": "Foo Team",
                        "current_name": "Foo Team",
                        "aliases": ["Foo Team"],
                        "relation_type": "sporting_successor",
                        "confidence": "high",
                        "current_leagues": [],
                        "source_urls": [],
                    }
                ],
                "players": [],
            },
            canonicalize_league_fn=canonical_league,
            is_international_league_fn=is_international_league,
        )
        self.assertEqual(report["summary"]["warn_count"], 1)
        row = report["team_rows"][0]
        self.assertEqual(row["status"], "warn")
        self.assertIn("missing_current_leagues", row["issues"])
        self.assertIn("missing_sources", row["issues"])

    def test_flags_team_alias_collision_as_error(self) -> None:
        report = audit_identity_registry(
            {
                "teams": [
                    {
                        "entity_key": "alpha",
                        "canonical_name": "Alpha",
                        "current_name": "Alpha",
                        "aliases": ["Shared Alias"],
                        "current_leagues": ["LCK"],
                        "source_urls": ["https://example.com/a"],
                    },
                    {
                        "entity_key": "beta",
                        "canonical_name": "Beta",
                        "current_name": "Beta",
                        "aliases": ["Shared Alias"],
                        "current_leagues": ["LPL"],
                        "source_urls": ["https://example.com/b"],
                    },
                ],
                "players": [],
            },
            canonicalize_league_fn=canonical_league,
            is_international_league_fn=is_international_league,
        )
        self.assertEqual(report["summary"]["error_count"], 2)
        self.assertTrue(all(row["status"] == "error" for row in report["team_rows"]))


if __name__ == "__main__":
    unittest.main()
