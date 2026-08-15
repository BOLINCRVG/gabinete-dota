import unittest

from mlcore.regression_fixtures import (
    build_players_fixture_key,
    resolve_odds_fixture_snapshot,
    resolve_players_fixture_snapshot,
    upsert_odds_fixture_snapshot,
    upsert_players_fixture_snapshot,
)


class TestBuildPlayersFixtureKey(unittest.TestCase):

    def test_key_format(self):
        key = build_players_fixture_key("a1", "p1", "c1")
        self.assertEqual(key, "a=a1|p=p1|c=c1")

    def test_empty_strings(self):
        key = build_players_fixture_key("", "", "")
        self.assertEqual(key, "a=|p=|c=")

    def test_strips_whitespace(self):
        key = build_players_fixture_key(" a1 ", " p1 ", " c1 ")
        self.assertEqual(key, "a=a1|p=p1|c=c1")

    def test_returns_string(self):
        self.assertIsInstance(build_players_fixture_key("a", "b", "c"), str)


class TestResolveOddsFixtureSnapshot(unittest.TestCase):

    def test_legacy_match(self):
        legacy = {"artifact_sha256": "sha-a", "matchups": [{"team_a": "A"}]}
        self.assertIsNotNone(resolve_odds_fixture_snapshot(legacy, "sha-a"))

    def test_legacy_no_match(self):
        legacy = {"artifact_sha256": "sha-a", "matchups": [{"team_a": "A"}]}
        self.assertIsNone(resolve_odds_fixture_snapshot(legacy, "sha-b"))

    def test_empty_sha_returns_none(self):
        self.assertIsNone(resolve_odds_fixture_snapshot({}, ""))

    def test_empty_payload_returns_none(self):
        self.assertIsNone(resolve_odds_fixture_snapshot({}, "sha-x"))

    def test_versioned_format(self):
        payload = upsert_odds_fixture_snapshot({}, "sha-a", {"matchups": [{"team_a": "A"}]})
        snap = resolve_odds_fixture_snapshot(payload, "sha-a")
        self.assertIsNotNone(snap)
        self.assertEqual((snap or {}).get("artifact_sha256"), "sha-a")

    def test_versioned_wrong_sha(self):
        payload = upsert_odds_fixture_snapshot({}, "sha-a", {"matchups": [{"team_a": "A"}]})
        self.assertIsNone(resolve_odds_fixture_snapshot(payload, "sha-z"))


class TestUpsertOddsFixtureSnapshot(unittest.TestCase):

    def test_schema_set(self):
        payload = upsert_odds_fixture_snapshot({}, "sha-a", {"matchups": [{"team_a": "A"}]})
        self.assertEqual(payload.get("fixture_schema"), "artifact_snapshot_v1")

    def test_latest_sha_set(self):
        payload = upsert_odds_fixture_snapshot({}, "sha-a", {"matchups": [{"team_a": "A"}]})
        self.assertEqual(payload.get("latest_artifact_sha256"), "sha-a")

    def test_missing_matchups_list_noop(self):
        payload = upsert_odds_fixture_snapshot({}, "sha-a", {"other": "data"})
        self.assertEqual(payload, {})

    def test_multiple_snapshots_preserved(self):
        p = upsert_odds_fixture_snapshot({}, "sha-a", {"matchups": [{"team_a": "A"}]})
        p = upsert_odds_fixture_snapshot(p, "sha-b", {"matchups": [{"team_a": "B"}]})
        self.assertIsNotNone(resolve_odds_fixture_snapshot(p, "sha-a"))
        self.assertIsNotNone(resolve_odds_fixture_snapshot(p, "sha-b"))

    def test_returns_dict(self):
        result = upsert_odds_fixture_snapshot({}, "sha-a", {"matchups": []})
        self.assertIsInstance(result, dict)


class TestPlayersFixtureRoundtrip(unittest.TestCase):

    def _upsert(self, a="a1", p="p1", c="c1"):
        return upsert_players_fixture_snapshot(
            {},
            artifact_sha256=a,
            players_artifact_sha256=p,
            csv_sha256=c,
            snapshot={"matchups": [{"team_a": "A", "team_b": "B"}]},
        )

    def test_schema_set(self):
        payload = self._upsert()
        self.assertEqual(payload.get("fixture_schema"), "artifact_players_csv_snapshot_v1")

    def test_latest_snapshot_key(self):
        payload = self._upsert()
        expected = build_players_fixture_key("a1", "p1", "c1")
        self.assertEqual(payload.get("latest_snapshot_key"), expected)

    def test_resolve_match(self):
        payload = self._upsert()
        self.assertIsNotNone(resolve_players_fixture_snapshot(payload, "a1", "p1", "c1"))

    def test_resolve_wrong_csv_sha(self):
        payload = self._upsert()
        self.assertIsNone(resolve_players_fixture_snapshot(payload, "a1", "p1", "c2"))

    def test_legacy_match(self):
        legacy = {
            "artifact_sha256": "a1",
            "players_artifact_sha256": "p1",
            "csv_sha256": "c1",
            "matchups": [{"team_a": "A"}],
        }
        self.assertIsNotNone(resolve_players_fixture_snapshot(legacy, "a1", "p1", "c1"))

    def test_legacy_no_match(self):
        legacy = {
            "artifact_sha256": "a1",
            "players_artifact_sha256": "p1",
            "csv_sha256": "c1",
            "matchups": [{"team_a": "A"}],
        }
        self.assertIsNone(resolve_players_fixture_snapshot(legacy, "a2", "p1", "c1"))

    def test_no_matchups_list_noop(self):
        payload = upsert_players_fixture_snapshot(
            {}, artifact_sha256="a1", players_artifact_sha256="p1",
            csv_sha256="c1", snapshot={"other": "data"}
        )
        self.assertEqual(payload, {})


if __name__ == "__main__":
    unittest.main()

