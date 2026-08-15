import unittest

from app_core.matchup_league_resolution_core import (
    build_matchup_integrity_row,
    merge_matchup_audit_cases,
    resolve_runtime_team_leagues,
    seed_default_matchup_audit_cases,
)


class TestResolveRuntimeTeamLeagues(unittest.TestCase):

    def test_uses_per_team_inference(self):
        out = resolve_runtime_team_leagues(
            team_a="Partizan Sangal",
            team_b="Unicorns of Love Sexy Edition",
            year=2026,
            league_mode="auto",
            fixed_league="",
            infer_team_league_fn=lambda team, _year: "EBL" if team == "Partizan Sangal" else "PRM",
            infer_match_league_fn=lambda *_args: "EBL",
            tier_from_league_fn=lambda lg: "ERL" if lg else "",
        )
        self.assertEqual(out["league_a"], "EBL")
        self.assertEqual(out["league_b"], "PRM")
        self.assertEqual(out["same_league"], "0")
        self.assertEqual(out["source"], "team_pair")

    def test_falls_back_to_match(self):
        out = resolve_runtime_team_leagues(
            team_a="A",
            team_b="B",
            year=2026,
            league_mode="auto",
            fixed_league="",
            infer_team_league_fn=lambda *_args: "",
            infer_match_league_fn=lambda *_args: "LCK",
            tier_from_league_fn=lambda lg: "T1" if lg else "",
        )
        self.assertEqual(out["league_a"], "LCK")
        self.assertEqual(out["league_b"], "LCK")
        self.assertEqual(out["same_league"], "1")
        self.assertEqual(out["source"], "match_fallback")

    def test_uses_identity_alias_candidates(self):
        out = resolve_runtime_team_leagues(
            team_a="Los Ratones",
            team_b="Partizan Sangal",
            year=2026,
            league_mode="auto",
            fixed_league="",
            infer_team_league_fn=lambda team, _year: "EM" if team == "Witchcraft" else ("EBL" if team == "Partizan Sangal" else ""),
            infer_match_league_fn=lambda ta, tb, _year: "EM" if ta == "Witchcraft" and tb == "Partizan Sangal" else "",
            tier_from_league_fn=lambda lg: "ERL" if lg else "",
            team_aliases_fn=lambda team: ["Witchcraft"] if team == "Los Ratones" else [team],
        )
        self.assertEqual(out["league_a"], "EM")
        self.assertEqual(out["league_b"], "EBL")
        self.assertEqual(out["source"], "team_pair")

    def test_fixed_league_overrides(self):
        out = resolve_runtime_team_leagues(
            team_a="T1",
            team_b="GEN",
            year=2026,
            league_mode="fixed",
            fixed_league="LCK",
            infer_team_league_fn=lambda *_args: "LPL",
            infer_match_league_fn=lambda *_args: "LPL",
            tier_from_league_fn=lambda lg: "T1" if lg else "",
        )
        self.assertEqual(out["league_a"], "LCK")
        self.assertEqual(out["league_b"], "LCK")
        self.assertEqual(out["same_league"], "1")

    def test_returns_dict(self):
        out = resolve_runtime_team_leagues(
            team_a="A",
            team_b="B",
            year=2026,
            league_mode="auto",
            fixed_league="",
            infer_team_league_fn=lambda *_args: "",
            infer_match_league_fn=lambda *_args: "",
            tier_from_league_fn=lambda lg: "",
        )
        self.assertIsInstance(out, dict)


class TestBuildMatchupIntegrityRow(unittest.TestCase):

    def test_flags_collapsed_trace(self):
        row = build_matchup_integrity_row(
            team_a="Partizan Sangal",
            team_b="Unicorns of Love Sexy Edition",
            league_a="EBL",
            league_b="PRM",
            source="team_pair",
            trace_league_a="EBL",
            trace_league_b="EBL",
        )
        self.assertEqual(row["status"], "review")
        self.assertIn("trace_collapsed_to_same_league", row["issues"])

    def test_ok_when_no_issues(self):
        row = build_matchup_integrity_row(
            team_a="T1",
            team_b="GEN",
            league_a="LCK",
            league_b="LCK",
            source="team_pair",
            trace_league_a="LCK",
            trace_league_b="LCK",
        )
        self.assertEqual(row["status"], "ok")

    def test_returns_dict(self):
        row = build_matchup_integrity_row(
            team_a="A",
            team_b="B",
            league_a="LCK",
            league_b="LCK",
            source="match",
            trace_league_a="LCK",
            trace_league_b="LCK",
        )
        self.assertIsInstance(row, dict)


class TestSeedAndMergeAuditCases(unittest.TestCase):

    def test_seed_returns_list(self):
        cases = seed_default_matchup_audit_cases()
        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)

    def test_seed_each_has_team_a_and_b(self):
        for case in seed_default_matchup_audit_cases():
            self.assertIn("team_a", case)
            self.assertIn("team_b", case)

    def test_merge_deduplicates(self):
        base = [{"team_a": "T1", "team_b": "GEN", "year": 2026}]
        extra = [{"team_a": "T1", "team_b": "GEN", "year": 2026}]
        merged = merge_matchup_audit_cases(base, extra)
        self.assertEqual(len(merged), 1)

    def test_merge_combines_unique(self):
        base = [{"team_a": "T1", "team_b": "GEN", "year": 2026}]
        extra = [{"team_a": "JDG", "team_b": "BLG", "year": 2026}]
        merged = merge_matchup_audit_cases(base, extra)
        self.assertEqual(len(merged), 2)

    def test_merge_skips_empty_teams(self):
        cases = [{"team_a": "", "team_b": "GEN", "year": 2026}]
        merged = merge_matchup_audit_cases(cases)
        self.assertEqual(len(merged), 0)


if __name__ == "__main__":
    unittest.main()
