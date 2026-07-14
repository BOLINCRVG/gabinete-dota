"""Tests for competition-phase annotation (EWC qualify vs main, etc.)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app_core.competition_phases_core import (
    apply_phase_overrides,
    load_phase_rules,
    apply_competition_phases,
    apply_ewc_regional_phases,
    region_of_league,
)


def _df():
    return pd.DataFrame([
        {"league": "EWC", "year": 2026, "date": "2026-04-20", "split": "", "playoffs": 0},
        {"league": "EWC", "year": 2026, "date": "2026-05-20", "split": "", "playoffs": 0},
        {"league": "LPL", "year": 2026, "date": "2026-05-20", "split": "Split 2", "playoffs": 0},
    ])


class TestApplyPhaseOverrides(unittest.TestCase):

    def test_noop_when_no_rules(self):
        df = _df()
        out = apply_phase_overrides(df, [])
        pd.testing.assert_frame_equal(out, df)

    def test_splits_ewc_by_date(self):
        rules = [
            {"league": "EWC", "year": 2026, "date_from": "2026-04-14", "date_to": "2026-05-01", "split": "Qualify"},
            {"league": "EWC", "year": 2026, "date_from": "2026-05-02", "date_to": "2026-05-26", "split": "Main"},
        ]
        out = apply_phase_overrides(_df(), rules)
        ewc = out[out["league"] == "EWC"].sort_values("date")
        self.assertEqual(list(ewc["split"]), ["Qualify", "Main"])
        # LPL untouched.
        self.assertEqual(out[out["league"] == "LPL"]["split"].iloc[0], "Split 2")

    def test_year_filter_respected(self):
        rules = [{"league": "EWC", "year": 2025, "date_from": "2026-04-14",
                  "date_to": "2026-05-26", "split": "X"}]
        out = apply_phase_overrides(_df(), rules)
        # Year 2025 rule must not touch 2026 rows.
        self.assertTrue((out[out["league"] == "EWC"]["split"] == "").all())

    def test_playoffs_override(self):
        rules = [{"league": "EWC", "date_from": "2026-05-02", "date_to": "2026-05-26",
                  "split": "Main", "playoffs": True}]
        out = apply_phase_overrides(_df(), rules)
        main = out[(out["league"] == "EWC") & (out["split"] == "Main")]
        self.assertTrue(bool(main["playoffs"].iloc[0]))


class TestEwcRegionalPhases(unittest.TestCase):

    def _ewc_df(self):
        # Each team has domestic (non-EWC) games defining its region, plus EWC games.
        rows = [
            # domestic games (define region)
            {"team": "T1", "opponent": "GenG", "league": "LCK", "split": "Split 2", "date": "2026-03-01"},
            {"team": "GenG", "opponent": "T1", "league": "LCK", "split": "Split 2", "date": "2026-03-01"},
            {"team": "FlyQuest", "opponent": "TL", "league": "LCS", "split": "Split 2", "date": "2026-03-01"},
            {"team": "TL", "opponent": "FlyQuest", "league": "LCS", "split": "Split 2", "date": "2026-03-01"},
            # EWC games
            {"team": "T1", "opponent": "GenG", "league": "EWC", "split": "", "date": "2026-05-25"},      # intra KR
            {"team": "FlyQuest", "opponent": "TL", "league": "EWC", "split": "", "date": "2026-04-14"},  # intra NA
            {"team": "T1", "opponent": "FlyQuest", "league": "EWC", "split": "", "date": "2026-05-20"},  # cross
        ]
        return pd.DataFrame(rows)

    def test_intra_region_labeled_qualify(self):
        out = apply_ewc_regional_phases(self._ewc_df())
        ewc = out[out["league"] == "EWC"]
        kr = ewc[(ewc["team"] == "T1") & (ewc["opponent"] == "GenG")]
        na = ewc[(ewc["team"] == "FlyQuest") & (ewc["opponent"] == "TL")]
        self.assertEqual(kr["split"].iloc[0], "Qualify KR")
        self.assertEqual(na["split"].iloc[0], "Qualify NA")

    def test_cross_region_labeled_main(self):
        out = apply_ewc_regional_phases(self._ewc_df())
        cross = out[(out["league"] == "EWC") & (out["team"] == "T1") & (out["opponent"] == "FlyQuest")]
        self.assertEqual(cross["split"].iloc[0], "Main")

    def test_domestic_rows_untouched(self):
        out = apply_ewc_regional_phases(self._ewc_df())
        dom = out[out["league"] == "LCK"]
        self.assertTrue((dom["split"] == "Split 2").all())

    def test_noop_when_no_ewc(self):
        df = pd.DataFrame([{"team": "T1", "opponent": "GenG", "league": "LCK", "split": "S2", "date": "2026-03-01"}])
        out = apply_ewc_regional_phases(df)
        pd.testing.assert_frame_equal(out, df)

    def test_unknown_region_inherits_opponent(self):
        # A team that only appears in EWC (no domestic games → unknown region)
        # must inherit the opponent's region (qualifiers are intra-region), NOT
        # be labeled a spurious cross-region "Main". Mirrors Team Secret x GAM.
        rows = [
            {"team": "GAM", "opponent": "X", "league": "LCP", "split": "S2", "date": "2026-03-01"},
            {"team": "TeamSecret", "opponent": "GAM", "league": "EWC", "split": "", "date": "2026-05-20"},
            {"team": "GAM", "opponent": "TeamSecret", "league": "EWC", "split": "", "date": "2026-05-20"},
        ]
        out = apply_ewc_regional_phases(pd.DataFrame(rows))
        ewc = out[out["league"] == "EWC"]
        self.assertTrue((ewc["split"] == "Qualify APAC").all())
        self.assertNotIn("Main", set(ewc["split"]))

    def test_region_of_league(self):
        self.assertEqual(region_of_league("LCK"), "KR")
        self.assertEqual(region_of_league("LPL"), "CN")
        self.assertEqual(region_of_league("LEC"), "EU")
        self.assertEqual(region_of_league("UNKNOWN"), "UNKNOWN")  # fallback = code


class TestLoadPhaseRules(unittest.TestCase):

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_phase_rules("/nonexistent/path.json"), [])

    def test_incomplete_rules_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "phases.json"
            p.write_text(json.dumps({"rules": [
                {"league": "EWC"},  # incomplete → skip
                {"league": "EWC", "date_from": "2026-04-14", "date_to": "2026-05-01", "split": "Qualify"},
            ]}), encoding="utf-8")
            rules = load_phase_rules(p)
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0]["split"], "Qualify")

    def test_template_file_rules_are_valid(self):
        # 2026-07-05: o arquivo deixou de ser template vazio — carrega a regra
        # REAL do EWC 2026 main event (decisão do usuário). O guard agora é:
        # toda regra shipped precisa ser completa/carregável (nada silencioso).
        template = Path(__file__).resolve().parents[2] / "competition_phases.json"
        if template.exists():
            raw = json.loads(template.read_text(encoding="utf-8"))
            n_raw = len(raw.get("rules", []) or [])
            rules = load_phase_rules(template)
            self.assertEqual(len(rules), n_raw, "regra shipped invalida foi descartada em silencio")
            for rule in rules:
                self.assertIn("league", rule)
                self.assertIn("split", rule)

    def test_apply_competition_phases_noop_missing(self):
        df = _df()
        out = apply_competition_phases(df, path="/nonexistent.json")
        pd.testing.assert_frame_equal(out, df)


if __name__ == "__main__":
    unittest.main()
