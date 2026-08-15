# -*- coding: utf-8 -*-
"""Testes do Pool do confronto (redesign Campeões, 2026-07-16)."""
import unittest

import pandas as pd

from app_core.matchup_pool_core import (
    build_team_pool,
    cross_pool_with_meta,
    pool_concentration,
    pool_verdict_lines,
)


def _rows():
    # T1 joga muito Corki (4-1) e Azir (2-1); Gen.G só Rumble (0-2)
    data = []
    for i, (team, champ, res, player) in enumerate([
        ("T1", "Corki", 1, "Faker"), ("T1", "Corki", 1, "Faker"),
        ("T1", "Corki", 1, "Faker"), ("T1", "Corki", 1, "Faker"),
        ("T1", "Corki", 0, "Faker"),
        ("T1", "Azir", 1, "Faker"), ("T1", "Azir", 1, "Faker"),
        ("T1", "Azir", 0, "Faker"),
        ("Gen.G", "Rumble", 0, "Kiin"), ("Gen.G", "Rumble", 0, "Kiin"),
    ]):
        data.append({
            "team": team, "champion": champ, "result": res,
            "playername": player, "date": f"2026-06-{i + 1:02d}", "year": 2026,
        })
    return pd.DataFrame(data)


class TestBuildTeamPool(unittest.TestCase):
    def test_pool_counts_and_order(self):
        p = build_team_pool(_rows(), "T1", year=2026)
        self.assertEqual(p.iloc[0]["champion"], "Corki")
        self.assertEqual(int(p.iloc[0]["games"]), 5)
        self.assertAlmostEqual(float(p.iloc[0]["wr"]), 0.8)
        self.assertEqual(p.iloc[0]["player"], "Faker")
        self.assertEqual(len(p), 2)

    def test_team_match_case_insensitive_and_year_filter(self):
        p = build_team_pool(_rows(), "t1 ", year=2025)
        self.assertTrue(p.empty)
        p2 = build_team_pool(_rows(), "GEN.G", year=2026)
        self.assertEqual(len(p2), 1)

    def test_empty_input_never_explodes(self):
        self.assertTrue(build_team_pool(None, "T1").empty)
        self.assertTrue(build_team_pool(pd.DataFrame(), "T1").empty)


class TestMetaCrossAndVerdict(unittest.TestCase):
    def test_cross_adds_tier(self):
        p = build_team_pool(_rows(), "T1", year=2026)
        pm = cross_pool_with_meta(p, {"Corki": {"tier": "S", "presence": 70.0}})
        self.assertEqual(pm[pm["champion"] == "Corki"].iloc[0]["tier"], "S")
        self.assertEqual(pm[pm["champion"] == "Azir"].iloc[0]["tier"], "")

    def test_concentration_flags_shallow_pool(self):
        p = build_team_pool(_rows(), "T1", year=2026)
        conc = pool_concentration(p)
        self.assertAlmostEqual(conc, 1.0)  # top-3 cobre tudo (só 2 champs)

    def test_verdict_mentions_comfort_vs_meta(self):
        p = cross_pool_with_meta(
            build_team_pool(_rows(), "T1", year=2026),
            {"Corki": {"tier": "S"}},
        )
        lines = pool_verdict_lines("T1", p)
        self.assertTrue(any("comfort" in ln for ln in lines), lines)
        self.assertTrue(any("1/1" in ln for ln in lines), lines)

    def test_verdict_empty_pool(self):
        lines = pool_verdict_lines("X", pd.DataFrame())
        self.assertTrue(any("sem amostra" in ln.lower() for ln in lines))


if __name__ == "__main__":
    unittest.main()
