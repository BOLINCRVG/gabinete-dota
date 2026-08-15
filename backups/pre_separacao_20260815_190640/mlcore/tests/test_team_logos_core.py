"""Tests for app_core/team_logos_core.py — resolver de logo com fallback."""
from __future__ import annotations

import unittest

from app_core.team_logos_core import _norm, has_logo, team_logo_html


class TestNorm(unittest.TestCase):

    def test_normalization(self):
        self.assertEqual(_norm("Gen.G"), "geng")
        self.assertEqual(_norm("T1"), "t1")
        self.assertEqual(_norm("  LNG Esports "), "lngesports")
        self.assertEqual(_norm("Bilibili Gaming"), "bilibiligaming")

    def test_mojibake_and_empty(self):
        self.assertEqual(_norm("L�S"), "ls")
        self.assertEqual(_norm(None), "")
        self.assertEqual(_norm("   "), "")


class TestLogoFallback(unittest.TestCase):

    def test_no_asset_has_logo_false(self):
        # Nenhum logo de teste no repo → has_logo False (não quebra).
        self.assertFalse(has_logo("Gen.G__inexistente__xyz"))

    def test_fallback_to_badge(self):
        # Sem arquivo → cai no badge de iniciais (sem <img>).
        # Badge é <span> (2026-06: div-em-span quebrava o header da Visão Geral).
        html = team_logo_html("Gen.G__inexistente__xyz", 40)
        self.assertNotIn("<img", html)
        self.assertIn("span", html.lower())

    def test_empty_team(self):
        # Nome vazio não deve quebrar.
        html = team_logo_html("", 40)
        self.assertIsInstance(html, str)


if __name__ == "__main__":
    unittest.main()
