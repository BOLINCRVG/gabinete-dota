# -*- coding: utf-8 -*-
"""Torneio de exibição (KeSPA etc.) — aviso na UI (achado 24/07)."""
import unittest

from app_core.exhibition_league_core import exhibition_warning, is_exhibition_league


class ExhibitionLeagueTests(unittest.TestCase):
    def test_kespa_e_exibicao(self):
        for lg in ("KeSPA Cup", "KeSPA", "kespa cup", "LCK KeSPA Cup"):
            self.assertTrue(is_exhibition_league(lg), lg)

    def test_demacia_e_allstar_tambem(self):
        self.assertTrue(is_exhibition_league("Demacia Cup"))
        self.assertTrue(is_exhibition_league("All-Star Event"))

    def test_ligas_serias_nao_sao(self):
        for lg in ("LCK", "LPL", "LEC", "LCS", "MSI", "WLDs", "LCKC", "EWC", ""):
            self.assertFalse(is_exhibition_league(lg), lg)

    def test_none_nao_quebra(self):
        self.assertFalse(is_exhibition_league(None))
        self.assertEqual(exhibition_warning(None), "")

    def test_aviso_so_para_exibicao(self):
        self.assertIn("exibição", exhibition_warning("KeSPA Cup"))
        self.assertEqual(exhibition_warning("LCK"), "")


class SiglasInternacionaisTests(unittest.TestCase):
    """Varredura 24/07: o registro so conhecia nomes por extenso; as SIGLAS do
    Oracle's Elixir caiam fora (mesma familia do 'WLDs'). Cada uma verificada
    nos dados: junta times de 4-19 ligas domesticas."""

    def test_siglas_de_torneio_sao_internacionais(self):
        from mlcore.tiers import is_international_league
        for lg in ("EM", "AC", "IC", "Asia Master", "EPL", "CD", "WLDs", "MSC"):
            self.assertTrue(is_international_league(lg), lg)

    def test_ligas_domesticas_nao_viram_internacionais(self):
        from mlcore.tiers import is_international_league
        for lg in ("LCK", "LPL", "LEC", "LCS", "LFL", "PRM", "LCKC", "NACL",
                   "CBLOL", "LLA", "PCS", "VCS", "LJL", "TCL", "LIT", "EBL",
                   "HLL", "LTA N", "LTA S", "LDL", "EM Masters BR"):
            self.assertFalse(is_international_league(lg), lg)
