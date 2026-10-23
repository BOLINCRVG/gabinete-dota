# -*- coding: utf-8 -*-
"""Jogo do Mundial nao acusa "liga errada" (23/10/2026).

LOS (CBLOL) contra um time da LCK, vindo da agenda como "Mundial", recebia
"este confronto e da LCK, confira o time". Arquivo identico nos dois repos."""
from __future__ import annotations

import inspect
import unittest

from app_core import matchup_selection_core as sel
from app_core.matchup_selection_core import eh_evento_internacional

LIGAS = {"LOS": "CBLOL", "T1": "LCK", "Gen.G": "LCK", "paiN Gaming": "CBLOL"}


class TestEventoInternacional(unittest.TestCase):

    def test_mundial_e_internacional(self):
        self.assertTrue(eh_evento_internacional("Mundial", LIGAS))

    def test_liga_domestica_nao_e(self):
        self.assertFalse(eh_evento_internacional("LCK", LIGAS))
        self.assertFalse(eh_evento_internacional("cblol", LIGAS))

    def test_sem_liga_na_agenda_nao_afirma_nada(self):
        self.assertFalse(eh_evento_internacional("", LIGAS))
        self.assertFalse(eh_evento_internacional("Mundial", {}))

    def test_a_tela_usa_a_regra(self):
        fonte = inspect.getsource(sel)
        self.assertIn("eh_evento_internacional(_liga_agenda, _ligas)", fonte)
        self.assertIn('session_state["_agenda_liga_do_jogo"]', fonte)


if __name__ == "__main__":
    unittest.main()
