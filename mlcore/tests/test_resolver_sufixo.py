# -*- coding: utf-8 -*-
"""Resolvedor de nomes: sufixo "Gaming"/"Esports" (28/10/2026, "Lyon Gaming" x "LYON")."""
import unittest

from app_core.schedule_import_core import resolve_schedule_team

TIMES = ["Invictus Gaming", "LYON", "LYON Academy", "Team Liquid", "G2 Esports", "G2 NORD", "FURIA"]


class TestSufixo(unittest.TestCase):
    def test_casa_tirando_o_sufixo_dos_dois_lados(self):
        self.assertEqual(resolve_schedule_team("Lyon Gaming", TIMES, {}), ("LYON", True))
        self.assertEqual(resolve_schedule_team("Invictus", TIMES, {}), ("Invictus Gaming", True))
        self.assertEqual(resolve_schedule_team("FURIA Esports", TIMES, {}), ("FURIA", True))
        self.assertEqual(resolve_schedule_team("G2", TIMES, {}), ("G2 Esports", True))

    def test_na_duvida_nao_chuta(self):
        self.assertFalse(resolve_schedule_team("Lyon Gaming", TIMES + ["Lyon Esports"], {})[1])
        self.assertFalse(resolve_schedule_team("Paris Gaming", TIMES, {})[1])


if __name__ == "__main__":
    unittest.main()
