# -*- coding: utf-8 -*-
"""Linha descritiva do perfil da comp (24/11/2026). Identico nos repos."""
from __future__ import annotations

import unittest

from app_core import perfil_comp_core as PC


class TestPerfilComp(unittest.TestCase):

    def test_dota(self):
        p = {"1": {"papeis": ["Initiator", "Disabler"], "ranged": False, "alcance": 150},
             "2": {"papeis": ["Initiator", "Support"], "ranged": True, "alcance": 600}}
        t = PC.descrever("dota", ["1", "2", "9"], p)
        self.assertTrue(t.startswith("Iniciação 2 · Controle 1"))
        self.assertIn("1 de alcance (médio 375)", t)
        self.assertIn("1 sem perfil", t)                 # nada some calado

    def test_lol_pela_chave_normalizada(self):
        p = {"wukong": {"classes": ["Diver"], "ranged": False, "alcance": 175, "magico": False},
             "orianna": {"classes": ["Burst"], "longo_alcance": True, "ranged": True, "alcance": 525, "magico": True}}
        t = PC.descrever("lol", ["MonkeyKing", "Orianna"], p)
        self.assertIn("Engage 1", t)
        self.assertIn("Poke 1", t)
        self.assertIn("dano 1 mág./1 fís.", t)


if __name__ == "__main__":
    unittest.main()
