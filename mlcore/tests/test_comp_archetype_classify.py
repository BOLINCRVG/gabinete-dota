# -*- coding: utf-8 -*-
"""Trava a classificação de arquétipo de comp (radar_confronto_core).

Bug achado com ele (24/07): comp de engage puro (Malphite/Amumu/Galio/Jinx/Leona)
era classificada como SCALING. A fórmula foi recalibrada; estes testes travam
os casos ÓBVIOS pra não regredir — o "suporte" de draft só vale se a leitura
for verdadeira."""
import unittest

from app_core.champion_knowledge_core import load_champion_knowledge_registry
from app_core.comp_archetype_core import build_comp_profile
from app_core.radar_confronto_core import ROLE_ORDER, classify_comp_archetype
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol


_REG = load_champion_knowledge_registry()


def _arch(picks):
    prof = build_comp_profile(registry=_REG, picks=picks, role_order=list(ROLE_ORDER))
    return classify_comp_archetype(prof)[0]


class ClassificaArquetiposObviosTests(unittest.TestCase):

    @so_no_lol("arquetipos de composicao com campeoes do LoL")
    def test_engage_puro_e_engage_nao_scaling(self):
        # o caso que estava errado: CC pesado / teamfight, sem dive
        self.assertEqual(_arch(["Malphite", "Amumu", "Galio", "Jinx", "Leona"]), "Engage")
        self.assertEqual(_arch(["Ornn", "Sejuani", "Orianna", "Kai'Sa", "Rakan"]), "Engage")

    @so_no_lol("arquetipos de composicao com campeoes do LoL")
    def test_poke_e_poke(self):
        self.assertEqual(_arch(["Jayce", "Nidalee", "Xerath", "Ezreal", "Karma"]), "Poke")
        self.assertEqual(_arch(["Kennen", "Nidalee", "Ziggs", "Ezreal", "Lulu"]), "Poke")

    @so_no_lol("arquetipos de composicao com campeoes do LoL")
    def test_scaling_e_scaling(self):
        self.assertEqual(_arch(["Kayle", "Kindred", "Azir", "Jinx", "Nami"]), "Scaling")

    def test_split_push_e_split(self):
        self.assertEqual(_arch(["Camille", "Trundle", "Yone", "Ezreal", "Braum"]), "Split-push")

    def test_engage_nunca_vira_scaling(self):
        # regra dura: uma comp com engage/frontline altos NÃO pode sair Scaling
        for picks in (["Malphite", "Amumu", "Galio", "Jinx", "Leona"],
                      ["Ornn", "Sejuani", "Orianna", "Kai'Sa", "Rakan"]):
            self.assertNotEqual(_arch(picks), "Scaling")


if __name__ == "__main__":
    unittest.main()
