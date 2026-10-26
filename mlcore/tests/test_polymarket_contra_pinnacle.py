# -*- coding: utf-8 -*-
"""Polymarket x Pinnacle (26/10/2026): a ultima leitura pre-jogo de cada evento,
e o placar no formato das outras reguas."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import tools.polymarket_contra_pinnacle as pc
except ImportError as _sem_ferramenta:
    import unittest as _u
    raise _u.SkipTest("tools.polymarket_contra_pinnacle nao existe neste repositorio: " + str(_sem_ferramenta))
from tools.regua_pinnacle_comum import resumo


class TestRegua(unittest.TestCase):

    def test_fica_a_ultima_leitura_de_cada_evento(self):
        with tempfile.TemporaryDirectory() as d:
            arq = Path(d) / "h.jsonl"
            arq.write_text("\n".join(json.dumps(x) for x in [
                {"evento": "a", "coletado_em": "2026-10-23T20:00", "p_a": 0.4},
                {"evento": "a", "coletado_em": "2026-10-23T21:00", "p_a": 0.45},
                {"evento": "b", "coletado_em": "2026-10-23T21:00", "p_a": 0.7}]), encoding="utf-8")
            with mock.patch.object(pc, "HISTORICO_POLY", arq):
                u = pc.ultima_poly_por_jogo()
        self.assertEqual(u["a"]["p_a"], 0.45)
        self.assertEqual(len(u), 2)

    def test_placar_polymarket_melhor_tem_diferenca_negativa(self):
        casos = [{"partida": i, "p_poly": 0.8, "p_casa": 0.6, "y": 1} for i in range(10)]
        r = resumo(casos, "p_poly", minimo=pc.MINIMO)
        self.assertLess(r["diferenca"], 0)
        self.assertFalse(r["conclusivo"])      # 10 de 200: ainda nao


if __name__ == "__main__":
    unittest.main()
