# -*- coding: utf-8 -*-
"""Contagem de kills AJUSTADA pelos adversarios (app_core/cruzamento_estatistica_core.py)."""
import unittest

import pandas as pd

from app_core import cruzamento_estatistica_core as ce


def _tg():
    linhas = []
    # "Over" joga muito acima da liga; "A" pegou o Over 3 vezes e fez 32 nas tres
    for i in range(12):
        linhas.append(("g%d" % i, "2026-01-%02d" % (i + 1), "L", "Over", "X%d" % i, 20, 16))
        linhas.append(("g%d" % i, "2026-01-%02d" % (i + 1), "L", "X%d" % i, "Over", 16, 20))
    for i in range(3):
        gid = "a%d" % i
        linhas.append((gid, "2026-02-%02d" % (i + 1), "L", "A", "Over", 16, 16))
        linhas.append((gid, "2026-02-%02d" % (i + 1), "L", "Over", "A", 16, 16))
    for i in range(30):
        gid = "f%d" % i
        linhas.append((gid, "2025-12-%02d" % (i % 28 + 1), "L", "F%d" % i, "G%d" % i, 12, 12))
        linhas.append((gid, "2025-12-%02d" % (i % 28 + 1), "L", "G%d" % i, "F%d" % i, 12, 12))
    return pd.DataFrame(linhas, columns=["gameid", "date", "league", "team", "opponent", "kills_for", "kills_against"])


class TestAjuste(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        if (Path(ce.__file__).resolve().parents[1] / "dota_paths.py").exists():
            self.skipTest("ajuste medido so no LoL; no Dota a contagem segue crua")

    def test_adversario_over_infla_e_o_ajuste_desconta(self):
        tg = ce.com_tendencia_dos_adversarios(_tg())
        a = tg[tg.team == "A"]
        self.assertTrue((a["tend_adv_kills"] > 5).all())          # o Over puxava ~36 contra liga ~25
        r = ce.acertos({"tipo": "total", "lado": "over", "pontos": 29.5, "unidade": "kills"},
                       ce.ultimos(tg, "A"), ce.ultimos(tg, "A").iloc[0:0])
        self.assertEqual(r["acertos"], 3)                          # cru: 32 > 29,5 nas tres
        self.assertLess(r["acertos_ajustados"], 3)                 # ajustado: some
        self.assertGreater(r["adversarios_kills"], 5)

    def test_torres_nao_ajusta(self):
        tg = ce.com_tendencia_dos_adversarios(_tg()).assign(towers_for=5, towers_against=5)
        r = ce.acertos({"tipo": "total", "lado": "over", "pontos": 9.5, "unidade": "towers"},
                       ce.ultimos(tg, "A"), ce.ultimos(tg, "A").iloc[0:0])
        self.assertNotIn("acertos_ajustados", r)


if __name__ == "__main__":
    unittest.main()
