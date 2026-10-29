"""K/D/A de jogador pelo confronto (29/10/2026)."""
import unittest

import pandas as pd

from app_core import player_props_confronto_core as m


def _jogos():
    rows = []
    for i in range(20):
        # time forte A: 20 kills, 8 mortes; jogador "Faker" faz 30% das kills
        rows.append({"gameid": f"g{i}", "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), "team": "A",
                     "player": "Faker", "pos": "mid", "kills": 6, "deaths": 2, "assists": 8,
                     "teamkills": 20, "teamdeaths": 8})
        rows.append({"gameid": f"h{i}", "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), "team": "B",
                     "player": "Bdl", "pos": "mid", "kills": 2, "deaths": 4, "assists": 4,
                     "teamkills": 10, "teamdeaths": 20})
    return pd.DataFrame(rows)


class TestPropsPeloConfronto(unittest.TestCase):
    def test_fatia_e_nao_media(self):
        d = _jogos()
        pr = m.priors_por_rota(d)
        tx = m.taxas_do_jogador(d, "Faker", "mid", pr)
        self.assertAlmostEqual(tx["media"]["kills"], 6.0)
        tot = m.total_do_confronto(d, "A", "B")
        self.assertAlmostEqual(tot, 29.0)
        # contra um adversario do mesmo nivel (50%), a projecao cai bem abaixo da media de 6
        r50 = m.linha_de_prop(jogador="Faker", rota="mid", time="A", stat="kills", linha=None,
                              taxas=tx, total=tot, p_time=0.5)
        self.assertLess(r50["proj"], 5.0)
        r80 = m.linha_de_prop(jogador="Faker", rota="mid", time="A", stat="kills", linha=4.5,
                              taxas=tx, total=tot, p_time=0.8)
        self.assertGreater(r80["proj"], r50["proj"])
        self.assertAlmostEqual(r80["p_over"] + r80["p_under"], 1.0)
        self.assertEqual(r50["line"], r50["linha_justa"])

    def test_prob_mais(self):
        self.assertAlmostEqual(m.prob_mais(3.0, 2.5), 0.5318, places=3)
        self.assertEqual(m.prob_mais(3.0, -0.5), 1.0)
        self.assertGreater(m.prob_mais(5.0, 2.5), m.prob_mais(3.0, 2.5))


if __name__ == "__main__":
    unittest.main()
