# -*- coding: utf-8 -*-
"""Ligas do Dota (16/11/2026): liga real que a OpenDota marca "excluded" entra no produto, e o que
ficar de fora tem motivo escrito. Regra dele: jogo nao pode faltar calado."""
from __future__ import annotations

import unittest

import pandas as pd

import ligas_dota as ld

TIERS = {1: "professional", 2: "excluded", 3: "excluded", 4: "excluded", 5: "excluded"}


def _base():
    linhas = []
    mid = 0

    def partida(liga_id, liga, a, b, dia):
        nonlocal mid
        mid += 1
        for t in (a, b):
            linhas.append({"match_id": mid, "date": f"2026-{1 + dia // 28:02d}-{1 + dia % 28:02d}T12:00:00Z", "league": liga,
                           "league_id": liga_id, "team_id": t, "teamname": f"Time {t}"})
    for i in range(60):                               # times 10 e 11: 60 partidas em liga profissional
        partida(1, "Liga Pro 2026", 10, 11, i)
    for i in range(30):                               # liga 2: os mesmos times profissionais -> promovida
        partida(2, "European Pro League  2025-2026 Season", 10, 11, 100 + i)
    for i in range(30):                               # liga 3: so amadores
        partida(3, "Copa do Bairro", 90 + i % 3, 95 + i % 3, 100 + i)
    for i in range(3):                                # liga 4: edicao nova da liga 2, amadores por enquanto
        partida(4, "European Pro League  2026-2027 Season", 80, 81, 200 + i)
    for i in range(44):                               # liga 5: o time 20 so joga aqui, contra o 10
        partida(5, "Asgard Championship", 10, 20, 120 + i)
    return pd.DataFrame(linhas)


class TestHisterese(unittest.TestCase):
    """24/11/2026: a Ultras (51,2%) caiu abaixo de 50% com partidas novas e levou 2.881 jogos de uma vez."""

    def _com_liga_no_limite(self):
        d = _base()
        linhas, mid = [], int(d["match_id"].max())
        for i in range(22):                           # liga 6: 18 jogos pro x amador e 4 amador x amador = 41%
            mid += 1
            a, b = (10, 70) if i < 18 else (71, 72)
            for t in (a, b):
                linhas.append({"match_id": mid, "date": f"2026-03-{1 + i:02d}T12:00:00Z", "league": "Ultras Pro League",
                               "league_id": 6, "team_id": t, "teamname": f"Time {t}"})
        return pd.concat([d, pd.DataFrame(linhas)], ignore_index=True), {**TIERS, 6: "excluded"}

    def test_ja_promovida_so_sai_abaixo_de_40(self):
        d, tiers = self._com_liga_no_limite()
        nova = ld.classificar(d, tiers)
        self.assertFalse(nova[6]["promovida"])                     # para ENTRAR precisa de 50%
        mantida = ld.classificar(d, tiers, ja_promovidas={6})
        self.assertTrue(mantida[6]["promovida"])                   # ja promovida: fica (41% >= 40%)
        self.assertIn("so sai abaixo de 40%", mantida[6]["motivo"])


class TestClassificar(unittest.TestCase):
    def setUp(self):
        self.r = ld.classificar(_base(), TIERS)

    def test_raiz_do_nome(self):
        self.assertEqual(ld.raiz_do_nome("European Pro League  2026-2027 Season"), ld.raiz_do_nome("European Pro League 41"))
        self.assertNotEqual(ld.raiz_do_nome("Asgard Championship"), ld.raiz_do_nome("European Pro League"))

    def test_liga_de_times_profissionais_e_promovida(self):
        self.assertTrue(self.r[2]["promovida"])
        self.assertEqual(self.r[2]["pct_pro"], 100.0)

    def test_liga_amadora_fica_fora_com_motivo(self):
        self.assertFalse(self.r[3]["promovida"])
        self.assertIn("0%", self.r[3]["motivo"])

    def test_edicao_nova_herda_pelo_nome(self):
        self.assertTrue(self.r[4]["promovida"])
        self.assertIn("edicao nova", self.r[4]["motivo"])

    def test_liga_profissional_nao_aparece(self):
        self.assertNotIn(1, self.r)

    def test_time_que_so_joga_em_liga_promovida_conta_na_rodada_seguinte(self):
        # liga 5: metade das participacoes e do time 10 (profissional) -> 50% -> promovida
        self.assertTrue(self.r[5]["promovida"])


class TestNadaSomeCalado(unittest.TestCase):
    def test_lista_o_que_ficou_fora_e_por_que(self):
        d = _base()
        tiers = dict(TIERS)
        tiers[2] = "professional"                                     # ja promovida
        itens = ld.fora_do_produto([90], dias=4000, d=d, tiers=tiers, agora=pd.Timestamp("2026-12-01", tz="UTC"))
        self.assertEqual([(i["liga"], i["partidas"]) for i in itens], [("Copa do Bairro", 10)])
        txt = ld.texto_fora_do_produto("Time 90", itens)
        self.assertIn("FORA da análise", txt)
        self.assertIn("Copa do Bairro", txt)
        self.assertEqual(ld.fora_do_produto([10], dias=4000, d=d, tiers={k: "professional" for k in TIERS},
                                            agora=pd.Timestamp("2026-12-01", tz="UTC")), [])
        self.assertEqual(ld.texto_fora_do_produto("x", []), "")

    def test_ids_pelo_nome(self):
        self.assertEqual(ld.ids_do_time("time 10", _base()), {10})
        self.assertEqual(ld.ids_do_time("", _base()), set())

    def test_promocao_vira_professional(self):
        antes = ld.promovidas
        try:
            ld.promovidas = lambda: {2, 5}
            self.assertEqual(ld.aplicar_promocao({1: "professional", 2: "excluded", 3: "excluded"}),
                             {1: "professional", 2: "professional", 3: "excluded", 5: "professional"})
        finally:
            ld.promovidas = antes


if __name__ == "__main__":
    unittest.main()
