# -*- coding: utf-8 -*-
"""Draft ao vivo pela Valve (24/11/2026): leitura do GetLiveLeagueGames. Identico nos repos."""
from __future__ import annotations

import unittest

from app_core import valve_ao_vivo_core as VA

PAPEIS = {"5": {"papeis": ["Support"]}, "26": {"papeis": ["Support", "Disabler"]}}


def _jogo(dur, nw_r=None):
    def lado(herois, contas, nw):
        return {"score": 7, "picks": [{"hero_id": h} for h in herois], "bans": [{"hero_id": 90}, {"hero_id": 91}],
                "players": [{"player_slot": i, "account_id": c, "hero_id": h, "net_worth": (nw[i] if nw else None)}
                            for i, (h, c) in enumerate(zip(herois, contas))]}
    return {"result": {"games": [{
        "match_id": 123, "league_id": 18867,
        "radiant_team": {"team_name": "Team Synapse"}, "dire_team": {"team_name": "Angry Birds"},
        "scoreboard": {"duration": dur,
                       "radiant": lado([5, 81, 16, 33, 50], [1, 2, 3, 4, 5], nw_r),
                       "dire": lado([26, 41, 74, 1, 2], [6, 7, 8, 9, 10], None)}},
        {"match_id": 999, "scoreboard": {"radiant": {"picks": [{"hero_id": 1}]}, "dire": {}}}]}}   # draft incompleto


class TestValve(unittest.TestCase):

    def test_le_draft_lado_e_contas(self):
        js = VA.ler_jogos(_jogo(0), PAPEIS)
        self.assertEqual(len(js), 1)                       # draft incompleto fica de fora
        j = js[0]
        self.assertEqual((j["radiant"], j["dire"]), ("Team Synapse", "Angry Birds"))
        self.assertEqual(j["picks"][0][-1], "5")           # antes do jogo: suporte vai para o fim
        self.assertEqual(j["picks"][1][-1], "26")
        self.assertEqual(j["contas"]["81"], "2")
        self.assertEqual(j["bans"], (["90", "91"], ["90", "91"]))

    def test_jogo_andando_ordena_pelo_patrimonio(self):
        j = VA.ler_jogos(_jogo(900, [1000, 9000, 5000, 3000, 2000]), PAPEIS)[0]
        self.assertEqual(j["picks"][0], ["81", "16", "33", "50", "5"])

    def test_acha_pelo_nome_resolvido(self):
        js = VA.ler_jogos(_jogo(0), PAPEIS)
        res = {"Team Synapse": "Team Synapse", "Angry Birds": "Angry Birds"}.get
        self.assertEqual(VA.achar(js, "Angry Birds", "Team Synapse", lambda n: res(n) or "")["radiant_base"], "Team Synapse")
        self.assertIsNone(VA.achar(js, "X", "Team Synapse", lambda n: res(n) or ""))


if __name__ == "__main__":
    unittest.main()
