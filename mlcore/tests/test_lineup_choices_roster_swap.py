# -*- coding: utf-8 -*-
"""Simulador de troca de roster (pedido dele 22/07).

"todo split tem mudança de roster; quero colocar tal player em tal time para
ver se o ML ainda é condizente com a realidade."

O seletor de lineup já existia, mas as opções eram SÓ de quem já jogou naquele
time — dava para testar reserva, não transferência. Com include_all_teams=True
o pool inteiro da posição entra. Estes testes travam as duas coisas que
importam: (a) nada muda com a flag desligada; (b) com ela ligada, os do próprio
time continuam vindo primeiro.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.lineup_runtime_core import build_ml_lineup_choices

POS = ["Top", "Jungle", "Mid", "Bot", "Support"]
_MAP = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}


def _npos(p):
    return _MAP.get(str(p).strip().lower(), "")


def _df():
    return pd.DataFrame([
        {"team": "T1", "player": "Faker", "playerid": "faker", "pos": "mid", "gameid": "g1", "date": "2026-01-01"},
        {"team": "T1", "player": "Faker", "playerid": "faker", "pos": "mid", "gameid": "g2", "date": "2026-01-08"},
        {"team": "T1", "player": "Poby", "playerid": "poby", "pos": "mid", "gameid": "g3", "date": "2026-01-15"},
        {"team": "T1", "player": "Oner", "playerid": "oner", "pos": "jng", "gameid": "g1", "date": "2026-01-01"},
        {"team": "HLE", "player": "Zeka", "playerid": "zeka", "pos": "mid", "gameid": "g4", "date": "2026-02-01"},
        {"team": "HLE", "player": "Zeus", "playerid": "zeus", "pos": "top", "gameid": "g4", "date": "2026-02-01"},
    ])


def _call(team, **kw):
    return build_ml_lineup_choices(df_pr=_df(), team_ui=team, pos_canon_ml=POS,
                                   norm_pos_ml_fn=_npos, **kw)


class FlagDesligadaTests(unittest.TestCase):
    """Comportamento histórico intacto — nenhuma regressão."""

    def test_so_jogadores_do_proprio_time(self):
        out, _ = _call("T1")
        self.assertEqual(set(out["Mid"]), {"faker", "poby"})
        self.assertNotIn("zeka", out["Mid"])

    def test_posicao_sem_jogador_no_time_fica_vazia(self):
        out, _ = _call("T1")
        self.assertEqual(out["Top"], [])  # T1 não tem top no fixture

    def test_ordena_por_relevancia(self):
        # Faker tem 2 jogos, Poby 1 -> Faker primeiro
        out, _ = _call("T1")
        self.assertEqual(out["Mid"][0], "faker")


class FlagLigadaTests(unittest.TestCase):
    """Pool completo — permite simular transferência."""

    def test_traz_jogador_de_outro_time(self):
        out, _ = _call("T1", include_all_teams=True)
        self.assertIn("zeka", out["Mid"])

    def test_do_proprio_time_vem_primeiro(self):
        """Crítico p/ a UX: o lineup real não pode ser empurrado para baixo."""
        out, _ = _call("T1", include_all_teams=True)
        self.assertEqual(out["Mid"][:2], ["faker", "poby"])

    def test_posicao_vazia_no_time_recebe_o_pool(self):
        out, _ = _call("T1", include_all_teams=True)
        self.assertEqual(out["Top"], ["zeus"])  # Zeus, da HLE

    def test_rotulo_mostra_o_time_atual_do_de_fora(self):
        _, disp = _call("T1", include_all_teams=True)
        self.assertIn("HLE", disp.get("zeus", ""))
        self.assertIn("Zeus", disp.get("zeus", ""))

    def test_nao_duplica_quem_ja_esta_no_time(self):
        out, _ = _call("T1", include_all_teams=True)
        self.assertEqual(out["Mid"].count("faker"), 1)

    def test_time_sem_historico_ainda_recebe_pool(self):
        # time novo (sem jogos): com a flag, ainda dá para montar o lineup
        out, _ = _call("TimeNovo", include_all_teams=True)
        self.assertTrue(out["Mid"])
        self.assertTrue(out["Top"])


class BordasTests(unittest.TestCase):
    def test_df_vazio_nao_quebra(self):
        out, disp = build_ml_lineup_choices(df_pr=pd.DataFrame(), team_ui="T1",
                                            pos_canon_ml=POS, norm_pos_ml_fn=_npos,
                                            include_all_teams=True)
        self.assertEqual(out, {p: [] for p in POS})
        self.assertEqual(disp, {})

    def test_df_none_nao_quebra(self):
        out, _ = build_ml_lineup_choices(df_pr=None, team_ui="T1", pos_canon_ml=POS,
                                          norm_pos_ml_fn=_npos, include_all_teams=True)
        self.assertEqual(out, {p: [] for p in POS})


if __name__ == "__main__":
    unittest.main()
