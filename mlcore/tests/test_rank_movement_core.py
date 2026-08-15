# -*- coding: utf-8 -*-
"""Movimento do ranking (↑/↓) — o elemento que a pesquisa (22/07) apontou como
o nº1 de uma ranking de esports. Testa o cálculo puro + o ciclo de snapshot."""
from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app_core.rank_movement_core import (
    compute_movement,
    load_snapshot,
    movement_badge_html,
    rank_map_from_df,
    save_snapshot,
    snapshot_is_stale,
)


class RankMapTests(unittest.TestCase):
    def test_extrai_time_para_posicao(self):
        df = pd.DataFrame({"rank": [1, 2, 3], "team": ["Gen.G", "T1", "HLE"]})
        m = rank_map_from_df(df)
        self.assertEqual(m[list(m)[0]], 1)
        self.assertEqual(len(m), 3)

    def test_usa_a_ordem_quando_nao_ha_coluna_rank(self):
        df = pd.DataFrame({"team": ["A", "B"]})
        m = rank_map_from_df(df)
        self.assertEqual(set(m.values()), {1, 2})

    def test_vazio_nao_quebra(self):
        self.assertEqual(rank_map_from_df(pd.DataFrame()), {})
        self.assertEqual(rank_map_from_df(None), {})


class MovementTests(unittest.TestCase):
    def test_subir_da_positivo(self):
        atual = {"geng": 1, "t1": 2}
        snap = {"ranks": {"geng": 3, "t1": 2}}
        mv = compute_movement(atual, snap)
        self.assertEqual(mv["geng"]["mov"], 2)   # 3 -> 1 = subiu 2
        self.assertFalse(mv["geng"]["novo"])

    def test_cair_da_negativo(self):
        mv = compute_movement({"t1": 5}, {"ranks": {"t1": 2}})
        self.assertEqual(mv["t1"]["mov"], -3)

    def test_parado_da_zero(self):
        mv = compute_movement({"a": 4}, {"ranks": {"a": 4}})
        self.assertEqual(mv["a"]["mov"], 0)

    def test_time_novo_marcado(self):
        mv = compute_movement({"novo": 10}, {"ranks": {"a": 1}})
        self.assertTrue(mv["novo"]["novo"])

    def test_aceita_snapshot_sem_wrapper(self):
        mv = compute_movement({"a": 1}, {"a": 3})
        self.assertEqual(mv["a"]["mov"], 2)


class BadgeTests(unittest.TestCase):
    def test_subiu_verde_com_seta_cima(self):
        h = movement_badge_html({"mov": 2, "novo": False})
        self.assertIn("▲2", h)
        self.assertIn("#22c55e", h)

    def test_caiu_vermelho(self):
        h = movement_badge_html({"mov": -1, "novo": False})
        self.assertIn("▼1", h)
        self.assertIn("#ef4444", h)

    def test_novo_tem_selo(self):
        self.assertIn("✦", movement_badge_html({"novo": True}))

    def test_none_nao_quebra(self):
        self.assertIn("—", movement_badge_html(None))


class SnapshotCicloTests(unittest.TestCase):
    def test_salva_e_carrega(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "snap.json"
            self.assertTrue(save_snapshot(p, {"a": 1, "b": 2}))
            snap = load_snapshot(p)
            self.assertEqual(snap["ranks"]["a"], 1)
            self.assertEqual(snap["date"], date.today().isoformat())

    def test_ausente_carrega_vazio(self):
        self.assertEqual(load_snapshot("/nao/existe/x.json"), {})

    def test_snapshot_novo_de_hoje_nao_e_stale(self):
        snap = {"date": date.today().isoformat(), "ranks": {"a": 1}}
        self.assertFalse(snapshot_is_stale(snap, min_age_days=7))

    def test_snapshot_velho_e_stale(self):
        old = (date.today() - timedelta(days=10)).isoformat()
        self.assertTrue(snapshot_is_stale({"date": old, "ranks": {"a": 1}}, min_age_days=7))

    def test_snapshot_vazio_e_stale(self):
        self.assertTrue(snapshot_is_stale({}, min_age_days=7))
        self.assertTrue(snapshot_is_stale({"ranks": {}}, min_age_days=7))


if __name__ == "__main__":
    unittest.main()
