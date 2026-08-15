# -*- coding: utf-8 -*-
"""Testes dos helpers PUROS do resumo_tab_core (o #2 hotspot de churn).

Item D da fila do teto (20/07): em vez de fatiar o arquivo (indireção que não
move o componente monolito, pois plays_app domina), travamos a lógica pura que
mudava sem rede — derivação de pick, folha de métrica, normalização de lado e o
HTML dos badges temporais. Zero risco (não toca no fonte), ganho real de
testabilidade num arquivo que o usuário edita muito.
"""
from __future__ import annotations

import unittest

from app_core.resumo_tab_core import (
    _metric_leaf,
    _row_to_pick_payload,
    _side_norm,
    _temporal_bars_html,
)


class MetricLeafTests(unittest.TestCase):
    def test_ml_dos_dois_lados_vira_ml(self):
        self.assertEqual(_metric_leaf("00_ml_A"), "ml")
        self.assertEqual(_metric_leaf("ml_B"), "ml")
        self.assertEqual(_metric_leaf("zz_x_ml_A"), "ml")

    def test_combo_mltot_vira_combo(self):
        self.assertEqual(_metric_leaf("zz_mltot_A_kills"), "combo")

    def test_metrica_simples_passa_direto(self):
        self.assertEqual(_metric_leaf("kills"), "kills")

    def test_vazio_nao_quebra(self):
        self.assertEqual(_metric_leaf(""), "")
        self.assertEqual(_metric_leaf(None), "")


class SideNormTests(unittest.TestCase):
    def test_acima_e_over_viram_over(self):
        self.assertEqual(_side_norm("ACIMA"), "Over")
        self.assertEqual(_side_norm("over"), "Over")

    def test_abaixo_e_under_viram_under(self):
        self.assertEqual(_side_norm("ABAIXO"), "Under")
        self.assertEqual(_side_norm("under"), "Under")

    def test_desconhecido_preserva_original(self):
        self.assertEqual(_side_norm("A"), "A")
        self.assertEqual(_side_norm(""), "")


class RowToPickPayloadTests(unittest.TestCase):
    def _row(self, **kw):
        base = {"_metric": "kills", "Lado": "ACIMA", "_line": 27.5}
        base.update(kw)
        return base

    def test_over_de_mercado_normal(self):
        p = _row_to_pick_payload(self._row(), "A", "B", "LCK")
        self.assertEqual(p["metric_key"], "kills")
        self.assertEqual(p["side_key"], "Over")
        self.assertEqual(p["line"], 27.5)
        self.assertEqual((p["team_a"], p["team_b"], p["league"]), ("A", "B", "LCK"))

    def test_ml_a_sem_linha(self):
        p = _row_to_pick_payload(self._row(_metric="00_ml_A", _line=None), "A", "B", "LCK")
        self.assertEqual(p["metric_key"], "ml")
        self.assertEqual(p["side_key"], "A")
        self.assertIsNone(p["line"])

    def test_ml_b_sem_linha(self):
        p = _row_to_pick_payload(self._row(_metric="ml_B", _line=None), "A", "B", "LCK")
        self.assertEqual(p["metric_key"], "ml")
        self.assertEqual(p["side_key"], "B")

    def test_under_normaliza(self):
        p = _row_to_pick_payload(self._row(Lado="ABAIXO"), "A", "B", "LCK")
        self.assertEqual(p["side_key"], "Under")

    def test_linha_nao_numerica_vira_none(self):
        p = _row_to_pick_payload(self._row(_line="x"), "A", "B", "LCK")
        self.assertIsNone(p["line"])


class TemporalBarsHtmlTests(unittest.TestCase):
    """HTML dos badges temporais — trava presença dos elementos-chave sem
    depender de estilos exatos."""

    class _W:
        def __init__(self, label, pct, n, hits):
            self.label, self.pct, self.n, self.hits = label, pct, n, hits

    class _TP:
        def __init__(self, windows, trend, trend_label):
            self.windows, self.trend, self.trend_label = windows, trend, trend_label

    def test_janela_com_dados_mostra_pct_e_contagem(self):
        tp = self._TP([self._W("L5", 80.0, 5, 4)], "improving", "melhorando")
        html = _temporal_bars_html(tp)
        self.assertIn("L5", html)
        self.assertIn("80%", html)
        self.assertIn("(4/5)", html)
        self.assertIn("↑", html)

    def test_janela_vazia_mostra_travessao(self):
        tp = self._TP([self._W("Ano", None, 0, 0)], "flat", "estável")
        html = _temporal_bars_html(tp)
        self.assertIn("Ano —", html)

    def test_cor_muda_por_faixa_de_pct(self):
        alto = _temporal_bars_html(self._TP([self._W("L5", 70.0, 10, 7)], "flat", "x"))
        baixo = _temporal_bars_html(self._TP([self._W("L5", 20.0, 10, 2)], "flat", "x"))
        self.assertIn("#22c55e", alto)   # verde
        self.assertIn("#ef4444", baixo)  # vermelho


if __name__ == "__main__":
    unittest.main()
