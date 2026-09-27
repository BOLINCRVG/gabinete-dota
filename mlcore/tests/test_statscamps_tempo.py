# -*- coding: utf-8 -*-
"""Testes do recorte de TEMPO na tela de stats do campeonato.

Pedido do dono em 13/09/2026: a tela tinha recorte de kills, torres, dragoes,
nashors e inibidores, mas nao de TEMPO — justamente a linha que ele usa
(32:30). O mercado "Tempo" ja existia no registro central, mas com a lista de
limiares VAZIA, entao nenhuma coluna aparecia.

Estes testes travam as tres partes: a linha padrao, o entendimento de "32:30"
e a coluna saindo na tabela com o rotulo em horario.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.market import parse_time_lines_text
from app_core.stats_config import STATS_MARKET_CONFIG


class TestLinhaPadrao(unittest.TestCase):

    def test_o_mercado_tempo_tem_3230_como_padrao(self):
        cfg = STATS_MARKET_CONFIG["Tempo"]
        self.assertEqual(str(cfg["default_thresholds"]), "32.5")

    def test_3230_e_o_mesmo_que_325(self):
        self.assertEqual(parse_time_lines_text("32:30"), [32.5])

    def test_o_padrao_bate_com_a_linha_padrao_do_mercado(self):
        """Os dois campos do registro nao podem discordar entre si."""
        cfg = STATS_MARKET_CONFIG["Tempo"]
        self.assertIn(float(cfg["default_line"]),
                      parse_time_lines_text(str(cfg["default_thresholds"])))

    def test_o_default_do_app_vem_do_registro(self):
        """Fonte unica: mexer no registro tem de refletir no estado do app."""
        from app_core.app_state import APP_STATE_DEFAULTS
        cfg = STATS_MARKET_CONFIG["Tempo"]
        self.assertEqual(APP_STATE_DEFAULTS.get(str(cfg["threshold_key"])),
                         str(cfg["default_thresholds"]))


class TestParserDeTempo(unittest.TestCase):
    """O campo aceita horario, nao so decimal."""

    def test_formatos_aceitos(self):
        self.assertEqual(parse_time_lines_text("32"), [32.0])
        self.assertEqual(parse_time_lines_text("32.5"), [32.5])
        self.assertEqual(parse_time_lines_text("32:00"), [32.0])
        self.assertEqual(parse_time_lines_text("32:30"), [32.5])

    def test_varias_linhas_de_uma_vez(self):
        self.assertEqual(parse_time_lines_text("32:30, 35"), [32.5, 35.0])

    def test_vazio_nao_estoura(self):
        self.assertEqual(parse_time_lines_text(""), [])
        self.assertEqual(parse_time_lines_text(None), [])

    def test_o_parser_comum_NAO_serve_para_tempo(self):
        """Motivo de haver um parser separado: o comum quebra "32:30".

        Se um dia alguem "simplificar" usando o parser comum aqui, o dono
        digita 32:30 e ve linhas erradas. Este teste explica por que nao.
        """
        from app_core.market import parse_lines_text
        comum = parse_lines_text("32:30")
        self.assertNotEqual(comum, [32.5],
                            "se o parser comum passar a entender 32:30, "
                            "reveja se ainda faz sentido ter dois")


class TestColunaNaTabela(unittest.TestCase):
    """A coluna sai mesmo, com o rotulo em horario."""

    def _render(self, texto_tempo: str = "") -> list:
        """Roda a tela de verdade com um `st` de mentira e devolve as colunas."""
        from app_core.statscamps_tab_core import render_statscamps_tab

        capturado: list = []

        class _Col:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        class _FakeSt:
            session_state: dict = {}

            def __init__(self):
                self.session_state = {"sc_th_time": texto_tempo}

            def __getattr__(self, nome):
                def _qualquer(*a, **k):
                    return None
                return _qualquer

            def columns(self, spec):
                n = spec if isinstance(spec, int) else len(spec)
                return [_Col() for _ in range(n)]

            def selectbox(self, label, options=None, **k):
                # Escolhe a primeira opcao NAO VAZIA. Os seletores de liga/ano/
                # split comecam com uma opcao em branco, e devolver ela fazia a
                # tela sair cedo com "Selecione uma liga..." — o teste capturava
                # zero colunas e parecia defeito do codigo.
                opts = [o for o in list(options or []) if str(o).strip()]
                return opts[0] if opts else None

            def text_input(self, label, key=None, **k):
                return str(self.session_state.get(key, "") or "")

            def dataframe(self, data=None, **k):
                if isinstance(data, pd.DataFrame):
                    capturado.append([str(c) for c in data.columns])

        dias = pd.date_range("2026-07-01", periods=12, freq="D")
        linhas = []
        for i, d in enumerate(dias):
            for time_nome in ("Alfa", "Beta"):
                linhas.append({
                    "team": time_nome, "opponent": "X", "league": "LCS",
                    "year": 2026, "split": "Summer", "playoffs": False,
                    "date": d, "win": 1 if i % 2 else 0,
                    "game_time_min": 30.0 + i,          # 30..41
                    "map_number": 1,
                    "firstblood": 0, "firstherald": 0, "firsttower": 0,
                    "firstdragon": 0, "firstbaron": 0,
                    "gold_for": 60000.0, "kills_for": 15.0, "deaths_for": 13.0,
                    "towers_for": 6.0, "total_gold": 120000.0, "total_kills": 28.0,
                    "total_towers": 12.0, "total_dragons": 4.0,
                    "total_nashors": 1.0, "total_inhibitors": 1.0,
                })
        jogos = pd.DataFrame(linhas)

        render_statscamps_tab(
            st=_FakeSt(),
            team_games=jogos,
            Filters=lambda **k: None,
            apply_filters_fn=lambda df, f: df,
            parse_lines_fn=lambda t: [],
            parse_time_lines_fn=parse_time_lines_text,
            min_to_mmss_fn=lambda x: "{:d}:{:02d}".format(int(x), int(round((float(x) % 1) * 60))),
            format_num_fn=lambda x, d: "{:.{}f}".format(float(x), int(d)),
            df_contract_view_fn=lambda df, *a, **k: df,
        )
        return capturado

    def test_a_coluna_de_tempo_aparece_por_padrao(self):
        colunas = self._render()
        todas = [c for grupo in colunas for c in grupo]
        tempo = [c for c in todas if "Tempo" in c]
        self.assertTrue(tempo, "nenhuma coluna de tempo saiu; colunas={}".format(todas[:20]))
        self.assertIn("% Tempo > 32:30", tempo)

    def test_o_rotulo_usa_HORARIO_e_nao_decimal(self):
        todas = [c for grupo in self._render() for c in grupo]
        self.assertNotIn("% Tempo > 32.5", todas,
                         "o rotulo deve sair como 32:30, que e como a linha e vendida")

    def test_linha_digitada_SOMA_a_padrao_em_vez_de_substituir(self):
        """Mesma regra dos outros mercados: digitar nao apaga o padrao."""
        todas = [c for grupo in self._render("35") for c in grupo]
        tempo = sorted(c for c in todas if "Tempo" in c)
        self.assertIn("% Tempo > 32:30", tempo)
        self.assertIn("% Tempo > 35:00", tempo)

    def test_aceita_horario_digitado(self):
        todas = [c for grupo in self._render("28:30") for c in grupo]
        tempo = [c for c in todas if "Tempo" in c]
        self.assertIn("% Tempo > 28:30", tempo)


if __name__ == "__main__":
    unittest.main()
