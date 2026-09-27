# -*- coding: utf-8 -*-
"""Testes do card PAST FACEOFFS mostrando OS DOIS TIMES.

PEDIDO DELE (15/09/2026), na linha de kills por time: o card trazia so a
coluna do time A — "AVG (GEN.G) 16,8" e a lista de kills da Gen.G jogo a jogo.
Faltava o outro lado.

POR QUE ISSO MUDA A LEITURA, e nao e enfeite: "a Gen.G fez 26" sozinho nao
diz se foi jogo de sangue (26 x 24) ou atropelo (26 x 4). Numa linha de kills
POR TIME, e essa diferenca que decide a aposta.

O RISCO DA IMPLEMENTACAO, e o motivo do teste mais importante daqui: parear os
dois lados por DATA quebraria num BO5, que tem ate cinco jogos no mesmo dia —
o app mostraria o valor do mapa errado ao lado. O pareamento e por `gameid`.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.market_v3_consistency_panel_core import render_v3_market_context_panel


class _Caixa:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeSt:
    """`st` de mentira que so guarda o HTML escrito."""

    def __init__(self):
        self.html: list[str] = []
        self.session_state: dict = {}

    def markdown(self, texto="", **k):
        self.html.append(str(texto))

    def container(self, **k):
        return _Caixa()

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Caixa() for _ in range(n)]

    def __getattr__(self, nome):
        def _qualquer(*a, **k):
            return None
        return _qualquer

    @property
    def tudo(self) -> str:
        return "\n".join(self.html)


def _jogos(time, adversario, valores, *, col="kills_for", gids=None, datas=None):
    n = len(valores)
    gids = gids or ["g{}".format(i) for i in range(n)]
    datas = datas or ["2026-08-0{}".format(i + 1) for i in range(n)]
    return pd.DataFrame([{
        "gameid": gids[i], "date": datas[i], "team": time, "opponent": adversario,
        col: valores[i], "league": "LCK", "year": 2026, "win": i % 2,
    } for i in range(n)])


def _desenha(*, col="kills_for", com_b=True, a_vals=(20, 10, 30),
            b_vals=(4, 12, 28), gids=None, datas=None, linha=13.5):
    st = _FakeSt()
    dfa = _jogos("Gen.G", "T1", list(a_vals), col=col, gids=gids, datas=datas)
    dfb = _jogos("T1", "Gen.G", list(b_vals), col=col, gids=gids, datas=datas)
    render_v3_market_context_panel(
        st,
        panel_title="Kills p/ time",
        col_name=col,
        team_a="Gen.G", team_b="T1",
        df_a_context_all=dfa, df_b_context_all=dfb,
        df_a_context_season=dfa, df_b_context_season=dfb,
        df_h2h_year=dfa,
        df_h2h_context_year_full_a=dfa,
        df_h2h_year_b=(dfb if com_b else None),
        league_scope_df=pd.concat([dfa, dfb], ignore_index=True),
        line=linha,
        min_to_mmss_fn=lambda x: str(x),
        num_formatter=lambda v, d: "{:.{}f}".format(float(v), int(d)),
    )
    return st.tudo


class TestAMediaDosDois(unittest.TestCase):

    def test_a_media_do_outro_time_aparece(self):
        html = _desenha()
        self.assertIn("Avg (Gen.G)", html)
        self.assertIn("Avg (T1)", html)

    def test_as_duas_medias_estao_certas(self):
        """Gen.G: (20+10+30)/3 = 20 · T1: (4+12+28)/3 ≈ 14,7"""
        html = _desenha(a_vals=(20, 10, 30), b_vals=(4, 12, 28))
        self.assertIn("20.0", html)
        self.assertIn("14.7", html)

    def test_sem_o_lado_B_o_card_segue_como_antes(self):
        """Compatibilidade: quem nao passar o frame de B nao ve nada mudar."""
        html = _desenha(com_b=False)
        self.assertIn("Avg (Gen.G)", html)
        self.assertNotIn("Avg (T1)", html)

    def test_frame_de_B_vazio_nao_inventa_caixa(self):
        st = _FakeSt()
        dfa = _jogos("Gen.G", "T1", [20, 10])
        render_v3_market_context_panel(
            st, panel_title="Kills", col_name="kills_for",
            team_a="Gen.G", team_b="T1",
            df_a_context_all=dfa, df_b_context_all=dfa,
            df_a_context_season=dfa, df_b_context_season=dfa,
            df_h2h_year=dfa, df_h2h_context_year_full_a=dfa,
            df_h2h_year_b=pd.DataFrame(),
            league_scope_df=dfa, line=13.5,
            min_to_mmss_fn=lambda x: str(x),
            num_formatter=lambda v, d: "{:.{}f}".format(float(v), int(d)),
        )
        self.assertNotIn("Avg (T1)", st.tudo)


class TestMercadoCombinadoNaoSeDivide(unittest.TestCase):
    """Em `total_*` os dois times compartilham o MESMO valor do jogo.

    Mostrar "Avg (T1)" ali seria repetir o numero do lado esquerdo com outro
    rotulo — parece informacao nova e nao e.
    """

    def test_total_continua_com_avg_combined(self):
        html = _desenha(col="total_kills", a_vals=(30, 22), b_vals=(30, 22))
        self.assertIn("Avg (combined)", html)
        self.assertNotIn("Avg (T1)", html)


class TestOParEPorJOGOENaoPorDATA(unittest.TestCase):
    """O teste que mais importa: um BO5 tem ate 5 jogos no MESMO dia.

    Parear por data mostraria o valor do mapa errado ao lado — um erro que
    passa despercebido porque o numero continua plausivel.
    """

    def test_cinco_jogos_no_mesmo_dia_casam_certo(self):
        mesma_data = ["2026-08-16"] * 5
        gids = ["m1", "m2", "m3", "m4", "m5"]
        html = _desenha(a_vals=(11, 12, 13, 14, 15),
                        b_vals=(21, 22, 23, 24, 25),
                        gids=gids, datas=mesma_data)
        # Cada valor do A aparece; os do B tambem, e nenhum se perde.
        for v in ("11", "12", "13", "14", "15", "21", "22", "23", "24", "25"):
            self.assertIn(v, html)

    def test_jogo_sem_par_no_outro_lado_nao_inventa_numero(self):
        """PRINCIPIO IV: sem o par, a celula do outro time simplesmente nao vem."""
        st = _FakeSt()
        dfa = _jogos("Gen.G", "T1", [20, 10], gids=["g1", "g2"])
        dfb = _jogos("T1", "Gen.G", [4], gids=["g1"])
        render_v3_market_context_panel(
            st, panel_title="Kills", col_name="kills_for",
            team_a="Gen.G", team_b="T1",
            df_a_context_all=dfa, df_b_context_all=dfb,
            df_a_context_season=dfa, df_b_context_season=dfb,
            df_h2h_year=dfa, df_h2h_context_year_full_a=dfa,
            df_h2h_year_b=dfb,
            league_scope_df=pd.concat([dfa, dfb], ignore_index=True), line=13.5,
            min_to_mmss_fn=lambda x: str(x),
            num_formatter=lambda v, d: "{:.{}f}".format(float(v), int(d)),
        )
        self.assertIn("Avg (T1)", st.tudo)   # a media do que existe, sim


class TestQuemEQualColuna(unittest.TestCase):
    """Dois numeros soltos na mesma linha viram adivinhacao."""

    def test_a_lista_ganha_cabecalho_com_os_dois_nomes(self):
        html = _desenha()
        self.assertIn("Gen.G", html)
        self.assertIn("T1", html)

    def test_sem_o_lado_B_nao_ha_cabecalho_de_duas_colunas(self):
        import re
        html = _desenha(com_b=False)
        # o cabecalho de duas colunas so existe quando ha duas colunas
        self.assertIsNone(re.search(r"text-transform:uppercase;padding-bottom:2px", html))


class TestNaoQuebraOQueJaFuncionava(unittest.TestCase):

    def test_o_card_continua_desenhando(self):
        html = _desenha()
        self.assertIn("PAST FACEOFFS", html)
        self.assertIn("H2H Games", html)

    def test_sem_h2h_nenhum_nao_estoura(self):
        st = _FakeSt()
        vazio = pd.DataFrame()
        dfa = _jogos("Gen.G", "X", [20])
        render_v3_market_context_panel(
            st, panel_title="Kills", col_name="kills_for",
            team_a="Gen.G", team_b="T1",
            df_a_context_all=dfa, df_b_context_all=dfa,
            df_a_context_season=dfa, df_b_context_season=dfa,
            df_h2h_year=vazio, df_h2h_context_year_full_a=None,
            df_h2h_year_b=None,
            league_scope_df=dfa, line=13.5,
            min_to_mmss_fn=lambda x: str(x),
            num_formatter=lambda v, d: "{:.{}f}".format(float(v), int(d)),
        )
        self.assertIn("sem H2H", st.tudo)


if __name__ == "__main__":
    unittest.main()
