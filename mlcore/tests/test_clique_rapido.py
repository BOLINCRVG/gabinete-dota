# -*- coding: utf-8 -*-
"""Os dois caches que tiraram 6,6 s de cada clique -- sem mudar resposta.

ELE DISSE (23/09/2026): "a interface eu sinto ela muito lenta... eu queria
algo rapido, que eu clique e ele va".

O PERFIL DE UM CLIQUE (a segunda passada do script, caches cheios): 10,5 s.

  - `build_lineup_name_choices`: 6,0 s. Chamada 16 vezes por clique; a cada
    chamada COPIAVA a tabela inteira de jogadores (~291 mil linhas) e aplicava
    funcao Python linha a linha. E a aba Jogadores roda em todo clique, mesmo
    com ele em outra aba -- o Streamlit desenha todas.
  - `team_canon_maps`: 1,4 s. Relia o CSV DO DISCO a cada clique, embora ja
    recebesse a assinatura do arquivo (caminho, tamanho, data) -- feita
    justamente para servir de chave de cache.

DEPOIS: um clique em 3,9 s.

O QUE ESTES TESTES PROTEGEM, na ordem de gravidade:
  1. a RESPOSTA nao mudou (cache que devolve outra coisa e pior que lento);
  2. o cache nao serve dado de outra tabela/arquivo;
  3. quem recebe o resultado pode mexer nele sem estragar o cache.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app_core import lineup_runtime_core as lr

POS = ["Top", "Jungle", "Mid", "Bot", "Support"]


def _norm(p):
    return str(p or "").strip().title()


def _jogadores():
    linhas = []
    d0 = pd.Timestamp("2026-01-01")
    for i, (time, jog, pos) in enumerate([
            ("T1", "Faker", "mid"), ("T1", "Faker", "mid"), ("T1", "Zeus", "top"),
            ("T1", "Oner", "jungle"), ("Gen.G", "Chovy", "mid"), ("Gen.G", "Kiin", "top"),
            ("HLE", "Zeus2", "top"), ("T1", "Gumayusi", "bot"), ("T1", "Keria", "support")]):
        linhas.append({"team": time, "player": jog, "pos": pos,
                       "gameid": "G%d" % i, "date": d0 + pd.Timedelta(days=i)})
    return pd.DataFrame(linhas)


class TestAsOpcoesDeEscalacaoNaoMudaram(unittest.TestCase):

    def setUp(self):
        lr._CACHE_OPCOES.update({"quadro": None, "norm": None, "resultados": {}})

    def _as_duas(self, df, time, todos):
        com = lr.build_lineup_name_choices(df_pr=df, team_ui=time, pos_canon=POS,
                                           norm_pos_fn=_norm, include_all_teams=todos)
        sem = lr._build_lineup_name_choices_sem_cache(
            df_pr=df, team_ui=time, pos_canon=POS, norm_pos_fn=_norm,
            include_all_teams=todos)
        return com, sem

    def test_mesma_resposta_com_e_sem_cache(self):
        df = _jogadores()
        for time in ("T1", "Gen.G", "HLE", "NAO_EXISTE"):
            for todos in (False, True):
                com, sem = self._as_duas(df, time, todos)
                self.assertEqual(com, sem, "%s / todos=%s mudou" % (time, todos))

    def test_segunda_chamada_igual_a_primeira(self):
        df = _jogadores()
        a = lr.build_lineup_name_choices(df_pr=df, team_ui="T1", pos_canon=POS,
                                         norm_pos_fn=_norm)
        b = lr.build_lineup_name_choices(df_pr=df, team_ui="T1", pos_canon=POS,
                                         norm_pos_fn=_norm)
        self.assertEqual(a, b)

    def test_tabela_NOVA_nao_herda_o_cache_da_velha(self):
        """Quando o CSV muda, a tabela e outra; o resultado tem de mudar."""
        df1 = _jogadores()
        a = lr.build_lineup_name_choices(df_pr=df1, team_ui="T1", pos_canon=POS,
                                         norm_pos_fn=_norm)
        df2 = df1.copy()
        df2.loc[len(df2)] = {"team": "T1", "player": "Novato", "pos": "top",
                             "gameid": "G99", "date": pd.Timestamp("2026-02-01")}
        b = lr.build_lineup_name_choices(df_pr=df2, team_ui="T1", pos_canon=POS,
                                         norm_pos_fn=_norm)
        self.assertNotEqual(a, b)
        self.assertIn("Novato", b[0]["Top"])

    def test_mexer_no_resultado_nao_estraga_a_proxima_chamada(self):
        df = _jogadores()
        a, _ = lr.build_lineup_name_choices(df_pr=df, team_ui="T1", pos_canon=POS,
                                            norm_pos_fn=_norm)
        a["Mid"].append("ESTRAGADO")
        b, _ = lr.build_lineup_name_choices(df_pr=df, team_ui="T1", pos_canon=POS,
                                            norm_pos_fn=_norm)
        self.assertNotIn("ESTRAGADO", b["Mid"])

    def test_funcao_de_posicao_diferente_nao_reusa(self):
        df = _jogadores()
        lr.build_lineup_name_choices(df_pr=df, team_ui="T1", pos_canon=POS,
                                     norm_pos_fn=_norm)
        outra = lambda p: "Mid"          # noqa: E731 -- todo mundo vira Mid
        b, _ = lr.build_lineup_name_choices(df_pr=df, team_ui="T1", pos_canon=POS,
                                            norm_pos_fn=outra)
        self.assertEqual(b["Top"], [])

    def test_tabela_vazia_nao_estoura(self):
        v = lr.build_lineup_name_choices(df_pr=pd.DataFrame(), team_ui="T1",
                                         pos_canon=POS, norm_pos_fn=_norm)
        self.assertEqual(v[0], {p: [] for p in POS})
        n = lr.build_lineup_name_choices(df_pr=None, team_ui="T1",
                                         pos_canon=POS, norm_pos_fn=_norm)
        self.assertEqual(n[0], {p: [] for p in POS})


class TestOsNomesDeTimeNaoRelemODisco(unittest.TestCase):

    def test_a_mesma_assinatura_nao_le_o_arquivo_de_novo(self):
        from app_core import matchup_selection_core as ms
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        csvs = sorted(raiz.glob("CSV_DIARIO/*OraclesElixir*.csv"))
        if not csvs:
            self.skipTest("sem CSV nesta maquina")
        c = csvs[-1]
        st = c.stat()
        sig = (str(c), int(st.st_size), int(st.st_mtime))
        ms._team_canon_maps_em_cache.cache_clear()
        ms.team_canon_maps(sig)
        ms.team_canon_maps(sig)
        info = ms._team_canon_maps_em_cache.cache_info()
        self.assertEqual(info.misses, 1, "leu o disco mais de uma vez")
        self.assertGreaterEqual(info.hits, 1)

    def test_assinatura_diferente_le_de_novo(self):
        """Quando ele atualiza o CSV, tamanho e data mudam -- e o mapa tem de
        ser refeito, nunca servido velho."""
        from app_core import matchup_selection_core as ms
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        csvs = sorted(raiz.glob("CSV_DIARIO/*OraclesElixir*.csv"))
        if not csvs:
            self.skipTest("sem CSV nesta maquina")
        c = csvs[-1]
        st = c.stat()
        ms._team_canon_maps_em_cache.cache_clear()
        ms.team_canon_maps((str(c), int(st.st_size), int(st.st_mtime)))
        ms.team_canon_maps((str(c), int(st.st_size), int(st.st_mtime) + 1))
        self.assertEqual(ms._team_canon_maps_em_cache.cache_info().misses, 2)

    def test_mexer_no_mapa_devolvido_nao_estraga_o_cache(self):
        from app_core import matchup_selection_core as ms
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        csvs = sorted(raiz.glob("CSV_DIARIO/*OraclesElixir*.csv"))
        if not csvs:
            self.skipTest("sem CSV nesta maquina")
        c = csvs[-1]
        st = c.stat()
        sig = (str(c), int(st.st_size), int(st.st_mtime))
        a = ms.team_canon_maps(sig)
        a["best_list"].append("ESTRAGADO")
        b = ms.team_canon_maps(sig)
        self.assertNotIn("ESTRAGADO", b["best_list"])


if __name__ == "__main__":
    unittest.main()
