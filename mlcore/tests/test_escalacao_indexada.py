# -*- coding: utf-8 -*-
"""A escalacao de cada jogo sai de um INDICE, nao de varrer a tabela.

O QUE ELE DISSE (23/09/2026): "a interface eu sinto ela muito lenta, nao e
algo fluido, rapido... eu queria algo rapido, que eu clique e ele va".

O PERFIL DEU O NUMERO. Uma passada completa do script do app: 808 segundos.
796 deles na manutencao automatica, e 660 dentro de `lineup_for_game` --
chamada 29.178 vezes, uma por jogo do historico. A cada chamada ela fazia:

    player_rows[player_rows["gameid"].astype(str) == str(gameid)]

isto e, convertia e varria a coluna inteira (~291 mil linhas) para achar 10
linhas. Sao bilhoes de comparacoes de texto, e o custo cresce ao quadrado. Ele
atualiza o CSV todo dia, entao pagava isso todo dia.

MEDIDO DEPOIS DO CONSERTO: os mesmos 29.178 jogos levam 0,06 s.

O RISCO DESTA TROCA, e o motivo da metade dos testes aqui: um indice que
devolva escalacao DIFERENTE da anterior seria muito pior que lentidao -- a
analise sairia sobre outros jogadores. Por isso o primeiro teste compara as
duas versoes jogo a jogo, e nao so o tempo.
"""
from __future__ import annotations

import unittest

import pandas as pd

from mlcore.players_layer import POS_CANON, lineup_for_game


def _versao_antiga(player_rows, gameid):
    """A que varria a coluna. Fica aqui como referencia de comportamento."""
    if player_rows is None or player_rows.empty:
        return {}, {}
    g = player_rows[player_rows["gameid"].astype(str) == str(gameid)].copy()
    if g.empty:
        return {}, {}
    blue, red = {}, {}
    for pos in POS_CANON:
        b = g[(g["side"] == "blue") & (g["pos_can"] == pos)]["player_key"].astype(str).tolist()
        r = g[(g["side"] == "red") & (g["pos_can"] == pos)]["player_key"].astype(str).tolist()
        if b:
            blue[pos] = str(b[0]).strip()
        if r:
            red[pos] = str(r[0]).strip()
    return blue, red


def _quadro(linhas):
    return pd.DataFrame(linhas, columns=["gameid", "side", "pos_can", "player_key"])


def _jogo(gid, sufixo=""):
    fora = []
    for lado in ("blue", "red"):
        for pos in POS_CANON:
            fora.append([gid, lado, pos, "%s_%s_%s%s" % (gid, lado, pos, sufixo)])
    return fora


class TestDaOMesmoResultadoDeAntes(unittest.TestCase):
    """O que importa nao e a velocidade: e nao mudar a resposta."""

    def test_caso_simples_bate_com_a_versao_antiga(self):
        q = _quadro(_jogo("G1") + _jogo("G2"))
        for gid in ("G1", "G2"):
            self.assertEqual(lineup_for_game(q, gid), _versao_antiga(q, gid))

    def test_a_PRIMEIRA_linha_continua_vencendo_quando_ha_repetida(self):
        """Substituicao no meio do jogo gera duas linhas para a mesma posicao.
        A regra antiga ficava com a primeira; a nova tem de fazer igual."""
        linhas = _jogo("G1")
        linhas.append(["G1", "blue", POS_CANON[0], "SUBSTITUTO"])
        q = _quadro(linhas)
        self.assertEqual(lineup_for_game(q, "G1"), _versao_antiga(q, "G1"))
        self.assertNotEqual(lineup_for_game(q, "G1")[0][POS_CANON[0]], "SUBSTITUTO")

    def test_jogo_que_nao_existe_devolve_vazio(self):
        q = _quadro(_jogo("G1"))
        self.assertEqual(lineup_for_game(q, "NAO_EXISTE"), ({}, {}))

    def test_quadro_vazio_nao_estoura(self):
        self.assertEqual(lineup_for_game(_quadro([]), "G1"), ({}, {}))
        self.assertEqual(lineup_for_game(None, "G1"), ({}, {}))

    def test_posicao_desconhecida_e_ignorada(self):
        linhas = _jogo("G1") + [["G1", "blue", "coach", "TREINADOR"]]
        q = _quadro(linhas)
        self.assertEqual(lineup_for_game(q, "G1"), _versao_antiga(q, "G1"))

    def test_lado_estranho_nao_entra_em_nenhuma_escalacao(self):
        linhas = _jogo("G1") + [["G1", "spectator", POS_CANON[0], "X"]]
        q = _quadro(linhas)
        azul, verm = lineup_for_game(q, "G1")
        self.assertNotIn("X", azul.values())
        self.assertNotIn("X", verm.values())

    def test_gameid_numerico_casa_com_texto(self):
        q = pd.DataFrame([[123, "blue", POS_CANON[0], "P"]],
                         columns=["gameid", "side", "pos_can", "player_key"])
        self.assertEqual(lineup_for_game(q, "123")[0], {POS_CANON[0]: "P"})

    def test_faltando_coluna_devolve_vazio_em_vez_de_estourar(self):
        q = pd.DataFrame([["G1", "blue"]], columns=["gameid", "side"])
        self.assertEqual(lineup_for_game(q, "G1"), ({}, {}))


class TestOIndiceNaoTrocaOsJogadores(unittest.TestCase):
    """O perigo do cache: devolver a escalacao de OUTRO quadro."""

    def test_quadro_diferente_com_o_mesmo_tamanho_nao_reusa_por_engano(self):
        """O Python reaproveita endereco de memoria. Se a chave fosse so o
        endereco, um quadro novo poderia herdar o indice do antigo."""
        import gc
        q1 = _quadro(_jogo("G1", "_A"))
        r1 = lineup_for_game(q1, "G1")
        del q1
        gc.collect()
        q2 = _quadro(_jogo("G1", "_B"))
        r2 = lineup_for_game(q2, "G1")
        self.assertNotEqual(r1, r2, "o indice antigo vazou para o quadro novo")
        self.assertTrue(all(v.endswith("_B") for v in r2[0].values()))

    def test_mexer_no_que_foi_devolvido_nao_estraga_a_proxima_chamada(self):
        """O indice e compartilhado; devolver o proprio dicionario deixaria
        quem chama corromper o cache."""
        q = _quadro(_jogo("G1"))
        azul, _ = lineup_for_game(q, "G1")
        azul[POS_CANON[0]] = "ESTRAGADO"
        azul2, _ = lineup_for_game(q, "G1")
        self.assertNotEqual(azul2[POS_CANON[0]], "ESTRAGADO")

    def test_o_cache_guarda_UM_quadro_so(self):
        """Cada quadro pesa dezenas de MB; guardar varios vazaria memoria."""
        from mlcore.players_layer import _INDICE_ESCALACOES
        for i in range(6):
            q = _quadro(_jogo("G%d" % i))
            lineup_for_game(q, "G%d" % i)
        self.assertLessEqual(len([k for k in _INDICE_ESCALACOES if k == "quadro"]), 1)

    def test_quadros_de_MESMO_TAMANHO_em_sequencia_nao_se_misturam(self):
        """O bug que a suite pegou em 23/09/2026: com a chave sendo o endereco
        de memoria, um quadro novo herdava o indice de um quadro morto do
        mesmo tamanho -- e devolvia a escalacao de outro jogo."""
        import gc
        anteriores = []
        for i in range(8):
            q = _quadro(_jogo("MESMO_ID", "_v%d" % i))
            azul, _ = lineup_for_game(q, "MESMO_ID")
            anteriores.append(azul[POS_CANON[0]])
            del q
            gc.collect()
        self.assertEqual(len(set(anteriores)), 8,
                         "algum quadro devolveu a escalacao de outro: %s"
                         % anteriores)


class TestContraOHistoricoDeVerdade(unittest.TestCase):
    """Casos montados a mao nao cobrem o que 291 mil linhas reais cobrem."""

    def test_as_duas_versoes_concordam_no_CSV_real(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        csvs = sorted(raiz.glob("CSV_DIARIO/*OraclesElixir*.csv"))
        if not csvs:
            self.skipTest("os CSVs nao estao nesta maquina")
        from mlcore.players_layer import load_match_player_rows
        pr = load_match_player_rows([str(c) for c in csvs])
        if pr is None or pr.empty:
            self.skipTest("sem linhas de jogador")
        gids = pr["gameid"].astype(str).drop_duplicates().tolist()
        amostra = gids[::400]          # espalhado por todo o historico
        self.assertGreater(len(amostra), 20)
        for gid in amostra:
            self.assertEqual(lineup_for_game(pr, gid), _versao_antiga(pr, gid),
                             "divergiu no jogo %s" % gid)
