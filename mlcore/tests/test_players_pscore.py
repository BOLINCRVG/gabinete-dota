# -*- coding: utf-8 -*-
"""Testes da nota de jogador por DESEMPENHO.

O que estes testes protegem, em ordem de importancia:

1. PONTO NO TEMPO. Se a nota de um jogo puder enxergar o proprio jogo, ou os
   pesos puderem enxergar o futuro, a medicao de +0,00402 vira ficcao. Os
   testes de vazamento sao os que NAO podem ser afrouxados.
2. CHAVE DO JOGADOR. Se divergir da que o app usa, a nota nao casa com a
   escalacao e a camada vira ruido silencioso.
3. AUSENCIA. Jogador sem nota nao pode virar zero disfarcado de nota.
"""
from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from mlcore.players_pscore_core import (
    BETA,
    ajustar,
    calcular_pscore,
    delta_da_escalacao,
    notas_finais,
    notas_ponto_no_tempo,
)

ROTAS = ["Top", "Jungle", "Mid", "Bot", "Support"]


def _quadro(n_jogos: int = 400, semente: int = 7) -> pd.DataFrame:
    """Monta linhas de jogador sinteticas com sinal de verdade.

    O lado azul recebe estatisticas melhores quando vence, para que exista
    relacao aprendivel entre desempenho e resultado.
    """
    rng = np.random.default_rng(semente)
    linhas = []
    d0 = pd.Timestamp("2024-01-01")
    for g in range(n_jogos):
        venceu_azul = int(rng.integers(0, 2))
        for lado in ("blue", "red"):
            ganhou = int(venceu_azul == (lado == "blue"))
            for i, rota in enumerate(ROTAS):
                pid = int(rng.integers(0, 40))
                empurra = 0.8 if ganhou else -0.8
                linhas.append({
                    "gameid": "g{}".format(g),
                    "date": d0 + pd.Timedelta(days=g),
                    "participantid": (i + 1) if lado == "blue" else (i + 6),
                    "position": rota,
                    "playerid": "p{}".format(pid),
                    "playername": "nome{}".format(pid),
                    "teamname": "time_{}".format(lado),
                    "result": ganhou,
                    "kills": max(0.0, rng.normal(3 + empurra, 1)),
                    "deaths": max(0.0, rng.normal(3 - empurra, 1)),
                    "assists": max(0.0, rng.normal(5 + empurra, 1)),
                    "teamkills": 12.0,
                    "dpm": rng.normal(500 + 80 * empurra, 50),
                    "damageshare": rng.normal(0.2 + 0.02 * empurra, 0.02),
                    "earnedgoldshare": rng.normal(0.2 + 0.02 * empurra, 0.02),
                    "cspm": rng.normal(8 + empurra, 1),
                    "visionscore": rng.normal(40 + 5 * empurra, 5),
                    "golddiffat15": rng.normal(300 * empurra, 200),
                })
    df = pd.DataFrame(linhas)
    df["pos_can"] = df["position"]
    df["player_key"] = df["playerid"]
    df["lado"] = np.where(df["participantid"] <= 5, "blue", "red")
    tk = df["teamkills"]
    df["kp"] = ((df["kills"] + df["assists"]) / tk).fillna(0.0)
    return df.sort_values(["date", "gameid"]).reset_index(drop=True)


class TestAjuste(unittest.TestCase):

    def test_aprende_um_peso_por_rota(self):
        df = _quadro()
        aj = ajustar(df, ate=None)
        self.assertEqual(set(aj["pesos_por_rota"]), set(ROTAS))
        for rota in ROTAS:
            # um peso por atributo, mais o intercepto
            self.assertEqual(len(aj["pesos_por_rota"][rota]),
                             len(aj["atributos"]) + 1)

    def test_recusa_quando_ha_pouco_dado_antes_do_corte(self):
        df = _quadro(n_jogos=400)
        # corte logo no comeco: quase nada antes dele
        with self.assertRaises(ValueError):
            ajustar(df, ate=date(2024, 1, 5))

    def test_o_corte_MUDA_os_pesos(self):
        """Se o corte nao mudasse nada, ele nao estaria sendo respeitado."""
        df = _quadro(n_jogos=700)
        a1 = ajustar(df, ate=date(2025, 1, 1))
        a2 = ajustar(df, ate=None)
        iguais = all(
            np.allclose(a1["pesos_por_rota"][r], a2["pesos_por_rota"][r])
            for r in ROTAS
        )
        self.assertFalse(iguais, "o corte nao alterou os pesos — ele esta sendo ignorado")


class TestVazamento(unittest.TestCase):
    """Os testes que NAO podem ser afrouxados."""

    def test_pesos_ignoram_tudo_depois_do_corte(self):
        df = _quadro(n_jogos=700)
        corte = date(2025, 1, 1)
        a_limpo = ajustar(df, ate=corte)

        # Estraga TODAS as linhas posteriores ao corte. Se os pesos mudarem,
        # e porque estao olhando para depois do corte.
        sujo = df.copy()
        pos = sujo["date"].dt.date >= corte
        for col in ("dpm", "cspm", "visionscore", "golddiffat15", "kp", "deaths"):
            sujo.loc[pos, col] = sujo.loc[pos, col] * -17.0 + 999.0
        sujo.loc[pos, "result"] = 1 - sujo.loc[pos, "result"]

        a_sujo = ajustar(sujo, ate=corte)
        for rota in ROTAS:
            np.testing.assert_allclose(
                a_limpo["pesos_por_rota"][rota],
                a_sujo["pesos_por_rota"][rota],
                err_msg="VAZAMENTO: os pesos da rota {} mudaram ao sujar o futuro".format(rota),
            )

    def test_a_nota_de_um_jogo_NAO_enxerga_o_proprio_jogo(self):
        df = _quadro(n_jogos=300)
        aj = ajustar(df, ate=None)
        ps = calcular_pscore(df, aj)
        notas = notas_ponto_no_tempo(df, ps)

        # O primeiro jogo de cada jogador nao tem passado: a nota tem de ser 0.
        d = df[["player_key", "date", "gameid"]].copy()
        d["nota"] = notas.to_numpy()
        d = d.sort_values(["player_key", "date", "gameid"])
        primeiros = d.groupby("player_key", sort=False).head(1)
        np.testing.assert_allclose(
            primeiros["nota"].to_numpy(), 0.0, atol=1e-12,
            err_msg="a nota do PRIMEIRO jogo nao e zero — ela esta vendo o proprio jogo",
        )

    def test_mudar_o_desempenho_de_um_jogo_nao_muda_a_nota_DAQUELE_jogo(self):
        df = _quadro(n_jogos=300)
        aj = ajustar(df, ate=None)

        n1 = notas_ponto_no_tempo(df, calcular_pscore(df, aj))
        alvo = len(df) - 1          # ultima linha: ninguem depois dela
        sujo = df.copy()
        sujo.loc[alvo, "dpm"] = 99999.0
        sujo.loc[alvo, "kp"] = 50.0
        n2 = notas_ponto_no_tempo(sujo, calcular_pscore(sujo, aj))

        self.assertAlmostEqual(
            float(n1.iloc[alvo]), float(n2.iloc[alvo]), places=12,
            msg="a nota mudou ao mexer no proprio jogo — ha vazamento",
        )


class TestNota(unittest.TestCase):

    def test_beta_segura_quem_tem_poucos_jogos(self):
        """Dois jogos otimos nao podem valer o mesmo que cem."""
        df = _quadro(n_jogos=300)
        aj = ajustar(df, ate=None)
        ps = calcular_pscore(df, aj)
        finais = notas_finais(df, ps)
        # com BETA no denominador, |nota| < |media| sempre
        d = df[["player_key", "pos_can"]].copy()
        d["pscore"] = ps.to_numpy()
        ag = d.groupby(["player_key", "pos_can"], sort=False)["pscore"].agg(["mean", "count"])
        for (chave, rota), linha in ag.iterrows():
            nota = finais["{}|||{}".format(chave, rota)]["n"]
            encolhe = float(linha["count"]) / (float(linha["count"]) + BETA)
            self.assertAlmostEqual(nota, float(linha["mean"]) * encolhe, places=9)

    def test_a_chave_casa_com_a_do_app(self):
        """A chave tem de ser 'jogador|||Rota', igual a de players_layer.

        Usa 300 jogos e nao 120: cada jogo gera 10 linhas, e abaixo de
        MIN_JOGOS_TOTAL=2000 a funcao se recusa a ajustar — com razao. A
        primeira versao deste teste pedia 120 jogos (1.200 linhas) e a trava
        disparou. A trava estava certa; o teste e que era degenerado.
        """
        df = _quadro(n_jogos=300)
        aj = ajustar(df, ate=None)
        finais = notas_finais(df, calcular_pscore(df, aj))
        exemplo = next(iter(finais))
        self.assertIn("|||", exemplo)
        chave, rota = exemplo.split("|||")
        self.assertIn(rota, ROTAS)
        self.assertTrue(chave.startswith("p"), "a chave deveria ser o playerid")


class TestDelta(unittest.TestCase):

    def _notas(self):
        return {
            "a|||Top": {"n": 1.0, "gp": 50},
            "b|||Mid": {"n": 0.5, "gp": 50},
            "x|||Top": {"n": -1.0, "gp": 50},
            "y|||Mid": {"n": 0.0, "gp": 50},
        }

    def test_soma_azul_menos_vermelho(self):
        d, n = delta_da_escalacao(
            self._notas(),
            blue_lineup={"Top": "a", "Mid": "b"},
            red_lineup={"Top": "x", "Mid": "y"},
        )
        self.assertAlmostEqual(d, (1.0 + 0.5) - (-1.0 + 0.0))
        self.assertEqual(n, 2)

    def test_jogador_sem_nota_NAO_vira_zero(self):
        """Principio IV: ausencia nao vira numero.

        O desconhecido fica de fora da soma E some da contagem, para quem
        chamar poder ver que a cobertura caiu.
        """
        d, n = delta_da_escalacao(
            self._notas(),
            blue_lineup={"Top": "a", "Mid": "NINGUEM_CONHECE"},
            red_lineup={"Top": "x", "Mid": "y"},
        )
        self.assertAlmostEqual(d, 1.0 - (-1.0 + 0.0))
        self.assertEqual(n, 1, "a cobertura tem de cair quando falta jogador")

    def test_escalacao_vazia_da_zero_e_cobertura_zero(self):
        d, n = delta_da_escalacao(self._notas(), blue_lineup={}, red_lineup={})
        self.assertEqual(d, 0.0)
        self.assertEqual(n, 0)

    def test_trocar_os_lados_inverte_o_sinal(self):
        notas = self._notas()
        a, _ = delta_da_escalacao(notas, blue_lineup={"Top": "a"}, red_lineup={"Top": "x"})
        b, _ = delta_da_escalacao(notas, blue_lineup={"Top": "x"}, red_lineup={"Top": "a"})
        self.assertAlmostEqual(a, -b)


if __name__ == "__main__":
    unittest.main()
