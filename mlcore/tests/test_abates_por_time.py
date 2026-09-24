# -*- coding: utf-8 -*-
"""Handicap de kills e total de kills por time (pedido dele, 24/09/2026).

MEDIDO (tools/gerar_abates_por_time.py), ajuste 2024-2025, prova CEGA 2026,
7.100 jogos, nas linhas em volta da media (as que a casa oferece):

    handicap    Brier 0,24442 x taxa-base 0,26274  ganho +0,0183
                IC95 [+0,0159, +0,0208]  calibracao 40,9% -> 40,9%
    por time    Brier 0,22478 x taxa-base 0,23910  ganho +0,0143
                IC95 [+0,0131, +0,0157]  erro medio de calibracao 0,56 pp

O QUE ESTES TESTES SEGURAM
  1. a medicao usou a MESMA conta que a tela serve (a incremental bate com a
     de producao);
  2. o handicap nao tem lado: os dois lados da mesma linha somam 1;
  3. o artefato so existe porque passou, e 2026 nao entrou no ajuste;
  4. a aba precifica as duas coisas, e o total por time so quando sabe DE
     QUAL time e a linha.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from mlcore import abates_por_time as ab

RAIZ = Path(__file__).resolve().parents[2]


def _quadro():
    """Tres times, jogos em datas diferentes, kills variados."""
    linhas = []
    jogos = [("2026-01-01", "A", "B", 15, 8), ("2026-01-03", "B", "C", 12, 12),
             ("2026-01-05", "A", "C", 20, 5), ("2026-01-09", "C", "B", 9, 14),
             ("2026-01-12", "B", "A", 7, 18), ("2026-01-15", "C", "A", 11, 16),
             ("2026-01-20", "A", "B", 13, 10), ("2026-01-26", "B", "C", 16, 6)]
    for i, (d, x, y, kx, ky) in enumerate(jogos * 3):
        dia = pd.Timestamp(d) + pd.Timedelta(days=30 * (i // len(jogos)))
        gid = "g%03d" % i
        linhas.append({"gameid": gid, "date": dia, "team": x, "opponent": y,
                       "kills_for": kx, "kills_against": ky})
        linhas.append({"gameid": gid, "date": dia, "team": y, "opponent": x,
                       "kills_for": ky, "kills_against": kx})
    return pd.DataFrame(linhas)


class TestAMedicaoUsouAContaDaTela(unittest.TestCase):

    def test_o_incremental_bate_com_a_producao(self):
        """Para cada jogo medido, `medias_dc` com o quadro de ANTES dele da o
        mesmo numero. (A data de referencia nao muda nada: o decaimento
        multiplica numerador e denominador pelo mesmo fator.)"""
        from tools.gerar_abates_por_time import registros_ponto_no_tempo
        from mlcore import lines_engine as le
        tg = _quadro()
        velho = le.DC_BLEND_MIN_EFF
        try:
            le.DC_BLEND_MIN_EFF = 2.0      # o quadro de teste e pequeno
            regs = registros_ponto_no_tempo(tg)
            self.assertGreater(len(regs), 5)
            for r in regs:
                antes = tg[tg["date"] < tg.loc[tg["gameid"] == r["gameid"], "date"].iloc[0]]
                par = tg[tg["gameid"] == r["gameid"]].sort_values("team")
                x, y = str(par.iloc[0]["team"]), str(par.iloc[1]["team"])
                m = ab.medias_dc(antes, x, y)
                self.assertIsNotNone(m, r["gameid"])
                self.assertAlmostEqual(m[0], r["ex"], places=9)
                self.assertAlmostEqual(m[1], r["ey"], places=9)
        finally:
            le.DC_BLEND_MIN_EFF = velho

    def test_time_desconhecido_nao_vira_numero(self):
        self.assertIsNone(ab.medias_dc(_quadro(), "A", "Nao Existe"))


class TestOHandicapNaoTemLado(unittest.TestCase):

    def setUp(self):
        self.art = ab.carregar()
        if "handicap" not in self.art:
            self.skipTest("artefato ausente")

    def test_os_dois_lados_da_mesma_linha_somam_um(self):
        for ea, eb, pts in ((18, 11, -7.5), (14, 14, 0.5), (9, 16, 4.5), (20, 8, -12.5)):
            p1 = ab.prob_cobre_handicap(ea, eb, pts)
            p2 = ab.prob_cobre_handicap(eb, ea, -pts)
            self.assertAlmostEqual(p1 + p2, 1.0, delta=1.0 / 400)

    def test_quem_recebe_mais_pontos_tem_mais_chance(self):
        ps = [ab.prob_cobre_handicap(15, 12, x + 0.5) for x in range(-12, 12)]
        for a, b in zip(ps, ps[1:]):
            self.assertLessEqual(a, b)

    def test_total_do_time_linha_maior_chance_menor(self):
        if "total_time" not in self.art:
            self.skipTest("total por time nao passou neste repositorio")
        ps = [ab.prob_total_do_time_acima(15, x + 0.5) for x in range(5, 25)]
        for a, b in zip(ps, ps[1:]):
            self.assertGreaterEqual(a, b)

    def test_sem_artefato_nao_inventa(self):
        self.assertIsNone(ab.prob_cobre_handicap(15, 12, -2.5, artefato={}))
        self.assertIsNone(ab.prob_total_do_time_acima(15, 14.5, artefato={}))


class TestOArtefatoSoExistePorquePassou(unittest.TestCase):

    def setUp(self):
        alvo = RAIZ / "abates_por_time_forma.json"
        if not alvo.exists():
            self.skipTest("artefato ausente")
        self.d = json.loads(alvo.read_text(encoding="utf-8"))

    def test_2026_nao_entrou_no_ajuste(self):
        self.assertNotIn(2026, self.d["_anos_de_ajuste"])

    def test_a_prova_cega_esta_gravada_e_passou(self):
        """So o mercado que passou esta no arquivo (no Dota, 24/09/2026, o
        total por time nao passou e nao esta)."""
        blocos = [m for m in ("handicap", "total_time") if m in self.d]
        self.assertTrue(blocos)
        for m in blocos:
            p = self.d["_prova_cega"][m]
            self.assertTrue(p["passou"], m)
            self.assertGreater(p["ic95"][0], 0.0, m)
            self.assertLess(p["ece"], 0.02, m)


def _linha(**k):
    base = {"jogo": 1, "periodo": 1, "unidade": "Kills", "odd": 1.9}
    base.update(k)
    return base


class TestAAbaPrecifica(unittest.TestCase):

    def setUp(self):
        if "handicap" not in ab.carregar():
            self.skipTest("artefato ausente")

    def _precisa_total_por_time(self):
        if "total_time" not in ab.carregar():
            self.skipTest("total por time nao passou neste repositorio (Dota)")

    def _montar(self, linhas, medias=(18.0, 11.0)):
        from app_core.oportunidades_core import montar_oportunidades
        odds = {"jogos": [{"id": 1, "casa": "T1", "fora": "GenG", "liga": "LCK"}],
                "linhas": linhas}
        return montar_oportunidades(odds, lambda a, b, m: None,
                                    prob_over=lambda t, x: 0.5,
                                    medias_kills=lambda a, b: medias)

    def test_handicap_de_kills_sai_com_o_numero_do_modelo(self):
        r = self._montar([
            _linha(tipo="spread", lado="home", pontos=-6.5, mercado="Handicap - Kills - mapa 1"),
            _linha(tipo="spread", lado="away", pontos=6.5, mercado="Handicap - Kills - mapa 1")])
        ps = sorted(x["probabilidade"] for x in r["todas"])
        self.assertEqual(len(ps), 2)
        self.assertAlmostEqual(sum(ps), 1.0, delta=1.0 / 400)
        casa = [x for x in r["todas"] if x["lado"] == "home"][0]
        self.assertAlmostEqual(casa["probabilidade"], ab.prob_cobre_handicap(18, 11, -6.5))

    def test_total_do_time_usa_O_TIME_CERTO(self):
        self._precisa_total_por_time()
        r = self._montar([
            _linha(tipo="team_total", lado="over", pontos=14.5, time="away",
                   mercado="Total do time - Kills - mapa 1")])
        self.assertEqual(len(r["todas"]), 1)
        self.assertAlmostEqual(r["todas"][0]["probabilidade"],
                               ab.prob_total_do_time_acima(11.0, 14.5))

    def test_total_do_time_SEM_saber_o_time_fica_de_fora_com_motivo(self):
        r = self._montar([
            _linha(tipo="team_total", lado="over", pontos=14.5,
                   mercado="Total do time - Kills - mapa 1")])
        self.assertEqual(r["todas"], [])
        self.assertIn("de qual time", r["mercado_completo"][0]["motivo"])

    def test_confronto_sem_modelo_diz_isso(self):
        r = self._montar([_linha(tipo="spread", lado="home", pontos=-6.5,
                                 mercado="Handicap - Kills - mapa 1")], medias=None)
        self.assertEqual(r["todas"], [])
        self.assertIn("nao conseguiu", r["mercado_completo"][0]["motivo"])

    def test_vencedor_de_kills_continua_fora(self):
        """Quem faz mais kills tem EMPATE possivel, e a regra de empate da
        casa nao esta no dado: fica sem numero, com motivo."""
        r = self._montar([_linha(tipo="moneyline", lado="home",
                                 mercado="Vencedor - Kills - mapa 1")])
        self.assertEqual(r["todas"], [])


class TestMercadoReprovadoDizIsso(unittest.TestCase):

    def test_motivo_de_mercado_que_nao_passou(self):
        from app_core.oportunidades_core import _motivo_kills_por_time
        guardado = dict(ab._CACHE)
        try:
            ab._CACHE.clear()
            ab._CACHE[str(ab.ARTEFATO)] = {"handicap": {"quantis": list(range(60))}}
            m = _motivo_kills_por_time(_linha(tipo="team_total", lado="over", pontos=14.5,
                                              time="home"))
            self.assertIn("nao passou na prova cega", m)
            m = _motivo_kills_por_time(_linha(tipo="spread", lado="home", pontos=-4.5))
            self.assertIn("nao conseguiu", m)
        finally:
            ab._CACHE.clear()
            ab._CACHE.update(guardado)


if __name__ == "__main__":
    unittest.main()
