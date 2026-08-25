# -*- coding: utf-8 -*-
"""Fotos por minuto do Dota: extração e troca de lado.

A aba "Durante o mapa (estado → vitória)" nunca funcionou no Dota por dois
motivos: o orquestrador de treino não existe no clone, e o CSV não tinha as
fotos por minuto que o modelo consome. Este é o segundo motivo, resolvido.

O erro que estes testes existem para pegar: RADIANT é o lado "Blue". Na linha do
dire, o próprio time é o `opp_` do store e vice-versa. Trocar errado inverteria
a vantagem de ouro de metade da base — e o modelo aprenderia o contrário, sem
nada apitar.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opendota_timeline_sync import CAMPOS, MINUTOS, colunas, extrair


def _partida(dur_s: int) -> dict:
    def jog(rad: bool, ouro: int, kills_em: list[int]) -> dict:
        return {"isRadiant": rad,
                "gold_t": [ouro * i for i in range(60)],
                "xp_t": [ouro * 2 * i for i in range(60)],
                "lh_t": [i for i in range(60)],
                "kills_log": [{"time": t} for t in kills_em]}
    return {"match_id": 42, "duration": dur_s,
            "players": [jog(True, 100, [300]) for _ in range(5)]
                       + [jog(False, 90, [600, 1200]) for _ in range(5)]}


class TestExtracao(unittest.TestCase):
    def test_colunas_sao_as_que_o_modelo_espera(self) -> None:
        c = colunas()
        for t in MINUTOS:
            for campo in CAMPOS:
                self.assertIn(f"{campo}{t}", c)
                self.assertIn(f"opp_{campo}{t}", c)

    def test_ouro_soma_o_time_inteiro(self) -> None:
        r = extrair(_partida(3000))
        self.assertEqual(r["goldat10"], 5 * 100 * 10)     # 5 jogadores
        self.assertEqual(r["opp_goldat10"], 5 * 90 * 10)

    def test_mortes_de_um_lado_sao_os_abates_do_outro(self) -> None:
        r = extrair(_partida(3000))
        for t in MINUTOS:
            self.assertEqual(r[f"deathsat{t}"], r[f"opp_killsat{t}"])
            self.assertEqual(r[f"opp_deathsat{t}"], r[f"killsat{t}"])

    def test_kills_contam_so_o_que_ocorreu_ate_o_minuto(self) -> None:
        r = extrair(_partida(3000))
        # radiant: 5 jogadores x 1 abate aos 300s (5min) -> ja conta aos 10
        self.assertEqual(r["killsat10"], 5.0)
        # dire: abates aos 600s (10min) e 1200s (20min)
        self.assertEqual(r["opp_killsat10"], 5.0)    # so o de 600s
        self.assertEqual(r["opp_killsat25"], 10.0)   # os dois

    def test_partida_curta_fica_VAZIA_e_nunca_zero(self) -> None:
        """Zero significaria 'ninguem tinha ouro' — mentira que o modelo comeria."""
        r = extrair(_partida(12 * 60))   # acabou aos 12min
        self.assertIsNotNone(r["goldat10"])
        for t in (15, 20, 25):
            self.assertIsNone(r[f"goldat{t}"])
            self.assertIsNone(r[f"opp_goldat{t}"])

    def test_assistencias_ficam_zero_e_isso_e_declarado(self) -> None:
        """A API nao traz assistencia por minuto: feature constante, modelo ignora."""
        r = extrair(_partida(3000))
        for t in MINUTOS:
            self.assertEqual(r[f"assistsat{t}"], 0.0)
            self.assertEqual(r[f"opp_assistsat{t}"], 0.0)

    def test_partida_sem_jogadores_devolve_nada(self) -> None:
        self.assertIsNone(extrair({"match_id": 1, "duration": 2000, "players": []}))


if __name__ == "__main__":
    unittest.main()
