# -*- coding: utf-8 -*-
"""O artifact tem que dizer a VERDADE sobre até onde os ratings foram.

Achado 16/08/2026: `data_max_date` grava o fim de TODOS os dados, mas o treino
só aplica aos ratings até o corte do holdout. Com isso o update_incremental lia
"estou em dia" e se recusava a fechar um buraco de 30 dias — o modelo publicado
do Dota ficava permanentemente atrasado (Iron Wing: 16 jogos em 2026,
games_played=0).

`ratings_max_date` é o carimbo honesto. A distância entre os dois É a defasagem.
"""
from __future__ import annotations

import unittest

from mlcore.update_incremental import _parse_dt


def _escolhe(meta: dict):
    """Mesma precedência do update_incremental: ratings_max_date manda."""
    return _parse_dt(meta.get("ratings_max_date")) or _parse_dt(meta.get("data_max_date"))


class TestRatingsMaxDate(unittest.TestCase):
    def test_prefere_o_carimbo_honesto(self) -> None:
        d = _escolhe({"data_max_date": "2026-08-16", "ratings_max_date": "2026-07-17"})
        self.assertEqual(d.date().isoformat(), "2026-07-17")

    def test_artifact_antigo_sem_a_chave_nao_muda_de_comportamento(self) -> None:
        d = _escolhe({"data_max_date": "2026-08-16"})
        self.assertEqual(d.date().isoformat(), "2026-08-16")

    def test_em_dia_quando_os_dois_batem(self) -> None:
        meta = {"data_max_date": "2026-08-16", "ratings_max_date": "2026-08-16"}
        self.assertEqual(_escolhe(meta), _parse_dt(meta["data_max_date"]))

    def test_sem_carimbo_nenhum_devolve_nada(self) -> None:
        self.assertIsNone(_escolhe({}))

    def test_a_defasagem_e_a_distancia_entre_os_dois(self) -> None:
        meta = {"data_max_date": "2026-08-16", "ratings_max_date": "2026-07-17"}
        atraso = (_parse_dt(meta["data_max_date"]) - _parse_dt(meta["ratings_max_date"])).days
        self.assertEqual(atraso, 30)


if __name__ == "__main__":
    unittest.main()
