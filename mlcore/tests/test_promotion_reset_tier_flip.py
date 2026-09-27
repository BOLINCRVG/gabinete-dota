# -*- coding: utf-8 -*-
"""Trava o BUG CRÍTICO do reset por ida-e-volta de tier (achado com ele 24/07).

LCK (T1) -> KeSPA Cup (MINOR) -> LCK (T1) era lido como "promoção" e ZERAVA
rating + games_played. Os 10 times da LCK ficaram com 2-14 jogos em vez de
300-500 (Gen.G: 5 resets, o último 2 dias antes do fim dos dados -> gp=2).
Promoção REAL (LCKC -> LCK pela primeira vez) tem que continuar resetando.
"""
import unittest

from mlcore.train_offline import _maybe_reset_on_promotion


class _FakeModel:
    def __init__(self, team, rating=250.0, gp=400):
        self.ratings = {team: rating}
        self.games_played = {team: gp}


class ResetPorTrocaDeTierTests(unittest.TestCase):

    def test_ida_e_volta_em_torneio_nao_zera(self):
        """Gen.G: LCK(T1) -> KeSPA(MINOR) -> LCK(T1). Historico tem que sobreviver."""
        m = _FakeModel("Gen.G")
        last: dict = {}
        _maybe_reset_on_promotion(m, "Gen.G", "T1", last_tier=last)     # LCK
        _maybe_reset_on_promotion(m, "Gen.G", "MINOR", last_tier=last)  # KeSPA Cup
        _maybe_reset_on_promotion(m, "Gen.G", "T1", last_tier=last)     # volta LCK
        self.assertEqual(m.games_played["Gen.G"], 400, "voltar da KeSPA zerou o time (bug)")
        self.assertEqual(m.ratings["Gen.G"], 250.0)

    def test_varias_idas_e_voltas_nao_zeram(self):
        m = _FakeModel("T1")
        last: dict = {}
        for tier in ("T1", "MINOR", "T1", "MINOR", "T1", "MINOR", "T1"):
            _maybe_reset_on_promotion(m, "T1", tier, last_tier=last)
        self.assertEqual(m.games_played["T1"], 400)

    def test_promocao_real_ainda_zera(self):
        """Time de liga fraca que sobe pela PRIMEIRA vez: reset continua valendo."""
        m = _FakeModel("Academy", rating=900.0, gp=120)
        last: dict = {}
        _maybe_reset_on_promotion(m, "Academy", "MINOR", last_tier=last)
        _maybe_reset_on_promotion(m, "Academy", "T1", last_tier=last)  # promoção real
        self.assertEqual(m.games_played["Academy"], 0)
        self.assertEqual(m.ratings["Academy"], 0.0)

    def test_torneio_internacional_nunca_zera(self):
        """LCS(T2) -> MSI(T1) -> LCS(T2): ir ao internacional nao e promocao.
        Medido 24/07: 61 times sofriam isso; FlyQuest/Team Liquid 8x cada."""
        m = _FakeModel("FlyQuest", rating=180.0, gp=300)
        last: dict = {}
        _maybe_reset_on_promotion(m, "FlyQuest", "T2", last_tier=last, league="LCS")
        _maybe_reset_on_promotion(m, "FlyQuest", "T1", last_tier=last, league="MSI")
        _maybe_reset_on_promotion(m, "FlyQuest", "T2", last_tier=last, league="LCS")
        _maybe_reset_on_promotion(m, "FlyQuest", "T1", last_tier=last, league="WLDs")
        self.assertEqual(m.games_played["FlyQuest"], 300, "internacional zerou o time (bug)")
        self.assertEqual(m.ratings["FlyQuest"], 180.0)

    def test_promocao_domestica_real_ainda_zera_com_liga(self):
        """Subir de liga DOMESTICA (LCKC -> LCK) continua resetando."""
        m = _FakeModel("Academy", rating=900.0, gp=120)
        last: dict = {}
        _maybe_reset_on_promotion(m, "Academy", "MINOR", last_tier=last, league="LCKC")
        _maybe_reset_on_promotion(m, "Academy", "T1", last_tier=last, league="LCK")
        self.assertEqual(m.games_played["Academy"], 0)

    def test_primeiro_tier_nao_zera(self):
        m = _FakeModel("Novo", gp=5)
        last: dict = {}
        _maybe_reset_on_promotion(m, "Novo", "T1", last_tier=last)
        self.assertEqual(m.games_played["Novo"], 5)


if __name__ == "__main__":
    unittest.main()
