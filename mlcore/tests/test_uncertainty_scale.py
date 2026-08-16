# -*- coding: utf-8 -*-
"""Camada de incerteza (unc_scale_c): MEDIDA E REJEITADA em 16/08/2026.

Fica desligada por default (c=0.0) e o codigo permanece porque a medicao vale:
no TI 2026 ela ganhava +0.13 de logloss, mas cobrava -0.02 a -0.06 na season
inteira em TODA combinacao (c x n0) testada — botao de temperatura global, nao
remedio dirigido. Ver o gate em project_dota_ti_premium_bug.

Estes testes travam as propriedades que importam se alguem ligar o parametro:
sobretudo que o DEFAULT nao muda nada (senao o REJECT vaza pra producao calado).
"""
from __future__ import annotations

import unittest

from mlcore.elo import (
    EloModel,
    EloParams,
    flatten_p_by_uncertainty,
    uncertainty_scale_mult,
)


def _mk(c: float, n_blue: int, n_red: int, n0: float = 50.0, alpha: float = 0.0) -> float:
    m = EloModel(EloParams(alpha=alpha, unc_scale_c=c, unc_n_prior=n0))
    m.ratings = {"top": 200.0, "novato": 0.0}
    m.games_played = {"top": n_blue, "novato": n_red}
    return m.p_blue("top", "novato")


class TestUncertaintyScale(unittest.TestCase):
    def test_default_desligado_nao_muda_nada(self) -> None:
        """O default TEM que ser inerte — a camada foi reprovada no gate."""
        self.assertEqual(EloParams().unc_scale_c, 0.0)
        self.assertEqual(uncertainty_scale_mult(0, 0, 0.0, 50.0), 1.0)
        self.assertEqual(flatten_p_by_uncertainty(0.9, 0, 0, 0.0, 50.0), 0.9)
        self.assertAlmostEqual(_mk(0.0, 450, 0), _mk(0.0, 450, 0), places=12)

    def test_artifact_antigo_carrega_como_legado(self) -> None:
        m = EloModel(EloParams(unc_scale_c=1.5, unc_n_prior=25.0))
        d = m.to_dict()
        self.assertEqual(EloModel.from_dict(d).params.unc_scale_c, 1.5)
        self.assertEqual(EloModel.from_dict(d).params.unc_n_prior, 25.0)
        d["params"].pop("unc_scale_c")
        d["params"].pop("unc_n_prior")
        velho = EloModel.from_dict(d)
        self.assertEqual(velho.params.unc_scale_c, 0.0)
        self.assertEqual(velho.params.unc_n_prior, 50.0)

    def test_nunca_inverte_o_favorito_nem_aumenta_confianca(self) -> None:
        for p in (0.05, 0.3, 0.5, 0.72, 0.99):
            q = flatten_p_by_uncertainty(p, 0, 0, 2.0, 50.0)
            self.assertGreaterEqual((p - 0.5) * (q - 0.5), 0.0, f"inverteu em p={p}")
            self.assertLessEqual(abs(q - 0.5), abs(p - 0.5) + 1e-12, f"mais confiante em p={p}")

    def test_achata_mais_quem_o_modelo_conhece_menos(self) -> None:
        desconhecido = abs(flatten_p_by_uncertainty(0.9, 0, 0, 2.0, 50.0) - 0.5)
        conhecido = abs(flatten_p_by_uncertainty(0.9, 500, 500, 2.0, 50.0) - 0.5)
        self.assertLess(desconhecido, conhecido)

    def test_vive_com_alpha_zero_que_e_o_core_whr(self) -> None:
        """O core WHR (default) zera alpha; amarrar a incerteza nele matava a camada."""
        legado = _mk(0.0, 450, 0, alpha=0.0)
        com = _mk(2.0, 450, 0, alpha=0.0)
        self.assertLess(com, legado - 0.02)

    def test_os_dois_caminhos_batem(self) -> None:
        """p_blue e o helper usado na calibracao do WHR precisam concordar."""
        legado = _mk(0.0, 450, 0, alpha=0.0)
        self.assertAlmostEqual(
            _mk(2.0, 450, 0, alpha=0.0),
            flatten_p_by_uncertainty(legado, 450, 0, 2.0, 50.0),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
