"""Trava a simetria do swap na Visão Geral: p(A,B) == 1 − p(B,A).

Bug histórico: trocar Time1/Time2 mudava as odds do MESMO jogo. A causa: a
simetria era aplicada só no core, e microseg/fused (dependentes de ordem) a
quebravam depois. O fix simetriza a SAÍDA (pipeline completo nas duas direções).

Aqui usamos mocks DELIBERADAMENTE assimétricos no core, no microseg E no fused —
se a saída sair simétrica mesmo assim, a simetria está garantida estruturalmente.
"""
from __future__ import annotations

import unittest

from app_core.ml_prediction_core import compute_ml_core_prediction

STR = {"A": 3.0, "B": 1.0}      # forças diferentes
LG = {"A": "LCK", "B": "LPL"}   # ligas diferentes (microseg/fused dependem da liga)


def _resolve_leagues(ta, tb, yr, lm, fl):
    return {"league_a": LG.get(ta, ""), "league_b": LG.get(tb, ""), "tier_a": "T1", "tier_b": "T1"}


def _p_map_v2(artifact, ta, tb, as_of, **kw):
    # core ASSIMÉTRICO: razão de força + viés de lado (blue) de 0.04.
    base = STR[ta] / (STR[ta] + STR[tb])
    p = min(0.98, max(0.02, base + 0.04))
    return {"p_raw": p, "p_cal": p}


def _microseg(p, **kw):
    # mapa linear ASSIMÉTRICO (a+2b != 1) → microseg(1-p) != 1-microseg(p).
    return (0.8 * float(p) + 0.05, {"applied": True})


def _fused(p, **kw):
    return (0.9 * float(p) + 0.03, {"applied": True})


def _compute(team_a, team_b):
    return compute_ml_core_prediction(
        team_a=team_a, team_b=team_b, artifact_path="x", as_of_date=None,
        achievement_path=None,
        ml_runtime_cfg={"ml_microseg_enabled": True, "ml_fused_post_cal_use_fitted": True},
        filter_year=2026, filter_league=None, filter_split=None, filter_playoffs=None,
        bo=3, league_mode="auto", csv_sig=("x",), art_sig=("x",), fit_params_path=None,
        resolve_leagues_fn=_resolve_leagues, p_map_v2_fn=_p_map_v2,
        build_microseg_table_fn=lambda **kw: {"dummy": 1},
        apply_microseg_fn=_microseg, apply_fused_postcal_fn=_fused,
    )


class TestSwapSymmetry(unittest.TestCase):

    def test_p_map_cal_symmetric_under_swap(self):
        r_ab = _compute("A", "B")
        r_ba = _compute("B", "A")
        p_ab = r_ab["p_map_cal"]
        p_ba = r_ba["p_map_cal"]
        self.assertAlmostEqual(p_ab, 1.0 - p_ba, places=9,
                               msg=f"swap assimétrico: p(A,B)={p_ab} vs 1-p(B,A)={1 - p_ba}")

    def test_p_map_raw_symmetric_under_swap(self):
        r_ab = _compute("A", "B")
        r_ba = _compute("B", "A")
        self.assertAlmostEqual(r_ab["p_map_raw"], 1.0 - r_ba["p_map_raw"], places=9)

    def test_actually_nontrivial(self):
        # Sanidade: não é só 0.5 (A é bem mais forte que B → p(A) > 0.5).
        r_ab = _compute("A", "B")
        self.assertGreater(r_ab["p_map_cal"], 0.5)


if __name__ == "__main__":
    unittest.main()
