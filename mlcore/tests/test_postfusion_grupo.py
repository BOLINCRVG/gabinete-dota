# -*- coding: utf-8 -*-
"""Paridade entre quem AJUSTA o calibrador e quem o APLICA.

Este teste existe por causa do achado de 26/08/2026: o problema do pré-jogo
nasceu de reimplementação — três versões da mesma probabilidade, cada uma
parando num ponto diferente, porque o caminho servido não era chamável. É
literalmente o "divergent code paths" da literatura de training/serving skew.

Então o calibrador com grupo entra com o teste que impede o mesmo erro: o que
o app calcula tem que bater, dígito a dígito, com o que o ajuste produziu.

Trava também o comportamento sem coeficientes: sem o bloco em params, nada
muda — é o que torna a mudança reversível removendo o bloco.
"""
from __future__ import annotations

import math
import random
import unittest

from app_core.postfusion_calibration_core import _calibrador_com_grupo
from mlcore.calibration import _fit_logistic
from mlcore.utils import sigmoid


def _ajusta_como_no_gate(amostras, chave, niveis, l2_grupo):
    """A MESMA matemática do tools/gate_calibrador_final.py::_ajusta.

    Repetida aqui de propósito: se um dia as duas divergirem, este teste
    quebra — que é o ponto."""
    def feats(a, p):
        z = math.log(p / (1 - p))
        f = [z, 1.0]
        for nv in niveis:
            ind = 1.0 if chave(a) == nv else 0.0
            f += [ind * z, ind]
        return f
    X = [feats(a, p) for a, p, _ in amostras]
    pen = [1e-6, 1e-6] + [l2_grupo] * (2 * len(niveis))
    w = _fit_logistic(X, [float(y) for _, _, y in amostras], l2=pen, max_iter=300)
    return w, feats


class PostfusionGrupoTest(unittest.TestCase):

    def _amostras(self, n=1500, semente=11):
        """(objeto com bo/phase, p, y). Cada grupo com distorção PRÓPRIA — é o
        que o calibrador com grupo deve conseguir corrigir e o global não."""
        rnd = random.Random(semente)
        grupos = [("1", "regular", 0.55), ("3", "playoffs", 1.45), ("5", "regular", 1.0)]
        fora = []
        for _ in range(n):
            bo, fase, distorcao = grupos[rnd.randrange(len(grupos))]
            z = rnd.gauss(0.0, 1.5)
            p_true = sigmoid(z)
            p_obs = min(max(sigmoid(z * distorcao), 1e-6), 1 - 1e-6)
            obj = type("A", (), {"bo": bo, "phase": fase})()
            fora.append((obj, p_obs, 1 if rnd.random() < p_true else 0))
        return fora

    def test_sem_coeficientes_nao_faz_nada(self):
        """Reversibilidade: sem o bloco em params, nao ha calibrador."""
        for vazio in (None, {}, {"niveis": [], "w": []}, {"niveis": ["1|regular"], "w": [1.0]}):
            self.assertIsNone(_calibrador_com_grupo(vazio, "1", "regular"),
                              f"deveria devolver None para {vazio!r}")

    def test_o_app_calcula_o_MESMO_que_o_ajuste(self):
        """A paridade que o projeto inteiro aprendeu a exigir."""
        am = self._amostras()
        chave = lambda a: f"{a.bo}|{a.phase}"          # noqa: E731
        niveis = sorted({chave(a) for a, _, _ in am})[1:]
        w, feats = _ajusta_como_no_gate(am, chave, niveis, l2_grupo=1.0)

        aplicar_por_grupo = {}
        for nv in [*niveis, sorted({chave(a) for a, _, _ in am})[0]]:
            bo, fase = nv.split("|")
            aplicar_por_grupo[nv] = _calibrador_com_grupo({"niveis": niveis, "w": w}, bo, fase)

        for a, p, _ in am[:400]:
            esperado = sigmoid(sum(wi * fi for wi, fi in zip(w, feats(a, p))))
            obtido = aplicar_por_grupo[chave(a)](p)
            self.assertAlmostEqual(obtido, esperado, places=10,
                                   msg=f"divergiu no grupo {chave(a)} com p={p:.4f}")

    def test_grupo_de_referencia_usa_so_os_termos_globais(self):
        """O primeiro nivel nao aparece na lista: e a referencia, e tem que
        sair exatamente igual ao calibrador global."""
        w = [1.3, -0.2, 0.7, 0.1]
        cal_ref = _calibrador_com_grupo({"niveis": ["3|playoffs"], "w": w}, "1", "regular")
        for p in (0.05, 0.3, 0.5, 0.77, 0.95):
            z = math.log(p / (1 - p))
            self.assertAlmostEqual(cal_ref(p), sigmoid(w[0] * z + w[1]), places=10)

    def test_corrige_distorcao_que_o_global_nao_corrige(self):
        """O ponto da coisa: com distorcao DIFERENTE por grupo, o calibrador
        com grupo tem que bater o global. Se isso quebrar, a feature perdeu o
        motivo de existir."""
        am = self._amostras(n=3000, semente=5)
        chave = lambda a: f"{a.bo}|{a.phase}"          # noqa: E731
        niveis = sorted({chave(a) for a, _, _ in am})[1:]
        w_g, _ = _ajusta_como_no_gate(am, chave, niveis, l2_grupo=1.0)
        w_o, _ = _ajusta_como_no_gate(am, lambda a: "*", [], l2_grupo=1.0)

        def perda(usar_grupo):
            tot = 0.0
            for a, p, y in am:
                if usar_grupo:
                    q = _calibrador_com_grupo({"niveis": niveis, "w": w_g}, a.bo, a.phase)(p)
                else:
                    z = math.log(p / (1 - p))
                    q = sigmoid(w_o[0] * z + w_o[1])
                q = min(max(q, 1e-12), 1 - 1e-12)
                tot += -(math.log(q) if y == 1 else math.log(1 - q))
            return tot / len(am)

        self.assertLess(perda(True), perda(False),
                        "com grupo tem que ganhar do global quando a distorcao difere por grupo")


if __name__ == "__main__":
    unittest.main()
