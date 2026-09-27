# -*- coding: utf-8 -*-
"""Testes da chave de grupo do calibrador — mlcore/agrupamentos.py.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ Dívida D10 (06/09/2026): o aplicador cravava a chave como f"{bo}|{phase}",  ║
║ então dos quatro agrupamentos que a validação cruzada pode escolher só UM   ║
║ era representável. Ao reajustar contra o modelo de 06/09 a VC escolheu      ║
║ "formato BO", a paridade acusou 2,8e-01 e o ajustador abortou — deixando o  ║
║ bloco em produção preso a um ajuste sobre modelo velho.                     ║
║                                                                             ║
║ O que estes testes travam:                                                  ║
║  - o artefato se DESCREVE: o bloco diz seu agrupamento e o app lê, em vez   ║
║    de adivinhar. O MLflow guarda a assinatura junto do modelo e valida a    ║
║    entrada "sem exigir lógica de validação própria no código de serviço".   ║
║    https://mlflow.org/docs/latest/ml/model/signatures/                      ║
║    (A leitura "isso evita cravar suposições no serviço" é MINHA, não da     ║
║    página — mas aqui foi medida: cravar `bo|phase` bloqueou o reajuste.)    ║
║  - há UMA definição de chave, usada por quem ajusta e por quem serve. O     ║
║    Feast: as definições de treino têm que bater EXATAMENTE com as do        ║
║    serviço online, e quando não batem "introduz-se divergência              ║
║    treino/serviço, que pode causar problemas catastróficos e difíceis de    ║
║    depurar". https://feast.dev/blog/what-is-a-feature-store/                ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import unittest

from mlcore.agrupamentos import (
    AGRUPAMENTOS,
    PADRAO_LEGADO,
    chave,
    chave_de_amostra,
    nome_do_bloco,
)


class _Amostra:
    def __init__(self, bo, phase):
        self.bo = bo
        self.phase = phase


class TestChaveDeGrupo(unittest.TestCase):

    def test_os_quatro_agrupamentos_montam_chave(self):
        """O caso da D10: nenhum deles pode ficar sem representacao."""
        esperado = {
            "nenhum (so global)": "*",
            "formato BO": "3",
            "fase": "regular",
            "formato + fase": "3|regular",
        }
        self.assertEqual(set(esperado), set(AGRUPAMENTOS),
                         "a lista de agrupamentos mudou sem os testes junto")
        for nome, alvo in esperado.items():
            with self.subTest(nome=nome):
                self.assertEqual(chave(nome, "3", "regular"), alvo)

    def test_agrupamento_desconhecido_LEVANTA(self):
        """Melhor falhar alto do que servir o global calado achando que
        aplicou o grupo — que e o modo de falha da D10."""
        with self.assertRaises(KeyError):
            chave("formato + liga + lua cheia", "3", "regular")

    def test_quem_ajusta_e_quem_serve_usam_a_MESMA_chave(self):
        """A regra da fonte 2: uma definicao so."""
        for nome in AGRUPAMENTOS:
            for bo, ph in (("1", "regular"), ("3", "playoffs"), ("5", "regular")):
                with self.subTest(nome=nome, bo=bo, ph=ph):
                    self.assertEqual(chave(nome, bo, ph),
                                     chave_de_amostra(nome, _Amostra(bo, ph)))

    def test_bo_e_phase_sao_STRING(self):
        """Comparar com int faria a chave nunca casar e o grupo sumir calado."""
        self.assertEqual(chave("formato BO", 3, "regular"), "3")
        self.assertEqual(chave("formato + fase", 3, "regular"), "3|regular")

    def test_valor_ausente_nao_estoura(self):
        self.assertEqual(chave("formato + fase", None, None), "|")
        self.assertEqual(chave("fase", "3", None), "")

    def test_bloco_declara_o_agrupamento(self):
        self.assertEqual(nome_do_bloco({"agrupamento": "formato BO"}), "formato BO")

    def test_bloco_ANTIGO_sem_o_campo_cai_no_legado(self):
        """Compatibilidade: o bloco gravado antes de 06/09 nao tem o campo, e
        so podia estar em producao com o unico agrupamento que o app sabia."""
        self.assertEqual(nome_do_bloco({"niveis": [], "w": []}), PADRAO_LEGADO)
        self.assertEqual(PADRAO_LEGADO, "formato + fase")
        self.assertEqual(nome_do_bloco(None), PADRAO_LEGADO)


if __name__ == "__main__":
    unittest.main()
