# -*- coding: utf-8 -*-
"""A validade das declaracoes da regua das 28 tem que MORDER.

As fontes do desenho (declaracao mora no repositorio medido; expiracao
trimestral contra "compliance drift") estao no bloco FONTES de
tools/regua_28.py, itens 6 e 7. Aqui so se prova que o mecanismo funciona:
carimbo que ninguem confere nao e controle, e decoracao.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
try:
    from mlcore.tests._repo import E_O_DOTA as _E_O_DOTA
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import E_O_DOTA as _E_O_DOTA
if _E_O_DOTA:
    raise unittest.SkipTest("so no LoL: so vale para o LoL")
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("_r28", ROOT / "tools" / "regua_28.py")
r28 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r28)


def _dias_atras(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


class TestValidadeDeclaracao(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        r28._declaracoes.__dict__.pop("_c", None)
        self.addCleanup(lambda: r28._declaracoes.__dict__.pop("_c", None))

    def _escreve(self, *, repo=None, **campos):
        p = self.raiz / r28.ARQ_DECLARACOES
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"repo": repo or self.raiz.name,
                                 "declaracoes": {"dados 1": campos}}), encoding="utf-8")
        return r28.declarado("dados", 1)(self.raiz)

    def test_sem_arquivo_nenhum_vale_zero(self):
        nota, ev = r28.declarado("dados", 1)(self.raiz)
        self.assertEqual(nota, 0.0)
        self.assertIn("NAO DECLARADO", ev)

    def test_item_ausente_no_arquivo_vale_zero(self):
        p = self.raiz / r28.ARQ_DECLARACOES
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"repo": self.raiz.name,
                                 "declaracoes": {"modelo 2": {"nota": 1.0}}}), encoding="utf-8")
        self.assertEqual(r28.declarado("dados", 1)(self.raiz)[0], 0.0)

    def test_sem_data_vale_zero(self):
        nota, ev = self._escreve(nota=1.0, justificativa="confie em mim")
        self.assertEqual(nota, 0.0)
        self.assertIn("SEM DATA", ev)

    def test_vencida_vale_zero(self):
        nota, ev = self._escreve(nota=1.0, justificativa="era verdade um dia",
                                 verificado_em=_dias_atras(r28.VALIDADE_DECLARACAO_DIAS + 1))
        self.assertEqual(nota, 0.0)
        self.assertIn("VENCIDA", ev)

    def test_dentro_do_prazo_vale_a_nota(self):
        nota, ev = self._escreve(nota=0.5, justificativa="meio ponto",
                                 verificado_em=_dias_atras(10))
        self.assertEqual(nota, 0.5)
        self.assertIn("vence em", ev)

    def test_data_ilegivel_vale_zero(self):
        self.assertEqual(self._escreve(nota=1.0, justificativa="x",
                                       verificado_em="ontem")[0], 0.0)

    def test_declaracao_nao_vaza_entre_repositorios(self):
        """O bug que originou tudo isto: o .py e o mesmo nos dois Gabinetes, e
        medir o Dota herdava as afirmacoes escritas olhando o LoL."""
        self._escreve(nota=1.0, justificativa="so vale aqui", verificado_em=_dias_atras(1))
        with tempfile.TemporaryDirectory() as outro:
            nota, ev = r28.declarado("dados", 1)(Path(outro))
        self.assertEqual(nota, 0.0)
        self.assertIn("NAO DECLARADO", ev)

    def test_declaracao_copiada_para_outro_repo_vale_zero(self):
        """Mudar de pasta nao basta: o ARQUIVO tambem se copia. A declaracao
        nomeia o seu sujeito, e sujeito errado nao vira credito emprestado."""
        nota, _ = self._escreve(repo="gabinete-de-outra-pessoa", nota=1.0,
                                justificativa="roubada", verificado_em=_dias_atras(1))
        self.assertEqual(nota, 0.0)
        _, recusa = r28._declaracoes(self.raiz)
        self.assertIn("gabinete-de-outra-pessoa", recusa)

    def test_sem_campo_repo_vale_zero(self):
        p = self.raiz / r28.ARQ_DECLARACOES
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"declaracoes": {"dados 1": {"nota": 1.0}}}), encoding="utf-8")
        self.assertEqual(r28.declarado("dados", 1)(self.raiz)[0], 0.0)

    def test_toda_declaracao_deste_repo_tem_data(self):
        sem_data = [f"{a} {n} ({t})" for a, n, t, v in r28.ITENS
                    if "SEM DATA" in v(ROOT)[1]]
        self.assertEqual(sem_data, [], f"declaracoes sem data: {sem_data}")


if __name__ == "__main__":
    unittest.main()
