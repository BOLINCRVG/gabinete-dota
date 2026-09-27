# -*- coding: utf-8 -*-
"""Testes da varredura de falha silenciosa — tools/find_silent_failures.py.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ A familia 5 (`getattr` com default lendo campo de outra camada) nasceu da   ║
║ divida D5, em 08/09/2026: o replay lia `getattr(s, "delta_players_eff",     ║
║ 0.0)` de um objeto `Sample` que NAO TEM campo de delta. O default voltava   ║
║ calado em 100% dos jogos, e a nota oficial passou a medir o Elo sozinho.    ║
║                                                                             ║
║ "Errors should never pass silently"; `except: pass` e o pior anti-padrao;   ║
║ e "e melhor falhar rapido do que esconder um erro que causa bug la na       ║
║ frente". O `getattr` com default e a mesma coisa para leitura de atributo.  ║
║ https://pybit.es/articles/python-errors-should-not-pass-silently/           ║
║ (conferida abrindo a pagina em 08/09/2026)                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import ast
import importlib.util as _iu
import subprocess
import sys
import unittest
try:
    from mlcore.tests._repo import E_O_DOTA as _E_O_DOTA
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import E_O_DOTA as _E_O_DOTA
if _E_O_DOTA:
    raise unittest.SkipTest("so no LoL: so vale para o LoL")
from pathlib import Path


def _raiz() -> Path:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                       cwd=str(Path(__file__).resolve().parent),
                       capture_output=True, text=True, check=True)
    return Path(r.stdout.strip())


ROOT = _raiz()
_spec = _iu.spec_from_file_location("_fsf", ROOT / "tools" / "find_silent_failures.py")
_fsf = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_fsf)


def _roda():
    """`sys.executable`, nunca "python" do PATH: o PATH pode apontar para
    outro interpretador, e o teste passaria a medir outra maquina. E a saida
    e lida como UTF-8 de proposito — no Windows o padrao e cp1252 e a
    ferramenta imprime acento."""
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "find_silent_failures.py")],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace")


def _acha(codigo: str):
    return _fsf.getattr_com_default(ast.parse(codigo))


class TestGetattrComDefault(unittest.TestCase):

    def test_o_caso_real_da_D5(self):
        achados = _acha('x = getattr(s, "delta_players_eff", 0.0)')
        self.assertEqual([(1, "delta_players_eff")], achados)

    def test_default_None_NAO_conta(self):
        """`None` e o idioma CERTO: obriga quem chama a tratar."""
        self.assertEqual([], _acha('t = getattr(row, "tier", None)'))

    def test_defaults_plausiveis_contam(self):
        for d in ("0.0", "0", '""', "[]", "{}"):
            with self.subTest(default=d):
                self.assertEqual(1, len(_acha(f'v = getattr(o, "campo", {d})')),
                                 f"default {d} deveria contar")

    def test_getattr_de_dois_argumentos_nao_conta(self):
        """Sem default ele ESTOURA — que e o comportamento desejado."""
        self.assertEqual([], _acha('v = getattr(o, "campo")'))

    def test_nome_dinamico_nao_conta(self):
        """So da para conferir o que e literal."""
        self.assertEqual([], _acha('v = getattr(o, nome, 0.0)'))

    def test_a_linha_certa_e_reportada(self):
        achados = _acha('a = 1\nb = 2\nc = getattr(o, "x", 0)')
        self.assertEqual([(3, "x")], achados)


class TestOutrasFamilias(unittest.TestCase):
    """As familias que ja existiam continuam funcionando."""

    def test_excecao_engolida(self):
        achados = _fsf.excecao_engolida(ast.parse(
            "try:\n    f()\nexcept ValueError:\n    pass\n"))
        self.assertEqual(1, len(achados))

    def test_excecao_que_faz_algo_nao_conta(self):
        achados = _fsf.excecao_engolida(ast.parse(
            "try:\n    f()\nexcept ValueError:\n    raise\n"))
        self.assertEqual([], achados)

    def test_coerce_com_fillna(self):
        linhas = _fsf.coerce_com_fillna(ast.parse(
            'x = pd.to_numeric(s, errors="coerce").fillna(0)'))
        self.assertEqual([1], linhas)


class TestAVarreduraRodaNoRepo(unittest.TestCase):

    def test_a_ferramenta_roda_sem_estourar(self):
        r = _roda()
        self.assertEqual(0, r.returncode, r.stderr[-400:])
        self.assertIn("=== 5)", r.stdout)

    def test_o_caso_da_D5_APARECE_no_relatorio(self):
        """A primeira versao ACHAVA e ESCONDIA: o relatorio mostrava os 12
        maiores por CONTAGEM, e o caso que originou a familia tem UMA
        ocorrencia. Contagem alta em arquivo de tela e ruido; uma ocorrencia
        no nucleo de medicao apaga uma nota."""
        trecho = _roda().stdout.split("=== 5)")[-1]
        self.assertIn("replay_pre_jogo", trecho,
                      "o caso da D5 sumiu do relatorio da familia 5")


if __name__ == "__main__":
    unittest.main()
