# -*- coding: utf-8 -*-
"""Testes do selo de versao do codigo.

POR QUE ELE EXISTE (15/09/2026)
Tres vezes seguidas ele disse "nao mudou nada" sobre conserto que eu ja tinha
publicado. Nas tres o codigo estava no GitHub e o app rodava versao antiga: a
hospedagem recarrega o arquivo principal quando chega um commit, mas nao
recarrega os modulos importados — so o reinicio baixa tudo.

O problema NAO era o conserto. Era nao dar para ver: ele olhando a tela, eu
olhando o repositorio, sem nada em comum para comparar. Agora a tela diz a
versao, e "mudou ou nao?" vira um olhar.

PRINCIPIO IV manda aqui tambem: quando nao da para saber a versao, o selo diz
que nao sabe. Um commit inventado seria pior que selo nenhum — daria a
confianca sem a informacao.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from app_core.versao_do_codigo_core import versao_do_codigo


class _ComPasta(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _git(self, head: str, *, ref_valor: str = "", packed: str = ""):
        g = self.raiz / ".git"
        g.mkdir(exist_ok=True)
        (g / "HEAD").write_text(head, encoding="utf-8")
        if ref_valor:
            ref = head.split(":", 1)[1].strip()
            alvo = g / ref
            alvo.parent.mkdir(parents=True, exist_ok=True)
            alvo.write_text(ref_valor, encoding="utf-8")
        if packed:
            (g / "packed-refs").write_text(packed, encoding="utf-8")

    def _codigo(self, nome: str = "app_core/x.py"):
        p = self.raiz / nome
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# nada\n", encoding="utf-8")
        return p


class TestLeOCommit(_ComPasta):
    """Le o .git na mao: a hospedagem pode nao ter o programa `git`."""

    def test_le_o_commit_do_ramo_atual(self):
        self._git("ref: refs/heads/master\n",
                  ref_valor="ebb2cbb1234567890abcdef1234567890abcdef\n")
        self.assertEqual(versao_do_codigo(self.raiz)["commit"], "ebb2cbb")

    def test_le_o_commit_quando_o_HEAD_esta_solto(self):
        """Copia da hospedagem as vezes fica em HEAD destacado."""
        self._git("ebb2cbb1234567890abcdef1234567890abcdef\n")
        self.assertEqual(versao_do_codigo(self.raiz)["commit"], "ebb2cbb")

    def test_le_do_packed_refs_quando_o_ref_nao_tem_arquivo(self):
        """Clone recente empacota os refs; sem isto o selo ficaria mudo."""
        self._git("ref: refs/heads/master\n",
                  packed="# pack-refs with: peeled\n"
                         "ebb2cbb1234567890abcdef1234567890abcdef refs/heads/master\n")
        self.assertEqual(versao_do_codigo(self.raiz)["commit"], "ebb2cbb")

    def test_git_como_ARQUIVO_aponta_para_a_pasta(self):
        """Worktree: .git e um arquivo com 'gitdir: <caminho>'."""
        real = self.raiz / "git_de_verdade"
        real.mkdir()
        (real / "HEAD").write_text("abc1234567890\n", encoding="utf-8")
        (self.raiz / ".git").write_text("gitdir: {}\n".format(real), encoding="utf-8")
        self.assertEqual(versao_do_codigo(self.raiz)["commit"], "abc1234")


class TestQuandoNaoDaParaSaber(_ComPasta):
    """PRINCIPIO IV: commit inventado da a confianca sem a informacao."""

    def test_sem_git_o_commit_e_None(self):
        self._codigo()
        self.assertIsNone(versao_do_codigo(self.raiz)["commit"])

    def test_git_quebrado_nao_estoura(self):
        (self.raiz / ".git").mkdir()
        (self.raiz / ".git" / "HEAD").write_text("lixo\n", encoding="utf-8")
        v = versao_do_codigo(self.raiz)
        self.assertTrue(isinstance(v, dict))

    def test_ref_apontando_para_o_nada_devolve_None(self):
        self._git("ref: refs/heads/inexistente\n")
        self.assertIsNone(versao_do_codigo(self.raiz)["commit"])

    def test_pasta_vazia_diz_que_NAO_SABE(self):
        v = versao_do_codigo(self.raiz)
        self.assertIsNone(v["commit"])
        self.assertIsNone(v["baixado_em"])
        self.assertIn("desconhecida", v["texto"])


class TestQuandoOCodigoFoiBaixado(_ComPasta):
    """Numa copia clonada, a data do arquivo mais novo e quando a hospedagem
    baixou — que e exatamente a pergunta "o meu conserto chegou?"."""

    def test_pega_a_data_do_arquivo_de_codigo_mais_novo(self):
        antigo = self._codigo("app_core/antigo.py")
        novo = self._codigo("mlcore/novo.py")
        os.utime(antigo, (1_600_000_000, 1_600_000_000))
        os.utime(novo, (1_700_000_000, 1_700_000_000))
        v = versao_do_codigo(self.raiz)
        self.assertEqual(int(v["baixado_em"].timestamp()), 1_700_000_000)

    def test_olha_as_tres_pastas_de_codigo(self):
        for pasta in ("app_core", "mlcore", "tools"):
            self.raiz.joinpath(pasta).mkdir(exist_ok=True)
        p = self._codigo("tools/z.py")
        os.utime(p, (1_700_000_000, 1_700_000_000))
        self.assertIsNotNone(versao_do_codigo(self.raiz)["baixado_em"])

    def test_ignora_arquivo_que_nao_e_codigo(self):
        (self.raiz / "app_core").mkdir()
        (self.raiz / "app_core" / "dados.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(versao_do_codigo(self.raiz)["baixado_em"])


class TestOTextoQueAparece(_ComPasta):

    def test_mostra_commit_e_data_juntos(self):
        self._git("ref: refs/heads/master\n",
                  ref_valor="ebb2cbb1234567890abcdef1234567890abcdef\n")
        self._codigo()
        t = versao_do_codigo(self.raiz)["texto"]
        self.assertIn("ebb2cbb", t)
        self.assertIn("baixado", t)

    def test_so_com_data_ainda_diz_algo_util(self):
        self._codigo()
        t = versao_do_codigo(self.raiz)["texto"]
        self.assertIn("baixado", t)
        self.assertNotIn("desconhecida", t)

    def test_o_selo_entra_no_topo_do_app(self):
        import inspect

        from app_core import viewer_mode_core as vm
        src = inspect.getsource(vm.render_viewer_badge)
        self.assertIn("versao_do_codigo_core", src)


class TestNoRepositorioDeVerdade(unittest.TestCase):
    """O caso que importa: rodando neste repositorio, o selo diz a versao."""

    def test_diz_o_commit_deste_repositorio(self):
        raiz = Path(__file__).resolve().parents[2]
        v = versao_do_codigo(raiz)
        if not (raiz / ".git").exists():
            self.skipTest("sem .git nesta copia")
        self.assertIsNotNone(v["commit"], "o selo ficou mudo no proprio repo")
        self.assertEqual(len(v["commit"]), 7)


if __name__ == "__main__":
    unittest.main()
