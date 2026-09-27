# -*- coding: utf-8 -*-
"""O SETUP.md diz a verdade? — dívida D3, fechada em 06/09/2026.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ O SETUP.md apodreceu do jeito mais traiçoeiro: alguém corrigiu o MEIO e     ║
║ esqueceu as PONTAS. O passo 4 dizia (certo) que o modelo vem no clone,      ║
║ enquanto o cabeçalho dizia (errado) que ele "vai por um zip separado, não   ║
║ entra no git de propósito". Quem lesse de cima para baixo recebia duas      ║
║ instruções opostas. O caminho do make_release.ps1 também estava errado.     ║
║                                                                             ║
║ POR QUE ISSO ACONTECE, e por que teste é a cura:                            ║
║ mudança de código passa por revisão, CI e merge; mudança de documento não   ║
║ passa por nada — e o erro só aparece meses depois, com alguém seguindo      ║
║ instrução que não corresponde mais à realidade.                             ║
║ https://understandingdata.com/posts/doc-drift-detection-ci/                 ║
║                                                                             ║
║ A FORMA: "docs as tests" — cada instrução carrega alegações sobre o         ║
║ produto (que um arquivo existe, que um comando está disponível), e essas    ║
║ alegações viram ações executáveis. Um teste que falha revela "ou um guia    ║
║ desatualizado, ou que uma atualização do produto quebrou um caminho que     ║
║ funcionava".                                                                ║
║ https://www.cherryleaf.com/2026/08/docs-as-tests/                           ║
║ (as duas conferidas abrindo as páginas em 06/09/2026)                       ║
║                                                                             ║
║ LIMITE HONESTO DESTE ARQUIVO: ele confere fatos verificáveis — o que está   ║
║ versionado, se as ferramentas citadas existem, se os caminhos batem. NÃO    ║
║ roda o passo a passo numa máquina limpa. Instrução certa porém incompleta   ║
║ continua passando aqui.                                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol


def _raiz() -> Path:
    saida = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           cwd=str(Path(__file__).resolve().parent),
                           capture_output=True, text=True, check=True)
    return Path(saida.stdout.strip())


ROOT = _raiz()
SETUP = ROOT / "SETUP.md"

# O que o passo 4 promete que vem no clone.
VERSIONADOS = ("ml_artifact.json", "params_fitted.json", "players_artifact.json",
               "CSV_HISTORICO", "CSV_DIARIO", "live_models", "evidence")
# O que ele promete que NAO vem, e por isso manda copiar a mao.
FORA_DO_GIT = ("ops/baselines", ".streamlit/secrets.toml")
# Toda ferramenta que o documento manda rodar.
FERRAMENTAS = ("tools/setup_check.py", "tools/atualizar_modelo.py",
               "tools/ml_quality_diagnose.py", "tools/ml_fusion_scorecard.py",
               "tools/rollback_baseline.py", "tools/make_release.ps1",
               "plays_app.py", "requirements.txt", ".python-version")


def _rastreado(caminho: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--", caminho],
                       cwd=str(ROOT), capture_output=True, text=True)
    return bool(r.stdout.strip())


@so_no_lol("depende de SETUP.md, que so existe no LoL")
class TestSetupDizAVerdade(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.texto = SETUP.read_text(encoding="utf-8")

    def test_o_que_ele_promete_no_clone_esta_MESMO_no_git(self):
        """A promessa central do passo 4."""
        for alvo in VERSIONADOS:
            with self.subTest(alvo=alvo):
                self.assertTrue(
                    _rastreado(alvo),
                    f"o SETUP.md promete que '{alvo}' vem no clone, e nao vem")

    def test_o_que_ele_manda_copiar_a_mao_continua_fora_do_git(self):
        """Se passar a vir no clone, a instrucao vira trabalho inutil."""
        for alvo in FORA_DO_GIT:
            with self.subTest(alvo=alvo):
                self.assertFalse(
                    _rastreado(alvo),
                    f"o SETUP.md manda copiar '{alvo}' a mao, mas ele ja vem "
                    f"no clone — a instrucao ficou obsoleta")

    def test_toda_ferramenta_citada_existe(self):
        """O caso real: o caminho estava `.tools\\make_release.ps1`."""
        for alvo in FERRAMENTAS:
            with self.subTest(alvo=alvo):
                self.assertTrue(
                    (ROOT / alvo).exists(),
                    f"o SETUP.md manda usar '{alvo}', que nao existe")

    # REMOVIDO (06/09/2026): havia aqui um teste que varria o texto com regex
    # atras de todo token terminado em .py/.ps1 e cobrava que o caminho
    # existisse. A ideia era pegar ferramenta renomeada sem o documento saber.
    # Na pratica o regex produzia LIXO -- fragmentos como '.py' e
    # 'Activate.ps1', tirados de coisas como `-p "test_*.py"` -- e o teste
    # ficava vermelho com o documento correto.
    #
    # Preferi apagar a remendar, pela regra que este projeto ja aprendeu duas
    # vezes esta semana: teste que chora sem motivo ensina a ignorar teste.
    # A lista FERRAMENTAS acima cobre o mesmo risco de forma legivel, e quem
    # acrescentar ferramenta ao guia acrescenta uma linha la.
    # O preco: ferramenta citada SO em prosa, fora da lista, nao e conferida.

    def test_o_cabecalho_nao_contradiz_o_passo_4(self):
        """A D3 exata: o topo dizia que o modelo NAO vem no git."""
        topo = self.texto.split("## Requisitos")[0]
        for frase in ("vão por um zip separado", "nao entram no git",
                      "não entram no git"):
            self.assertNotIn(
                frase, topo,
                "o cabecalho voltou a dizer que modelo/dados nao vem no git, "
                "contradizendo o passo 4")

    def test_a_versao_do_python_bate_com_o_arquivo(self):
        declarada = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
        self.assertIn(declarada, self.texto,
                      f".python-version diz {declarada!r} e o SETUP.md nao cita")

    def test_avisa_do_caminho_sem_acento(self):
        """Armadilha que custou um dia inteiro: nao pode sumir do guia."""
        self.assertIn("ACENTO", self.texto.upper(),
                      "o aviso de caminho sem acento sumiu do SETUP.md")


if __name__ == "__main__":
    unittest.main()
