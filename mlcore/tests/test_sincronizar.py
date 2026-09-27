# -*- coding: utf-8 -*-
"""A atualizacao ao abrir o app -- testada com repositorios de VERDADE.

POR QUE ELA EXISTE (23/09/2026): duas vezes em dois dias os dados na maquina
dele estavam velhos enquanto o robo trabalhava direito. LoL, 17 atualizacoes
paradas (odds de um dia antes na aba Oportunidades). Dota, 35 paradas (a PGL
Wallachia Season 9 simplesmente nao existia). Ele autorizou: "Faca isso".

POR QUE REPOSITORIOS DE VERDADE: logica de git testada com simulacao testa a
simulacao. Aqui cada teste monta um "GitHub" (repositorio nu), um "robo" que
publica nele, e a "maquina dele" -- e roda o git de verdade.

A REGRA QUE MAIS IMPORTA, e que metade dos testes guarda: NADA do que pode
ser trabalho dele se perde. Codigo, apelidos, configuracao -- se o robo e ele
mexeram no mesmo arquivo desses, a atualizacao PARA e avisa.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol

RAIZ = Path(__file__).resolve().parents[2]
ARQ = RAIZ / "tools" / "sincronizar.py"


def _carregar():
    spec = importlib.util.spec_from_file_location("sincronizar", ARQ)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _g(cwd, *args):
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), r.stderr))
    return r.stdout


def _config(repo):
    _g(repo, "config", "user.email", "teste@exemplo")
    _g(repo, "config", "user.name", "teste")
    _g(repo, "config", "core.autocrlf", "false")


class _Cenario(unittest.TestCase):
    """GitHub falso + robo + a maquina dele."""

    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git nao esta instalado")
        self.m = _carregar()
        self.tmp = Path(tempfile.mkdtemp(prefix="sync_"))
        self.github = self.tmp / "github.git"
        _g(self.tmp, "init", "--bare", "-b", "main", str(self.github))
        # o robo cria o repositorio com um arquivo de dado e um de codigo
        self.robo = self.tmp / "robo"
        _g(self.tmp, "clone", str(self.github), str(self.robo))
        _config(self.robo)
        _g(self.robo, "checkout", "-b", "main")
        (self.robo / "dados.csv").write_text("v1\n", encoding="utf-8")
        (self.robo / "modelo.json").write_text('{"v":1}\n', encoding="utf-8")
        (self.robo / "app.py").write_text("print(1)\n", encoding="utf-8")
        _g(self.robo, "add", ".")
        _g(self.robo, "commit", "-m", "inicio")
        _g(self.robo, "push", "-u", "origin", "main")
        # a maquina dele clona
        self.pc = self.tmp / "pc"
        _g(self.tmp, "clone", str(self.github), str(self.pc))
        _config(self.pc)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def robo_publica(self, arquivo, conteudo, msg="robo"):
        (self.robo / arquivo).write_text(conteudo, encoding="utf-8")
        _g(self.robo, "add", arquivo)
        _g(self.robo, "commit", "-m", msg)
        _g(self.robo, "push")

    def ler(self, arquivo):
        return (self.pc / arquivo).read_text(encoding="utf-8")


class TestOCaminhoFeliz(_Cenario):

    @so_no_lol("so vale para o LoL")
    def test_em_dia_nao_faz_nada(self):
        r = self.m.sincronizar(self.pc)
        self.assertTrue(r["ok"])
        self.assertEqual(r["trazidas"], 0)

    @so_no_lol("so vale para o LoL")
    def test_atras_e_limpo_traz_tudo(self):
        self.robo_publica("dados.csv", "v2\n")
        self.robo_publica("dados.csv", "v3\n")
        r = self.m.sincronizar(self.pc)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["trazidas"], 2)
        self.assertEqual(self.ler("dados.csv"), "v3\n")


class TestArquivoGeradoMexidoDosDoisLados(_Cenario):
    """O caso real: o app reescreve o modelo aqui, o robo publica outro."""

    @so_no_lol("so vale para o LoL")
    def test_troca_pelo_do_robo_e_GUARDA_o_daqui(self):
        (self.pc / "modelo.json").write_text('{"v":"local"}\n', encoding="utf-8")
        self.robo_publica("modelo.json", '{"v":2}\n')
        r = self.m.sincronizar(self.pc)
        self.assertTrue(r["ok"], r)
        self.assertEqual(self.ler("modelo.json"), '{"v":2}\n')
        self.assertIn("modelo.json", r["substituidos"])
        copias = list((self.tmp / "_sync_backups" / "pc").glob("*/modelo.json"))
        self.assertEqual(len(copias), 1, "nao guardou a copia fisica")
        self.assertEqual(copias[0].read_text(encoding="utf-8"), '{"v":"local"}\n')

    @so_no_lol("so vale para o LoL")
    def test_arquivo_novo_aqui_que_o_robo_tambem_cria(self):
        (self.pc / "relatorio.json").write_text("local\n", encoding="utf-8")
        self.robo_publica("relatorio.json", "robo\n")
        r = self.m.sincronizar(self.pc)
        self.assertTrue(r["ok"], r)
        self.assertEqual(self.ler("relatorio.json"), "robo\n")


class TestNADADoQueEleFezSePerde(_Cenario):
    """A regra que mais importa."""

    @so_no_lol("so vale para o LoL")
    def test_codigo_mexido_dos_dois_lados_PARA_e_nao_toca_em_nada(self):
        (self.pc / "app.py").write_text("print('meu trabalho')\n", encoding="utf-8")
        self.robo_publica("app.py", "print(2)\n")
        r = self.m.sincronizar(self.pc)
        self.assertFalse(r["ok"])
        self.assertIn("app.py", r["aviso"])
        self.assertEqual(self.ler("app.py"), "print('meu trabalho')\n")

    @so_no_lol("so vale para o LoL")
    def test_apelidos_mexidos_dos_dois_lados_PARA(self):
        (self.robo / "user_team_aliases.json").write_text("{}\n", encoding="utf-8")
        _g(self.robo, "add", ".")
        _g(self.robo, "commit", "-m", "apelidos")
        _g(self.robo, "push")
        self.m.sincronizar(self.pc)
        (self.pc / "user_team_aliases.json").write_text('{"liquid":"Team Liquid"}\n',
                                                         encoding="utf-8")
        self.robo_publica("user_team_aliases.json", '{"x":"y"}\n')
        r = self.m.sincronizar(self.pc)
        self.assertFalse(r["ok"])
        self.assertIn("Team Liquid", self.ler("user_team_aliases.json"))

    @so_no_lol("so vale para o LoL")
    def test_codigo_mexido_SO_aqui_e_preservado_e_a_atualizacao_segue(self):
        """O caso do livro_caixa.py do Dota: mudanca dele, que o robo nao toca."""
        (self.pc / "app.py").write_text("print('meu')\n", encoding="utf-8")
        self.robo_publica("dados.csv", "v2\n")
        r = self.m.sincronizar(self.pc)
        self.assertTrue(r["ok"], r)
        self.assertEqual(self.ler("dados.csv"), "v2\n")
        self.assertEqual(self.ler("app.py"), "print('meu')\n")

    @so_no_lol("so vale para o LoL")
    def test_commits_feitos_nesta_maquina_sao_mantidos(self):
        (self.pc / "novo.py").write_text("x = 1\n", encoding="utf-8")
        _g(self.pc, "add", "novo.py")
        _g(self.pc, "commit", "-m", "trabalho local")
        self.robo_publica("dados.csv", "v2\n")
        r = self.m.sincronizar(self.pc)
        self.assertTrue(r["ok"], r)
        self.assertEqual(self.ler("dados.csv"), "v2\n")
        self.assertIn("trabalho local", _g(self.pc, "log", "--oneline"))

    @so_no_lol("so vale para o LoL")
    def test_juncao_que_falha_e_DESFEITA_e_devolve_os_arquivos(self):
        (self.pc / "dados.csv").write_text("commit local\n", encoding="utf-8")
        _g(self.pc, "commit", "-am", "local mexeu no dado")
        (self.pc / "modelo.json").write_text('{"v":"local"}\n', encoding="utf-8")
        self.robo_publica("dados.csv", "robo mexeu no dado\n")
        self.robo_publica("modelo.json", '{"v":2}\n')
        r = self.m.sincronizar(self.pc)
        self.assertFalse(r["ok"])
        self.assertFalse((self.pc / ".git" / "MERGE_HEAD").exists(),
                         "deixou o repositorio no meio de uma juncao")
        self.assertEqual(self.ler("dados.csv"), "commit local\n")
        self.assertEqual(self.ler("modelo.json"), '{"v":"local"}\n',
                         "o arquivo substituido nao foi devolvido")


class TestOAppAbreDeQualquerJeito(_Cenario):

    @so_no_lol("so vale para o LoL")
    def test_sem_conexao_so_avisa(self):
        _g(self.pc, "remote", "set-url", "origin", str(self.tmp / "nao_existe.git"))
        r = self.m.sincronizar(self.pc)
        self.assertFalse(r["ok"])
        self.assertIn("conexao", r["aviso"])

    @so_no_lol("so vale para o LoL")
    def test_sem_github_configurado_so_avisa(self):
        _g(self.pc, "branch", "--unset-upstream")
        r = self.m.sincronizar(self.pc)
        self.assertFalse(r["ok"])

    @so_no_lol("so vale para o LoL")
    def test_no_meio_de_outra_operacao_nao_mexe(self):
        (self.pc / ".git" / "MERGE_HEAD").write_text("x", encoding="utf-8")
        r = self.m.sincronizar(self.pc)
        self.assertFalse(r["ok"])
        self.assertIn("outra operacao", r["aviso"])

    @so_no_lol("so vale para o LoL")
    def test_main_sempre_devolve_zero(self):
        """O lancador so segue se o codigo de saida deixar."""
        import inspect
        fonte = inspect.getsource(self.m.main)
        self.assertIn("return 0", fonte)
        self.assertNotIn("return 1", fonte)


class TestOQueEProtegido(unittest.TestCase):

    def setUp(self):
        self.m = _carregar()

    @so_no_lol("so vale para o LoL")
    def test_codigo_e_configuracao_sao_protegidos(self):
        for c in ("app.py", "tools/x.py", "CLAUDE.md", "RUN.bat", "a.ps1",
                  "requirements.txt", ".github/workflows/odds.yml",
                  "user_team_aliases.json", ".streamlit/secrets.toml",
                  ".gitignore"):
            self.assertTrue(self.m.protegido(c), c)

    @so_no_lol("so vale para o LoL")
    def test_dados_e_modelos_nao_sao(self):
        for c in ("data.csv", "live_models/minute_10/lgb_model.txt",
                  "evidence/regua_28.json", "odds_pinnacle.json",
                  "ml_artifact.json", "rank_snapshot.json"):
            self.assertFalse(self.m.protegido(c), c)


class TestOsDoisRepositoriosTemOMesmoArquivo(unittest.TestCase):
    """Regra de arquivo compartilhado do Dota: identico nos dois."""

    @so_no_lol("so vale para o LoL")
    def test_copia_do_dota_e_identica(self):
        dota = RAIZ.parent / "GabineteDOTA" / "sincronizar.py"
        if not dota.exists():
            self.skipTest("o repositorio do Dota nao esta ao lado")
        a = ARQ.read_bytes().replace(b"\r\n", b"\n")
        b = dota.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(a, b, "sincronizar.py divergiu entre LoL e Dota")


if __name__ == "__main__":
    unittest.main()
