# -*- coding: utf-8 -*-
"""Traz o que o robo publicou, antes de o app abrir -- sem nunca perder nada seu.

POR QUE ISTO EXISTE (23/09/2026). Duas vezes em dois dias, nos dois apps, os
dados na maquina dele estavam VELHOS enquanto o robo trabalhava direito:

  - LoL, 22/09: 17 atualizacoes do robo paradas. A aba Oportunidades mostrava
    odds de um dia antes.
  - Dota, 23/09: 35 atualizacoes paradas. O data.csv parava em 05/09, e a PGL
    Wallachia Season 9 (que comecou em 19/09) simplesmente nao existia.

A causa foi a mesma: nenhum dos dois apps puxa do GitHub sozinho, e arquivos
que o proprio app reescreve (relatorios, modelos) travavam a atualizacao manual.

ELE AUTORIZOU ("Faca isso") que os dois apps se atualizem ao abrir.

AS TRAVAS -- o que este script NUNCA faz:

  1. Nao descarta NADA que o robo nao tenha mudado. So entra na conta o
     arquivo que foi mudado dos dois lados (aqui e no robo).
  2. Nao descarta codigo, apelidos, configuracao ou documento, mesmo que o
     robo tambem tenha mudado. Nesse caso NAO atualiza e avisa, porque ali
     pode estar trabalho dele.
  3. Nao substitui nada sem copia fisica antes (fora do repositorio, as
     ultimas 5 guardadas). O projeto do Dota ja ensinou que modelo nao se
     restaura pelo git: o fim de linha muda e o arquivo engorda.
  4. Nao impede o app de abrir. Sem internet, sem git, repositorio no meio de
     outra operacao -- qualquer imprevisto vira um aviso, e o app sobe.
  5. Nao envia nada para o GitHub. So traz.

O arquivo e IDENTICO nos dois repositorios (regra de arquivo compartilhado).
A raiz vem de `git rev-parse --show-toplevel`, nunca da pasta do arquivo --
no Dota as ferramentas ficam na raiz e no LoL em tools/, e usar a pasta do
arquivo ja fez uma ferramenta medir a pasta errada por semanas.

Uso: python sincronizar.py            (o lancador do app chama sozinho)
"""
from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

# Arquivos que podem ser trabalho dele: NUNCA descartados. Se o robo mudou um
# destes e ele tambem, a atualizacao para e avisa.
PROTEGIDOS = (
    "*.py", "*.md", "*.bat", "*.ps1", "*.cmd", "*.sh",
    "*.toml", "*.yml", "*.yaml", "*.cfg", "*.ini",
    "requirements*.txt", ".gitignore", ".gitattributes",
    ".github/*", ".streamlit/*",
    "*alias*.json", "user_*.json",
)

TEMPO_REDE = 45          # segundos para falar com o GitHub
COPIAS_GUARDADAS = 5


def _git(raiz, *args, tempo=60):
    """(codigo, saida) de um comando git. Nunca levanta excecao."""
    try:
        r = subprocess.run(["git", *args], cwd=str(raiz), capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=tempo)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def protegido(caminho: str) -> bool:
    c = caminho.replace("\\", "/")
    nome = c.rsplit("/", 1)[-1]
    for p in PROTEGIDOS:
        if fnmatch.fnmatch(nome, p) or fnmatch.fnmatch(c, p):
            return True
        if p.endswith("/*") and c.startswith(p[:-1]):
            return True
    return False


def _raiz():
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return Path(r.stdout.strip())


def _no_meio_de_outra_operacao(raiz: Path) -> bool:
    g = raiz / ".git"
    return any((g / n).exists() for n in
               ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD"))


def _mexidos_aqui(raiz: Path):
    """{caminho: codigo} dos arquivos mudados ou novos nesta maquina."""
    cod, out = _git(raiz, "status", "--porcelain", "--untracked-files=all")
    if cod != 0:
        return None
    fora = {}
    for linha in out.splitlines():
        if len(linha) < 4:
            continue
        c = linha[3:].strip().strip('"')
        if " -> " in c:
            c = c.split(" -> ", 1)[1]
        fora[c.replace("\\", "/")] = linha[:2]
    return fora


def _pasta_de_copias(raiz: Path) -> Path:
    """Fora do repositorio, para a copia nao virar mais um arquivo mexido."""
    return raiz.parent / "_sync_backups" / raiz.name


def _anotar(raiz: Path, texto: str) -> None:
    try:
        p = _pasta_de_copias(raiz)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "sincronizacoes.log", "a", encoding="utf-8") as fh:
            fh.write("%s  %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), texto))
    except OSError:
        pass


def _copiar(raiz: Path, caminhos, destino: Path) -> None:
    for c in caminhos:
        orig = raiz / c
        if not orig.exists():
            continue
        alvo = destino / c
        alvo.parent.mkdir(parents=True, exist_ok=True)
        if orig.is_dir():
            shutil.copytree(orig, alvo, dirs_exist_ok=True)
        else:
            shutil.copy2(orig, alvo)


def _devolver(raiz: Path, caminhos, origem: Path) -> None:
    """Volta as copias ao lugar (so em caso de falha)."""
    for c in caminhos:
        src = origem / c
        if src.exists() and src.is_file():
            (raiz / c).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, raiz / c)


def _podar_copias(pasta: Path) -> None:
    try:
        antigas = sorted([p for p in pasta.iterdir() if p.is_dir()],
                         key=lambda p: p.name)
        for p in antigas[:-COPIAS_GUARDADAS]:
            shutil.rmtree(p, ignore_errors=True)
    except OSError:
        pass


def sincronizar(raiz: Path) -> dict:
    """Faz o trabalho. Devolve um resumo; nunca levanta excecao."""
    res = {"ok": False, "trazidas": 0, "substituidos": [], "aviso": ""}

    if _no_meio_de_outra_operacao(raiz):
        res["aviso"] = "o repositorio esta no meio de outra operacao do git; nao mexi"
        return res
    cod, _ = _git(raiz, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if cod != 0:
        res["aviso"] = "este repositorio nao acompanha nenhum GitHub; nada a trazer"
        return res

    cod, out = _git(raiz, "fetch", "--quiet", tempo=TEMPO_REDE)
    if cod != 0:
        res["aviso"] = "sem conexao com o GitHub agora; o app abre com o que ja tem"
        return res

    cod, out = _git(raiz, "rev-list", "--left-right", "--count", "HEAD...@{u}")
    try:
        frente, atras = (int(x) for x in out.split()[:2])
    except ValueError:
        res["aviso"] = "nao consegui comparar com o GitHub (%s)" % out.strip()[:120]
        return res
    if atras == 0:
        res["ok"] = True
        return res

    # O que o robo mudou desde que os dois caminhos se separaram.
    cod, out = _git(raiz, "diff", "--name-only", "HEAD...@{u}")
    vindos = {l.strip() for l in out.splitlines() if l.strip()} if cod == 0 else set()
    mexidos = _mexidos_aqui(raiz)
    if mexidos is None:
        res["aviso"] = "nao consegui ler o estado dos arquivos; nao mexi"
        return res
    dos_dois_lados = sorted(c for c in mexidos if c in vindos)

    travados = [c for c in dos_dois_lados if protegido(c)]
    if travados:
        res["aviso"] = ("NAO atualizei: voce e o robo mudaram o(s) mesmo(s) arquivo(s) "
                        "que pode(m) ser trabalho seu: %s. Nada foi tocado."
                        % ", ".join(travados[:5]))
        return res

    # Copia fisica antes de qualquer substituicao.
    pasta = _pasta_de_copias(raiz)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = pasta / carimbo
    if dos_dois_lados:
        try:
            _copiar(raiz, dos_dois_lados, destino)
        except OSError as e:
            res["aviso"] = "nao consegui fazer a copia de seguranca (%s); nao mexi" % e
            return res
        for c in dos_dois_lados:
            estado = mexidos.get(c, "")
            if estado.startswith("??"):
                try:
                    (raiz / c).unlink()          # novo aqui; o robo traz o dele
                except OSError:
                    pass
            else:
                _git(raiz, "checkout", "--", c)
        res["substituidos"] = dos_dois_lados

    if frente == 0:
        cod, out = _git(raiz, "merge", "--ff-only", "@{u}", tempo=180)
    else:
        # Ha commits so daqui (trabalho feito nesta maquina). Junta os dois
        # caminhos em vez de reescrever o historico dele.
        cod, out = _git(raiz, "merge", "--no-edit", "@{u}", tempo=180)
    if cod != 0:
        _git(raiz, "merge", "--abort")
        if dos_dois_lados:
            _devolver(raiz, dos_dois_lados, destino)
        res["aviso"] = ("NAO atualizei: a juncao com o GitHub falhou e foi desfeita, "
                        "com os seus arquivos devolvidos. (%s)" % out.strip()[:160])
        return res

    _podar_copias(pasta)
    res["ok"] = True
    res["trazidas"] = atras
    return res


def main() -> int:
    raiz = _raiz()
    if raiz is None:
        print("[atualizacao] esta pasta nao e um repositorio git; o app abre como esta.")
        return 0
    inicio = time.time()
    try:
        r = sincronizar(raiz)
    except Exception as e:  # noqa: BLE001 -- o app TEM de abrir, aconteca o que for
        r = {"ok": False, "trazidas": 0, "substituidos": [],
             "aviso": "erro inesperado na atualizacao (%s); o app abre como esta" % e}
    seg = time.time() - inicio

    if r["ok"] and r["trazidas"]:
        msg = "trouxe %d atualizacao(oes) do robo em %.0f s" % (r["trazidas"], seg)
        if r["substituidos"]:
            msg += "; %d arquivo(s) gerado(s) trocado(s) pela versao do robo (copia guardada)" \
                   % len(r["substituidos"])
    elif r["ok"]:
        msg = "ja estava em dia"
    else:
        msg = r["aviso"]
    print("[atualizacao] " + msg)
    _anotar(raiz, msg)
    try:
        (_pasta_de_copias(raiz)).mkdir(parents=True, exist_ok=True)
        (_pasta_de_copias(raiz) / "ultima_sincronizacao.json").write_text(
            json.dumps({"quando": datetime.now().isoformat(timespec="seconds"),
                        "mensagem": msg, **r}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    except OSError:
        pass
    return 0           # SEMPRE: o app abre de qualquer jeito


if __name__ == "__main__":
    raise SystemExit(main())
