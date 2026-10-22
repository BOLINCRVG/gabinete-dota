# -*- coding: utf-8 -*-
"""VOLTAR ATRAS: restaura os arquivos do modelo a partir de um commit anterior.

POR QUE (22/10/2026). O "da para reverter" do LoL era tools/rollback_baseline.py,
que restaura copias guardadas em ops/baselines/ -- pasta que so existe no PC
dele, fora do git. O Dota nao tinha nada. Desde que os robos passaram a
versionar o modelo, o historico do git E a copia de seguranca: voltar atras e
trazer os arquivos do modelo de um commit anterior.

O ENSAIO roda na suite (mlcore/tests/test_reverter_modelo.py), que o robo roda
todo dia: num repositorio temporario, publica v1, publica v2, reverte, e confere
que voltou a v1 byte a byte. Rollback que nunca foi ensaiado e o que falha no
dia em que precisa.

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
infraestrutura "Models can be rolled back to a previous version", e a regra de
pontuacao: "A full point is awarded if there is a system in place to run that
test automatically on a repeated basis."
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Arquivo IDENTICO nos dois repositorios. So mexe nos arquivos do modelo e NAO
faz commit: depois de conferir, quem decide commitar e voce.

Uso:
  python tools/reverter_modelo.py                 # mostra o que faria (alvo: antes do ultimo modelo)
  python tools/reverter_modelo.py --aplicar       # restaura
  python tools/reverter_modelo.py --para <commit> --aplicar
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# o que e "o modelo" em cada repo; so entra o que existe no historico
ARQUIVOS_DO_MODELO = (
    "ml_artifact.json", "players_artifact.json", "players_pscore.json",
    "params_fitted.json", "live_models",
)


def _git(raiz: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(raiz), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)


def raiz_do_repo(inicio: Path | None = None) -> Path:
    r = _git(Path(inicio or Path(__file__).resolve().parent), "rev-parse", "--show-toplevel")
    if r.returncode != 0:
        raise SystemExit("nao estou dentro de um repositorio git")
    return Path(r.stdout.strip())


def presentes(raiz: Path) -> list[str]:
    return [a for a in ARQUIVOS_DO_MODELO
            if _git(raiz, "ls-files", "--error-unmatch", a).returncode == 0
            or (raiz / a).is_dir() and _git(raiz, "ls-files", a).stdout.strip()]


def alvo_padrao(raiz: Path, arquivos: list[str]) -> str:
    """O commit ANTERIOR ao ultimo que mudou o modelo principal."""
    principal = "ml_artifact.json" if "ml_artifact.json" in arquivos else arquivos[0]
    r = _git(raiz, "log", "-2", "--format=%H", "--", principal)
    linhas = r.stdout.split()
    if len(linhas) < 2:
        raise SystemExit(f"{principal} nao tem versao anterior no historico")
    return linhas[1]


def reverter(raiz: Path, para: str | None = None, aplicar: bool = False) -> dict:
    arquivos = presentes(raiz)
    if not arquivos:
        raise SystemExit("nenhum arquivo de modelo versionado neste repositorio")
    alvo = para or alvo_padrao(raiz, arquivos)
    if _git(raiz, "cat-file", "-e", alvo + "^{commit}").returncode != 0:
        raise SystemExit(f"commit desconhecido: {alvo}")
    no_alvo = [a for a in arquivos if _git(raiz, "cat-file", "-e", f"{alvo}:{a}").returncode == 0]
    mudaria = _git(raiz, "diff", "--stat", alvo, "--", *no_alvo).stdout.strip()
    if aplicar and no_alvo:
        r = _git(raiz, "checkout", alvo, "--", *no_alvo)
        if r.returncode != 0:
            raise SystemExit(f"falhou ao restaurar: {r.stderr.strip()}")
    return {"alvo": alvo, "arquivos": no_alvo, "mudaria": mudaria, "aplicado": bool(aplicar)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--para", default=None, help="commit de destino (padrao: antes do ultimo modelo)")
    ap.add_argument("--aplicar", action="store_true", help="restaura de fato (sem isto, so mostra)")
    a = ap.parse_args()
    res = reverter(raiz_do_repo(), a.para, a.aplicar)
    print(f"alvo: {res['alvo']}\narquivos: {', '.join(res['arquivos'])}")
    print(res["mudaria"] or "(nada difere do alvo)")
    print("RESTAURADO no disco; confira e commite." if res["aplicado"]
          else "so mostrei. Para restaurar: --aplicar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
