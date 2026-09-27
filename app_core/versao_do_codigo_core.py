# -*- coding: utf-8 -*-
"""Qual versao do codigo este app esta rodando AGORA.

POR QUE ISTO EXISTE (15/09/2026)
Tres vezes seguidas ele disse "nao mudou nada" sobre conserto que eu ja tinha
publicado. Nas tres, o codigo estava no GitHub e o app rodava versao antiga —
a hospedagem recarrega o arquivo principal quando chega um commit, mas nao
recarrega os modulos importados, e so o reinicio baixa tudo de novo:
  https://discuss.streamlit.io/t/apps-deployed-on-streamlit-community-cloud-reacts-only-on-main-app-file-changes-but-not-on-changes-in-the-py-modules/36626
  https://github.com/streamlit/streamlit/issues/366

O problema nao era o conserto: era NAO DAR PARA VER. Nos dois adivinhavamos —
ele olhando a tela, eu olhando o repositorio, sem nada em comum para comparar.

Agora a tela diz a versao. "Mudou ou nao?" vira um olhar, para sempre.

COMO SE DESCOBRE A VERSAO, em ordem de confianca:
  1. o HEAD do git, quando a pasta .git veio junto (a hospedagem clona o
     repositorio, entao normalmente veio);
  2. a data do arquivo de codigo mais novo — que numa copia clonada e QUANDO A
     HOSPEDAGEM BAIXOU os arquivos. E justamente o que interessa saber.
Se nem isso der, diz que nao sabe. PRINCIPIO IV: nao inventa versao.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Pastas onde o codigo do app vive. A data do arquivo mais novo delas e a
# melhor pista de quando a copia em execucao foi baixada.
PASTAS_DE_CODIGO = ("app_core", "mlcore", "tools")


def _commit_do_git(raiz: Path) -> Optional[str]:
    """Os 7 primeiros caracteres do commit, lendo .git na mao.

    Sem depender do programa `git` instalado nem de biblioteca: a hospedagem
    pode nao ter nenhum dos dois, e uma dependencia a mais para mostrar um
    rotulo seria troca ruim.
    """
    git = raiz / ".git"
    try:
        if git.is_file():
            # repositorio "worktree": .git e um arquivo apontando para a pasta
            alvo = git.read_text(encoding="utf-8").strip()
            if alvo.startswith("gitdir:"):
                git = Path(alvo.split(":", 1)[1].strip())
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None

    if not head.startswith("ref:"):
        return head[:7] if len(head) >= 7 else None   # HEAD solto (detached)

    ref = head.split(":", 1)[1].strip()
    try:
        return (git / ref).read_text(encoding="utf-8").strip()[:7]
    except OSError:
        pass
    # o ref pode estar empacotado em packed-refs
    try:
        for linha in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            partes = linha.split()
            if len(partes) == 2 and partes[1] == ref:
                return partes[0][:7]
    except OSError:
        pass
    return None


def _codigo_mais_novo(raiz: Path) -> Optional[datetime]:
    """Data do arquivo .py mais novo do app.

    Numa copia clonada pela hospedagem, isto e QUANDO ELA BAIXOU os arquivos —
    que e exatamente a pergunta "o meu conserto chegou?".
    """
    mais_nova: Optional[float] = None
    for pasta in PASTAS_DE_CODIGO:
        alvo = raiz / pasta
        if not alvo.is_dir():
            continue
        try:
            for p in alvo.rglob("*.py"):
                try:
                    t = p.stat().st_mtime
                except OSError:
                    continue
                if mais_nova is None or t > mais_nova:
                    mais_nova = t
        except OSError:
            continue
    if mais_nova is None:
        return None
    try:
        return datetime.fromtimestamp(mais_nova)
    except (OSError, OverflowError, ValueError):
        return None


def versao_do_codigo(raiz: Path) -> Dict[str, Any]:
    """{commit, baixado_em, texto} — ou 'nao sei', nunca um palpite."""
    commit = _commit_do_git(Path(raiz))
    quando = _codigo_mais_novo(Path(raiz))

    partes = []
    if commit:
        partes.append("codigo {}".format(commit))
    if quando is not None:
        partes.append("baixado {}".format(quando.strftime("%d/%m %H:%M")))

    return {
        "commit": commit,
        "baixado_em": quando,
        "texto": (" · ".join(partes) if partes else "versao do codigo desconhecida"),
    }


def render_versao(st: Any, raiz: Path) -> None:
    """Uma linha discreta com a versao em execucao."""
    try:
        v = versao_do_codigo(raiz)
    except Exception:
        return
    st.caption("🧩 " + str(v["texto"]))
