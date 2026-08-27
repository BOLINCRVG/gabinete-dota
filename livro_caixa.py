# -*- coding: utf-8 -*-
"""LIVRO-CAIXA DO SILÊNCIO — toda falha que está sendo TOLERADA, com IDADE.

O problema recorrente deste projeto não é "bug": é silêncio. Coisa errada
sobre a qual nada avisa. E o que machuca não é quebrar — é ficar semanas
quebrado sem ninguém ver. A aba StatsCamps do Dota estava morta desde
10/07/2026 e só apareceu em 26/08 porque um teste voltou a rodar.

Por isso o campo principal aqui é a IDADE, não o estado. Uma lista ordenada
por idade transforma "descobri por acaso" em "está na lista há 3 dias".

Três famílias, escolhidas por terem pegado achados REAIS:

  1) TESTE QUEBRADO — escondeu a StatsCamps. A idade é contada a partir da
     primeira vez que este livro-caixa viu a falha (carimbo gravado no
     próprio ledger); não dá para reconstruir o passado sem rodar a suíte em
     cada commit antigo, e isso custaria horas para responder o que o carimbo
     responde de graça daqui para frente.

  2) ARQUIVO LIDO E NÃO VERSIONADO — a Streamlit Cloud só puxa do git, então
     arquivo ignorado simplesmente não existe lá e o recurso some CALADO.
     Aconteceu duas vezes: live_models (07/2026) e evidence/totals_scorecard
     (26/08/2026).

     LIMITE HONESTO desta família: separar "insumo que a nuvem precisa" de
     "estado que o app cria sozinho" não é confiável por heurística. Tentei
     três (quem escreve na mesma linha; quem lê; o app escreve?) e todas
     deixam resto, porque o caminho costuma vir de uma constante definida
     em outro módulo. Então esta família funciona como LINHA DE BASE: você
     revisa a lista uma vez, e daí em diante o que importa é o item NOVO e a
     IDADE dos que ficaram. É assim que linter sério lida com isso.

  3) PASSO DE ESTEIRA COM continue-on-error — falha por projeto, o que é
     legítimo, mas precisa estar na lista para ninguém esquecer que está lá.

Uso:
    python tools/livro_caixa.py [--root OUTRO_REPO] [--suite-log ARQUIVO]

O ledger fica em evidence/livro_caixa.json e é ACUMULATIVO: o carimbo de
"visto pela primeira vez" nunca é reescrito enquanto o item continuar na lista.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]

# Extensões de dado que o app costuma LER em tempo de execução. Código (.py)
# fica de fora: módulo não versionado quebra o import na cara, não em silêncio.
EXT_DADO = (".json", ".csv", ".txt", ".parquet", ".toml")

# Pastas que nunca contam: saída de execução, backup e cache não são "o app
# lendo um arquivo que devia estar versionado".
IGNORAR_PASTAS = ("backups", "__pycache__", ".venv", "evidence/_tmp_tests",
                  "scratchpad", "node_modules", ".git")


def _rodar(args: list[str], cwd: Path) -> str:
    try:
        r = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _e_ignorado(caminho: Path, raiz: Path) -> bool:
    """git check-ignore: devolve rc=0 quando o caminho ESTÁ ignorado."""
    try:
        r = subprocess.run(["git", "check-ignore", "-q", str(caminho)],
                           cwd=str(raiz), capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _rastreado(caminho: Path, raiz: Path) -> bool:
    saida = _rodar(["git", "ls-files", "--error-unmatch", str(caminho)], raiz)
    return bool(saida.strip())


# QUEM LE e o que decide, nao quem escreve. Tentei primeiro filtrar por
# "o repo produz esse arquivo?" e o filtro estava errado de origem: os DOIS
# incidentes reais (live_models em 07/2026, evidence/totals_scorecard em
# 26/08/2026) sao produzidos pelo proprio repo. Ser produzido nao e o risco.
#
# O risco e o APP precisar do arquivo em tempo de execucao: a Streamlit Cloud
# so puxa do git, entao arquivo ignorado nao existe la e o recurso some calado.
# Ferramenta lendo estado local do PC nao tem esse problema.
LIDO_PELO_APP = ("app_core/", "plays_app.py", "gabinete_dota_app.py")


def _le_no_app(rel: str) -> bool:
    return any(rel.startswith(x) or rel == x for x in LIDO_PELO_APP)


VERBOS_DE_ESCRITA = ("write_text", "write_bytes", "to_csv", "to_json", "dump(",
                     "to_parquet", 'open(', "savefig")


def _app_escreve(basename: str, raiz: Path) -> bool:
    """O PROPRIO app escreve esse arquivo?

    Fecha a definicao: perigoso e o arquivo que o app LE e NAO escreve. Se o app
    escreve, ele se vira sozinho na nuvem (cache, trava, relatorio, estado do
    usuario nascem la mesmo). Se so ferramenta/robo escreve, a nuvem — que nao
    roda ferramenta — fica sem, e o recurso some calado. Foi assim com
    live_models (07/2026) e evidence/totals_scorecard (26/08/2026)."""
    for base in LIDO_PELO_APP:
        alvo = raiz / base
        arquivos = sorted(alvo.rglob("*.py")) if alvo.is_dir() else ([alvo] if alvo.exists() else [])
        for py in arquivos:
            try:
                for linha in py.read_text(encoding="utf-8", errors="replace").splitlines():
                    if basename in linha and any(v in linha for v in VERBOS_DE_ESCRITA):
                        return True
            except (OSError, ValueError):
                continue
    return False


def arquivos_lidos_nao_versionados(raiz: Path) -> list[dict]:
    """Literais de caminho no código que existem em disco e o git ignora.

    Heurística deliberadamente estreita: só nomes com extensão de DADO e que
    de fato existem na raiz do repo. Falso positivo aqui custa ruído; falso
    negativo custa um recurso sumindo calado na nuvem."""
    achados: dict[str, dict] = {}
    for py in sorted(raiz.rglob("*.py")):
        rel = py.relative_to(raiz).as_posix()
        if any(rel.startswith(x) or f"/{x}/" in f"/{rel}" for x in IGNORAR_PASTAS):
            continue
        try:
            arvore = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError, ValueError):
            continue
        for no in ast.walk(arvore):
            if not (isinstance(no, ast.Constant) and isinstance(no.value, str)):
                continue
            valor = no.value.strip()
            if not valor.endswith(EXT_DADO) or len(valor) > 120 or "*" in valor:
                continue
            alvo = raiz / valor
            if not alvo.exists() or not alvo.is_file():
                continue
            chave = alvo.relative_to(raiz).as_posix()
            if any(chave.startswith(x) for x in IGNORAR_PASTAS) or chave in achados:
                continue
            if _rastreado(alvo, raiz) or not _e_ignorado(alvo, raiz):
                continue
            if not _le_no_app(rel):
                continue   # so ferramenta le: sem risco de sumir na nuvem
            if _app_escreve(alvo.name, raiz):
                continue   # o app cria sozinho: nao precisa estar no git
            achados[chave] = {"familia": "lido_nao_versionado", "item": chave,
                              "onde": f"{rel}:{no.lineno}"}
    return list(achados.values())


def passos_com_continue_on_error(raiz: Path) -> list[dict]:
    achados = []
    for yml in sorted((raiz / ".github" / "workflows").glob("*.y*ml")):
        linhas = yml.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, linha in enumerate(linhas):
            if "continue-on-error" not in linha or "true" not in linha:
                continue
            nome = ""
            for j in range(i, max(-1, i - 8), -1):
                m = re.search(r"-\s*name:\s*(.+)", linhas[j])
                if m:
                    nome = m.group(1).strip()
                    break
            achados.append({"familia": "continue_on_error",
                            "item": f"{yml.name}: {nome or 'passo sem nome'}",
                            "onde": f"{yml.relative_to(raiz).as_posix()}:{i + 1}"})
    return achados


def testes_quebrados(log: Path | None) -> list[dict]:
    if log is None or not log.exists():
        return []
    achados = []
    texto = log.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"^(FAIL|ERROR):\s*(\S+)", texto, re.M):
        achados.append({"familia": "teste_quebrado",
                        "item": f"{m.group(1)}: {m.group(2)}", "onde": ""})
    vistos, unicos = set(), []
    for a in achados:
        if a["item"] not in vistos:
            vistos.add(a["item"])
            unicos.append(a)
    return unicos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="varre outro repositório")
    ap.add_argument("--suite-log", default=None,
                    help="log da suíte, para a família 'teste quebrado'")
    ap.add_argument("--out", default="evidence/livro_caixa.json")
    args = ap.parse_args()

    raiz = Path(args.root).resolve() if args.root else ROOT
    print(f"[raiz] {raiz}")

    itens = (arquivos_lidos_nao_versionados(raiz)
             + passos_com_continue_on_error(raiz)
             + testes_quebrados(Path(args.suite_log) if args.suite_log else None))

    # Ledger ACUMULATIVO: o carimbo de "visto pela primeira vez" só nasce uma
    # vez. É ele que faz a idade existir — sem isso, toda varredura acharia
    # tudo "novo" e a lista nunca envelheceria, que é justamente o defeito
    # que este arquivo existe para corrigir.
    destino = raiz / str(args.out)
    antigo = {}
    if destino.exists():
        try:
            antigo = {x["chave"]: x for x in
                      (json.loads(destino.read_text(encoding="utf-8")).get("itens") or [])}
        except (OSError, ValueError, KeyError):
            antigo = {}

    hoje = date.today().isoformat()
    saida = []
    for it in itens:
        chave = f"{it['familia']}|{it['item']}"
        visto = antigo.get(chave, {}).get("visto_em", hoje)
        idade = (date.today() - datetime.fromisoformat(visto).date()).days
        saida.append({**it, "chave": chave, "visto_em": visto, "idade_dias": idade})
    saida.sort(key=lambda x: (-x["idade_dias"], x["familia"], x["item"]))

    resolvidos = [k for k in antigo if k not in {x["chave"] for x in saida}]

    rot = {"teste_quebrado": "TESTE QUEBRADO",
           "lido_nao_versionado": "LIDO PELO APP E NÃO VERSIONADO",
           "continue_on_error": "PASSO QUE FALHA SEM DERRUBAR A ESTEIRA"}
    print(f"\n{'idade':>7}  {'família':<34} item")
    for x in saida:
        marca = "  <-- " if x["idade_dias"] >= 30 else "      "
        print(f"{x['idade_dias']:>5}d{marca}{rot.get(x['familia'], x['familia']):<34}{x['item'][:70]}")
    print(f"\ntotal tolerado: {len(saida)}")
    velhos = [x for x in saida if x["idade_dias"] >= 30]
    if velhos:
        print(f"ATENÇÃO: {len(velhos)} item(ns) com 30 dias ou mais sem decisão.")
    if resolvidos:
        print(f"saíram da lista desde a última varredura: {len(resolvidos)}")
    if not saida:
        print("nada tolerado — ou a varredura está cega. Confira antes de comemorar.")

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"gerado_em": datetime.now().isoformat(timespec="seconds"),
                                   "itens": saida}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"[salvo] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
