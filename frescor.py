# -*- coding: utf-8 -*-
"""FRESCOR DAS RÉGUAS — a nota que você está lendo é de quando?

Pergunta dele: "como a nota caiu de 9,25 para 5 sem ninguém ver?". A resposta
estrutural é que **a nota é um programa que só roda quando alguém roda**. Velha,
ela repete o número antigo com toda a confiança do mundo. O sintoma aparece
como "nota estranha", que não aponta para a causa.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║                                                                             ║
║ 1. POR QUE O ALARME É POR AUSÊNCIA, e não por erro:                         ║
║    padrão do "dead man's switch" — "alerte-me quando o SILÊNCIO for o       ║
║    problema". Quem checa não pode ser quem executa; o vigia é externo.       ║
║                                                                             ║
║ 2. POR QUE CADA RÉGUA TEM PRAZO PRÓPRIO, e não um prazo único:              ║
║    prática de SLA de frescor de dados — "defina o prazo em relação à        ║
║    frequência de atualização: o que atualiza a cada hora tem prazo de 2h,   ║
║    não de 61 minutos", e "o erro mais comum é usar o mesmo prazo para       ║
║    tudo: 1h numa tabela semanal gera ruído; 24h numa tabela de tempo real   ║
║    perde todo incidente".                                                    ║
║                                                                             ║
║ 3. POR QUE DUAS FAIXAS (atrasado / velho) e não uma:                        ║
║    mesma fonte — "é mais prático definir duas faixas do que um limiar só".  ║
║    E "comece generoso e aperte depois; começar apertado cria fadiga de      ║
║    alerta", por isso o prazo é o DOBRO da frequência, não 1,1x.             ║
║                                                                             ║
║ 4. POR QUE "FREQUÊNCIA NÃO DECLARADA" É UM ACHADO, e não um caso a ignorar: ║
║    sem cadência declarada não existe prazo possível — e um artefato que     ║
║    ninguém sabe quando deveria ser refeito é exatamente o que envelhece     ║
║    calado. Ele aparece na lista com essa marca.                             ║
╚═════════════════════════════════════════════════════════════════════════════╝

Uso: python tools/frescor.py [--root OUTRO_REPO]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

# Cadencia de cada regua, em dias. Prazo = 2x (atrasada) e 3x (velha), fonte 3.
#
# VERIFICADO em 27/08/2026 contra as esteiras, nao inferido — a primeira versao
# destes numeros era chute meu, e 5 dos 14 estavam errados:
#   .github/workflows/nota_semanal.yml roda toda SEGUNDA e produz diretamente
#     ml_fusion_scorecard, product_scorecard, totals_scorecard, nota_por_liga,
#     livro_caixa e frescor;
#   product_scorecard.py --run (chamado ali) regenera as "reguas baratas":
#     data, live, reliability, code, players, portability — ou seja, TAMBEM
#     semanais, e nao mensais como eu tinha suposto;
#   a esteira do Dota roda 2x por dia e produz dota_scorecard.
# O que NAO tem produtor automatico fica FORA desta tabela de proposito: sem
# cadencia declarada nao existe prazo, e isso e o proprio achado (fonte 4).
# O emulador de estresse e o caso: 34,8 dias e nenhuma esteira o regenera.
CADENCIA = {
    "fusion_scorecard.json": ("nota oficial (vigia semanal)", 7),
    "product_scorecard.json": ("nota do produto (vigia semanal)", 7),
    "totals_scorecard.json": ("regua dos totais (vigia semanal)", 7),
    "nota_por_liga.json": ("nota por liga — APARECE NA TELA", 7),
    "livro_caixa.json": ("livro-caixa do silencio", 7),
    "frescor.json": ("esta propria varredura", 7),
    "data_scorecard.json": ("regua de dados (product --run)", 7),
    "live_scorecard.json": ("regua da live (product --run)", 7),
    "reliability_scorecard.json": ("regua de confiabilidade (product --run)", 7),
    "code_scorecard.json": ("regua de codigo (product --run)", 7),
    "players_scorecard.json": ("regua de jogadores (product --run)", 7),
    "portability_scorecard.json": ("regua de portabilidade (product --run)", 7),
    "dota_scorecard.json": ("regua do Dota (esteira 2x ao dia)", 1),
    "dota_totals_scorecard.json": ("totais do Dota", 7),
}
CAMPOS_CARIMBO = ("gerado_em", "generated_at", "ajustado_em", "timestamp", "data")


def _raiz_do_repo(inicio: Path) -> Path:
    """A raiz do repositorio, sem depender de ONDE este arquivo mora.

    Terceira vez que este bug aparece (livro_caixa.py e regua_28.py foram as
    outras duas): no LoL este script vive em tools/ e no Dota na raiz, entao
    parents[1] resolvia para a Area de Trabalho. Aqui ele ao menos falhava
    alto ("sem pasta evidence em C:/Users/.../Desktop"); a regua das 28 falhou
    CALADA, medindo o Desktop e reportando nota.

    `git rev-parse --show-toplevel` e o jeito canonico e independente de onde o
    script foi chamado; travessia atras de .git fica de reserva se o git nao
    estiver no PATH. (A copia esta em tres arquivos de proposito: um modulo
    comum precisaria da raiz para ser importado, que e o que se quer descobrir.)
    """
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           cwd=str(inicio.parent), capture_output=True,
                           text=True, encoding="utf-8", timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).resolve()
    except (OSError, subprocess.SubprocessError):
        pass
    atual = inicio.resolve().parent
    for cand in [atual, *atual.parents]:
        if (cand / ".git").exists():
            return cand
    return atual


def _carimbo(caminho: Path):
    try:
        d = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    for campo in CAMPOS_CARIMBO:
        v = d.get(campo)
        if isinstance(v, str) and v.strip():
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00").split("+")[0])
            except ValueError:
                continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="evidence/frescor.json")
    args = ap.parse_args()

    raiz = Path(args.root).resolve() if args.root else _raiz_do_repo(Path(__file__))
    ev = raiz / "evidence"
    if not ev.is_dir():
        print(f"sem pasta evidence em {raiz}")
        return 1

    agora = datetime.now()
    linhas = []
    for arq in sorted(ev.glob("*.json")):
        if arq.name.startswith("_"):
            continue
        rotulo, cad = CADENCIA.get(arq.name, (None, None))
        c = _carimbo(arq)
        idade = None if c is None else (agora - c).total_seconds() / 86400.0
        if cad is None:
            estado = "frequencia nao declarada"
        elif c is None:
            estado = "SEM CARIMBO"
        elif idade > 3 * cad:
            estado = "VELHA"
        elif idade > 2 * cad:
            estado = "atrasada"
        else:
            estado = "ok"
        linhas.append({"arquivo": arq.name, "rotulo": rotulo or "(nao declarada)",
                       "cadencia_dias": cad,
                       "idade_dias": None if idade is None else round(idade, 1),
                       "estado": estado})

    ordem = {"VELHA": 0, "SEM CARIMBO": 1, "atrasada": 2, "frequencia nao declarada": 3, "ok": 4}
    linhas.sort(key=lambda x: (ordem[x["estado"]], -(x["idade_dias"] or 0)))

    print(f"[raiz] {raiz}\n")
    print(f"{'regua':<34}{'cadencia':>10}{'idade':>9}   estado")
    for x in linhas:
        cad = f"{x['cadencia_dias']}d" if x["cadencia_dias"] else "-"
        idade = f"{x['idade_dias']:.1f}d" if x["idade_dias"] is not None else "-"
        marca = "  <<<" if x["estado"] in ("VELHA", "SEM CARIMBO") else ""
        print(f"{x['rotulo'][:33]:<34}{cad:>10}{idade:>9}   {x['estado']}{marca}")

    velhas = [x for x in linhas if x["estado"] == "VELHA"]
    sem = [x for x in linhas if x["estado"] == "SEM CARIMBO"]
    nao_decl = [x for x in linhas if x["estado"] == "frequencia nao declarada"]
    print()
    if velhas:
        print(f"ATENCAO: {len(velhas)} regua(s) VELHA(s) — passaram de 3x a propria cadencia.")
    if sem:
        print(f"ATENCAO: {len(sem)} regua(s) SEM CARIMBO — nao da pra saber a idade,")
        print("         que e o mesmo que nao ter vigia nenhum ali.")
    if nao_decl:
        print(f"{len(nao_decl)} artefato(s) sem cadencia declarada: nao da pra ter prazo,")
        print("         e artefato sem prazo e o que envelhece calado (fonte 4).")
    if not (velhas or sem):
        print("todas as reguas declaradas estao dentro do prazo.")

    destino = raiz / str(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"gerado_em": agora.isoformat(timespec="seconds"),
                                   "reguas": linhas}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"[salvo] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
