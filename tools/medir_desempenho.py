# -*- coding: utf-8 -*-
"""DESEMPENHO COMPUTACIONAL: carregar o modelo e prever ficou lento? (22/10/2026)

Item "desempenho computacional nao regrediu" da regua das 28, em zero nos dois
repos: tempo de treino e de resposta nao eram medidos em lugar nenhum.

Mede, na mesma maquina e do mesmo jeito toda vez: (1) tempo para carregar o
ml_artifact.json, (2) tempo para 2.000 previsoes de confronto, (3) a duracao da
suite inteira (de evidence/suite_esteira.json). Guarda o historico em
evidence/desempenho.json (ultimas 90 medicoes) e REGRIDE se a medicao de hoje
passar de 2x a mediana das ultimas 10 E de um piso absoluto -- o piso evita
alarme por ruido em tempos de milissegundos.

Roda no robo da suite (suite.yml), todo dia. Arquivo identico nos dois repos.

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
monitoramento "The model has not experienced a dramatic or slow-leak
regression in training speed, serving latency, throughput, or RAM usage".
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Uso: python tools/medir_desempenho.py     (sai com 1 se regrediu)
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARQ = RAIZ / "evidence" / "desempenho.json"
PREVISOES = 2000
FATOR = 2.0
JANELA = 10
PISO_S = {"carregar_s": 1.0, "prever_s": 0.5, "suite_s": 120.0}


def medir() -> dict:
    sys.path.insert(0, str(RAIZ))
    from mlcore.artifact import MLArtifact
    t0 = time.perf_counter()
    art = MLArtifact.load_json(str(RAIZ / "ml_artifact.json"))
    carregar = time.perf_counter() - t0
    times = list((getattr(art.model, "ratings", {}) or {}))[:200]
    pares = [(times[i % len(times)], times[(i * 7 + 1) % len(times)]) for i in range(PREVISOES)]
    t0 = time.perf_counter()
    for a, b in pares:
        if a != b:
            art.model.p_blue(a, b)
    prever = time.perf_counter() - t0
    suite = None
    try:
        suite = float(json.loads((RAIZ / "evidence" / "suite_esteira.json")
                                 .read_text(encoding="utf-8")).get("segundos"))
    except (OSError, ValueError, TypeError):
        pass
    return {"em": datetime.now().isoformat(timespec="seconds"), "carregar_s": round(carregar, 4),
            "prever_s": round(prever, 4), "suite_s": suite}


def regrediu(atual: dict, historico: list[dict]) -> list[str]:
    """Pura (testavel): quais medidas passaram de FATOR x a mediana recente."""
    problemas = []
    for chave, piso in PISO_S.items():
        v = atual.get(chave)
        antes = [h.get(chave) for h in historico[-JANELA:] if isinstance(h.get(chave), (int, float))]
        if v is None or len(antes) < 5:
            continue
        med = statistics.median(antes)
        if v > FATOR * med and v > piso:
            problemas.append(f"{chave}: {v:.3f}s contra mediana {med:.3f}s (> {FATOR}x)")
    return problemas


def main() -> int:
    try:
        hist = json.loads(ARQ.read_text(encoding="utf-8")).get("historico", [])
    except (OSError, ValueError):
        hist = []
    atual = medir()
    problemas = regrediu(atual, hist)
    hist = (hist + [atual])[-90:]
    ARQ.parent.mkdir(parents=True, exist_ok=True)
    ARQ.write_text(json.dumps({"gerado_em": atual["em"], "ultimo": atual, "historico": hist},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[medir_desempenho] carregar {atual['carregar_s']}s | {PREVISOES} previsoes "
          f"{atual['prever_s']}s | suite {atual['suite_s']}s")
    for p in problemas:
        print("REGREDIU:", p)
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
