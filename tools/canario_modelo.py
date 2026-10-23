# -*- coding: utf-8 -*-
"""CANARIO: o modelo novo preve parecido com o que esta no ar? (23/10/2026)

Item "canario antes de servir" da regua das 28 (LoL 0,5, Dota 0). As guardas
pos-retreino conferem se o modelo novo carrega, preve e (no Dota) se a nota
nao caiu. Nenhuma olhava o que o canario olha: se, nos MESMOS jogos, o modelo
novo da numeros muito diferentes do que esta servindo hoje. Um retreino diario
mexe pouco; um que vira o jogo de muitos times de uma vez e quebra (dado
duplicado, liga mapeada errada, corte de data trocado), mesmo com nota boa.

Compara o modelo no disco (o novo, recem-treinado pelo robo) com o do ultimo
commit (o que esta no ar), ambos pela MESMA funcao de previsao (model.p_blue),
nos jogos mais recentes do CSV. Barra se a mediana de |dp| ou a fatia de jogos
com |dp| grande passarem dos tetos.

TETOS medidos (23/10/2026), nao chutados: tools/canario_modelo.py --historico
mede o |dp| entre cada par de modelos consecutivos do git. MEDIDO nos 15
ultimos modelos de cada repo, 500 jogos: pior mediana 0,0009 (LoL) e 0,0126
(Dota); pior fatia com |dp| > 0,15: 1,8% (LoL) e 2,0% (Dota). Os tetos (0,05 e
10%) deixam folga de 4-5x sobre o pior dia normal.

Roda nos robos antes do commit do modelo. Sai 1 se barrar; o robo restaura o
modelo antigo. Arquivo identico nos dois repositorios.

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
infraestrutura "The model is tested via a canary process before it enters
production serving environments".
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Uso: python tools/canario_modelo.py               (novo no disco x HEAD)
     python tools/canario_modelo.py --historico   (mede os tetos no git)
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

N_JOGOS = 500
TETO_MEDIANA = 0.05
TETO_GRANDE = 0.15       # o que e "|dp| grande"
TETO_FATIA_GRANDE = 0.10  # fatia maxima de jogos com |dp| grande


def _git(*a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=str(RAIZ), capture_output=True,
                          encoding="utf-8", errors="replace", timeout=300)


def _modelo_do_commit(commit: str, pasta: Path):
    from mlcore.artifact import MLArtifact
    r = subprocess.run(["git", "show", f"{commit}:ml_artifact.json"], cwd=str(RAIZ),
                       capture_output=True, timeout=300)
    if r.returncode != 0:
        return None
    p = pasta / f"{commit[:10]}.json"
    p.write_bytes(r.stdout)
    return MLArtifact.load_json(str(p)).model


def _jogos():
    import pandas as pd
    from mlcore.io_oracle import load_oracle_elixir_games
    from mlcore.tiers import tier_for_league
    csvs = sorted((RAIZ / "CSV_DIARIO").glob("*.csv"))
    g = load_oracle_elixir_games(str(csvs[-1]), keep_playoffs=None)
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    saida = []
    for r in g.sort_values("date").tail(N_JOGOS).itertuples(index=False):
        lg = str(getattr(r, "league", "") or "OTHER")
        saida.append((str(r.blue), str(r.red), lg, str(tier_for_league(lg))))
    return saida


def diferencas(velho, novo, jogos) -> list[float]:
    def p(m, j):
        return float(m.p_blue(j[0], j[1], blue_league=j[2], red_league=j[2],
                              blue_tier=j[3], red_tier=j[3]))
    return [abs(p(novo, j) - p(velho, j)) for j in jogos]


def decidir(dps: list[float]) -> dict:
    """Pura: {'ok', 'mediana', 'fatia_grande', 'motivos'}."""
    if not dps:
        return {"ok": True, "mediana": 0.0, "fatia_grande": 0.0, "motivos": ["sem jogos para comparar"]}
    med = statistics.median(dps)
    fatia = sum(d > TETO_GRANDE for d in dps) / len(dps)
    motivos = []
    if med > TETO_MEDIANA:
        motivos.append(f"mediana de |dp| {med:.3f} > {TETO_MEDIANA}")
    if fatia > TETO_FATIA_GRANDE:
        motivos.append(f"{fatia:.1%} dos jogos com |dp| > {TETO_GRANDE} (max {TETO_FATIA_GRANDE:.0%})")
    return {"ok": not motivos, "mediana": round(med, 4), "fatia_grande": round(fatia, 4),
            "max": round(max(dps), 4), "motivos": motivos}


def historico(n=15) -> int:
    commits = _git("log", f"-{n + 1}", "--format=%H", "--", "ml_artifact.json").stdout.split()
    jogos = _jogos()
    with tempfile.TemporaryDirectory() as d:
        modelos = [(c, _modelo_do_commit(c, Path(d))) for c in commits]
    linhas = []
    for (c_novo, novo), (c_velho, velho) in zip(modelos, modelos[1:]):
        if novo is None or velho is None:
            continue
        r = decidir(diferencas(velho, novo, jogos))
        linhas.append(r)
        print(f"{c_velho[:8]} -> {c_novo[:8]}: mediana {r['mediana']:.4f}  "
              f"fatia>{TETO_GRANDE} {r['fatia_grande']:.3f}  max {r['max']:.3f}")
    if linhas:
        print(f"PIOR mediana {max(x['mediana'] for x in linhas):.4f}, "
              f"PIOR fatia {max(x['fatia_grande'] for x in linhas):.3f}")
    return 0


def main() -> int:
    if "--historico" in sys.argv:
        return historico()
    from mlcore.artifact import MLArtifact
    jogos = _jogos()
    novo = MLArtifact.load_json(str(RAIZ / "ml_artifact.json")).model
    with tempfile.TemporaryDirectory() as d:
        velho = _modelo_do_commit("HEAD", Path(d))
    if velho is None:
        print("[canario] sem modelo no HEAD para comparar; deixo passar")
        return 0
    r = decidir(diferencas(velho, novo, jogos))
    print(f"[canario] {len(jogos)} jogos: mediana |dp| {r['mediana']}, "
          f"fatia > {TETO_GRANDE}: {r['fatia_grande']}, max {r.get('max')}")
    for m in r["motivos"]:
        print("BARRADO:", m)
    print("[canario] " + ("PASSOU" if r["ok"] else "BARROU o modelo novo"))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
