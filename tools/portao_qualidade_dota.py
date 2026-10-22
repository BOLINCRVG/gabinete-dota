# -*- coding: utf-8 -*-
"""TRAVA DE QUALIDADE ANTES DE PUBLICAR o modelo do Dota (22/10/2026).

A esteira diaria ja conferia se o modelo CARREGA e preve (guarda de "nao esta
quebrado"). Faltava conferir se ele esta BOM antes de servir -- o item "qualidade
validada ANTES de servir" da regua das 28, que estava em zero.

O QUE MEDE: a nota que a propria esteira recalcula na mesma rodada
(dota_scorecard.py -> evidence/dota_scorecard.json, logloss prequencial da
temporada), comparada com a versao JA PUBLICADA no git (HEAD).

POR QUE NAO A METRICA DO HOLDOUT DO ARTEFATO: em 22/10/2026 ela dava logloss
0,705 -- pior que moeda --, mas mede ratings congelados no inicio dos 30 dias, e
a esteira os atualiza logo depois (update_incremental). Ela mede um modelo que
nunca e o servido; barrar por ela seria barrar todo dia.

REPROVA (exit 1) se: a nota nao foi re-medida nesta rodada; o logloss ficou pior
que moeda; piorou mais que 0,020 contra o publicado (o mesmo FAIL_DELTA do
alarme do LoL, tools/ml_quality_check.py); ou o numero de jogos medidos caiu mais
de 10% (sinal classico de CSV truncado).
A esteira, ao reprovar, publica os DADOS e mantem o MODELO anterior no ar.

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
infraestrutura "Model quality is validated before attempting to serve it".
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Uso: python tools/portao_qualidade_dota.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

MOEDA = math.log(2.0)          # 0,6931: logloss de cara ou coroa
FAIL_DELTA = 0.020
QUEDA_N = 0.10
ARQ = "evidence/dota_scorecard.json"


def decidir(novo: dict | None, antigo: dict | None) -> tuple[bool, str]:
    """Pura (testavel). novo/antigo = o bloco 'geral' do dota_scorecard."""
    if not novo:
        return False, "a nota nao foi medida nesta rodada"
    try:
        ll, n = float(novo["logloss"]), int(novo["n"])
    except (KeyError, TypeError, ValueError):
        return False, f"nota ilegivel: {novo}"
    if not math.isfinite(ll):
        return False, f"logloss nao e numero ({ll})"
    if ll >= MOEDA:
        return False, f"logloss {ll:.4f} e pior que moeda ({MOEDA:.4f})"
    if antigo:
        try:
            ll0, n0 = float(antigo["logloss"]), int(antigo["n"])
        except (KeyError, TypeError, ValueError):
            ll0, n0 = None, None
        if ll0 is not None and ll - ll0 > FAIL_DELTA:
            return False, f"logloss piorou {ll - ll0:+.4f} (de {ll0:.4f} para {ll:.4f}; limite {FAIL_DELTA})"
        if n0 and n < (1 - QUEDA_N) * n0:
            return False, f"jogos medidos cairam de {n0} para {n} (CSV truncado?)"
    return True, f"ok: logloss {ll:.4f} em {n} jogos"


def _publicado() -> dict | None:
    try:
        r = subprocess.run(["git", "show", f"HEAD:{ARQ}"], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout).get("geral") if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main() -> int:
    try:
        atual_txt = Path(ARQ).read_text(encoding="utf-8")
        novo = json.loads(atual_txt).get("geral")
    except (OSError, ValueError):
        novo = None
    # "a nota rodou nesta rodada?" quem sabe e a esteira (resultado do passo);
    # ela chama com --nao-medida quando o dota_scorecard falhou.
    if "--nao-medida" in sys.argv:
        novo = None
    antigo = _publicado()
    ok, motivo = decidir(novo, antigo)
    print(f"[portao_qualidade_dota] {'APROVADO' if ok else 'REPROVADO'} -- {motivo}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
