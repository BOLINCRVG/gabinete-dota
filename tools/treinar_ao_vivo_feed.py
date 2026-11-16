# -*- coding: utf-8 -*-
"""SUPER-LIVE: treina os modelos ao vivo SO com o que o feed ao vivo entrega.

O feed ao vivo do LoL (livestats da lolesports) nao traz a experiencia (XP) dos times; o do Dota
(OpenDota /live) so traz a vantagem de patrimonio e o placar de abates. Este script roda o gate
oficial do modelo ao vivo (tools/run_phase6_live_gate.py no LoL, run_phase6_live_dota.py no Dota)
com essas colunas zeradas (LightGBM com coluna constante = coluna fora) e grava em
live_models_feed/. Os modelos de live_models/ (com todas as colunas) ficam intactos.

Medido em 16/11/2026 (mesmo teste, logloss do modelo completo -> so feed):
  LoL  @10 0.5205->0.5226  @15 0.4748->0.4762  @20 0.4129->0.4148  @25 0.3426->0.3471
  Dota @10 0.5364->0.5422  @15 0.4962->0.5013  @20 0.4495->0.4514  @25 0.4086->0.4117
  (o pre-jogo nesses jogos: ~0.59-0.63). Perda pequena perto do ganho do estado de jogo.

Arquivo identico nos dois repositorios. Uso: python tools/treinar_ao_vivo_feed.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

E_DOTA = (RAIZ / "run_phase6_live_dota.py").exists()
# Colunas que o feed NAO tem (zeradas no treino e na previsao).
FORA_DO_FEED = ("xp_diff", "cs_diff", "assists_diff") if E_DOTA else ("xp_diff",)
PASTA = "live_models_feed"


def main() -> int:
    import mlcore.live_ml.features as F

    original = F.build_snapshot_features
    idx = [list(F.feature_names()).index(c) for c in FORA_DO_FEED]

    def so_feed(*a, **k):
        out = []
        for s in original(*a, **k):
            f = np.array(s.features, dtype=float)
            f[idx] = 0.0
            out.append(F.LiveSnapshot(gameid=s.gameid, minute=s.minute, side=s.side,
                                      p_pre=s.p_pre, features=f, y=s.y))
        return out

    arq = RAIZ / ("run_phase6_live_dota.py" if E_DOTA else "tools/run_phase6_live_gate.py")
    spec = importlib.util.spec_from_file_location("gate_ao_vivo", arq)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.build_snapshot_features = so_feed
    F.build_snapshot_features = so_feed
    saida = str(RAIZ / "evidence" / "ao_vivo_feed.json")
    (RAIZ / "evidence").mkdir(exist_ok=True)
    sys.argv = [str(arq), "--output", saida, "--save-models-dir", str(RAIZ / PASTA)]
    if not E_DOTA:
        sys.argv += ["--n-bootstrap", "200"]
    rc = gate.main()
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
