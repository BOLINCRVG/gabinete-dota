# -*- coding: utf-8 -*-
"""Mistura da NOTA POR JOGADOR na chance do mapa (Dota, 27/10/2026).

O arquivo nota_jogadores_dota.json (gerado pelo robo, nota_jogadores_dota.py)
traz a forca de cada time = media das notas do elenco. A chance final do mapa e
    logit = c * ((1-w)*logit(p_modelo) + w*logit(p_jogador))
com w e c da prova cega (gravados no proprio arquivo). Sem o arquivo, ou sem um
dos dois times nele, devolve None e o app segue com o modelo de sempre.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

ARQ = Path(__file__).resolve().parents[1] / "nota_jogadores_dota.json"
_CACHE: Dict[str, Any] = {"mtime": None, "dados": None}


def _dados() -> Optional[dict]:
    try:
        mt = ARQ.stat().st_mtime
    except OSError:
        return None
    if _CACHE["mtime"] != mt:
        try:
            _CACHE["dados"] = json.loads(ARQ.read_text(encoding="utf-8"))
            _CACHE["mtime"] = mt
        except (OSError, ValueError):
            return None
    return _CACHE["dados"]


def _lg(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def p_com_jogadores(time_a: str, time_b: str, p_modelo: float, dados: Optional[dict] = None) -> Optional[dict]:
    """{p_jogador, p_final, w, c, forca_a, forca_b} ou None."""
    d = dados if dados is not None else _dados()
    if not d or not math.isfinite(float(p_modelo)):
        return None
    times = d.get("times") or {}
    a, b = times.get(str(time_a)), times.get(str(time_b))
    if not a or not b:
        return None
    par = d.get("parametros") or {}
    w, c = float(par.get("w_jogador", 0.5)), float(par.get("temperatura", 1.1))
    p_j = 1.0 / (1.0 + 10 ** (-(float(a["forca"]) - float(b["forca"])) / 400.0))
    z = c * ((1 - w) * _lg(p_modelo) + w * _lg(p_j))
    return {"p_jogador": p_j, "p_final": 1.0 / (1.0 + math.exp(-z)), "w": w, "c": c,
            "forca_a": float(a["forca"]), "forca_b": float(b["forca"]),
            "elenco_a": bool(a.get("elenco")), "elenco_b": bool(b.get("elenco"))}
