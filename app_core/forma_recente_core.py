# -*- coding: utf-8 -*-
"""FORMA RECENTE no preco do mapa (LoL e Dota; aprovado por ele em 16/11/2026).

logit(p_mapa') = logit(p_mapa) + PESO * (forma_A - forma_B), forma = soma de (resultado - p previsto
na hora) nos ultimos 5 mapas do time (tools/forma_recente.py, robo diario). Medido as cegas:
LoL +0.0073 IC95 [+0.0046; +0.0098]; Dota +0.0041 [+0.0017; +0.0065]. A serie e recalculada do mapa
com a mesma funcao do app. Time sem forma no arquivo: sem ajuste, e a tela diz por que.
Arquivo identico nos dois repositorios.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

_CACHE: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}


def carregar(raiz: Any) -> Optional[Dict[str, Any]]:
    raiz = Path(raiz)
    esporte = "dota" if (raiz / "dota_paths.py").exists() else "lol"
    p = raiz / f"forma_recente_{esporte}.json"
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    c = _CACHE.get(str(p))
    if c and c[0] == mt:
        return c[1]
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = None
    _CACHE[str(p)] = (mt, d)
    return d


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def ajustar(p_mapa: float, time_a: str, time_b: str, doc: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """{'p', 'p_antes', 'delta', 'a', 'b', 'motivo'}: p = P(A vence o mapa) com a forma."""
    out = {"p": p_mapa, "p_antes": p_mapa, "delta": 0.0, "a": None, "b": None, "motivo": ""}
    if not (isinstance(p_mapa, (int, float)) and 0.0 < p_mapa < 1.0):
        out["motivo"] = "sem p do mapa"
        return out
    if not doc:
        out["motivo"] = "o robo ainda nao gerou a forma recente"
        return out
    times = doc.get("times") or {}
    fa, fb = times.get(str(time_a)), times.get(str(time_b))
    out["a"], out["b"] = fa, fb
    faltam = [t for t, f in ((time_a, fa), (time_b, fb)) if not f]
    if faltam:
        out["motivo"] = "sem forma recente de " + " e ".join(faltam) + " (sem mapa com previsao do modelo)"
        return out
    delta = float(doc.get("peso") or 0.0) * (float(fa["forma"]) - float(fb["forma"]))
    out["delta"] = delta
    out["p"] = 1.0 / (1.0 + math.exp(-(_logit(p_mapa) + delta)))
    return out


def aplicar_no_confronto(st: Any, raiz: Any, time_a: str, time_b: str, p_map: float, p_series: float,
                         bo: int, prob_win_series: Callable[[float, int], float]) -> Tuple[float, float]:
    """Ajusta p do mapa e da serie do confronto aberto e mostra o quanto mexeu (uma linha)."""
    if not str(time_a or "").strip() or not str(time_b or "").strip():
        return p_map, p_series  # nenhum confronto aberto ainda
    r = ajustar(p_map, time_a, time_b, carregar(raiz))
    st.session_state["_forma_recente_info"] = r
    if r["motivo"]:
        if r["motivo"] != "sem p do mapa":
            st.caption(f"Forma recente: {r['motivo']}; preço sem esse ajuste.")
        return p_map, p_series
    try:
        if int(bo) > 1:
            # a serie do app ja tem ajustes proprios (gamma, coerencia): mantem a diferenca dela para
            # a serie "pura" do mapa antigo e soma a mudanca que a forma trouxe
            base_velha = float(prob_win_series(float(r["p_antes"]), int(bo)))
            ps = float(prob_win_series(float(r["p"]), int(bo))) + (float(p_series) - base_velha)
            if not math.isfinite(ps):
                ps = float(prob_win_series(float(r["p"]), int(bo)))
            ps = min(max(ps, 0.001), 0.999)
        else:
            ps = float(r["p"])
    except Exception:  # noqa: BLE001
        ps = p_series
    fa, fb = r["a"], r["b"]
    st.caption(
        f"Forma recente (últimos 5 mapas, já no preço): **{time_a}** {fa['resultados']} ({fa['forma']:+.1f}) · "
        f"**{time_b}** {fb['resultados']} ({fb['forma']:+.1f}) → soma **{100 * (r['p'] - r['p_antes']):+.1f} pp** "
        f"ao mapa de {time_a} ({100 * r['p_antes']:.1f}% → {100 * r['p']:.1f}%, antes das travas de confiança "
        f"do modelo, que ainda puxam um pouco para 50%). Forma = vitórias acima do que o modelo previa; medido às cegas.")
    return float(r["p"]), ps
