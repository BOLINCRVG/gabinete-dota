# -*- coding: utf-8 -*-
"""A LINHA INICIAL da Analise de Linha: a do jogo, nao a padrao (24/10/2026).

Ele: "essas linhas padroes elas sao boas, mas as vezes elas enganam ... se dois
times com medias de 23,5 e 24,5 jogarem, e o modelo vier por padrao 27,5, logico
que vai ser excelente, mas nao vai ter na casa". Linha que a casa nao oferece
mostra um valor que ninguem consegue apostar.

ORDEM (a primeira que existir):
  1. a linha PRINCIPAL da Pinnacle para este jogo e este mapa (a que ela
     oferece de verdade; o robo coleta). A Polymarket hoje so tem vencedor;
  2. a MEDIA DAS MEDIAS dos dois times (no recorte que a tela ja usa),
     arredondada para o ,5 mais perto -- a linha onde o jogo "deveria" estar;
  3. a padrao do mercado (stats_config), como antes.
Ele continua podendo trocar a linha na mao; isto so muda de onde ela comeca.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]

# chave do mercado na Analise de Linha -> unidade da Pinnacle
UNIDADE_DO_MERCADO = {
    "kills": "kills", "towers": "turrets", "dragons": "dragons",
    "barons": "barons", "inhibitors": "inhibitors", "inibs": "inhibitors",
}


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def media_das_medias(serie_a, serie_b) -> Optional[float]:
    """Media de cada time, depois a media das duas, no ,5 mais perto."""
    import pandas as pd
    medias = []
    for s in (serie_a, serie_b):
        v = pd.to_numeric(s, errors="coerce").dropna() if s is not None else []
        if len(v):
            medias.append(float(v.mean()))
    if not medias:
        return None
    m = sum(medias) / len(medias)
    return math.floor(m) + 0.5


def _chave(nome: str) -> str:
    from app_core.schedule_import_core import normalize_schedule_team_key
    return normalize_schedule_team_key(str(nome or ""))


def _apelidos() -> Dict[str, str]:
    try:
        return json.loads((RAIZ / "user_team_aliases.json").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def _e_o_time(nome_casa: str, time_csv: str, apelidos: Mapping[str, str]) -> bool:
    k = _chave(nome_casa)
    if k == _chave(time_csv):
        return True
    return _chave(apelidos.get(k, "")) == _chave(time_csv)


def linha_da_pinnacle(odds: Mapping[str, Any], time_a: str, time_b: str, unidade: str,
                      mapa: int = 1, apelidos: Optional[Mapping[str, str]] = None) -> Optional[float]:
    """A linha principal (nao alternativa) do total desse mercado nesse mapa. Pura."""
    apelidos = apelidos if apelidos is not None else {}
    jogo_id = None
    for j in odds.get("jogos") or []:
        casa, fora = str(j.get("casa") or ""), str(j.get("fora") or "")
        if ((_e_o_time(casa, time_a, apelidos) and _e_o_time(fora, time_b, apelidos))
                or (_e_o_time(casa, time_b, apelidos) and _e_o_time(fora, time_a, apelidos))):
            jogo_id = j.get("id")
            break
    if jogo_id is None:
        return None
    candidatas: List[Tuple[bool, float]] = []
    for l in odds.get("linhas") or []:
        if not isinstance(l, dict) or l.get("jogo") != jogo_id or str(l.get("tipo") or "") != "total":
            continue
        if str(l.get("unidade") or "").strip().lower() != unidade:
            continue
        try:
            if int(l.get("periodo") or 0) != int(mapa):
                continue
        except (TypeError, ValueError):
            continue
        p = _num(l.get("pontos"))
        if p is not None:
            candidatas.append((bool(l.get("alternativa")), p))
    principais = [p for alt, p in candidatas if not alt]
    if principais:
        return principais[0]
    return None


def linha_inicial(mercado: str, time_a: str, time_b: str, serie_a, serie_b,
                  padrao: float, *, mapa: int = 1,
                  odds: Optional[Mapping[str, Any]] = None) -> Tuple[float, str]:
    """(linha, de onde veio). Ver a ordem no topo do arquivo."""
    unidade = UNIDADE_DO_MERCADO.get(str(mercado))
    if unidade:
        if odds is None:
            try:
                from app_core.odds_coletadas_core import carregar_odds_coletadas
                odds = carregar_odds_coletadas(RAIZ)
            except Exception:
                odds = {}
        p = linha_da_pinnacle(odds or {}, time_a, time_b, unidade, mapa, _apelidos())
        if p is not None:
            return float(p), "a linha principal da Pinnacle para este jogo"
    if str(mercado) in UNIDADE_DO_MERCADO or str(mercado) == "time":
        m = media_das_medias(serie_a, serie_b)
        if m is not None:
            return float(m), "a media das medias dos dois times (a Pinnacle nao tem este mercado aqui)"
    return float(padrao), "a linha padrao do mercado"
