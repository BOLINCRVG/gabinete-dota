# -*- coding: utf-8 -*-
"""PERFIL DA COMP (24/11/2026): uma linha DESCRITIVA por time na tela do draft ("Iniciacao 3 · Controle 4 ·
4 de alcance..."), para ele ler o draft do jeito dele. Ele pediu ("coloque, por favor") depois de ver a medicao:
as interacoes de estilo (alcance x engage, dive x protecao, burst x frontline, dano de um tipo so...) foram
testadas as cegas em 24/11/2026 e NAO melhoraram a previsao -- por isso a linha NAO mexe no preco e a tela diz.
Perfis: perfil_herois_<esporte>.json (tools/perfil_herois.py). Arquivo identico nos repositorios.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from app_core.camada_draft_core import chave_campeao

PAPEIS_DOTA = (("Initiator", "Iniciação"), ("Disabler", "Controle"), ("Durable", "Durável"), ("Carry", "Carry"),
               ("Nuker", "Dano explosivo"), ("Pusher", "Push"), ("Escape", "Fuga"), ("Support", "Suporte"))
GRUPOS_LOL = (("Engage", ("Vanguard", "Catcher", "Diver")), ("Frontline", ("Vanguard", "Warden", "Juggernaut")),
              ("Proteção", ("Warden", "Enchanter")), ("Dive", ("Diver", "Assassin")), ("Burst", ("Burst", "Assassin")),
              ("Poke", ("Artillery",)))


def carregar(raiz: Path, esporte: str) -> Dict[str, Any]:
    try:
        return json.loads((Path(raiz) / f"perfil_herois_{esporte}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def descrever(esporte: str, herois: Sequence[Any], perfis: Mapping[str, Any]) -> str:
    """Pura: os 5 do time -> 'Iniciação 3 · Controle 4 · ... · 4 de alcance (médio 520)'. Sem perfil: diz quantos."""
    ps: List[Mapping[str, Any]] = []
    falta = 0
    for h in herois:
        p = perfis.get(str(h)) or perfis.get(chave_campeao(h))
        if p:
            ps.append(p)
        else:
            falta += 1
    if not ps:
        return "perfil indisponível"
    partes = []
    if esporte == "dota":
        for chave, rot in PAPEIS_DOTA:
            n = sum(1 for p in ps if chave in (p.get("papeis") or []))
            if n:
                partes.append(f"{rot} {n}")
    else:
        for rot, classes in GRUPOS_LOL:
            n = sum(1 for p in ps if set(classes) & set(p.get("classes") or []) or (rot == "Poke" and p.get("longo_alcance")))
            if n:
                partes.append(f"{rot} {n}")
        mag = sum(1 for p in ps if p.get("magico"))
        partes.append(f"dano {mag} mág./{len(ps) - mag} fís.")
    rng = [p for p in ps if p.get("ranged")]
    media = round(sum(int(p.get("alcance") or 0) for p in ps) / len(ps))
    partes.append(f"{len(rng)} de alcance (médio {media})")
    if falta:
        partes.append(f"{falta} sem perfil")
    return " · ".join(partes)
