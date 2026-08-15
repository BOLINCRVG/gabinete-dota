"""Helpers compartilhados de campeões/draft — quebra o ciclo meta<->radar.

Antes: meta_tab_core importava _champ_icon_uri/_get_registry do radar_tab_core,
e radar_tab_core (+ champions_tab_core) importava _get_draft_rows do meta_tab_core
— um ciclo de import (contornado por imports lazy dentro de função). A régua de
código o detectava no grafo (componente 12).

Correção canônica (extrair para módulo compartilhado de baixo nível que nenhum
dos dois chama de volta): estas três funções — registro de campeões, ícone do
campeão e cache das linhas de draft — vivem aqui; meta e radar re-exportam para
uso interno e ambos passam a depender SÓ deste módulo. Ver
[[project_code_scorecard_sweep]].
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from app_core.champion_knowledge_core import load_champion_knowledge_registry
from app_core.meta_patch_core import load_draft_rows

_ICON_DIR = None


def _get_registry(session_state: Any) -> Mapping[str, Any]:
    reg = session_state.get("_radar_champ_registry")
    if not isinstance(reg, dict) or not reg.get("champions"):
        reg = load_champion_knowledge_registry()
        session_state["_radar_champ_registry"] = reg
    return reg


def _champ_icon_uri(champ_id: str) -> Optional[str]:
    """Data-URI do ícone do campeão (champ_icons/<id>.png, cache em módulo)."""
    global _ICON_DIR
    if _ICON_DIR is None:
        from pathlib import Path
        _ICON_DIR = Path(__file__).resolve().parents[1] / "champ_icons"
    cache = getattr(_champ_icon_uri, "_cache", None)
    if cache is None:
        cache = {}
        _champ_icon_uri._cache = cache
    cid = str(champ_id or "").strip()
    if not cid:
        return None
    if cid not in cache:
        p = _ICON_DIR / f"{cid}.png"
        if p.exists():
            import base64
            cache[cid] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
        else:
            cache[cid] = None
    return cache[cid]


def _get_draft_rows(session_state: Any, csv_sig: Any) -> pd.DataFrame:
    key = ("_meta_draft_rows", str(csv_sig))
    cached = session_state.get("_meta_draft_rows_cache")
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == key[1]:
        return cached[1]
    csv_path = csv_sig[0] if isinstance(csv_sig, (tuple, list)) else str(csv_sig)
    df = load_draft_rows(str(csv_path))
    session_state["_meta_draft_rows_cache"] = (key[1], df)
    return df
