# -*- coding: utf-8 -*-
"""Cliente PandaScore (plano Fixtures) — agenda de jogos futuros do Dota 2.

A chave vive em .streamlit/secrets.toml (PANDASCORE_API_KEY) ou na env var de
mesmo nome. Plano free: fixtures/resultados, 1000 req/hora — a Diaria usa ~1
chamada por atualizacao (cacheada 10min no app).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://api.pandascore.co"
_SECRETS = Path(__file__).with_name(".streamlit") / "secrets.toml"

# tiers PandaScore (tournament.tier): s > a > b > c > d
DEFAULT_TIERS = ("s", "a", "b")


def resolve_api_key() -> str:
    key = (os.getenv("PANDASCORE_API_KEY") or "").strip()
    if key:
        return key
    try:
        import streamlit as st
        key = str(st.secrets.get("PANDASCORE_API_KEY", "") or "").strip()
        if key:
            return key
    except Exception:
        pass
    try:
        for line in _SECRETS.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("PANDASCORE_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    key = resolve_api_key()
    if not key:
        raise RuntimeError("PANDASCORE_API_KEY nao configurada.")
    resp = requests.get(
        f"{BASE_URL}/{path.lstrip('/')}",
        params=dict(params or {}),
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        timeout=(8, 30),
    )
    resp.raise_for_status()
    return resp.json()


def upcoming_matches(hours_ahead: int = 48, per_page: int = 50,
                     include_running: bool = True) -> List[dict]:
    """Jogos futuros (ate hours_ahead) + em andamento, normalizados p/ a Diaria.

    include_running: jogos que JA comecaram continuam aparecendo (o usuario
    quer consultar info com o jogo rolando — decisao 2026-07-08)."""
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=int(hours_ahead))
    raw = _get("dota2/matches/upcoming", {
        "per_page": int(per_page),
        "sort": "scheduled_at",
        "range[scheduled_at]": f"{now:%Y-%m-%dT%H:%M:%SZ},{until:%Y-%m-%dT%H:%M:%SZ}",
    })
    items = list(raw) if isinstance(raw, list) else []
    if include_running:
        try:
            running = _get("dota2/matches/running", {"per_page": 30})
            if isinstance(running, list):
                items = running + items
        except Exception:
            pass
    out: List[dict] = []
    for it in items:
        opps = [(o or {}).get("opponent") or {} for o in (it.get("opponents") or [])]
        if len(opps) != 2:
            continue
        tour = it.get("tournament") or {}
        league = it.get("league") or {}
        serie = it.get("serie") or {}
        out.append({
            "match_id": it.get("id"),
            "scheduled_at": str(it.get("scheduled_at") or it.get("original_scheduled_at") or ""),
            "league": str(league.get("name") or ""),
            "serie": str(serie.get("full_name") or serie.get("name") or ""),
            "tournament": str(tour.get("name") or ""),
            "tier": str(tour.get("tier") or "").lower(),
            "bo": int(it.get("number_of_games") or 0),
            "team_a": str(opps[0].get("name") or ""),
            "team_a_acronym": str(opps[0].get("acronym") or ""),
            "team_b": str(opps[1].get("name") or ""),
            "team_b_acronym": str(opps[1].get("acronym") or ""),
        })
    return out
