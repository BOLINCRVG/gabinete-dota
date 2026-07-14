from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

BASE_URL = "https://api.opendota.com/api"

_CONNECT_TIMEOUT = 8        # seconds to establish TCP connection
_READ_TIMEOUT_MATCH = 25    # seconds to read a single /matches/{id} response
_READ_TIMEOUT_LIST = 35     # seconds for list endpoints (proMatches, leagues, heroStats)
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0     # first retry waits 2s, second waits 4s
_LEAGUES_CACHE_TTL = 3600   # cache /leagues for 1 hour (it barely changes)

_session: Optional[requests.Session] = None
_leagues_cache: Dict[str, Any] = {}


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=4)
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


def resolve_api_key() -> Tuple[str, str]:
    key = (os.getenv("OPENDOTA_API_KEY") or "").strip()
    if key:
        return key, "env"
    try:
        import streamlit as st
        secret = (st.secrets.get("OPENDOTA_API_KEY", "") or "").strip()
        if secret:
            return secret, "secrets"
    except Exception:
        pass
    return "", "public"


def auth_status() -> Dict[str, Any]:
    key, source = resolve_api_key()
    masked = ""
    if key:
        masked = f"{key[:4]}...{key[-4:]}" if len(key) >= 10 else "***"
    return {"enabled": bool(key), "source": source, "masked_key": masked}


def _request(path: str, params: Optional[Dict[str, Any]] = None, read_timeout: int = _READ_TIMEOUT_MATCH) -> Any:
    key, _ = resolve_api_key()
    url = f"{BASE_URL}/{path.lstrip('/')}"
    query = dict(params or {})
    if key:
        query["api_key"] = key

    timeout = (_CONNECT_TIMEOUT, read_timeout)
    session = _get_session()
    last_exc: Optional[Exception] = None
    delay = _RETRY_BASE_DELAY

    for attempt in range(_MAX_RETRIES):
        try:
            response = session.get(url, params=query, timeout=timeout)
            # invalid key: retry without it
            if response.status_code == 400 and key:
                try:
                    body = response.text or ""
                except Exception:
                    body = ""
                if "API key invalid" in body:
                    q_no_key = {k: v for k, v in query.items() if k != "api_key"}
                    response = session.get(url, params=q_no_key, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            if status == 429 and attempt < _MAX_RETRIES - 1:
                last_exc = exc
                time.sleep(delay * 3)
                delay *= 2
                continue
            raise

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Falha ao consultar OpenDota: {path}")


def get_team_summary(team_id: int) -> Dict[str, Any]:
    return _request(f"teams/{int(team_id)}")


def get_team_heroes(team_id: int) -> list:
    data = _request(f"teams/{int(team_id)}/heroes")
    return data if isinstance(data, list) else []


def get_team_players(team_id: int) -> list:
    data = _request(f"teams/{int(team_id)}/players")
    return data if isinstance(data, list) else []


def get_team_matches(team_id: int) -> list:
    data = _request(f"teams/{int(team_id)}/matches")
    return data if isinstance(data, list) else []


def get_hero_stats() -> list:
    data = _request("heroStats", read_timeout=_READ_TIMEOUT_LIST)
    return data if isinstance(data, list) else []


def get_pro_matches(limit: int = 100) -> list:
    data = _request("proMatches", read_timeout=_READ_TIMEOUT_LIST)
    if not isinstance(data, list):
        return []
    return data[:max(1, int(limit))]


def get_leagues(tier: Optional[str] = None) -> list:
    cache_key = tier or "__all__"
    entry = _leagues_cache.get(cache_key)
    if entry and (time.time() - entry["ts"]) < _LEAGUES_CACHE_TTL:
        return entry["data"]

    raw = _request("leagues", read_timeout=_READ_TIMEOUT_LIST)
    if not isinstance(raw, list):
        return []

    _leagues_cache["__all__"] = {"data": raw, "ts": time.time()}

    if tier:
        wanted = str(tier).strip().lower()
        filtered = [r for r in raw if str((r or {}).get("tier") or "").strip().lower() == wanted]
        _leagues_cache[cache_key] = {"data": filtered, "ts": time.time()}
        return filtered

    return raw


def get_league_matches(league_id: int) -> list:
    data = _request(f"leagues/{int(league_id)}/matches", read_timeout=_READ_TIMEOUT_LIST)
    return data if isinstance(data, list) else []


def get_match(match_id: int) -> Dict[str, Any]:
    return _request(f"matches/{int(match_id)}", read_timeout=_READ_TIMEOUT_MATCH)


def get_player_heroes(account_id: int) -> list:
    data = _request(f"players/{int(account_id)}/heroes")
    return data if isinstance(data, list) else []
