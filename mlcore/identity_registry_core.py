"""Local registry helpers for team/player identity, aliases and successors."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from mlcore.tiers import canonical_league, is_international_league


def _now_iso_local() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def normalize_identity_key(value: str) -> str:
    """Normaliza uma string para chave de identidade em lowercase alfanumérico sem espaços ou pontuação."""
    text = (value or "").strip().casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def _clean_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = re.split(r"[\n,;|]+", value)
    elif isinstance(value, Iterable):
        chunks = []
        for item in value:
            chunks.extend(re.split(r"[\n,;|]+", str(item or "")))
    else:
        chunks = [str(value)]
    out: list[str] = []
    for item in chunks:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def empty_identity_registry() -> dict[str, Any]:
    """Retorna um dicionário vazio com a estrutura padrão de registry de identidades."""
    return {
        "version": 1,
        "updated_at": "",
        "teams": [],
        "players": [],
    }


def _normalize_team_record(record: dict[str, Any]) -> dict[str, Any]:
    canonical_name = str(record.get("canonical_name") or record.get("current_name") or "").strip()
    current_name = str(record.get("current_name") or canonical_name).strip()
    previous_names = _clean_text_list(record.get("previous_names"))
    aliases = _clean_text_list(record.get("aliases"))
    if canonical_name and canonical_name not in aliases:
        aliases = [canonical_name, *aliases]
    if current_name and current_name not in aliases:
        aliases = [current_name, *aliases]
    for item in previous_names:
        if item and item not in aliases:
            aliases.append(item)
    return {
        "entity_key": normalize_identity_key(canonical_name or current_name),
        "canonical_name": canonical_name,
        "current_name": current_name,
        "previous_names": previous_names,
        "aliases": aliases,
        "relation_type": str(record.get("relation_type") or "alias").strip() or "alias",
        "confidence": str(record.get("confidence") or "medium").strip() or "medium",
        "summary": str(record.get("summary") or "").strip(),
        "current_leagues": _clean_text_list(record.get("current_leagues")),
        "kept_players": _clean_text_list(record.get("kept_players")),
        "added_players": _clean_text_list(record.get("added_players")),
        "removed_players": _clean_text_list(record.get("removed_players")),
        "kept_staff": _clean_text_list(record.get("kept_staff")),
        "notes": str(record.get("notes") or "").strip(),
        "source_urls": _clean_text_list(record.get("source_urls")),
        "updated_at": str(record.get("updated_at") or _now_iso_local()),
    }


def _normalize_player_record(record: dict[str, Any]) -> dict[str, Any]:
    canonical_name = str(record.get("canonical_name") or record.get("current_name") or "").strip()
    current_name = str(record.get("current_name") or canonical_name).strip()
    previous_names = _clean_text_list(record.get("previous_names"))
    aliases = _clean_text_list(record.get("aliases"))
    if canonical_name and canonical_name not in aliases:
        aliases = [canonical_name, *aliases]
    if current_name and current_name not in aliases:
        aliases = [current_name, *aliases]
    for item in previous_names:
        if item and item not in aliases:
            aliases.append(item)
    return {
        "entity_key": normalize_identity_key(canonical_name or current_name),
        "canonical_name": canonical_name,
        "current_name": current_name,
        "previous_names": previous_names,
        "aliases": aliases,
        "relation_type": str(record.get("relation_type") or "alias").strip() or "alias",
        "confidence": str(record.get("confidence") or "medium").strip() or "medium",
        "summary": str(record.get("summary") or "").strip(),
        "current_team": str(record.get("current_team") or "").strip(),
        "notes": str(record.get("notes") or "").strip(),
        "source_urls": _clean_text_list(record.get("source_urls")),
        "updated_at": str(record.get("updated_at") or _now_iso_local()),
    }


def normalize_identity_registry(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normaliza um payload de registry de identidades, validando times e jogadores."""
    data = empty_identity_registry()
    if isinstance(payload, dict):
        data.update(
            {
                "version": int(payload.get("version") or 1),
                "updated_at": str(payload.get("updated_at") or ""),
                "teams": payload.get("teams") or [],
                "players": payload.get("players") or [],
            }
        )
    data["teams"] = [
        item
        for item in (_normalize_team_record(dict(row or {})) for row in list(data.get("teams") or []))
        if item.get("entity_key")
    ]
    data["players"] = [
        item
        for item in (_normalize_player_record(dict(row or {})) for row in list(data.get("players") or []))
        if item.get("entity_key")
    ]
    data["updated_at"] = str(data.get("updated_at") or _now_iso_local())
    return data


def load_identity_registry(path: str | Path) -> dict[str, Any]:
    """Carrega e normaliza o registry de identidades do arquivo JSON indicado."""
    file_path = Path(path)
    if not file_path.exists():
        return empty_identity_registry()
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return empty_identity_registry()
    return normalize_identity_registry(payload if isinstance(payload, dict) else None)


def save_identity_registry(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Normaliza e persiste o registry de identidades no arquivo JSON indicado."""
    file_path = Path(path)
    normalized = normalize_identity_registry(payload)
    normalized["updated_at"] = _now_iso_local()
    file_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def _score_match(query_key: str, values: list[str]) -> tuple[int, str]:
    best_score = -1
    best_value = ""
    for value in values:
        key = normalize_identity_key(value)
        if not key:
            continue
        score = -1
        if key == query_key:
            score = 300
        elif key.startswith(query_key):
            score = 220
        elif query_key in key:
            score = 180
        if score > best_score:
            best_score = score
            best_value = value
    return best_score, best_value


def search_identity_records(
    registry: dict[str, Any],
    *,
    query: str,
    entity_type: str = "all",
) -> list[dict[str, Any]]:
    """Busca registros de identidade no registry por texto, retornando resultados ordenados por score."""
    query_key = normalize_identity_key(query)
    out: list[dict[str, Any]] = []
    want_teams = entity_type in ("all", "teams")
    want_players = entity_type in ("all", "players")

    def _push(record: dict[str, Any], kind: str) -> None:
        values = [
            str(record.get("canonical_name") or ""),
            str(record.get("current_name") or ""),
            *list(record.get("previous_names") or []),
            *list(record.get("aliases") or []),
        ]
        score, matched_value = _score_match(query_key, values) if query_key else (0, str(record.get("canonical_name") or ""))
        if query_key and score < 0:
            return
        out.append(
            {
                "entity_type": kind,
                "score": score,
                "matched_value": matched_value,
                "record": record,
            }
        )

    if want_teams:
        for record in list(registry.get("teams") or []):
            if isinstance(record, dict):
                _push(record, "team")
    if want_players:
        for record in list(registry.get("players") or []):
            if isinstance(record, dict):
                _push(record, "player")

    out.sort(
        key=lambda item: (
            -int(item.get("score") or 0),
            str(item.get("record", {}).get("canonical_name") or ""),
        )
    )
    return out


def resolve_team_identity_record(
    registry: dict[str, Any] | None,
    team_name: str,
) -> dict[str, Any]:
    """Retorna o melhor registro de identidade de time para o nome fornecido."""
    query = str(team_name or "").strip()
    if not query:
        return {}
    matches = search_identity_records(
        registry if isinstance(registry, dict) else empty_identity_registry(),
        query=query,
        entity_type="teams",
    )
    if not matches:
        return {}
    record = matches[0].get("record")
    return dict(record) if isinstance(record, dict) else {}


def resolve_player_identity_record(
    registry: dict[str, Any] | None,
    player_name: str,
) -> dict[str, Any]:
    """Retorna o melhor registro de identidade de jogador para o nome fornecido."""
    query = str(player_name or "").strip()
    if not query:
        return {}
    matches = search_identity_records(
        registry if isinstance(registry, dict) else empty_identity_registry(),
        query=query,
        entity_type="players",
    )
    if not matches:
        return {}
    record = matches[0].get("record")
    return dict(record) if isinstance(record, dict) else {}


def team_identity_candidates(
    registry: dict[str, Any] | None,
    team_name: str,
) -> list[str]:
    """Retorna lista de nomes candidatos (aliases e nomes anteriores) para um time dado."""
    query = str(team_name or "").strip()
    out: list[str] = []

    def _push(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)

    _push(query)
    record = resolve_team_identity_record(registry, query)
    if not record:
        return out

    _push(record.get("current_name"))
    _push(record.get("canonical_name"))
    for value in list(record.get("aliases") or []):
        _push(value)
    for value in list(record.get("previous_names") or []):
        _push(value)
    return out


def player_identity_candidates(
    registry: dict[str, Any] | None,
    player_name: str,
) -> list[str]:
    """Retorna lista de nomes candidatos (aliases e nomes anteriores) para um jogador dado."""
    query = str(player_name or "").strip()
    out: list[str] = []

    def _push(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)

    _push(query)
    record = resolve_player_identity_record(registry, query)
    if not record:
        return out

    _push(record.get("current_name"))
    _push(record.get("canonical_name"))
    for value in list(record.get("aliases") or []):
        _push(value)
    for value in list(record.get("previous_names") or []):
        _push(value)
    return out


def team_identity_league_candidates(
    registry: dict[str, Any] | None,
    team_name: str,
) -> list[str]:
    """Retorna lista de ligas candidatas para um time, priorizando domésticas antes de internacionais."""
    query = str(team_name or "").strip()
    if not query:
        return []

    record = resolve_team_identity_record(registry, query)
    raw_leagues = _clean_text_list((record or {}).get("current_leagues"))
    if not raw_leagues:
        return []

    buckets: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for idx, value in enumerate(raw_leagues):
        text = str(value or "").strip()
        if not text:
            continue
        canon = str(canonical_league(text) or text).strip()
        key = normalize_identity_key(canon or text)
        if not key or key in seen:
            continue
        seen.add(key)
        buckets.append(
            (
                1 if is_international_league(text) else 0,
                idx,
                canon or text,
            )
        )
    buckets.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[2] for item in buckets]


def upsert_team_identity(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    """Insere ou atualiza um registro de identidade de time no registry e persiste o arquivo."""
    registry = load_identity_registry(path)
    normalized = _normalize_team_record(record)
    rows = [row for row in list(registry.get("teams") or []) if str(row.get("entity_key")) != str(normalized.get("entity_key"))]
    rows.append(normalized)
    registry["teams"] = rows
    return save_identity_registry(path, registry)


def upsert_player_identity(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    """Insere ou atualiza um registro de identidade de jogador no registry e persiste o arquivo."""
    registry = load_identity_registry(path)
    normalized = _normalize_player_record(record)
    rows = [row for row in list(registry.get("players") or []) if str(row.get("entity_key")) != str(normalized.get("entity_key"))]
    rows.append(normalized)
    registry["players"] = rows
    return save_identity_registry(path, registry)
