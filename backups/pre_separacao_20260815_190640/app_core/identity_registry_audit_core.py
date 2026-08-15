from __future__ import annotations

from typing import Any, Callable, Dict, Iterable


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_preserve(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_str(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def audit_identity_registry(
    registry: dict[str, Any] | None,
    *,
    canonicalize_league_fn: Callable[[str], str] | None = None,
    is_international_league_fn: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    data = registry if isinstance(registry, dict) else {}
    teams = [dict(row or {}) for row in list(data.get("teams") or []) if isinstance(row, dict)]
    players = [dict(row or {}) for row in list(data.get("players") or []) if isinstance(row, dict)]

    alias_to_team_keys: Dict[str, set[str]] = {}
    alias_to_player_keys: Dict[str, set[str]] = {}

    def _push_alias(bucket: Dict[str, set[str]], alias: str, entity_key: str) -> None:
        key = _clean_str(alias).casefold()
        ek = _clean_str(entity_key)
        if not key or not ek:
            return
        bucket.setdefault(key, set()).add(ek)

    for row in teams:
        entity_key = _clean_str(row.get("entity_key"))
        for alias in list(row.get("aliases") or []) + list(row.get("previous_names") or []):
            _push_alias(alias_to_team_keys, str(alias), entity_key)
        _push_alias(alias_to_team_keys, row.get("canonical_name"), entity_key)
        _push_alias(alias_to_team_keys, row.get("current_name"), entity_key)

    for row in players:
        entity_key = _clean_str(row.get("entity_key"))
        for alias in list(row.get("aliases") or []) + list(row.get("previous_names") or []):
            _push_alias(alias_to_player_keys, str(alias), entity_key)
        _push_alias(alias_to_player_keys, row.get("canonical_name"), entity_key)
        _push_alias(alias_to_player_keys, row.get("current_name"), entity_key)

    team_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    error_count = 0
    warn_count = 0
    ok_count = 0

    for row in teams:
        issues: list[str] = []
        entity_key = _clean_str(row.get("entity_key"))
        canonical_name = _clean_str(row.get("canonical_name"))
        current_name = _clean_str(row.get("current_name"))
        relation_type = _clean_str(row.get("relation_type")) or "alias"
        confidence = _clean_str(row.get("confidence")) or "medium"
        source_urls = _dedupe_preserve(list(row.get("source_urls") or []))
        current_leagues = _dedupe_preserve(list(row.get("current_leagues") or []))

        if not entity_key or not canonical_name or not current_name:
            issues.append("missing_identity_core")
        if relation_type not in {"alias", "manual_note"} and not source_urls:
            issues.append("missing_sources")
        if confidence == "high" and not source_urls:
            issues.append("high_confidence_without_sources")
        if not current_leagues:
            issues.append("missing_current_leagues")

        if current_leagues and callable(is_international_league_fn):
            if all(bool(is_international_league_fn(str(item))) for item in current_leagues):
                issues.append("international_only_current_leagues")

        canonical_leagues: list[str] = []
        if current_leagues and callable(canonicalize_league_fn):
            canonical_leagues = _dedupe_preserve(canonicalize_league_fn(str(item)) for item in current_leagues)
            if not any(canonical_leagues):
                issues.append("uncanonicalizable_current_leagues")

        aliases = _dedupe_preserve(
            [canonical_name, current_name, *list(row.get("aliases") or []), *list(row.get("previous_names") or [])]
        )
        collision_aliases = [
            alias
            for alias in aliases
            if len(alias_to_team_keys.get(alias.casefold(), set())) > 1
        ]
        if collision_aliases:
            issues.append("team_alias_collision")

        status = "ok"
        if any(issue in {"missing_identity_core", "team_alias_collision"} for issue in issues):
            status = "error"
            error_count += 1
        elif issues:
            status = "warn"
            warn_count += 1
        else:
            ok_count += 1

        team_rows.append(
            {
                "entity_type": "team",
                "entity_key": entity_key,
                "canonical_name": canonical_name,
                "current_name": current_name,
                "relation_type": relation_type,
                "confidence": confidence,
                "current_leagues": current_leagues,
                "canonical_leagues": canonical_leagues,
                "status": status,
                "issues": issues,
                "collision_aliases": collision_aliases,
            }
        )

    for row in players:
        issues: list[str] = []
        entity_key = _clean_str(row.get("entity_key"))
        canonical_name = _clean_str(row.get("canonical_name"))
        current_name = _clean_str(row.get("current_name"))
        relation_type = _clean_str(row.get("relation_type")) or "alias"
        confidence = _clean_str(row.get("confidence")) or "medium"
        source_urls = _dedupe_preserve(list(row.get("source_urls") or []))
        current_team = _clean_str(row.get("current_team"))

        if not entity_key or not canonical_name or not current_name:
            issues.append("missing_identity_core")
        if relation_type not in {"alias", "manual_note"} and not source_urls:
            issues.append("missing_sources")
        if confidence == "high" and not source_urls:
            issues.append("high_confidence_without_sources")
        if confidence == "high" and not current_team:
            issues.append("missing_current_team")

        aliases = _dedupe_preserve(
            [canonical_name, current_name, *list(row.get("aliases") or []), *list(row.get("previous_names") or [])]
        )
        collision_aliases = [
            alias
            for alias in aliases
            if len(alias_to_player_keys.get(alias.casefold(), set())) > 1
        ]
        if collision_aliases:
            issues.append("player_alias_collision")

        status = "ok"
        if any(issue in {"missing_identity_core", "player_alias_collision"} for issue in issues):
            status = "error"
            error_count += 1
        elif issues:
            status = "warn"
            warn_count += 1
        else:
            ok_count += 1

        player_rows.append(
            {
                "entity_type": "player",
                "entity_key": entity_key,
                "canonical_name": canonical_name,
                "current_name": current_name,
                "relation_type": relation_type,
                "confidence": confidence,
                "current_team": current_team,
                "status": status,
                "issues": issues,
                "collision_aliases": collision_aliases,
            }
        )

    return {
        "summary": {
            "team_count": len(teams),
            "player_count": len(players),
            "ok_count": ok_count,
            "warn_count": warn_count,
            "error_count": error_count,
        },
        "team_rows": team_rows,
        "player_rows": player_rows,
    }
