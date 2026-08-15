from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


TRAIT_KEYS = (
    "early",
    "mid",
    "late",
    "lane_priority",
    "all_in",
    "poke",
    "waveclear",
    "engage",
    "peel",
    "pick",
    "dive",
    "frontline",
    "siege",
    "splitpush",
    "skirmish",
    "teamfight",
    "objective_control",
    "mobility",
    "execution",
    "range",
)

STYLE_TRAITS = (
    "engage",
    "peel",
    "pick",
    "dive",
    "frontline",
    "siege",
    "splitpush",
    "skirmish",
    "teamfight",
    "poke",
)


def _norm_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def default_registry_path() -> Path:
    return Path(__file__).resolve().parent.parent / "champion_knowledge_registry.json"


def load_champion_knowledge_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else default_registry_path()
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return {"meta": {"ok": False, "path": str(registry_path)}, "champions": {}, "aliases": {}}
    champions = data.get("champions")
    aliases = data.get("aliases")
    if not isinstance(champions, dict):
        champions = {}
    if not isinstance(aliases, dict):
        aliases = {}
    out = dict(data)
    out["champions"] = champions
    out["aliases"] = aliases
    return out


def _generic_role_fallback(role_hint: str) -> dict[str, Any]:
    role = str(role_hint or "").strip().lower()
    base_traits = {key: 0.0 for key in TRAIT_KEYS}
    if role == "top":
        base_traits.update({"early": 1.0, "mid": 1.0, "late": 0.5, "lane_priority": 1.0, "all_in": 1.0, "frontline": 1.5, "splitpush": 1.0, "skirmish": 1.0})
        styles = ["skirmish", "frontline"]
    elif role == "jng":
        base_traits.update({"early": 1.0, "mid": 1.5, "late": 0.5, "engage": 1.0, "pick": 1.0, "dive": 1.0, "objective_control": 1.5, "mobility": 1.0, "skirmish": 1.5})
        styles = ["engage", "skirmish"]
    elif role == "mid":
        base_traits.update({"early": 0.5, "mid": 1.5, "late": 1.5, "lane_priority": 1.0, "poke": 1.0, "waveclear": 1.5, "pick": 1.0, "teamfight": 1.0, "mobility": 1.0})
        styles = ["poke", "teamfight"]
    elif role == "bot":
        base_traits.update({"early": 0.0, "mid": 1.0, "late": 2.0, "range": 2.0, "siege": 1.5, "objective_control": 1.5, "teamfight": 1.0})
        styles = ["siege", "teamfight"]
    elif role == "sup":
        base_traits.update({"early": 0.5, "mid": 1.0, "late": 0.5, "engage": 1.0, "peel": 1.5, "pick": 0.5, "frontline": 0.5, "teamfight": 1.0})
        styles = ["peel", "engage"]
    else:
        styles = ["teamfight"]
    return {
        "name": "",
        "primary_role": role or "unknown",
        "secondary_roles": [],
        "styles": styles,
        "window": "mid",
        "traits": base_traits,
    }


def get_champion_knowledge(
    registry: Mapping[str, Any] | None,
    champion_name: str,
    *,
    role_hint: str | None = None,
) -> dict[str, Any]:
    champions = {}
    aliases = {}
    if isinstance(registry, Mapping):
        champions = registry.get("champions") or {}
        aliases = registry.get("aliases") or {}
    raw_name = str(champion_name or "").strip()
    if not raw_name:
        return _generic_role_fallback(str(role_hint or ""))
    norm = _norm_key(raw_name)
    resolved_key = str(aliases.get(norm) or norm)
    entry = champions.get(resolved_key)
    if isinstance(entry, Mapping):
        out = dict(entry)
        out.setdefault("name", raw_name)
        out.setdefault("primary_role", str(role_hint or "unknown"))
        out.setdefault("secondary_roles", [])
        out.setdefault("styles", [])
        out.setdefault("window", "mid")
        traits = {key: float((out.get("traits") or {}).get(key, 0.0) or 0.0) for key in TRAIT_KEYS}
        out["traits"] = traits
        return out
    fallback = _generic_role_fallback(str(role_hint or ""))
    fallback["name"] = raw_name
    return fallback


def champion_style_summary(entry: Mapping[str, Any] | None, limit: int = 2) -> str:
    if not isinstance(entry, Mapping):
        return "-"
    styles = [str(value).strip() for value in list(entry.get("styles") or []) if str(value).strip()]
    if styles:
        return " / ".join(styles[: max(1, int(limit))])
    traits = entry.get("traits") or {}
    ordered = sorted(
        [(name, float(traits.get(name, 0.0) or 0.0)) for name in STYLE_TRAITS],
        key=lambda item: item[1],
        reverse=True,
    )
    top = [name for name, score in ordered if score > 0][: max(1, int(limit))]
    return " / ".join(top) if top else "-"


__all__ = [
    "STYLE_TRAITS",
    "TRAIT_KEYS",
    "champion_style_summary",
    "default_registry_path",
    "get_champion_knowledge",
    "load_champion_knowledge_registry",
]
