from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from mlcore.identity_registry_core import normalize_identity_key
from mlcore.league_context import league_lookup_keys
from mlcore.league_context import league_macro_key
from mlcore.tiers import canonical_league, league_strength_band, tier_for_league
from mlcore.utils import clip_safe as _clip, safe_float as _f


def empty_league_transfer_features() -> dict[str, Any]:
    """Retorna um dicionário vazio com a estrutura padrão de features de transferência entre ligas."""
    return {
        "version": "league_transfer_features_v1",
        "generated_at": "",
        "meta": {},
        "by_league": {},
        "by_team": {},
        "by_player": {},
        "by_lineup": {},
        "by_pair": {},
        "team_alias_index": {},
        "player_alias_index": {},
    }


def normalize_league_transfer_features(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normaliza e valida um payload de features de transferência entre ligas, preenchendo valores ausentes."""
    out = empty_league_transfer_features()
    if not isinstance(payload, Mapping):
        return out
    out["version"] = str(payload.get("version") or out["version"])
    out["generated_at"] = str(payload.get("generated_at") or "")
    out["meta"] = dict(payload.get("meta") or {}) if isinstance(payload.get("meta"), Mapping) else {}
    out["by_league"] = dict(payload.get("by_league") or {}) if isinstance(payload.get("by_league"), Mapping) else {}
    out["by_team"] = dict(payload.get("by_team") or {}) if isinstance(payload.get("by_team"), Mapping) else {}
    out["by_player"] = dict(payload.get("by_player") or {}) if isinstance(payload.get("by_player"), Mapping) else {}
    out["by_lineup"] = dict(payload.get("by_lineup") or {}) if isinstance(payload.get("by_lineup"), Mapping) else {}
    out["by_pair"] = dict(payload.get("by_pair") or {}) if isinstance(payload.get("by_pair"), Mapping) else {}
    raw_alias = dict(payload.get("team_alias_index") or {}) if isinstance(payload.get("team_alias_index"), Mapping) else {}
    alias_index: dict[str, str] = {}
    for key, value in raw_alias.items():
        norm = normalize_identity_key(str(key or ""))
        ent = str(value or "").strip()
        if norm and ent:
            alias_index[norm] = ent
    out["team_alias_index"] = alias_index
    raw_player_alias = dict(payload.get("player_alias_index") or {}) if isinstance(payload.get("player_alias_index"), Mapping) else {}
    player_alias_index: dict[str, str] = {}
    for key, value in raw_player_alias.items():
        norm = normalize_identity_key(str(key or ""))
        ent = str(value or "").strip()
        if norm and ent:
            player_alias_index[norm] = ent
    out["player_alias_index"] = player_alias_index
    return out


def load_league_transfer_features(path_like: str | Path) -> dict[str, Any]:
    """Carrega e normaliza o arquivo JSON de features de transferência entre ligas."""
    path = Path(path_like)
    if not path.exists():
        return empty_league_transfer_features()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return empty_league_transfer_features()
    return normalize_league_transfer_features(payload if isinstance(payload, Mapping) else None)


def lookup_league_transfer_features(payload: Mapping[str, Any] | None, league: Any) -> dict[str, Any]:
    """Busca as features de transferência de uma liga pelo nome, tentando chaves de lookup alternativas."""
    data = payload if isinstance(payload, Mapping) else {}
    table = data.get("by_league") if isinstance(data.get("by_league"), Mapping) else {}
    for key in league_lookup_keys(league):
        row = table.get(key)
        if isinstance(row, Mapping):
            return dict(row)
    return {}


def lookup_pair_transfer_features(
    payload: Mapping[str, Any] | None,
    *,
    league_a: Any,
    league_b: Any,
) -> dict[str, Any]:
    """Busca features de transferência para o par de ligas usando chaves de tier, banda e macro-região."""
    data = payload if isinstance(payload, Mapping) else {}
    by_pair = data.get("by_pair") if isinstance(data.get("by_pair"), Mapping) else {}
    if not by_pair:
        return {}
    la = str(canonical_league(str(league_a or "")) or str(league_a or "")).strip()
    lb = str(canonical_league(str(league_b or "")) or str(league_b or "")).strip()
    if not la or not lb or la == lb:
        return {}
    tier_a = str(tier_for_league(la) or "OTHER")
    tier_b = str(tier_for_league(lb) or "OTHER")
    band_a = str(league_strength_band(la) or "local")
    band_b = str(league_strength_band(lb) or "local")
    macro_a = str(league_macro_key(la, unknown="OTHER"))
    macro_b = str(league_macro_key(lb, unknown="OTHER"))
    class_a = "major" if band_a.lower() in {"elite", "major", "upper"} else "minor"
    class_b = "major" if band_b.lower() in {"elite", "major", "upper"} else "minor"
    keys = [
        ("league_pair", f"league_pair:{la}->{lb}"),
        ("tier_pair", f"tier_pair:{tier_a}->{tier_b}"),
        ("strength_pair", f"strength_pair:{band_a}->{band_b}"),
        ("macro_pair", f"macro_pair:{macro_a}->{macro_b}"),
        ("class_pair", f"class_pair:{class_a}->{class_b}"),
    ]
    for scope, key in keys:
        row = by_pair.get(key)
        if isinstance(row, Mapping):
            out = dict(row)
            out.setdefault("lookup_key", key)
            out.setdefault("lookup_scope", scope)
            return out
    return {}


def _team_candidate_keys(team_name: str, aliases: Iterable[str] | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in [team_name, *(list(aliases or []))]:
        text = str(value or "").strip()
        if not text:
            continue
        norm = normalize_identity_key(text)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def lookup_team_transfer_features(
    payload: Mapping[str, Any] | None,
    team_name: Any,
    *,
    aliases: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Busca features de transferência de um time pelo nome ou aliases via índice de entidades."""
    data = payload if isinstance(payload, Mapping) else {}
    by_team = data.get("by_team") if isinstance(data.get("by_team"), Mapping) else {}
    alias_index = data.get("team_alias_index") if isinstance(data.get("team_alias_index"), Mapping) else {}
    for norm in _team_candidate_keys(str(team_name or ""), aliases):
        entity_key = str(alias_index.get(norm) or norm)
        row = by_team.get(entity_key)
        if isinstance(row, Mapping):
            out = dict(row)
            out.setdefault("entity_key", entity_key)
            out.setdefault("lookup_key", norm)
            return out
    return {}


def lookup_player_transfer_features(
    payload: Mapping[str, Any] | None,
    player_name: Any,
    *,
    aliases: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Busca features de transferência de um jogador pelo nome ou aliases via índice de entidades."""
    data = payload if isinstance(payload, Mapping) else {}
    by_player = data.get("by_player") if isinstance(data.get("by_player"), Mapping) else {}
    alias_index = data.get("player_alias_index") if isinstance(data.get("player_alias_index"), Mapping) else {}
    for norm in _team_candidate_keys(str(player_name or ""), aliases):
        entity_key = str(alias_index.get(norm) or norm)
        row = by_player.get(entity_key)
        if isinstance(row, Mapping):
            out = dict(row)
            out.setdefault("entity_key", entity_key)
            out.setdefault("lookup_key", norm)
            return out
    return {}


def lookup_lineup_transfer_features(
    payload: Mapping[str, Any] | None,
    team_name: Any,
    *,
    aliases: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Busca features de transferência do lineup de um time a partir da entidade do time."""
    data = payload if isinstance(payload, Mapping) else {}
    by_lineup = data.get("by_lineup") if isinstance(data.get("by_lineup"), Mapping) else {}
    team_row = lookup_team_transfer_features(data, team_name, aliases=aliases)
    entity_key = str(team_row.get("entity_key") or "")
    if not entity_key:
        return {}
    row = by_lineup.get(entity_key)
    return dict(row) if isinstance(row, Mapping) else {}


def build_interleague_transfer_adjustment(
    *,
    payload: Mapping[str, Any] | None,
    interregional: bool,
    league_a: Any,
    league_b: Any,
    team_a: Any = None,
    team_b: Any = None,
    team_aliases_a: Iterable[str] | None = None,
    team_aliases_b: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Calcula ajustes de desempenho inter-ligas com edge de times, ligas e par, bônus direcional e shrink."""
    out: dict[str, Any] = {
        "enabled": bool(interregional),
        "applied": False,
        "reason": "",
        "league_a": str(league_a or ""),
        "league_b": str(league_b or ""),
        "team_a": str(team_a or ""),
        "team_b": str(team_b or ""),
        "team_entity_a": "",
        "team_entity_b": "",
        "league_reliability_a": None,
        "league_reliability_b": None,
        "league_depth_a": None,
        "league_depth_b": None,
        "league_translation_a": None,
        "league_translation_b": None,
        "league_inflation_a": None,
        "league_inflation_b": None,
        "team_domestic_power_a": None,
        "team_domestic_power_b": None,
        "team_transfer_power_a": None,
        "team_transfer_power_b": None,
        "team_sample_games_a": None,
        "team_sample_games_b": None,
        "team_confidence": 0.0,
        "league_confidence": 0.0,
        "edge_team": 0.0,
        "edge_league": 0.0,
        "edge_pair": 0.0,
        "edge_total": 0.0,
        "dir_bonus": 0.0,
        "dir_cap_bonus": 0.0,
        "shrink_adj": 0.0,
        "pair_lookup_scope": "",
        "pair_lookup_key": "",
        "pair_confidence": 0.0,
        "pair_portability_edge": 0.0,
        "pair_result_edge": 0.0,
        "pair_transfer_edge": 0.0,
    }
    if not bool(interregional):
        out["reason"] = "same_league"
        return out
    data = payload if isinstance(payload, Mapping) else {}
    if not data:
        out["reason"] = "missing_payload"
        return out

    league_row_a = lookup_league_transfer_features(data, league_a)
    league_row_b = lookup_league_transfer_features(data, league_b)
    team_row_a = lookup_team_transfer_features(data, team_a, aliases=team_aliases_a)
    team_row_b = lookup_team_transfer_features(data, team_b, aliases=team_aliases_b)
    pair_row = lookup_pair_transfer_features(data, league_a=league_a, league_b=league_b)
    out["team_entity_a"] = str(team_row_a.get("entity_key") or "")
    out["team_entity_b"] = str(team_row_b.get("entity_key") or "")

    rel_a = _clip(league_row_a.get("league_reliability"), 0.0, 1.0) if league_row_a else 0.50
    rel_b = _clip(league_row_b.get("league_reliability"), 0.0, 1.0) if league_row_b else 0.50
    depth_a = _clip(league_row_a.get("league_depth"), 0.0, 1.0) if league_row_a else 0.50
    depth_b = _clip(league_row_b.get("league_depth"), 0.0, 1.0) if league_row_b else 0.50
    trans_a = _clip(league_row_a.get("league_translation_score"), 0.0, 1.0) if league_row_a else 0.50
    trans_b = _clip(league_row_b.get("league_translation_score"), 0.0, 1.0) if league_row_b else 0.50
    infl_a = _clip(league_row_a.get("league_domestic_inflation"), 0.0, 1.0) if league_row_a else 0.50
    infl_b = _clip(league_row_b.get("league_domestic_inflation"), 0.0, 1.0) if league_row_b else 0.50
    out["league_reliability_a"] = rel_a
    out["league_reliability_b"] = rel_b
    out["league_depth_a"] = depth_a
    out["league_depth_b"] = depth_b
    out["league_translation_a"] = trans_a
    out["league_translation_b"] = trans_b
    out["league_inflation_a"] = infl_a
    out["league_inflation_b"] = infl_b

    team_tp_a = _clip(team_row_a.get("team_transfer_power"), 0.0, 1.0) if team_row_a else 0.50
    team_tp_b = _clip(team_row_b.get("team_transfer_power"), 0.0, 1.0) if team_row_b else 0.50
    team_dp_a = _clip(team_row_a.get("team_domestic_power"), 0.0, 1.0) if team_row_a else 0.50
    team_dp_b = _clip(team_row_b.get("team_domestic_power"), 0.0, 1.0) if team_row_b else 0.50
    team_games_a = float(max(0.0, _f(team_row_a.get("sample_games"), 0.0))) if team_row_a else 0.0
    team_games_b = float(max(0.0, _f(team_row_b.get("sample_games"), 0.0))) if team_row_b else 0.0
    out["team_domestic_power_a"] = team_dp_a
    out["team_domestic_power_b"] = team_dp_b
    out["team_transfer_power_a"] = team_tp_a
    out["team_transfer_power_b"] = team_tp_b
    out["team_sample_games_a"] = team_games_a
    out["team_sample_games_b"] = team_games_b

    team_present = bool(team_row_a) and bool(team_row_b)
    team_conf = min(1.0, min(team_games_a, team_games_b) / 20.0) if team_present else 0.0
    pair_conf = _clip((pair_row.get("pair_confidence") if pair_row else 0.0), 0.0, 1.0)
    pair_edge = float(max(-1.0, min(1.0, _f(pair_row.get("pair_portability_edge"), 0.0)))) if pair_row else 0.0
    pair_result_edge = float(max(-1.0, min(1.0, _f(pair_row.get("pair_result_edge"), 0.0)))) if pair_row else 0.0
    pair_transfer_edge = float(max(-1.0, min(1.0, _f(pair_row.get("pair_transfer_edge"), 0.0)))) if pair_row else 0.0
    league_conf = float(max(0.0, min(1.0, 0.45 * ((rel_a + rel_b) / 2.0) + 0.30 * ((trans_a + trans_b) / 2.0) + 0.25 * pair_conf)))
    edge_team = float(team_tp_a - team_tp_b) if team_present else 0.0
    edge_league = float(
        (0.40 * (trans_a - trans_b))
        + (0.30 * (rel_a - rel_b))
        + (0.20 * (depth_a - depth_b))
        + (0.10 * (infl_b - infl_a))
    )
    pair_mix = float(max(0.0, min(0.55, 0.18 + (0.37 * pair_conf)))) if pair_row else 0.0
    edge_pair = float(pair_edge)
    edge_league = float(((1.0 - pair_mix) * edge_league) + (pair_mix * edge_pair))
    edge_total = float((0.66 * edge_team) + (0.34 * edge_league) if team_present else edge_league)
    confidence = float(max(0.0, min(1.0, 0.22 + (0.43 * team_conf) + (0.35 * league_conf))))

    dir_bonus_raw = float(edge_total * (0.004 + (0.012 * confidence)))
    if not team_present:
        dir_bonus_raw *= 0.75
    if pair_row:
        dir_bonus_raw += float(pair_edge * pair_conf * 0.004)
    dir_bonus = float(max(-0.018, min(0.018, dir_bonus_raw)))

    rel_mean = (rel_a + rel_b) / 2.0
    trans_mean = (trans_a + trans_b) / 2.0
    infl_mean = (infl_a + infl_b) / 2.0
    shrink_adj_raw = (
        (0.030 * (infl_mean - 0.50))
        - (0.024 * (rel_mean - 0.50))
        - (0.020 * (trans_mean - 0.50))
        - (0.016 * (team_conf - 0.50))
    )
    if pair_row:
        shrink_adj_raw -= float(0.012 * pair_conf)
        shrink_adj_raw -= float(0.010 * max(0.0, abs(pair_edge) - 0.08))
    if abs(edge_total) < 0.02:
        shrink_adj_raw *= 0.75
    shrink_adj = float(max(-0.028, min(0.030, shrink_adj_raw)))
    dir_cap_bonus = float(
        max(
            -0.004,
            min(
                0.006,
                (0.004 * (rel_mean - 0.50))
                + (0.003 * (trans_mean - 0.50))
                - (0.003 * (infl_mean - 0.50)),
            ),
        )
    )
    if pair_row:
        dir_cap_bonus = float(
            max(
                -0.004,
                min(0.007, dir_cap_bonus + (0.0025 * pair_conf) + (0.0015 * max(0.0, abs(pair_edge)))),
            )
        )

    out["team_confidence"] = float(team_conf)
    out["league_confidence"] = float(league_conf)
    out["edge_team"] = float(edge_team)
    out["edge_league"] = float(edge_league)
    out["edge_pair"] = float(edge_pair)
    out["edge_total"] = float(edge_total)
    out["dir_bonus"] = float(dir_bonus)
    out["dir_cap_bonus"] = float(dir_cap_bonus)
    out["shrink_adj"] = float(shrink_adj)
    out["pair_lookup_scope"] = str(pair_row.get("lookup_scope") or "") if pair_row else ""
    out["pair_lookup_key"] = str(pair_row.get("lookup_key") or "") if pair_row else ""
    out["pair_confidence"] = float(pair_conf)
    out["pair_portability_edge"] = float(pair_edge)
    out["pair_result_edge"] = float(pair_result_edge)
    out["pair_transfer_edge"] = float(pair_transfer_edge)
    out["reason"] = "ok" if (league_row_a or league_row_b or team_present) else "no_lookup"
    out["applied"] = bool(abs(dir_bonus) > 1e-9 or abs(shrink_adj) > 1e-9 or abs(dir_cap_bonus) > 1e-9)
    return out


def _norm_pos(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"top", "t"}:
        return "Top"
    if text in {"jng", "jg", "jungle", "j"}:
        return "Jungle"
    if text in {"mid", "m"}:
        return "Mid"
    if text in {"bot", "adc", "bottom", "b"}:
        return "Bot"
    if text in {"support", "sup", "s"}:
        return "Support"
    return ""


def build_lineup_transfer_summary(
    *,
    payload: Mapping[str, Any] | None,
    team_a: Any,
    team_b: Any,
    blue_lineup: Mapping[str, Any] | None,
    red_lineup: Mapping[str, Any] | None,
    team_aliases_a: Iterable[str] | None = None,
    team_aliases_b: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Calcula resumo de transferência dos lineups dos dois times com portabilidade, estabilidade e ajuste delta."""
    role_weights = {"Top": 0.18, "Jungle": 0.22, "Mid": 0.24, "Bot": 0.22, "Support": 0.14}
    out: dict[str, Any] = {
        "enabled": bool(isinstance(payload, Mapping) and payload),
        "applied": False,
        "reason": "",
        "lineup_transfer_power_a": None,
        "lineup_transfer_power_b": None,
        "lineup_strength_a": None,
        "lineup_strength_b": None,
        "lineup_stability_a": None,
        "lineup_stability_b": None,
        "player_portability_avg_a": None,
        "player_portability_avg_b": None,
        "coverage_a": 0,
        "coverage_b": 0,
        "identity_successor_signal_a": None,
        "identity_successor_signal_b": None,
        "identity_match_ratio_a": None,
        "identity_match_ratio_b": None,
        "identity_relation_type_a": "",
        "identity_relation_type_b": "",
        "delta_adj": 0.0,
        "confidence": 0.0,
        "player_details_a": [],
        "player_details_b": [],
    }
    data = payload if isinstance(payload, Mapping) else {}
    if not data:
        out["reason"] = "missing_payload"
        return out

    def _norm_lineup(lineup: Mapping[str, Any] | None) -> dict[str, str]:
        res: dict[str, str] = {}
        for key, value in dict(lineup or {}).items():
            pos = _norm_pos(key)
            text = str(value or "").strip()
            if pos and text:
                res[pos] = text
        return res

    def _team_side(
        team_name: Any,
        lineup: Mapping[str, Any] | None,
        *,
        aliases: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        team_row = lookup_team_transfer_features(data, team_name, aliases=aliases)
        lineup_row = lookup_lineup_transfer_features(data, team_name, aliases=aliases)
        line = _norm_lineup(lineup)
        dom_sum = 0.0
        port_sum = 0.0
        weight_sum = 0.0
        sample_scores: list[float] = []
        details: list[dict[str, Any]] = []
        coverage = 0
        canonical_players = dict(lineup_row.get("player_entities_by_pos") or {}) if isinstance(lineup_row, Mapping) else {}
        relation_type = str(team_row.get("relation_type") or "alias")
        identity_conf_text = str(team_row.get("identity_confidence") or "medium").strip().lower()
        identity_conf = {"low": 0.35, "medium": 0.60, "high": 0.82}.get(identity_conf_text, 0.60)
        successor_expected = {
            normalize_identity_key(str(v or ""))
            for v in [*(list(team_row.get("kept_players") or [])), *(list(team_row.get("added_players") or []))]
            if str(v or "").strip()
        }
        for pos, player in line.items():
            weight = float(role_weights.get(pos, 0.20))
            row = lookup_player_transfer_features(data, player)
            if not row:
                details.append({"pos": pos, "player": str(player), "used": False})
                continue
            coverage += 1
            dom = _clip(row.get("player_domestic_rating"), 0.0, 1.0)
            port = _clip(row.get("player_portability"), 0.0, 1.0)
            sample = float(max(0.0, _f(row.get("sample_games"), 0.0)))
            dom_sum += weight * dom
            port_sum += weight * port
            weight_sum += weight
            sample_scores.append(min(1.0, sample / 18.0))
            details.append(
                {
                    "pos": pos,
                    "player": str(player),
                    "entity_key": str(row.get("entity_key") or ""),
                    "player_domestic_rating": float(dom),
                    "player_portability": float(port),
                    "sample_games": float(sample),
                    "used": True,
                    "matches_canonical_lineup": bool(canonical_players and str(canonical_players.get(pos) or "") == str(row.get("entity_key") or "")),
                }
            )
        if weight_sum <= 0.0:
            return {
                "team_entity": str(team_row.get("entity_key") or ""),
                "coverage": int(coverage),
                "lineup_strength": 0.50,
                "player_portability_avg": 0.50,
                "lineup_stability": 0.50,
                "lineup_transfer_power": 0.50,
                "sample_conf": 0.0,
                "details": details,
            }
        dom_avg = float(dom_sum / weight_sum)
        port_avg = float(port_sum / weight_sum)
        canonical_match_count = sum(1 for row in details if bool(row.get("matches_canonical_lineup")))
        coverage_ratio = float(coverage / 5.0)
        stability_base = float(_clip(lineup_row.get("lineup_stability"), 0.0, 1.0)) if lineup_row else 0.50
        synergy_base = float(_clip(lineup_row.get("lineup_synergy"), 0.0, 1.0)) if lineup_row else 0.50
        actual_player_keys = [normalize_identity_key(str(row.get("player") or "")) for row in details if bool(row.get("used"))]
        actual_player_keys = [v for v in actual_player_keys if v]
        identity_match_ratio = 0.0
        identity_successor_signal = 0.50
        if successor_expected:
            matched = sum(1 for key in actual_player_keys if key in successor_expected)
            denom = max(1, min(len(successor_expected), max(1, len(actual_player_keys))))
            identity_match_ratio = float(matched / denom)
            if relation_type in {"sporting_successor", "roster_successor", "rebrand"}:
                identity_successor_signal = _clip(
                    (0.55 * identity_match_ratio) + (0.25 * identity_conf) + (0.20 * coverage_ratio),
                    0.0,
                    1.0,
                )
            else:
                identity_successor_signal = _clip((0.50 * identity_match_ratio) + (0.50 * identity_conf), 0.0, 1.0)
        if canonical_players:
            stability = _clip((0.65 * (canonical_match_count / max(1, len(canonical_players)))) + (0.35 * stability_base), 0.0, 1.0)
        else:
            stability = _clip((0.60 * coverage_ratio) + (0.40 * stability_base), 0.0, 1.0)
        if successor_expected:
            stability = _clip((0.78 * stability) + (0.22 * identity_successor_signal), 0.0, 1.0)
        sample_conf = float(sum(sample_scores) / len(sample_scores)) if sample_scores else 0.0
        lineup_transfer_power = _clip(port_avg * (0.68 + (0.16 * stability) + (0.16 * synergy_base)), 0.0, 1.0)
        if successor_expected:
            lineup_transfer_power = _clip((0.88 * lineup_transfer_power) + (0.12 * identity_successor_signal), 0.0, 1.0)
        return {
            "team_entity": str(team_row.get("entity_key") or ""),
            "coverage": int(coverage),
            "lineup_strength": float(dom_avg),
            "player_portability_avg": float(port_avg),
            "lineup_stability": float(stability),
            "lineup_transfer_power": float(lineup_transfer_power),
            "sample_conf": float(sample_conf),
            "identity_successor_signal": float(identity_successor_signal),
            "identity_match_ratio": float(identity_match_ratio),
            "identity_relation_type": relation_type,
            "details": details,
        }

    side_a = _team_side(team_a, blue_lineup, aliases=team_aliases_a)
    side_b = _team_side(team_b, red_lineup, aliases=team_aliases_b)
    conf = float(_clip(0.50 * min(side_a["sample_conf"], side_b["sample_conf"]) + 0.50 * min(side_a["coverage"], side_b["coverage"]) / 5.0, 0.0, 1.0))
    edge = float(
        (0.75 * (float(side_a["lineup_transfer_power"]) - float(side_b["lineup_transfer_power"])))
        + (0.25 * (float(side_a["lineup_strength"]) - float(side_b["lineup_strength"])))
    )
    delta_adj = float(max(-0.055, min(0.055, edge * (0.030 + (0.045 * conf)))))
    out.update(
        {
            "applied": bool(abs(delta_adj) > 1e-9),
            "reason": "ok",
            "lineup_transfer_power_a": float(side_a["lineup_transfer_power"]),
            "lineup_transfer_power_b": float(side_b["lineup_transfer_power"]),
            "lineup_strength_a": float(side_a["lineup_strength"]),
            "lineup_strength_b": float(side_b["lineup_strength"]),
            "lineup_stability_a": float(side_a["lineup_stability"]),
            "lineup_stability_b": float(side_b["lineup_stability"]),
            "player_portability_avg_a": float(side_a["player_portability_avg"]),
            "player_portability_avg_b": float(side_b["player_portability_avg"]),
            "coverage_a": int(side_a["coverage"]),
            "coverage_b": int(side_b["coverage"]),
            "identity_successor_signal_a": float(side_a.get("identity_successor_signal", 0.50)),
            "identity_successor_signal_b": float(side_b.get("identity_successor_signal", 0.50)),
            "identity_match_ratio_a": float(side_a.get("identity_match_ratio", 0.0)),
            "identity_match_ratio_b": float(side_b.get("identity_match_ratio", 0.0)),
            "identity_relation_type_a": str(side_a.get("identity_relation_type") or ""),
            "identity_relation_type_b": str(side_b.get("identity_relation_type") or ""),
            "delta_adj": float(delta_adj),
            "confidence": float(conf),
            "player_details_a": list(side_a["details"]),
            "player_details_b": list(side_b["details"]),
        }
    )
    return out
