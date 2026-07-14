from __future__ import annotations

from typing import Any, Mapping

from app_core.champion_knowledge_core import get_champion_knowledge
from mlcore.utils import safe_float_0 as _safe_float



def _team_from_delta(delta: float, *, team_a: str, team_b: str, threshold: float = 0.75) -> str:
    if delta >= threshold:
        return str(team_a)
    if delta <= -threshold:
        return str(team_b)
    return "Equilibrado"


def _confidence_label(sample_score: float) -> str:
    if sample_score >= 1.75:
        return "Alta"
    if sample_score >= 0.9:
        return "Media"
    return "Baixa"


def _lane_pressure_score(traits: Mapping[str, Any]) -> float:
    return (
        _safe_float(traits.get("lane_priority"))
        + 0.70 * _safe_float(traits.get("poke"))
        + 0.45 * _safe_float(traits.get("range"))
        + 0.35 * _safe_float(traits.get("waveclear"))
    )


def _all_in_score(traits: Mapping[str, Any]) -> float:
    return (
        _safe_float(traits.get("all_in"))
        + 0.55 * _safe_float(traits.get("engage"))
        + 0.45 * _safe_float(traits.get("dive"))
        + 0.25 * _safe_float(traits.get("mobility"))
    )


def _fight_setup_score(traits: Mapping[str, Any]) -> float:
    return (
        _safe_float(traits.get("teamfight"))
        + 0.55 * _safe_float(traits.get("frontline"))
        + 0.45 * _safe_float(traits.get("engage"))
        + 0.35 * _safe_float(traits.get("peel"))
        + 0.20 * _safe_float(traits.get("objective_control"))
    )


def _scale_score(traits: Mapping[str, Any]) -> float:
    return (
        _safe_float(traits.get("late"))
        + 0.55 * _safe_float(traits.get("range"))
        + 0.45 * _safe_float(traits.get("waveclear"))
        + 0.35 * _safe_float(traits.get("siege"))
        + 0.25 * _safe_float(traits.get("teamfight"))
    )


def _side_score(traits: Mapping[str, Any]) -> float:
    return (
        _safe_float(traits.get("splitpush"))
        + 0.35 * _safe_float(traits.get("mobility"))
        + 0.30 * _safe_float(traits.get("waveclear"))
        + 0.20 * _safe_float(traits.get("lane_priority"))
    )


def _pick_score(traits: Mapping[str, Any]) -> float:
    return (
        _safe_float(traits.get("pick"))
        + 0.45 * _safe_float(traits.get("mobility"))
        + 0.25 * _safe_float(traits.get("poke"))
        + 0.20 * _safe_float(traits.get("engage"))
    )


def _supportive_score(traits: Mapping[str, Any]) -> float:
    return _safe_float(traits.get("peel")) + 0.45 * _safe_float(traits.get("teamfight")) + 0.30 * _safe_float(traits.get("frontline"))


def _lane_plan_phrase(role: str, champ_name: str, traits: Mapping[str, Any], *, window: str) -> str:
    role_key = str(role or "").strip().lower()
    champ = str(champ_name or "").strip() or "Esse pick"
    pressure = _lane_pressure_score(traits)
    all_in = _all_in_score(traits)
    fight = _fight_setup_score(traits)
    scale = _scale_score(traits)
    side = _side_score(traits)
    pick = _pick_score(traits)
    support = _supportive_score(traits)
    objective = _safe_float(traits.get("objective_control"))
    tempo = _safe_float(traits.get("early")) + 0.35 * _safe_float(traits.get("mobility"))

    if window == "early":
        if role_key == "top":
            if pressure >= all_in + 0.25 and pressure >= 1.35:
                return f"{champ} controla melhor wave e pressiona cedo"
            if all_in >= 1.25:
                return f"{champ} tem janela real de troca curta/all-in cedo"
            return f"{champ} tende a absorver a lane no early"
        if role_key == "jng":
            if all_in + pick >= objective + 0.35 and all_in + pick >= 1.65:
                return f"{champ} entra melhor nas primeiras jogadas e ganks"
            if objective + tempo >= 1.55:
                return f"{champ} estabiliza melhor cover e objetivo cedo"
            return f"{champ} joga cedo sem muita margem de erro"
        if role_key == "mid":
            if pressure >= pick + 0.25 and pressure >= 1.55:
                return f"{champ} empurra e joga primeiro no mapa"
            if pick >= 1.45:
                return f"{champ} ameaca pick e troca cedo"
            return f"{champ} segura wave e responde o mapa cedo"
        if role_key == "bot":
            if pressure >= all_in + 0.25 and pressure >= 1.75:
                return f"{champ} pressiona wave e placa cedo"
            if all_in >= 1.35:
                return f"{champ} tem janela de all-in cedo na bot lane"
            return f"{champ} quer segurar a lane e escalar"
        if role_key == "sup":
            if all_in + pick >= support + 0.20 and all_in + pick >= 1.55:
                return f"{champ} acha engage e roam cedo"
            if support >= 1.35:
                return f"{champ} protege melhor a lane cedo"
            return f"{champ} joga a lane de forma mais reativa cedo"
        if pressure >= all_in and pressure >= 1.45:
            return f"{champ} pressiona cedo"
        if all_in >= 1.25:
            return f"{champ} tem janela cedo"
        return f"{champ} joga um early equilibrado"

    if role_key == "top":
        if side >= fight + 0.20 and side >= 1.15:
            return f"{champ} cresce mais na side e no mapa aberto"
        if fight + scale >= 2.15:
            return f"{champ} entrega mais valor no 5v5 depois"
        return f"{champ} mantem utilidade quando a lane perde peso"
    if role_key == "jng":
        if fight >= pick + 0.20 and fight >= 1.55:
            return f"{champ} oferece engage e setup mais estaveis nas fights"
        if pick >= 1.45:
            return f"{champ} continua punindo alvo isolado depois"
        return f"{champ} mantem utilidade de mapa quando o jogo estabiliza"
    if role_key == "mid":
        if scale + fight >= pick + 0.25 and scale + fight >= 2.00:
            return f"{champ} controla melhor teamfight e objetivo depois"
        if pick >= 1.45:
            return f"{champ} continua criando acesso e pick no mid/side"
        return f"{champ} segura wave e mantem utilidade depois"
    if role_key == "bot":
        if scale + fight >= 2.20:
            return f"{champ} vira condicao forte de teamfight e objetivo"
        if pressure + objective >= 1.85:
            return f"{champ} continua ajudando a converter prioridade em estrutura"
        return f"{champ} mantem dano mais estavel depois"
    if role_key == "sup":
        if support >= all_in + 0.10 and support >= 1.55:
            return f"{champ} segura melhor o carry nas fights"
        if all_in + pick >= 1.55:
            return f"{champ} continua abrindo luta e pick depois"
        return f"{champ} mantem utilidade de engage/peel depois"
    if scale + fight >= 1.95:
        return f"{champ} entrega mais valor depois"
    return f"{champ} mantem utilidade quando a lane perde peso"


def _team_lane_plan(role: str, champ_name: str, traits: Mapping[str, Any], *, early_team: str, late_team: str, team_name: str) -> str:
    early_phrase = _lane_plan_phrase(role, champ_name, traits, window="early")
    late_phrase = _lane_plan_phrase(role, champ_name, traits, window="late")
    if str(early_team) == str(team_name) and str(late_team) == str(team_name):
        return f"{early_phrase}; {late_phrase}."
    if str(early_team) == str(team_name):
        return f"{early_phrase}."
    if str(late_team) == str(team_name):
        return f"{late_phrase}."
    return f"{early_phrase if role == 'jng' else late_phrase}."


def evaluate_lane_matchup(
    *,
    registry: Mapping[str, Any] | None,
    role: str,
    champ_a: str,
    champ_b: str,
    team_a: str,
    team_b: str,
    empirical_edge_pp: Any = None,
    h2h_games: Any = None,
    player_games_a: Any = None,
    player_games_b: Any = None,
) -> dict[str, Any]:
    role_key = str(role or "").strip().lower()
    info_a = get_champion_knowledge(registry, str(champ_a), role_hint=role_key)
    info_b = get_champion_knowledge(registry, str(champ_b), role_hint=role_key)
    traits_a = info_a.get("traits") or {}
    traits_b = info_b.get("traits") or {}

    early_delta = (
        (_safe_float(traits_a.get("early")) - _safe_float(traits_b.get("early")))
        + 0.60 * (_safe_float(traits_a.get("lane_priority")) - _safe_float(traits_b.get("lane_priority")))
        + 0.45 * (_safe_float(traits_a.get("all_in")) - _safe_float(traits_b.get("all_in")))
        + 0.35 * (_safe_float(traits_a.get("poke")) - _safe_float(traits_b.get("poke")))
        + 0.25 * (_safe_float(traits_a.get("range")) - _safe_float(traits_b.get("range")))
        + 0.20 * (_safe_float(traits_a.get("mobility")) - _safe_float(traits_b.get("mobility")))
    )
    late_delta = (
        (_safe_float(traits_a.get("late")) - _safe_float(traits_b.get("late")))
        + 0.55 * (_safe_float(traits_a.get("teamfight")) - _safe_float(traits_b.get("teamfight")))
        + 0.30 * (_safe_float(traits_a.get("frontline")) - _safe_float(traits_b.get("frontline")))
        + 0.20 * (_safe_float(traits_a.get("siege")) - _safe_float(traits_b.get("siege")))
        + 0.25 * (_safe_float(traits_a.get("splitpush")) - _safe_float(traits_b.get("splitpush")))
    )
    empirical_delta = 0.10 * _safe_float(empirical_edge_pp)
    combined_delta = (0.55 * early_delta) + (0.20 * late_delta) + empirical_delta

    early_team = _team_from_delta(early_delta, team_a=team_a, team_b=team_b, threshold=0.85)
    late_team = _team_from_delta(late_delta, team_a=team_a, team_b=team_b, threshold=0.85)
    lane_team = _team_from_delta(combined_delta, team_a=team_a, team_b=team_b, threshold=0.60)

    games_a = _safe_float(player_games_a)
    games_b = _safe_float(player_games_b)
    sample_score = min(games_a, 8.0) / 8.0 + min(games_b, 8.0) / 8.0 + min(_safe_float(h2h_games), 6.0) / 6.0
    confidence = _confidence_label(sample_score)

    early_phrase_a = _lane_plan_phrase(role_key, str(champ_a), traits_a, window="early")
    early_phrase_b = _lane_plan_phrase(role_key, str(champ_b), traits_b, window="early")
    late_phrase_a = _lane_plan_phrase(role_key, str(champ_a), traits_a, window="late")
    late_phrase_b = _lane_plan_phrase(role_key, str(champ_b), traits_b, window="late")
    plan_a = _team_lane_plan(role_key, str(champ_a), traits_a, early_team=early_team, late_team=late_team, team_name=str(team_a))
    plan_b = _team_lane_plan(role_key, str(champ_b), traits_b, early_team=early_team, late_team=late_team, team_name=str(team_b))

    if early_team != "Equilibrado" and late_team != "Equilibrado" and early_team != late_team:
        early_phrase = early_phrase_a if early_team == str(team_a) else early_phrase_b
        late_phrase = late_phrase_a if late_team == str(team_a) else late_phrase_b
        summary = f"{early_phrase}; {late_phrase}."
    elif lane_team == "Equilibrado" and early_team == "Equilibrado" and late_team == "Equilibrado":
        summary = "Skill matchup ou lane sem edge estrutural claro."
    elif lane_team != "Equilibrado" and late_team == lane_team:
        summary = f"{late_phrase_a if lane_team == str(team_a) else late_phrase_b}."
    elif lane_team != "Equilibrado":
        summary = f"{early_phrase_a if lane_team == str(team_a) else early_phrase_b}."
    elif late_team != "Equilibrado":
        summary = f"{late_phrase_a if late_team == str(team_a) else late_phrase_b}."
    else:
        summary = "Lane equilibrada, com pouca separacao estrutural."

    counter_side = lane_team if lane_team != "Equilibrado" else (early_team if early_team != "Equilibrado" else late_team)
    return {
        "role": role_key,
        "early_team": early_team,
        "late_team": late_team,
        "lane_team": lane_team,
        "counter_team": counter_side if counter_side != "Equilibrado" else "-",
        "early_delta": early_delta,
        "late_delta": late_delta,
        "combined_delta": combined_delta,
        "confidence": confidence,
        "summary": summary,
        "team_a_plan": plan_a,
        "team_b_plan": plan_b,
    }


__all__ = ["evaluate_lane_matchup"]
