from __future__ import annotations

from typing import Any, Mapping

from app_core.champion_knowledge_core import STYLE_TRAITS, champion_style_summary, get_champion_knowledge
from mlcore.utils import safe_float_0 as _safe_float

COMP_STYLE_KEYS = tuple(dict.fromkeys([*list(STYLE_TRAITS), "objective_control"]))
COMP_MACRO_KEYS = (
    "tempo_pressure",
    "carry_scaling",
    "fight_reliability",
    "siege_pressure",
    "side_pressure",
    "pick_threat",
    "objective_setup",
    "anti_dive",
    "dive_threat",
)



def _winner_label(delta: float, *, team_a: str, team_b: str, threshold: float = 0.8) -> str:
    if delta >= threshold:
        return str(team_a)
    if delta <= -threshold:
        return str(team_b)
    return "Equilibrado"


def _top_styles(style_scores: Mapping[str, float], limit: int = 2) -> list[str]:
    ordered = sorted(style_scores.items(), key=lambda item: float(item[1]), reverse=True)
    return [name for name, score in ordered if float(score) > 0][: max(1, int(limit))]


def _style_score(style_scores: Mapping[str, Any], name: str) -> float:
    return _safe_float(style_scores.get(name))


def _team_plan_profile(
    *,
    team_name: str,
    own_macro: Mapping[str, Any],
    opp_macro: Mapping[str, Any],
    own_styles: Mapping[str, Any],
    opp_styles: Mapping[str, Any],
) -> dict[str, str]:
    tempo = _safe_float(own_macro.get("tempo_pressure"))
    carry = _safe_float(own_macro.get("carry_scaling"))
    fight = _safe_float(own_macro.get("fight_reliability"))
    siege = _safe_float(own_macro.get("siege_pressure"))
    side = _safe_float(own_macro.get("side_pressure"))
    pick = _safe_float(own_macro.get("pick_threat"))
    objective = _safe_float(own_macro.get("objective_setup"))
    anti_dive = _safe_float(own_macro.get("anti_dive"))
    dive = _safe_float(own_macro.get("dive_threat"))

    opp_carry = _safe_float(opp_macro.get("carry_scaling"))
    opp_fight = _safe_float(opp_macro.get("fight_reliability"))
    opp_pick = _safe_float(opp_macro.get("pick_threat"))
    opp_objective = _safe_float(opp_macro.get("objective_setup"))
    opp_anti_dive = _safe_float(opp_macro.get("anti_dive"))
    opp_dive = _safe_float(opp_macro.get("dive_threat"))

    teamfight_style = _style_score(own_styles, "teamfight")
    peel_style = _style_score(own_styles, "peel")
    pick_style = _style_score(own_styles, "pick")
    dive_style = _style_score(own_styles, "dive")
    split_style = _style_score(own_styles, "splitpush")
    siege_style = _style_score(own_styles, "siege")
    skirmish_style = _style_score(own_styles, "skirmish")

    front_to_back_score = carry + 0.75 * fight + 0.45 * anti_dive + 0.35 * teamfight_style + 0.25 * peel_style
    dive_score = dive + 0.65 * tempo + 0.45 * pick + 0.30 * dive_style
    pick_score = pick + 0.35 * tempo + 0.25 * pick_style
    side_score = side + 0.30 * split_style + 0.20 * tempo
    siege_score = siege + 0.35 * objective + 0.25 * siege_style
    skirmish_score = tempo + 0.35 * skirmish_style + 0.25 * dive

    identity = "Fight de objetivo"
    win_condition = f"{team_name} quer jogar a partir de objetivo e resetar a fight melhor."
    avoid_condition = f"{team_name} nao quer ceder setup livre de objetivo."

    if side_score >= max(front_to_back_score, dive_score, pick_score, siege_score, skirmish_score) and side_score >= 5.2:
        identity = "Side / mapa aberto"
        win_condition = f"{team_name} quer abrir side antes do objetivo para quebrar o mapa e obrigar resposta lateral."
        avoid_condition = f"{team_name} nao quer acelerar 5v5 reto antes de abrir a side."
    elif dive_score >= max(front_to_back_score, pick_score, siege_score, skirmish_score) and dive_score >= 5.4:
        identity = "Acesso no backline"
        if opp_anti_dive <= anti_dive + 0.4:
            win_condition = f"{team_name} ganha se achar acesso no backline antes da fight estabilizar."
        else:
            win_condition = f"{team_name} precisa sincronizar engage e follow-up para atravessar o peel adversario."
        avoid_condition = f"{team_name} nao pode entrar quebrado nem gastar engage no frontline errado."
    elif front_to_back_score >= max(pick_score, siege_score, skirmish_score) and front_to_back_score >= 5.8:
        identity = "Front-to-back"
        win_condition = f"{team_name} quer alongar a luta, proteger o carry e bater de frente para tras."
        avoid_condition = f"{team_name} nao quer dar pick cedo nem fight curta desorganizada."
    elif pick_score >= max(siege_score, skirmish_score) and pick_score >= 4.8:
        identity = "Pick antes da fight"
        win_condition = f"{team_name} quer tirar um alvo antes do objetivo e transformar a luta em 5v4."
        avoid_condition = f"{team_name} nao quer front-to-back longo sem vantagem de acesso."
    elif siege_score >= max(skirmish_score, 4.8):
        identity = "Wave + cerco"
        win_condition = f"{team_name} quer empilhar wave, bater estrutura e converter prioridade em objetivo."
        avoid_condition = f"{team_name} nao pode subir wave sem visao contra engage livre."
    elif skirmish_score >= 4.8:
        identity = "Skirmish cedo"
        win_condition = f"{team_name} quer acelerar cedo, forcar 2v2/3v3 e jogar a partir de lead curta."
        avoid_condition = f"{team_name} nao quer deixar o jogo virar 5v5 lento e limpo."

    if front_to_back_score >= 5.8 and opp_dive >= dive + 0.8 and anti_dive <= opp_dive:
        avoid_condition = f"{team_name} nao pode expor o backline para o acesso do adversario."
    elif side_score >= 5.0 and opp_objective >= objective + 0.8:
        avoid_condition = f"{team_name} nao quer perder tempo de side e entregar objetivo gratis."
    elif pick_score >= 4.8 and opp_fight >= fight + 1.0:
        avoid_condition = f"{team_name} nao quer luta longa e frontal sem tirar alvo antes."
    elif siege_score >= 4.8 and opp_dive >= dive + 0.6:
        avoid_condition = f"{team_name} nao pode bater torre sem cobertura de engage e visao."
    elif skirmish_score >= 4.8 and opp_carry >= carry + 1.0:
        avoid_condition = f"{team_name} nao quer deixar o jogo alongar sem vantagem concreta."
    elif objective >= 4.6 and opp_pick >= pick + 0.6:
        avoid_condition = f"{team_name} nao pode entrar cego no setup de objetivo."

    return {
        "identity": identity,
        "win_condition": win_condition,
        "avoid_condition": avoid_condition,
    }


def _macro_from_traits(traits: Mapping[str, Any] | None) -> dict[str, float]:
    traits = dict(traits or {})
    early = _safe_float(traits.get("early"))
    late = _safe_float(traits.get("late"))
    lane_priority = _safe_float(traits.get("lane_priority"))
    all_in = _safe_float(traits.get("all_in"))
    poke = _safe_float(traits.get("poke"))
    waveclear = _safe_float(traits.get("waveclear"))
    engage = _safe_float(traits.get("engage"))
    peel = _safe_float(traits.get("peel"))
    pick = _safe_float(traits.get("pick"))
    dive = _safe_float(traits.get("dive"))
    frontline = _safe_float(traits.get("frontline"))
    siege = _safe_float(traits.get("siege"))
    splitpush = _safe_float(traits.get("splitpush"))
    skirmish = _safe_float(traits.get("skirmish"))
    teamfight = _safe_float(traits.get("teamfight"))
    objective_control = _safe_float(traits.get("objective_control"))
    mobility = _safe_float(traits.get("mobility"))
    range_score = _safe_float(traits.get("range"))
    return {
        "tempo_pressure": early + 0.55 * lane_priority + 0.45 * skirmish + 0.35 * all_in,
        "carry_scaling": late + 0.55 * range_score + 0.45 * siege + 0.35 * waveclear + 0.20 * mobility,
        "fight_reliability": teamfight + 0.55 * frontline + 0.35 * peel + 0.20 * engage,
        "siege_pressure": siege + 0.55 * range_score + 0.45 * waveclear + 0.25 * poke,
        "side_pressure": splitpush + 0.25 * lane_priority + 0.15 * mobility + 0.10 * waveclear,
        "pick_threat": pick + 0.55 * mobility + 0.35 * range_score + 0.25 * poke,
        "objective_setup": objective_control + 0.35 * lane_priority + 0.25 * frontline + 0.20 * waveclear,
        "anti_dive": peel + 0.45 * frontline + 0.25 * range_score,
        "dive_threat": engage + 0.70 * dive + 0.45 * all_in + 0.25 * mobility,
    }


def build_comp_profile(
    *,
    registry: Mapping[str, Any] | None,
    picks: list[str],
    players: list[str] | None = None,
    role_order: list[str] | None = None,
) -> dict[str, Any]:
    roles = list(role_order or ["top", "jng", "mid", "bot", "sup"])
    entries: list[dict[str, Any]] = []
    window_scores = {"early": 0.0, "mid": 0.0, "late": 0.0}
    style_scores = {name: 0.0 for name in COMP_STYLE_KEYS}
    macro_scores = {name: 0.0 for name in COMP_MACRO_KEYS}
    for idx, champ_name in enumerate(list(picks or [])[: len(roles)]):
        role_hint = roles[idx] if idx < len(roles) else ""
        info = get_champion_knowledge(registry, str(champ_name), role_hint=role_hint)
        traits = info.get("traits") or {}
        window_scores["early"] += _safe_float(traits.get("early"))
        window_scores["mid"] += _safe_float(traits.get("mid"))
        window_scores["late"] += _safe_float(traits.get("late"))
        for trait_name in COMP_STYLE_KEYS:
            style_scores[trait_name] += _safe_float(traits.get(trait_name))
        for macro_name, macro_value in _macro_from_traits(traits).items():
            macro_scores[macro_name] += float(macro_value)
        entries.append(
            {
                "role": role_hint,
                "champion": str(champ_name),
                "player": str((players or [""] * len(roles))[idx] or "") if idx < len(roles) else "",
                "window": str(info.get("window") or "mid"),
                "styles": list(info.get("styles") or []),
            }
        )

    top_styles = _top_styles({name: style_scores.get(name, 0.0) for name in STYLE_TRAITS}, limit=3)
    if not top_styles:
        top_styles = ["teamfight"]
    primary_style = " / ".join(top_styles[:2])
    if window_scores["early"] >= window_scores["mid"] and window_scores["early"] >= window_scores["late"] + 0.5:
        window_label = "Early"
    elif window_scores["late"] >= window_scores["mid"] and window_scores["late"] >= window_scores["early"] + 0.5:
        window_label = "Late"
    elif abs(window_scores["mid"] - max(window_scores["early"], window_scores["late"])) <= 0.4:
        window_label = "Balanceado"
    else:
        window_label = "Mid"

    return {
        "entries": entries,
        "window_scores": window_scores,
        "style_scores": style_scores,
        "macro_scores": macro_scores,
        "primary_style": primary_style,
        "window_label": window_label,
        "style_summary": primary_style or champion_style_summary({}, limit=2),
    }


def compare_comp_profiles(
    *,
    comp_a: Mapping[str, Any],
    comp_b: Mapping[str, Any],
    team_a: str,
    team_b: str,
) -> dict[str, Any]:
    window_a = comp_a.get("window_scores") or {}
    window_b = comp_b.get("window_scores") or {}
    styles_a = comp_a.get("style_scores") or {}
    styles_b = comp_b.get("style_scores") or {}
    macro_a = comp_a.get("macro_scores") or {}
    macro_b = comp_b.get("macro_scores") or {}

    early_delta = _safe_float(macro_a.get("tempo_pressure")) - _safe_float(macro_b.get("tempo_pressure"))
    mid_delta = (
        _safe_float(window_a.get("mid"))
        + 0.35 * _safe_float(macro_a.get("pick_threat"))
        + 0.25 * _safe_float(macro_a.get("objective_setup"))
        - _safe_float(window_b.get("mid"))
        - 0.35 * _safe_float(macro_b.get("pick_threat"))
        - 0.25 * _safe_float(macro_b.get("objective_setup"))
    )
    late_delta = (
        _safe_float(macro_a.get("carry_scaling"))
        + 0.55 * _safe_float(macro_a.get("fight_reliability"))
        + 0.30 * _safe_float(macro_a.get("anti_dive"))
        - _safe_float(macro_b.get("carry_scaling"))
        - 0.55 * _safe_float(macro_b.get("fight_reliability"))
        - 0.30 * _safe_float(macro_b.get("anti_dive"))
    )
    engage_delta = _safe_float(macro_a.get("dive_threat")) - _safe_float(macro_b.get("dive_threat"))
    objective_delta = _safe_float(macro_a.get("objective_setup")) - _safe_float(macro_b.get("objective_setup"))
    towers_delta = (
        0.70 * _safe_float(macro_a.get("siege_pressure"))
        + 1.00 * _safe_float(macro_a.get("side_pressure"))
        + 0.35 * _safe_float(macro_a.get("objective_setup"))
        + 0.15 * _safe_float(macro_a.get("tempo_pressure"))
        - 0.70 * _safe_float(macro_b.get("siege_pressure"))
        - 1.00 * _safe_float(macro_b.get("side_pressure"))
        - 0.35 * _safe_float(macro_b.get("objective_setup"))
        - 0.15 * _safe_float(macro_b.get("tempo_pressure"))
    )
    kills_pressure_a = (
        0.65 * _safe_float(macro_a.get("dive_threat"))
        + 0.50 * _safe_float(macro_a.get("pick_threat"))
        + 0.35 * _safe_float(macro_a.get("tempo_pressure"))
        - 0.25 * _safe_float(macro_b.get("anti_dive"))
    )
    kills_pressure_b = (
        0.65 * _safe_float(macro_b.get("dive_threat"))
        + 0.50 * _safe_float(macro_b.get("pick_threat"))
        + 0.35 * _safe_float(macro_b.get("tempo_pressure"))
        - 0.25 * _safe_float(macro_a.get("anti_dive"))
    )
    kills_delta = kills_pressure_a - kills_pressure_b
    time_signal = (
        _safe_float(macro_a.get("carry_scaling"))
        + _safe_float(macro_b.get("carry_scaling"))
        + 0.45 * (_safe_float(macro_a.get("fight_reliability")) + _safe_float(macro_b.get("fight_reliability")))
        - 0.55 * (_safe_float(macro_a.get("tempo_pressure")) + _safe_float(macro_b.get("tempo_pressure")))
        - 0.35 * (_safe_float(macro_a.get("dive_threat")) + _safe_float(macro_b.get("dive_threat")))
    )

    combined_side = _safe_float(macro_a.get("side_pressure")) + _safe_float(macro_b.get("side_pressure"))
    combined_siege = _safe_float(macro_a.get("siege_pressure")) + _safe_float(macro_b.get("siege_pressure"))
    combined_objective = _safe_float(macro_a.get("objective_setup")) + _safe_float(macro_b.get("objective_setup"))
    combined_fight = _safe_float(macro_a.get("fight_reliability")) + _safe_float(macro_b.get("fight_reliability"))
    combined_tempo = _safe_float(macro_a.get("tempo_pressure")) + _safe_float(macro_b.get("tempo_pressure"))
    combined_carry = _safe_float(macro_a.get("carry_scaling")) + _safe_float(macro_b.get("carry_scaling"))
    tower_total_signal = 1.45 * combined_side + 0.25 * combined_siege + 0.15 * combined_objective
    dragon_total_signal = 0.60 * combined_objective + 0.30 * combined_fight + 0.18 * combined_tempo + 0.12 * max(_safe_float(macro_a.get("tempo_pressure")), _safe_float(macro_b.get("tempo_pressure")))
    baron_total_signal = 0.50 * combined_objective + 0.38 * combined_siege + 0.30 * combined_carry + 0.18 * combined_fight + 0.12 * combined_side

    early_team = _winner_label(early_delta, team_a=team_a, team_b=team_b)
    mid_team = _winner_label(mid_delta, team_a=team_a, team_b=team_b)
    late_team = _winner_label(late_delta, team_a=team_a, team_b=team_b)
    engage_team = _winner_label(engage_delta, team_a=team_a, team_b=team_b)
    objective_team = _winner_label(objective_delta, team_a=team_a, team_b=team_b)
    towers_team = _winner_label(towers_delta, team_a=team_a, team_b=team_b)
    if kills_delta >= 0.9:
        kills_call = f"{team_a} acelera"
    elif kills_delta <= -0.9:
        kills_call = f"{team_b} acelera"
    else:
        kills_call = "Neutro"
    if towers_team == str(team_a):
        towers_call = f"{team_a} pressiona"
    elif towers_team == str(team_b):
        towers_call = f"{team_b} pressiona"
    else:
        towers_call = "Neutro"
    if objective_team == str(team_a):
        dragons_call = f"{team_a} setup"
        barons_call = f"{team_a} setup"
    elif objective_team == str(team_b):
        dragons_call = f"{team_b} setup"
        barons_call = f"{team_b} setup"
    else:
        dragons_call = "Neutro"
        barons_call = "Neutro"
    if objective_team == str(team_a) and early_team == str(team_a):
        dragons_call = f"{team_a} joga cedo em dragao"
    elif objective_team == str(team_b) and early_team == str(team_b):
        dragons_call = f"{team_b} joga cedo em dragao"
    if objective_team == str(team_a) and late_team == str(team_a):
        barons_call = f"{team_a} joga para barao"
    elif objective_team == str(team_b) and late_team == str(team_b):
        barons_call = f"{team_b} joga para barao"
    if time_signal >= 3.0:
        time_call = "Alonga"
    elif time_signal <= 0.2:
        time_call = "Encurta"
    else:
        time_call = "Neutro"

    tower_total_call = "Neutro"
    tower_total_bias = 0.0
    tower_total_reason = ""
    if combined_side <= 4.8 and max(_safe_float(macro_a.get("side_pressure")), _safe_float(macro_b.get("side_pressure"))) <= 2.8:
        tower_total_call = "Menos torres"
        tower_total_bias = -0.55
        tower_total_reason = "Pouca side/split real para quebrar o mapa; torre tende a depender mais de luta ganha do que de pressao lateral."
    elif tower_total_signal >= 14.0:
        tower_total_call = "Mais torres"
        tower_total_bias = 0.45
        tower_total_reason = "Os dois drafts somam pressao de side/siege suficiente para converter wave e prioridade em estrutura."
    elif tower_total_signal <= 8.8:
        tower_total_call = "Menos torres"
        tower_total_bias = -0.30
        tower_total_reason = "Falta pressao limpa de side ou siege para estourar muita estrutura sem erro do adversario."

    dragons_total_call = "Neutro"
    dragons_total_bias = 0.0
    dragons_total_reason = ""
    if combined_objective <= 6.2 and combined_fight <= 7.0 and combined_tempo <= 6.0:
        dragons_total_call = "Menos dragoes"
        dragons_total_bias = -0.18
        dragons_total_reason = "Os drafts nao somam prioridade limpa e setup consistente para empilhar muitos dragoes."
    elif dragon_total_signal >= 9.8 and time_signal >= 0.9:
        dragons_total_call = "Mais dragoes"
        dragons_total_bias = 0.20
        dragons_total_reason = "Ha setup, fight e tempo suficientes para o mapa girar varias vezes em torno do dragon."
    elif dragon_total_signal <= 6.8:
        dragons_total_call = "Menos dragoes"
        dragons_total_bias = -0.10
        dragons_total_reason = "Sem prioridade/controle de objetivo forte, o dragon tende a aparecer menos como eixo do jogo."

    barons_total_call = "Neutro"
    barons_total_bias = 0.0
    barons_total_reason = ""
    if time_signal <= 0.3 and combined_objective <= 6.0:
        barons_total_call = "Menos baroes"
        barons_total_bias = -0.14
        barons_total_reason = "O jogo tende a resolver cedo demais para abrir muitos setups limpos de barao."
    elif tower_total_call == "Menos torres" and combined_objective <= 7.4 and combined_side <= 5.2:
        barons_total_call = "Menos baroes"
        barons_total_bias = -0.10
        barons_total_reason = "Sem abrir bem o mapa e sem setup forte de objetivo, o barao tende a entrar menos no script do draft."
    elif baron_total_signal >= 10.4 and time_signal >= 1.1:
        barons_total_call = "Mais baroes"
        barons_total_bias = 0.15
        barons_total_reason = "Os drafts conseguem alongar o jogo e criar setup de 20+ com siege/controle suficiente para barao."
    elif baron_total_signal <= 7.2:
        barons_total_call = "Menos baroes"
        barons_total_bias = -0.08
        barons_total_reason = "Falta carry scaling ou setup de objetivo para transformar o barao em eixo frequente do spot."

    late_reason = ""
    carry_gap = _safe_float(macro_a.get("carry_scaling")) - _safe_float(macro_b.get("carry_scaling"))
    fight_gap = _safe_float(macro_a.get("fight_reliability")) - _safe_float(macro_b.get("fight_reliability"))
    if late_team == str(team_a):
        if carry_gap >= abs(fight_gap) + 0.8:
            late_reason = f"{team_a} cresce mais em dano/alcance para o late."
        else:
            late_reason = f"{team_a} tem setup de fight mais estavel quando o jogo alonga."
    elif late_team == str(team_b):
        if (-carry_gap) >= abs(fight_gap) + 0.8:
            late_reason = f"{team_b} cresce mais em dano/alcance para o late."
        else:
            late_reason = f"{team_b} tem setup de fight mais estavel quando o jogo alonga."

    reasons: list[str] = []
    if early_team != "Equilibrado":
        reasons.append(f"Early favorece {early_team}.")
    if late_team != "Equilibrado" and late_team != early_team:
        reasons.append(late_reason or f"Late favorece {late_team}.")
    if objective_team != "Equilibrado":
        reasons.append(f"Setup de objetivo favorece {objective_team}.")
    if towers_team != "Equilibrado" and towers_team != objective_team:
        reasons.append(f"Comp de torre/siege favorece {towers_team}.")
    if tower_total_reason:
        reasons.append(tower_total_reason)
    if dragons_total_reason:
        reasons.append(dragons_total_reason)
    if barons_total_reason:
        reasons.append(barons_total_reason)
    if engage_team != "Equilibrado":
        reasons.append(f"Engage/acesso de luta favorece {engage_team}.")
    if not reasons:
        reasons.append("Comps proximas em janela e estilo.")

    plan_profile_a = _team_plan_profile(
        team_name=str(team_a),
        own_macro=macro_a,
        opp_macro=macro_b,
        own_styles=styles_a,
        opp_styles=styles_b,
    )
    plan_profile_b = _team_plan_profile(
        team_name=str(team_b),
        own_macro=macro_b,
        opp_macro=macro_a,
        own_styles=styles_b,
        opp_styles=styles_a,
    )

    return {
        "styles": {str(team_a): str(comp_a.get("primary_style") or "-"), str(team_b): str(comp_b.get("primary_style") or "-")},
        "windows": {str(team_a): str(comp_a.get("window_label") or "-"), str(team_b): str(comp_b.get("window_label") or "-")},
        "macro_scores": {str(team_a): dict(macro_a), str(team_b): dict(macro_b)},
        "plan_profiles": {str(team_a): plan_profile_a, str(team_b): plan_profile_b},
        "early_team": early_team,
        "mid_team": mid_team,
        "late_team": late_team,
        "engage_team": engage_team,
        "objective_team": objective_team,
        "towers_team": towers_team,
        "early_delta": early_delta,
        "mid_delta": mid_delta,
        "late_delta": late_delta,
        "objective_delta": objective_delta,
        "towers_delta": towers_delta,
        "tower_total_signal": tower_total_signal,
        "tower_total_call": tower_total_call,
        "tower_total_bias": tower_total_bias,
        "tower_total_reason": tower_total_reason,
        "dragon_total_signal": dragon_total_signal,
        "dragons_total_call": dragons_total_call,
        "dragons_total_bias": dragons_total_bias,
        "dragons_total_reason": dragons_total_reason,
        "baron_total_signal": baron_total_signal,
        "barons_total_call": barons_total_call,
        "barons_total_bias": barons_total_bias,
        "barons_total_reason": barons_total_reason,
        "late_reason": late_reason,
        "kills_call": kills_call,
        "towers_call": towers_call,
        "dragons_call": dragons_call,
        "barons_call": barons_call,
        "time_call": time_call,
        "reasons": reasons,
    }


__all__ = ["build_comp_profile", "compare_comp_profiles"]
