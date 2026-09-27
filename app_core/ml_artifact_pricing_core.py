from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def team_meta_from_artifact(art: Any) -> Dict[str, Dict[str, str]]:
    meta = getattr(art, "meta", {}).get("team_meta") if hasattr(art, "meta") else None
    if isinstance(meta, dict) and meta:
        return {str(k): dict(v) for k, v in meta.items() if isinstance(v, dict)}
    model = getattr(art, "model", None)
    tm = getattr(model, "team_meta", None) if model is not None else None
    if isinstance(tm, dict) and tm:
        return {str(k): dict(v) for k, v in tm.items() if isinstance(v, dict)}
    return {}


def p_from_diff_points(
    diff_points: float,
    *,
    scale_points: float = 400.0,
    clip_prob_fn: Callable[[float], float],
) -> float:
    try:
        denom = float(scale_points) if float(scale_points) > 0 else 400.0
        p = float(1.0 / (1.0 + 10 ** (-float(diff_points) / denom)))
        return float(clip_prob_fn(p))
    except Exception:
        return float("nan")


def strength_breakdown(
    *,
    art: Any,
    team: str,
    as_of: Any,
    team_meta: Dict[str, Dict[str, str]],
    ach_layer: Dict[str, Any],
    norm_league_fn: Callable[[str], str],
    tier_from_league_fn: Callable[[str], str],
    tier_base_points_fn: Callable[[str], float],
    league_base_points_fn: Callable[[str], float],
    team_boost_points_fn: Callable[[str, Any, Dict[str, Any], Dict[str, float]], float],
    league_hint: Optional[str] = None,
    tier_hint: Optional[str] = None,
) -> Dict[str, Any]:
    model = getattr(art, "model", None)
    ratings = dict(getattr(model, "ratings", {}) or {}) if model is not None else {}
    rating_raw_fn = getattr(model, "rating_raw", None) if model is not None else None
    residual_eff_fn = getattr(model, "residual_eff", None) if model is not None else None
    components_fn = getattr(model, "components", None) if model is not None else None
    league_strength_fn = getattr(model, "_league_strength", None) if model is not None else None
    tier_strength_fn = getattr(model, "_tier_strength", None) if model is not None else None
    residual_raw = float(rating_raw_fn(team)) if callable(rating_raw_fn) else float(ratings.get(team, 0.0))
    residual = float(residual_eff_fn(team)) if callable(residual_eff_fn) else residual_raw
    n_games_fn = getattr(model, "n_games", None) if model is not None else None
    games_played = int(n_games_fn(team)) if callable(n_games_fn) else int(getattr(model, "games_played", {}).get(team, 0) if model is not None else 0)
    shrink_fn = getattr(model, "_shrink_factor", None) if model is not None else None
    if callable(shrink_fn):
        shrink_factor = float(shrink_fn(int(max(0, games_played))))
    else:
        try:
            shrink_factor = float(residual) / float(residual_raw) if float(residual_raw) != 0.0 else (0.0 if games_played <= 0 else 1.0)
        except Exception:
            shrink_factor = 1.0 if games_played > 0 else 0.0
    shrink_factor = float(max(0.0, min(1.0, shrink_factor)))

    meta = dict(team_meta.get(team) or {})
    league_raw = str(meta.get("league") or "OTHER")
    tier_raw = str(meta.get("tier") or "OTHER")

    league = norm_league_fn(league_hint or "") or league_raw
    tier = str(tier_hint or "").strip() or (tier_raw if not norm_league_fn(league_hint or "") else tier_from_league_fn(league))

    tier_base = float(tier_base_points_fn(tier))
    league_base = float(league_base_points_fn(league))
    params = getattr(model, "params", None) if model is not None else None
    tier_prior = float(getattr(params, "tier_prior_weight", 1.0) or 1.0) * float(tier_base)
    league_prior = float(getattr(params, "league_prior_weight", 1.0) or 1.0) * float(league_base)
    tier_strength = float(tier_strength_fn(tier)) if callable(tier_strength_fn) else 0.0
    league_strength = float(league_strength_fn(league)) if callable(league_strength_fn) else 0.0

    if callable(components_fn):
        try:
            comp = dict(components_fn(team, league=league, tier=tier, apply_rank_caps=False) or {})
        except Exception:
            comp = {}
        if comp:
            if comp.get("residual_raw") is not None:
                residual_raw = float(comp.get("residual_raw"))
            if comp.get("residual_eff") is not None:
                residual = float(comp.get("residual_eff"))
            if comp.get("games_played") is not None:
                games_played = int(comp.get("games_played"))
            if comp.get("shrink_factor") is not None:
                shrink_factor = float(comp.get("shrink_factor"))
            if comp.get("tier_strength") is not None:
                tier_strength = float(comp.get("tier_strength"))
            if comp.get("league_strength") is not None:
                league_strength = float(comp.get("league_strength"))
            if comp.get("tier_prior_points") is not None:
                tier_prior = float(comp.get("tier_prior_points"))
            if comp.get("league_prior_points") is not None:
                league_prior = float(comp.get("league_prior_points"))

    ach = 0.0
    if ach_layer.get("enabled") and ach_layer.get("weights_points"):
        ach = float(team_boost_points_fn(team, as_of, ach_layer.get("team_idx") or {}, ach_layer.get("weights_points") or {}))

    base_strength = float(residual + tier_strength + league_strength + tier_prior + league_prior)
    total = float(base_strength + ach)
    return {
        "team": team,
        "league": league,
        "tier": tier,
        "league_raw": league_raw,
        "tier_raw": tier_raw,
        "residual_elo": residual,
        "residual_raw": residual_raw,
        "games_played": games_played,
        "shrink_factor": shrink_factor,
        "tier_strength": tier_strength,
        "league_strength": league_strength,
        "tier_base": tier_prior,
        "league_base": league_prior,
        "tier_base_raw": tier_base,
        "league_base_raw": league_base,
        "base_strength": base_strength,
        "ach_boost": ach,
        "total_strength": total,
    }


def mlcore_p_map_v2(
    *,
    artifact_path: str,
    team_blue: str,
    team_red: str,
    as_of: Any,
    clip_prob_fn: Callable[[float], float],
    artifact_sig_fn: Callable[[str], Any],
    load_artifact_cached_fn: Callable[[Any], Any],
    team_meta_from_artifact_fn: Callable[[Any], Dict[str, Dict[str, str]]],
    ach_layer_for_artifact_fn: Callable[[Any, str, Optional[str]], Dict[str, Any]],
    strength_breakdown_fn: Callable[..., Dict[str, Any]],
    p_from_diff_points_fn: Callable[..., float],
    ach_override_path: Optional[str] = None,
    league_hint_blue: Optional[str] = None,
    league_hint_red: Optional[str] = None,
    tier_hint_blue: Optional[str] = None,
    tier_hint_red: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    path = Path(artifact_path)
    if not path.exists():
        return None

    art = load_artifact_cached_fn(artifact_sig_fn(str(path)))
    team_meta = team_meta_from_artifact_fn(art)
    ach_layer = ach_layer_for_artifact_fn(art, str(path), ach_override_path)

    model_params = getattr(getattr(art, "model", None), "params", None)
    scale_points = float(getattr(model_params, "scale_points", 400.0) or 400.0)
    alpha_points = float(getattr(model_params, "alpha", 80.0) or 80.0)

    blue = strength_breakdown_fn(
        art=art,
        team=team_blue,
        as_of=as_of,
        team_meta=team_meta,
        ach_layer=ach_layer,
        league_hint=league_hint_blue,
        tier_hint=tier_hint_blue,
    )
    red = strength_breakdown_fn(
        art=art,
        team=team_red,
        as_of=as_of,
        team_meta=team_meta,
        ach_layer=ach_layer,
        league_hint=league_hint_red,
        tier_hint=tier_hint_red,
    )

    model = getattr(art, "model", None)
    diff = float(blue["total_strength"] - red["total_strength"])
    p_raw = float("nan")
    pricing_source = "fallback_diff"
    cal_error = ""

    if model is not None and hasattr(model, "p_blue"):
        try:
            p_raw = float(
                model.p_blue(
                    team_blue,
                    team_red,
                    blue_league=blue.get("league"),
                    red_league=red.get("league"),
                    blue_tier=blue.get("tier"),
                    red_tier=red.get("tier"),
                )
            )
            p_raw = float(clip_prob_fn(p_raw))
            pricing_source = "artifact_model_p_blue"
            diff = float((float(scale_points) / math.log(10.0)) * math.log(p_raw / (1.0 - p_raw)))
        except Exception:
            p_raw = float("nan")

    if not math.isfinite(p_raw):
        p_raw = p_from_diff_points_fn(diff, scale_points=scale_points)
        pricing_source = "reconstructed_diff"

    # SIMETRIZADO NOS DOIS SENTIDOS (16/09/2026).
    #
    # ELE MANDOU DOIS PRINTS do MESMO confronto com numeros diferentes e
    # atribuiu ao CSV que tinha acabado de subir. O CSV nao tinha mexido nada
    # ali (0.567 antes, 0.567 depois; nenhum dos dois times jogou nos 54 jogos
    # novos). O que mudou foi a ORDEM: num print a NAVI estava no Time 1, no
    # outro o KOI.
    #
    # A CAUSA: o calibrador carrega o INTERCEPTO DO LADO AZUL. Ele manda
    # p=0.50 para 0.5253, e a vitoria do lado azul nos dados e 53,62% em 8.758
    # mapas -- o calibrador esta CERTO, aprendeu uma vantagem que existe. So
    # que o caminho servido passa o Time 1 como lado azul, e na tela o "Time 1"
    # e apenas a caixa onde ele digitou primeiro.
    #
    # Resultado: cal(p) + cal(1-p) dava 1.05, e o mesmo confronto mudava ate 5
    # pontos percentuais conforme a ordem -- bem no meio da tabela, onde as
    # decisoes de aposta acontecem.
    #
    # E O PROJETO JA FAZIA ISSO EM TRES LUGARES, menos aqui:
    #   mlcore/fit_samples_core.py      (as amostras de TREINO)
    #   app_core/live_between_maps_core.py
    #   tools/calibrate_key_expectations.py
    # Ou seja: o modelo era TREINADO com a probabilidade simetrizada e SERVIDO
    # sem ela. Divergencia treino/servico, que e o que mlcore/replay_pre_jogo
    # cita o Google para evitar ("garanta que treino e producao se imitem o
    # mais perto possivel").
    #
    # MEDIDO em 6.331 mapas (tools/medir_lado_azul.py), com a ordem dos times
    # arbitraria, que e o caso dele:
    #     hoje              logloss 0.58871   nota 8.28
    #     simetrizado       logloss 0.58760   nota 8.32
    # ganho +0.00111, intervalo 95% [+0.00125, +0.00133] no caminho inteiro --
    # pequeno, mas nao cabe zero dentro dele. E a convexidade do logloss
    # garante que simetrizar NUNCA piora quando a ordem e arbitraria.
    # A inconsistencia por confronto cai de 4,8 para 1,54 pontos (mediana).
    try:
        p_raw_rev = float("nan")
        if model is not None and hasattr(model, "p_blue"):
            p_raw_rev = float(
                model.p_blue(
                    team_red,
                    team_blue,
                    blue_league=red.get("league"),
                    red_league=blue.get("league"),
                    blue_tier=red.get("tier"),
                    red_tier=blue.get("tier"),
                )
            )
        if math.isfinite(p_raw_rev):
            cal_f = float(art.calibrator.predict_proba(float(p_raw)))
            cal_r = float(art.calibrator.predict_proba(float(p_raw_rev)))
            p_cal = float(clip_prob_fn(0.5 * (cal_f + (1.0 - cal_r))))
        else:
            # Sem o sentido inverso nao ha o que simetrizar. Fica o de sempre
            # -- pior que simetrizado, melhor que nada, e DITO no retorno.
            p_cal = float(clip_prob_fn(float(art.calibrator.predict_proba(float(p_raw)))))
            cal_error = "sem_sentido_inverso_nao_simetrizado"
    except Exception:
        p_cal = float(clip_prob_fn(float(p_raw)))
        cal_error = "calibrator_failed_fallback_to_p_raw"

    return {
        "p_raw": p_raw,
        "p_cal": p_cal,
        "diff_points": diff,
        "scale_points": scale_points,
        "alpha_points": alpha_points,
        "ach_enabled": bool(ach_layer.get("enabled")) if isinstance(ach_layer, dict) else False,
        "ach_path": str(ach_layer.get("path")) if isinstance(ach_layer, dict) else "",
        "ach_error": str(ach_layer.get("error") or "") if isinstance(ach_layer, dict) else "",
        "blue": blue,
        "red": red,
        "achievements_path": ach_layer.get("path"),
        "achievements_enabled": bool(ach_layer.get("enabled")),
        "has_ach_weights": bool(ach_layer.get("weights_points")),
        "artifact_version": getattr(art, "version", ""),
        "ml_pricing_source": pricing_source,
        "calibration_error": cal_error,
    }
