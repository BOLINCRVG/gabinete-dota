from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


def resolve_maintenance_preferences(settings: Mapping[str, Any], *, force_full: bool = False) -> Dict[str, Any]:
    fast_mode_user_pref = bool(settings.get("auto_maint_fast_mode", True))
    cmd_timeout_s = int(settings.get("auto_maint_cmd_timeout_s", 1200) or 1200)
    params_timeout_s = int(settings.get("auto_maint_params_timeout_s", max(cmd_timeout_s, 14000)) or max(cmd_timeout_s, 14000))
    return {
        "cmd_timeout_s": int(cmd_timeout_s),
        "params_timeout_s": int(max(cmd_timeout_s, params_timeout_s)),
        "fast_mode_user_pref": bool(fast_mode_user_pref),
        "fast_mode": bool(fast_mode_user_pref),
        "run_layer_validation": bool(settings.get("auto_maint_run_layer_validation", (not fast_mode_user_pref))),
        # IMPORTANTE: force_full (botão "Manutenção completa") força o full retrain
        # do MODELO (skip incremental → train_offline reconstrói ml+players), mas
        # NÃO o profile full do PARAMS. O fit profundo de params (horas) vem só do
        # setting explícito auto_maint_force_full (a "Qualidade profunda"). Senão o
        # botão de sync ficava preso horas no params_fit full sem necessidade.
        "policy_force_full": bool(settings.get("auto_maint_force_full", False)),
        "policy_force_fast": bool(settings.get("auto_maint_force_fast", False)),
        "fast_new_games_max": int(settings.get("auto_maint_fast_new_games_max", 30) or 30),
    }


def seed_maintenance_needs(
    need_retrain_model: bool,
    need_retrain_params: bool,
    need_walkforward: bool,
) -> Tuple[bool, bool]:
    if bool(need_retrain_model):
        return True, True
    if not (bool(need_retrain_model) or bool(need_retrain_params) or bool(need_walkforward)):
        return True, True
    return bool(need_retrain_params), bool(need_walkforward)


def adjust_needs_after_incremental(
    *,
    model_ok: bool,
    new_games_inc: int | None,
    need_patch_recalibration: bool,
    need_retrain_params: bool,
    need_walkforward: bool,
) -> Dict[str, Any]:
    if bool(model_ok) and (new_games_inc == 0) and (not bool(need_patch_recalibration)):
        return {
            "need_retrain_params": False,
            "need_walkforward": False,
            "log_message": "[modelo] incremental sem novos jogos; params/walk-forward pulados.",
        }
    return {
        "need_retrain_params": bool(need_retrain_params),
        "need_walkforward": bool(need_walkforward),
        "log_message": "",
    }


def force_params_after_model_refresh(
    *,
    need_retrain_model: bool,
    need_retrain_params: bool,
    model_updated_this_cycle: bool,
) -> Dict[str, Any]:
    forced = bool(need_retrain_model) and bool(model_updated_this_cycle) and (not bool(need_retrain_params))
    return {
        "need_retrain_params": bool(need_retrain_params) or forced,
        "forced": bool(forced),
        "log_message": ("[params] forçado: modelo atualizado no ciclo atual." if forced else ""),
    }


def decide_maintenance_policy(
    *,
    policy_force_full: bool,
    policy_force_fast: bool,
    need_retrain_model: bool,
    need_retrain_params: bool,
    need_walkforward: bool,
    need_patch_recalibration: bool,
    model_backfill_suspected: bool,
    model_had_full_retrain: bool,
    model_new_games: int | None,
    fast_new_games_max: int,
    fast_mode_user_pref: bool,
) -> Dict[str, Any]:
    if bool(policy_force_full):
        profile = "full"
        reason = "force_full_flag"
    elif bool(policy_force_fast):
        profile = "fast"
        reason = "force_fast_flag"
    elif bool(need_patch_recalibration):
        profile = "full"
        reason = "patch_liga_refit_warn"
    elif (not bool(need_retrain_model)) and bool(need_retrain_params) and (not bool(need_walkforward)):
        profile = "fast"
        reason = "params_only_fast"
    elif bool(model_backfill_suspected):
        profile = "full"
        reason = "artifact_backfill_suspected"
    elif bool(model_had_full_retrain):
        profile = "full"
        reason = "modelo_full_retrain_executado"
    elif (model_new_games is not None) and (int(model_new_games) > int(fast_new_games_max)):
        profile = "full"
        reason = f"new_games_gt_fast_max({int(model_new_games)}>{int(fast_new_games_max)})"
    elif not bool(fast_mode_user_pref):
        profile = "full"
        reason = "user_pref_full"
    else:
        profile = "fast"
        reason = (
            f"new_games_le_fast_max({int(model_new_games)}<={int(fast_new_games_max)})"
            if model_new_games is not None
            else "default_fast"
        )
    return {"profile": str(profile), "reason": str(reason), "fast_mode": bool(profile == "fast")}
