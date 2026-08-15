from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List

from .fit_io_core import _read_json_file_safe, _write_json_file_safe, _utc_now_iso, _safe_int
FIT_STAGE_ORDER = [
    "load_inputs",
    "player_weights",
    "global_scales",
    "form_combined",
    "segmented_form",
    "series_models",
    "context_models",
    "snapshots",
    "finalize",
]

def _normalize_fit_stage(stage: Any, *, allow_finalize: bool = True) -> str:
    stage_key = str(stage or "load_inputs").strip().lower()
    if not stage_key:
        stage_key = "load_inputs"
    if stage_key not in FIT_STAGE_ORDER:
        raise ValueError(f"invalid_start_stage:{stage_key}")
    if not allow_finalize and stage_key == "finalize":
        raise ValueError("invalid_start_stage:finalize")
    return stage_key


def _fit_stage_index(stage: Any) -> int:
    return FIT_STAGE_ORDER.index(_normalize_fit_stage(stage))


def _seed_stage_payload_from_params(base_params: Dict[str, Any], stage: str) -> Dict[str, Any]:
    params = dict(base_params or {}) if isinstance(base_params, dict) else {}
    if not params:
        return {}
    stage_key = _normalize_fit_stage(stage)
    if stage_key == "player_weights":
        lane = params.get("players_lane_weights")
        league = params.get("players_league_weights")
        if isinstance(lane, dict) and lane and isinstance(league, dict) and league:
            return {
                "players_lane_weights": dict(lane),
                "players_league_weights": dict(league),
            }
        return {}
    if stage_key == "global_scales":
        baseline = params.get("baseline")
        best = params.get("best")
        grid = params.get("grid")
        if isinstance(baseline, dict) and isinstance(best, dict) and isinstance(grid, list) and grid:
            baseline_seed = dict(baseline)
            baseline_folds_n = _safe_int(baseline_seed.get("folds"), 0)
            if not isinstance(baseline_seed.get("folds"), list):
                baseline_seed["folds"] = ([None] * max(0, baseline_folds_n))
            return {
                "baseline": baseline_seed,
                "best": dict(best),
                "grid_rows": list(grid),
                "best_report": {"folds": list(params.get("best_folds") or [])},
            }
        return {}
    if stage_key == "form_combined":
        best_form_node = dict(params.get("best_form") or {}) if isinstance(params.get("best_form"), dict) else {}
        best_combined_node = dict(params.get("best_combined") or {}) if isinstance(params.get("best_combined"), dict) else {}
        microseg_node = dict(params.get("microseg_calibration") or {}) if isinstance(params.get("microseg_calibration"), dict) else {}
        best_form_settings = dict(best_form_node.get("settings") or {})
        best_form = dict(best_form_node)
        best_form.pop("settings", None)
        if not best_form_settings and isinstance(best_combined_node.get("form_settings"), dict):
            best_form_settings = dict(best_combined_node.get("form_settings") or {})
        combined_folds_n = int(best_combined_node.get("folds") or 0)
        pred_rows_n = int(microseg_node.get("pred_rows_n") or 0)
        combined_report = {
            "logloss": best_combined_node.get("logloss"),
            "brier": best_combined_node.get("brier"),
            "n_total": best_combined_node.get("n_total"),
            "folds": ([None] * combined_folds_n),
            "pred_rows_n": pred_rows_n,
            "pred_rows": ([None] * pred_rows_n),
        }
        form_fit = {
            "baseline_no_form": dict(params.get("best_form_baseline_no_form") or {}),
            "best_form": dict(best_form),
            "grid_form": list(params.get("grid_form_top20") or []),
        }
        fused = params.get("fused_post_calibration")
        microseg_table = microseg_node.get("table")
        if isinstance(fused, dict) and isinstance(microseg_table, dict):
            return {
                "form_fit": form_fit,
                "best_form": dict(best_form),
                "best_form_settings": dict(best_form_settings),
                "combined_report": combined_report,
                "fused_post_calibration": dict(fused),
                "microseg_table": dict(microseg_table),
            }
        return {}
    if stage_key == "segmented_form":
        best_by_league = params.get("best_by_league")
        best_form_by_league = params.get("best_form_by_league")
        best_form_by_macro = params.get("best_form_by_macro")
        if isinstance(best_by_league, dict) and isinstance(best_form_by_league, dict) and isinstance(best_form_by_macro, dict):
            return {
                "best_by_league": dict(best_by_league),
                "best_form_by_league": dict(best_form_by_league),
                "best_form_by_macro": dict(best_form_by_macro),
            }
        return {}
    if stage_key == "series_models":
        gamma_global = params.get("series_gamma_global")
        gamma_by_league = params.get("series_gamma_by_league")
        gamma_by_macro = params.get("series_gamma_by_macro")
        bo_tuning = params.get("series_bo_dplite_tuning")
        blind = params.get("blind_backtest_by_league")
        if all(isinstance(x, dict) for x in [gamma_global, gamma_by_league, gamma_by_macro, bo_tuning, blind]):
            return {
                "series_gamma_global": dict(gamma_global),
                "series_gamma_by_league": dict(gamma_by_league),
                "series_gamma_by_macro": dict(gamma_by_macro),
                "series_bo_dplite_tuning": dict(bo_tuning),
                "blind_backtest": dict(blind),
            }
        return {}
    if stage_key == "context_models":
        meta_model = params.get("meta_blend_c_model")
        style_model = params.get("style_matchup_b_model")
        players_fit = params.get("players_context_fit")
        if all(isinstance(x, dict) for x in [meta_model, style_model, players_fit]):
            return {
                "meta_blend_c_model": dict(meta_model),
                "style_matchup_b_model": dict(style_model),
                "players_context_fit": dict(players_fit),
            }
        return {}
    if stage_key == "snapshots":
        areal = params.get("areal_prob_tuning")
        marginal = params.get("ml_marginal_trust_band_coeffs")
        if isinstance(areal, dict) and isinstance(marginal, dict):
            return {
                "areal_prob_tuning": dict(areal),
                "areal_prob_tuning_source": str(params.get("areal_prob_tuning_source") or ""),
                "ml_marginal_trust_band_coeffs": dict(marginal),
                "ml_marginal_trust_band_coeffs_source": str(params.get("ml_marginal_trust_band_coeffs_source") or ""),
            }
        return {}
    return {}


def _prime_seed_stage_checkpoints_from_params(
    checkpoint_dir: Path | None,
    *,
    run_signature: Dict[str, Any],
    base_params: Dict[str, Any],
    start_stage: str,
    base_params_path: Path | None = None,
) -> Dict[str, Any]:
    if checkpoint_dir is None:
        return {"seeded": [], "missing": []}
    start_stage_key = _normalize_fit_stage(start_stage)
    seeded: List[str] = []
    missing: List[str] = []
    for stage in FIT_STAGE_ORDER:
        if stage in {"load_inputs", "finalize"}:
            continue
        if _fit_stage_index(stage) >= _fit_stage_index(start_stage_key):
            continue
        if _load_stage_checkpoint(checkpoint_dir, stage, run_signature=run_signature):
            continue
        payload = _seed_stage_payload_from_params(base_params, stage)
        if not payload:
            missing.append(str(stage))
            continue
        summary: Dict[str, Any] = {
            "seeded_from_base": True,
            "start_stage": str(start_stage_key),
        }
        if base_params_path is not None:
            summary["base_params_path"] = str(base_params_path)
        _save_stage_checkpoint(
            checkpoint_dir,
            stage,
            run_signature=run_signature,
            payload=payload,
            summary=summary,
        )
        seeded.append(str(stage))
    return {"seeded": seeded, "missing": missing}


def _checkpoint_path(checkpoint_dir: Path | None, stage: str) -> Path | None:
    if checkpoint_dir is None:
        return None
    return checkpoint_dir / f"{str(stage or '').strip()}.json"


def _load_stage_checkpoint(
    checkpoint_dir: Path | None,
    stage: str,
    *,
    run_signature: Dict[str, Any],
) -> Dict[str, Any]:
    path = _checkpoint_path(checkpoint_dir, stage)
    if path is None:
        return {}
    obj = _read_json_file_safe(path)
    if str(obj.get("version") or "") != "fit_stage_checkpoint_v1":
        return {}
    if str(obj.get("stage") or "") != str(stage):
        return {}
    sig = obj.get("run_signature") if isinstance(obj.get("run_signature"), dict) else {}
    if str(sig.get("sha256") or "") != str((run_signature or {}).get("sha256") or ""):
        return {}
    return obj


def _save_stage_checkpoint(
    checkpoint_dir: Path | None,
    stage: str,
    *,
    run_signature: Dict[str, Any],
    payload: Dict[str, Any],
    summary: Dict[str, Any] | None = None,
) -> None:
    path = _checkpoint_path(checkpoint_dir, stage)
    if path is None:
        return
    obj = {
        "version": "fit_stage_checkpoint_v1",
        "saved_at": _utc_now_iso(),
        "stage": str(stage),
        "run_signature": {
            "sha256": str((run_signature or {}).get("sha256") or ""),
            "speed_profile": str((run_signature or {}).get("speed_profile") or ""),
            "min_train_games": int((run_signature or {}).get("min_train_games") or 0),
            "run_label": str((run_signature or {}).get("run_label") or ""),
            "start_stage": str((run_signature or {}).get("start_stage") or ""),
        },
        "summary": dict(summary or {}),
        "payload": dict(payload or {}),
    }
    _write_json_file_safe(path, obj)


def _write_fit_progress(
    progress_path: Path | None,
    *,
    stage: str,
    status: str,
    run_signature: Dict[str, Any],
    total_stages: int,
    completed_stages: List[str],
    detail: Dict[str, Any] | None = None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
) -> None:
    if progress_path is None:
        return
    payload = {
        "version": "fit_progress_v1",
        "updated_at": _utc_now_iso(),
        "stage": str(stage),
        "status": str(status),
        "total_stages": int(total_stages),
        "completed_count": len(completed_stages or []),
        "completed_stages": [str(x) for x in (completed_stages or [])],
        "run_signature": {
            "sha256": str((run_signature or {}).get("sha256") or ""),
            "speed_profile": str((run_signature or {}).get("speed_profile") or ""),
            "min_train_games": int((run_signature or {}).get("min_train_games") or 0),
            "run_label": str((run_signature or {}).get("run_label") or ""),
            "start_stage": str((run_signature or {}).get("start_stage") or ""),
        },
        "process": {"pid": int(os.getpid())},
        "started_at": (str(started_at or "") if str(started_at or "").strip() else None),
        "elapsed_s": (float(elapsed_s) if elapsed_s is not None and math.isfinite(float(elapsed_s)) else None),
        "detail": dict(detail or {}),
    }
    _write_json_file_safe(progress_path, payload)


def _print_fit_stage(stage: str, status: str, detail: Dict[str, Any] | None = None) -> None:
    msg = f"[fit_ml_params] stage={stage!s} status={status!s}"
    if isinstance(detail, dict) and detail:
        parts = []
        for k in sorted(detail.keys()):
            v = detail.get(k)
            if isinstance(v, float):
                parts.append(f"{k}={v:.6f}")
            else:
                parts.append(f"{k}={v}")
        if parts:
            msg += " | " + " ".join(parts)
    print(msg, flush=True)
