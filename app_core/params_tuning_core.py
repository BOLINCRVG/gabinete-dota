from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


def _read_json_file_safe(path_like: str | Path) -> Dict[str, Any]:
    try:
        p = Path(path_like).expanduser()
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}
    return {}


def _write_json_file_atomic(path_like: str | Path, payload: Dict[str, Any]) -> bool:
    try:
        p = Path(path_like).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception:
        return False


def is_valid_areal_prob_tuning(cfg: Any) -> bool:
    if not isinstance(cfg, dict):
        return False
    if not str(cfg.get("version") or "").startswith("areal_prob_tuning_v"):
        return False
    g = cfg.get("global") if isinstance(cfg.get("global"), dict) else {}
    if not g:
        return False
    try:
        _ = float(g.get("bayes_info_scale", 1.0))
        _ = float(g.get("bayes_mu_q_strength", 1.0))
        _ = float(g.get("quality_k_boost", 0.22))
        _ = float(g.get("qwl_surplus_mix", 0.70))
        _ = float(g.get("qwl_throw_weight", 1.20))
        _ = float(g.get("qwl_comeback_weight", 0.65))
        _ = float(g.get("qwl_phase_dynamic_strength", 0.35))
        _ = float(g.get("qwl_opponent_context_mix", 0.30))
    except Exception:
        return False
    return True


def is_valid_ml_marginal_band_coeffs(
    cfg: Any,
    *,
    sanitize_band_coeffs_fn: Callable[[Any], Dict[str, Dict[str, float]]],
) -> bool:
    return bool(sanitize_band_coeffs_fn(cfg))


def load_promoted_areal_prob_tuning(root: Path) -> Tuple[Dict[str, Any], str]:
    rep = _read_json_file_safe(root / "phase2b_areal_tuning_report.json")
    if isinstance(rep, dict) and str(rep.get("decision") or "").strip().lower() == "promote":
        tcfg = rep.get("tuning_cfg")
        if is_valid_areal_prob_tuning(tcfg):
            return dict(tcfg), "phase2b_areal_tuning_report.json"
    cur = _read_json_file_safe(root / "params_fitted.json")
    cur_t = cur.get("areal_prob_tuning") if isinstance(cur, dict) else None
    if is_valid_areal_prob_tuning(cur_t):
        return dict(cur_t), "params_fitted.json"
    return {}, ""


def load_promoted_ml_marginal_band_coeffs(
    root: Path,
    *,
    sanitize_band_coeffs_fn: Callable[[Any], Dict[str, Dict[str, float]]],
) -> Tuple[Dict[str, Any], str]:
    rep = _read_json_file_safe(root / "phase4b_marginal_tuning_report.json")
    if isinstance(rep, dict) and str(rep.get("decision") or "").strip().lower() == "promote":
        tcfg = rep.get("tuning_cfg") if isinstance(rep.get("tuning_cfg"), dict) else {}
        bcfg = tcfg.get("band_coeffs") if isinstance(tcfg.get("band_coeffs"), dict) else rep.get("band_coeffs")
        if is_valid_ml_marginal_band_coeffs(bcfg, sanitize_band_coeffs_fn=sanitize_band_coeffs_fn):
            return sanitize_band_coeffs_fn(bcfg), "phase4b_marginal_tuning_report.json"
    cur = _read_json_file_safe(root / "params_fitted.json")
    cur_t = cur.get("ml_marginal_trust_band_coeffs") if isinstance(cur, dict) else None
    if is_valid_ml_marginal_band_coeffs(cur_t, sanitize_band_coeffs_fn=sanitize_band_coeffs_fn):
        return sanitize_band_coeffs_fn(cur_t), "params_fitted.json"
    return {}, ""


def enrich_params_with_areal_prob_tuning(params_obj: Any, *, root: Path) -> Tuple[Dict[str, Any], str]:
    obj = dict(params_obj) if isinstance(params_obj, dict) else {}
    cur_t = obj.get("areal_prob_tuning")
    if is_valid_areal_prob_tuning(cur_t):
        return obj, "already_present"
    tune, src = load_promoted_areal_prob_tuning(root)
    if is_valid_areal_prob_tuning(tune):
        obj["areal_prob_tuning"] = dict(tune)
        obj["areal_prob_tuning_source"] = str(src or "fallback")
        return obj, str(src or "fallback")
    return obj, ""


def enrich_params_with_ml_marginal_band_coeffs(
    params_obj: Any,
    *,
    root: Path,
    sanitize_band_coeffs_fn: Callable[[Any], Dict[str, Dict[str, float]]],
) -> Tuple[Dict[str, Any], str]:
    obj = dict(params_obj) if isinstance(params_obj, dict) else {}
    cur_t = obj.get("ml_marginal_trust_band_coeffs")
    if is_valid_ml_marginal_band_coeffs(cur_t, sanitize_band_coeffs_fn=sanitize_band_coeffs_fn):
        obj["ml_marginal_trust_band_coeffs"] = sanitize_band_coeffs_fn(cur_t)
        return obj, "already_present"
    tune, src = load_promoted_ml_marginal_band_coeffs(
        root,
        sanitize_band_coeffs_fn=sanitize_band_coeffs_fn,
    )
    if is_valid_ml_marginal_band_coeffs(tune, sanitize_band_coeffs_fn=sanitize_band_coeffs_fn):
        obj["ml_marginal_trust_band_coeffs"] = sanitize_band_coeffs_fn(tune)
        obj["ml_marginal_trust_band_coeffs_source"] = str(src or "fallback")
        return obj, str(src or "fallback")
    return obj, ""


def postprocess_params_file_tuning(
    root: Path,
    *,
    sanitize_band_coeffs_fn: Callable[[Any], Dict[str, Dict[str, float]]],
) -> Tuple[bool, str]:
    p = root / "params_fitted.json"
    data = _read_json_file_safe(p)
    if not isinstance(data, dict) or not data:
        return False, "params_missing_or_invalid"

    out = dict(data)
    srcs: List[str] = []
    out, src_a = enrich_params_with_areal_prob_tuning(out, root=root)
    if src_a:
        srcs.append(f"areal:{src_a}")
    out, src_m = enrich_params_with_ml_marginal_band_coeffs(
        out,
        root=root,
        sanitize_band_coeffs_fn=sanitize_band_coeffs_fn,
    )
    if src_m:
        srcs.append(f"marginal:{src_m}")

    # Guardrail: never drop pre-existing keys during enrichment pass.
    if not set(data.keys()).issubset(set(out.keys())):
        return False, "config_key_loss_guard"

    if out == data:
        if srcs:
            return True, "already_present"
        return False, "no_promoted_tuning_available"

    ok = _write_json_file_atomic(p, out)
    return bool(ok), ("updated_from_" + ",".join(srcs) if ok else "write_failed")

