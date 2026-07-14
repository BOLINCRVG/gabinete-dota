from __future__ import annotations

import hashlib
import json
from typing import Any


def build_microseg_caption(info: dict[str, Any] | None) -> str:
    payload = dict(info or {})
    if not bool(payload.get("applied")):
        return ""
    try:
        bias_pp = float(payload.get("bias", 0.0) or 0.0) * 100.0
    except Exception:
        bias_pp = 0.0
    return (
        f"Microseg calibracao: aplicada ({payload.get('key')}) | "
        f"fonte={payload.get('source', 'live')} | "
        f"bo={payload.get('bo', '-')} | fase={payload.get('phase', '-')} | "
        f"n={int(payload.get('n', 0) or 0)} | ajuste={bias_pp:+.1f} pp"
    )


def resolve_shared_league_hint(
    *,
    ml_info: dict[str, Any] | None,
    fit_league_hint: Any = "",
    match_league: Any = "",
    league_a: Any = "",
    league_b: Any = "",
) -> str:
    lg = str(fit_league_hint or "").strip()
    if lg:
        return lg
    la = str(league_a or "").strip()
    lb = str(league_b or "").strip()
    if la and lb and la == lb:
        return la
    lg = str(match_league or "").strip()
    if lg and ((not la and not lb) or (la and not lb) or (lb and not la)):
        return lg
    try:
        blue = str((((ml_info or {}).get("blue") or {}).get("league")) or "").strip()
        red = str((((ml_info or {}).get("red") or {}).get("league")) or "").strip()
        if blue and blue == red:
            return blue
    except Exception:
        pass
    return ""


def build_input_signature_payload(
    *,
    matchup: dict[str, Any],
    filters: dict[str, Any],
    config: dict[str, Any],
    lineup: dict[str, Any],
    versions: dict[str, Any],
) -> dict[str, Any]:
    return {
        "matchup": dict(matchup or {}),
        "filters": dict(filters or {}),
        "config": dict(config or {}),
        "lineup": dict(lineup or {}),
        "versions": dict(versions or {}),
    }


def compute_input_signature(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(payload or {})
    signature_json = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    signature_hash = hashlib.sha256(signature_json.encode("utf-8")).hexdigest()
    return {
        "payload": raw,
        "json": signature_json,
        "sha256": signature_hash,
    }


def apply_runtime_tuning_to_session(
    *,
    session_state: Any,
    league_hint: str,
    bo: int,
    confidence_baseline: dict[str, Any] | None,
    confidence_governor_budget_fn: Any,
    series_coherence_post_cap_fn: Any,
    governor_default_pp: float,
) -> dict[str, Any]:
    gov_tuning = confidence_governor_budget_fn(
        league_hint,
        confidence_baseline=confidence_baseline,
    )
    gov_budget = float(
        (gov_tuning or {}).get("budget_pp", float(governor_default_pp))
        or float(governor_default_pp)
    )
    session_state["_ml_confidence_governor_budget_runtime_pp"] = float(gov_budget)
    session_state["_ml_confidence_governor_budget_runtime_info"] = (
        dict(gov_tuning) if isinstance(gov_tuning, dict) else {}
    )

    series_tuning = series_coherence_post_cap_fn(
        league_hint=league_hint,
        bo=int(bo),
        confidence_baseline=confidence_baseline,
    )
    series_cap = (
        float((series_tuning or {}).get("max_dev_pp"))
        if (series_tuning or {}).get("max_dev_pp") is not None
        else None
    )
    session_state["_ml_series_coherence_post_max_dev_pp_runtime"] = (
        float(series_cap) if series_cap is not None else None
    )
    session_state["_ml_series_coherence_post_tuning_info"] = (
        dict(series_tuning) if isinstance(series_tuning, dict) else {}
    )
    return {
        "gov_tuning": dict(gov_tuning) if isinstance(gov_tuning, dict) else {},
        "gov_budget_pp": float(gov_budget),
        "series_tuning": dict(series_tuning) if isinstance(series_tuning, dict) else {},
        "series_post_cap": (float(series_cap) if series_cap is not None else None),
    }
