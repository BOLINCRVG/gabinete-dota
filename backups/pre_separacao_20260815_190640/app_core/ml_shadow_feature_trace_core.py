"""Shadow feature-trace append helpers extracted from plays_app."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _r6(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except Exception:
        return None


def _dedupe_signature(
    *,
    input_signature_hash: str,
    ml_engine: str,
    p_map_old: Any,
    p_map_new: Any,
    p_series_old: Any,
    p_series_new: Any,
) -> str:
    return json.dumps(
        {
            "sig": str(input_signature_hash),
            "eng": str(ml_engine),
            "pm_old": _r6(p_map_old),
            "pm_new": _r6(p_map_new),
            "ps_old": _r6(p_series_old),
            "ps_new": _r6(p_series_new),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def append_ml_shadow_feature_traces(
    *,
    append_feature_trace_fn: Callable[..., Any],
    ff_ml_shadow_vnext: bool,
    trace_shadow_ml_vnext: Mapping[str, Any] | None,
    ff_ml_shadow_state_alite: bool,
    trace_shadow_state_alite: Mapping[str, Any] | None,
    ff_ml_shadow_state_areal: bool,
    trace_shadow_state_areal: Mapping[str, Any] | None,
    season_label: Any,
    teamA: str,
    teamB: str,
    bo: Any,
    ml_engine: Any,
    input_signature_hash: Any,
) -> Dict[str, bool]:
    emitted = {
        "ml_shadow_vnext_map_series_v1": False,
        "ml_shadow_state_alite_v1": False,
        "ml_shadow_state_areal_v1": False,
    }
    matchup = {
        "season": str(season_label),
        "teamA": str(teamA),
        "teamB": str(teamB),
        "bo": int(bo),
    }
    engine = str(ml_engine)
    sig = str(input_signature_hash)

    try:
        if bool(ff_ml_shadow_vnext):
            sv = _as_dict(trace_shadow_ml_vnext)
            append_feature_trace_fn(
                feature="ml_shadow_vnext_map_series_v1",
                event="shadow_compare",
                enabled=True,
                payload={
                    "matchup": matchup,
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(sv.get("applied", False)),
                    "reason": str(sv.get("reason", "") or ""),
                    "p_map_old": sv.get("p_map_old"),
                    "p_map_new": sv.get("p_map_new"),
                    "p_series_old": sv.get("p_series_old"),
                    "p_series_new": sv.get("p_series_new"),
                    "diff_pp_map": sv.get("diff_pp_map"),
                    "diff_pp_series": sv.get("diff_pp_series"),
                    "weights": (_as_dict(sv.get("weights"))),
                    "inputs": (_as_dict(sv.get("inputs"))),
                },
                dedupe_signature=_dedupe_signature(
                    input_signature_hash=sig,
                    ml_engine=engine,
                    p_map_old=sv.get("p_map_old"),
                    p_map_new=sv.get("p_map_new"),
                    p_series_old=sv.get("p_series_old"),
                    p_series_new=sv.get("p_series_new"),
                ),
                persist_files=False,
            )
            emitted["ml_shadow_vnext_map_series_v1"] = True
    except Exception:
        pass

    try:
        if bool(ff_ml_shadow_state_alite):
            ss = _as_dict(trace_shadow_state_alite)
            append_feature_trace_fn(
                feature="ml_shadow_state_alite_v1",
                event="shadow_compare",
                enabled=True,
                payload={
                    "matchup": matchup,
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(ss.get("applied", False)),
                    "reason": str(ss.get("reason", "") or ""),
                    "p_map_old": ss.get("p_map_old"),
                    "p_map_new": ss.get("p_map_new"),
                    "p_series_old": ss.get("p_series_old"),
                    "p_series_new": ss.get("p_series_new"),
                    "diff_pp_map": ss.get("diff_pp_map"),
                    "diff_pp_series": ss.get("diff_pp_series"),
                    "weights": (_as_dict(ss.get("weights"))),
                    "inputs": (_as_dict(ss.get("inputs"))),
                    "uncertainty": (_as_dict(ss.get("uncertainty"))),
                },
                dedupe_signature=_dedupe_signature(
                    input_signature_hash=sig,
                    ml_engine=engine,
                    p_map_old=ss.get("p_map_old"),
                    p_map_new=ss.get("p_map_new"),
                    p_series_old=ss.get("p_series_old"),
                    p_series_new=ss.get("p_series_new"),
                ),
                persist_files=False,
            )
            emitted["ml_shadow_state_alite_v1"] = True
    except Exception:
        pass

    try:
        if bool(ff_ml_shadow_state_areal):
            sr = _as_dict(trace_shadow_state_areal)
            append_feature_trace_fn(
                feature="ml_shadow_state_areal_v1",
                event="shadow_compare",
                enabled=True,
                payload={
                    "matchup": matchup,
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(sr.get("applied", False)),
                    "reason": str(sr.get("reason", "") or ""),
                    "p_map_old": sr.get("p_map_old"),
                    "p_map_new": sr.get("p_map_new"),
                    "p_series_old": sr.get("p_series_old"),
                    "p_series_new": sr.get("p_series_new"),
                    "diff_pp_map": sr.get("diff_pp_map"),
                    "diff_pp_series": sr.get("diff_pp_series"),
                    "weights": (_as_dict(sr.get("weights"))),
                    "inputs": (_as_dict(sr.get("inputs"))),
                    "state": (_as_dict(sr.get("state"))),
                    "qwl_trace": (_as_dict(sr.get("qwl_trace"))),
                },
                dedupe_signature=_dedupe_signature(
                    input_signature_hash=sig,
                    ml_engine=engine,
                    p_map_old=sr.get("p_map_old"),
                    p_map_new=sr.get("p_map_new"),
                    p_series_old=sr.get("p_series_old"),
                    p_series_new=sr.get("p_series_new"),
                ),
                persist_files=False,
            )
            emitted["ml_shadow_state_areal_v1"] = True
    except Exception:
        pass

    return emitted
