"""Guardrail helpers for Resumo tab."""

from __future__ import annotations

import math
from typing import Any, Mapping


def extract_resumo_operational_confidence(ml_trace_last: Mapping[str, Any] | None) -> tuple[float | None, float | None, float | None]:
    score = None
    marginal = None
    layer = None
    try:
        ml_trace_obj = ((ml_trace_last or {}).get("ml_trace") or {})

        raw_marg = ((ml_trace_obj.get("marginal_trust_guard") or {}).get("confidence_score"))
        if raw_marg is not None:
            marg = float(raw_marg)
            if math.isfinite(marg):
                marginal = float(marg)

        raw_layer = ((ml_trace_obj.get("layer_uncertainty_live") or {}).get("confidence_score"))
        if raw_layer is not None:
            lay = float(raw_layer)
            if math.isfinite(lay):
                if 0.0 <= lay <= 1.0:
                    lay = 100.0 * lay
                layer = float(lay)

        vals = []
        if marginal is not None:
            vals.append(float(marginal))
        if layer is not None:
            vals.append(float(layer))
        if vals:
            score = float(sum(vals) / float(len(vals)))
    except Exception:
        score = None
        marginal = None
        layer = None
    return score, marginal, layer


def build_resumo_guardrail_cfg(
    session_state: Mapping[str, Any],
    *,
    ml_trace_last: Mapping[str, Any] | None,
) -> dict[str, Any]:
    score, marginal, layer = extract_resumo_operational_confidence(ml_trace_last)
    return {
        "profile": str(session_state.get("resumo_guardrail_profile", "Equilibrado") or "Equilibrado"),
        "strength_major": float(session_state.get("resumo_guardrail_strength_major", 1.0) or 1.0),
        "strength_mid": float(session_state.get("resumo_guardrail_strength_mid", 1.0) or 1.0),
        "strength_minor": float(session_state.get("resumo_guardrail_strength_minor", 1.0) or 1.0),
        "target_major": float(session_state.get("resumo_guardrail_target_major", 1.0) or 1.0),
        "target_mid": float(session_state.get("resumo_guardrail_target_mid", 1.0) or 1.0),
        "target_minor": float(session_state.get("resumo_guardrail_target_minor", 1.0) or 1.0),
        "operational_confidence_score": score,
        "operational_confidence_marginal": marginal,
        "operational_confidence_layer": layer,
        "operational_confidence_weight": float(session_state.get("resumo_operational_confidence_weight", 0.25) or 0.25),
    }
