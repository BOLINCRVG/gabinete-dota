from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from mlcore.tiers import league_strength_score, tier_for_league, tier_priority


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return float(out) if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _odd_from_prob(prob: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(prob)))
    return float(1.0 / p)


def seed_pricing_distortion_cases() -> list[dict[str, Any]]:
    return [
        {"name": "JDG_vs_LOUD", "team_a": "JD Gaming", "team_b": "LOUD", "bo": 5},
        {"name": "GEN_vs_LOUD", "team_a": "Gen.G", "team_b": "LOUD", "bo": 5},
        {"name": "T1_vs_LOUD", "team_a": "T1", "team_b": "LOUD", "bo": 5},
        {"name": "BLG_vs_FURIA", "team_a": "Bilibili Gaming", "team_b": "FURIA", "bo": 5},
        {"name": "G2_vs_paiN", "team_a": "G2 Esports", "team_b": "paiN Gaming", "bo": 5},
        {"name": "CFO_vs_GAM", "team_a": "CTBC Flying Oyster", "team_b": "GAM Esports", "bo": 3},
        {"name": "GEN_vs_T1", "team_a": "Gen.G", "team_b": "T1", "bo": 5},
        {"name": "HLE_vs_DK", "team_a": "Hanwha Life Esports", "team_b": "Dplus Kia", "bo": 5},
    ]


def merge_pricing_distortion_cases(
    base_cases: Iterable[Mapping[str, Any]] | None,
    extra_cases: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for raw in list(base_cases or []) + list(extra_cases or []):
        team_a = str((raw or {}).get("team_a") or (raw or {}).get("teamA") or "").strip()
        team_b = str((raw or {}).get("team_b") or (raw or {}).get("teamB") or "").strip()
        bo = int((raw or {}).get("bo") or 5)
        if not team_a or not team_b:
            continue
        key = (team_a.casefold(), team_b.casefold(), int(bo))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "name": str((raw or {}).get("name") or f"{team_a}_vs_{team_b}"),
                "team_a": team_a,
                "team_b": team_b,
                "bo": int(bo),
            }
        )
    return rows


def pick_reference_probability(trace_payload: Mapping[str, Any] | None) -> tuple[float, str]:
    probs = ((trace_payload or {}).get("probabilities") or {}) if isinstance(trace_payload, Mapping) else {}
    if not isinstance(probs, Mapping):
        return float("nan"), ""
    for key in ("p_fused", "p_cal", "p_team", "p_raw"):
        value = _f(probs.get(key), float("nan"))
        if math.isfinite(value):
            return float(value), str(key)
    return float("nan"), ""


def stronger_side_from_leagues(league_a: Any, league_b: Any) -> dict[str, Any]:
    la = str(league_a or "").strip()
    lb = str(league_b or "").strip()
    score_a = float(league_strength_score(la))
    score_b = float(league_strength_score(lb))
    gap = abs(score_a - score_b)
    tier_a = str(tier_for_league(la) or "")
    tier_b = str(tier_for_league(lb) or "")
    pri_a = int(tier_priority(tier_a))
    pri_b = int(tier_priority(tier_b))
    tier_gap = abs(pri_a - pri_b)
    stronger_side = ""
    if gap >= 1e-6:
        stronger_side = "a" if score_a > score_b else "b"
    return {
        "league_a": la,
        "league_b": lb,
        "tier_a": tier_a,
        "tier_b": tier_b,
        "score_a": float(score_a),
        "score_b": float(score_b),
        "strength_gap": float(gap),
        "tier_gap": int(tier_gap),
        "stronger_side": stronger_side,
    }


def perspective_probability(p_team_a: Any, side: str) -> float:
    p_a = _f(p_team_a, float("nan"))
    if not math.isfinite(p_a):
        return float("nan")
    side_norm = str(side or "").strip().lower()
    if side_norm == "a":
        return float(p_a)
    if side_norm == "b":
        return float(1.0 - p_a)
    return float("nan")


def extract_top_adverse_moves(
    pipeline_steps: Any,
    *,
    stronger_side: str,
    limit: int = 5,
    min_pp: float = 0.25,
) -> list[dict[str, Any]]:
    if str(stronger_side or "").strip().lower() not in {"a", "b"}:
        return []
    rows: list[dict[str, Any]] = []
    sign = 1.0 if str(stronger_side).strip().lower() == "a" else -1.0
    if not isinstance(pipeline_steps, list):
        return []
    for step in pipeline_steps:
        if not isinstance(step, Mapping):
            continue
        before = _f(step.get("before"), float("nan"))
        after = _f(step.get("after"), float("nan"))
        if not (math.isfinite(before) and math.isfinite(after)):
            continue
        delta_pp = float(100.0 * sign * (after - before))
        adverse_pp = float(max(0.0, -delta_pp))
        if adverse_pp < float(min_pp):
            continue
        rows.append(
            {
                "stage": str(step.get("stage", "") or ""),
                "before": float(before),
                "after": float(after),
                "delta_pp": float(delta_pp),
                "adverse_pp": float(adverse_pp),
                "applied": bool(step.get("applied", False)),
            }
        )
    rows.sort(key=lambda row: (-float(row.get("adverse_pp", 0.0)), str(row.get("stage", ""))))
    return rows[: max(1, int(limit))]


def _expected_base_floor(strength_gap: float, tier_gap: int) -> float:
    floor = 0.50 + min(0.12, float(strength_gap) * 0.35)
    if int(tier_gap) >= 1:
        floor += 0.015
    if int(tier_gap) >= 2:
        floor += 0.010
    return float(min(0.72, max(0.52, floor)))


def _expected_final_floor(strength_gap: float, tier_gap: int) -> float:
    floor = 0.50 + min(0.16, float(strength_gap) * 0.42)
    if int(tier_gap) >= 1:
        floor += 0.020
    if int(tier_gap) >= 2:
        floor += 0.015
    return float(min(0.76, max(0.53, floor)))


def _compression_budget_pp(strength_gap: float, tier_gap: int) -> float:
    budget = 9.0 - min(4.5, float(strength_gap) * 18.0)
    if int(tier_gap) >= 1:
        budget -= 1.5
    if int(tier_gap) >= 2:
        budget -= 1.0
    return float(max(3.0, budget))


def evaluate_pricing_distortion(
    *,
    team_a: str,
    team_b: str,
    league_a: str,
    league_b: str,
    trace_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trace_map = dict(trace_payload or {}) if isinstance(trace_payload, Mapping) else {}
    prob_ref, prob_ref_source = pick_reference_probability(trace_map)
    probs = trace_map.get("probabilities") or {}
    pipeline = trace_map.get("pipeline") or {}
    pipeline_steps = list(pipeline.get("steps", [])) if isinstance(pipeline, Mapping) else []
    p_final_a = _f((probs or {}).get("p_map_used"), float("nan"))

    strength = stronger_side_from_leagues(league_a, league_b)
    stronger_side = str(strength.get("stronger_side") or "")
    stronger_team = str(team_a if stronger_side == "a" else (team_b if stronger_side == "b" else ""))
    weaker_team = str(team_b if stronger_side == "a" else (team_a if stronger_side == "b" else ""))
    p_base_stronger = perspective_probability(prob_ref, stronger_side)
    p_final_stronger = perspective_probability(p_final_a, stronger_side)
    compression_pp = (
        float(100.0 * (p_base_stronger - p_final_stronger))
        if math.isfinite(p_base_stronger) and math.isfinite(p_final_stronger)
        else float("nan")
    )
    base_floor = _expected_base_floor(float(strength.get("strength_gap", 0.0)), int(strength.get("tier_gap", 0)))
    final_floor = _expected_final_floor(float(strength.get("strength_gap", 0.0)), int(strength.get("tier_gap", 0)))
    budget_pp = _compression_budget_pp(float(strength.get("strength_gap", 0.0)), int(strength.get("tier_gap", 0)))
    top_moves = extract_top_adverse_moves(pipeline_steps, stronger_side=stronger_side)

    reasons: list[str] = []
    problem_class = "ok"
    if not stronger_side:
        problem_class = "same_strength"
    elif not (math.isfinite(p_base_stronger) and math.isfinite(p_final_stronger)):
        problem_class = "insufficient_trace"
        reasons.append("trace_incompleto")
    else:
        base_low = bool(
            float(strength.get("strength_gap", 0.0)) >= 0.12
            and p_base_stronger < float(base_floor)
        )
        final_low = bool(
            float(strength.get("strength_gap", 0.0)) >= 0.12
            and p_final_stronger < float(final_floor)
        )
        reversal = bool(
            float(strength.get("strength_gap", 0.0)) >= 0.10
            and p_final_stronger < 0.50
        )
        overcompressed = bool(
            math.isfinite(compression_pp)
            and float(strength.get("strength_gap", 0.0)) >= 0.12
            and compression_pp > float(budget_pp)
            and p_base_stronger > 0.53
        )
        if reversal:
            reasons.append("reversao_para_o_lado_mais_fraco")
        if base_low:
            reasons.append("base_ja_sai_fraca_para_o_gap")
        if final_low:
            reasons.append("final_abaixo_do_piso_esperado")
        if overcompressed:
            reasons.append("compressao_excessiva_pos_camadas")
        if top_moves and float(top_moves[0].get("adverse_pp", 0.0)) >= max(3.5, float(budget_pp) * 0.75):
            reasons.append(f"movimento_forte_em_{top_moves[0].get('stage', '')!s}")

        if reasons:
            if base_low and overcompressed:
                problem_class = "mixed"
            elif base_low:
                problem_class = "base_underpowered"
            elif overcompressed or final_low or reversal:
                problem_class = "post_layers_overcompressed"
            else:
                problem_class = "review"

    return {
        "team_a": str(team_a),
        "team_b": str(team_b),
        "league_a": str(league_a or ""),
        "league_b": str(league_b or ""),
        "tier_a": str(strength.get("tier_a") or ""),
        "tier_b": str(strength.get("tier_b") or ""),
        "strength_score_a": float(strength.get("score_a", 0.0)),
        "strength_score_b": float(strength.get("score_b", 0.0)),
        "strength_gap": float(strength.get("strength_gap", 0.0)),
        "tier_gap": int(strength.get("tier_gap", 0)),
        "stronger_side": stronger_side,
        "stronger_team": stronger_team,
        "weaker_team": weaker_team,
        "reference_prob_source": str(prob_ref_source or ""),
        "p_base_a": float(prob_ref) if math.isfinite(prob_ref) else None,
        "p_final_a": float(p_final_a) if math.isfinite(p_final_a) else None,
        "p_base_stronger": float(p_base_stronger) if math.isfinite(p_base_stronger) else None,
        "p_final_stronger": float(p_final_stronger) if math.isfinite(p_final_stronger) else None,
        "odd_base_stronger": (_odd_from_prob(p_base_stronger) if math.isfinite(p_base_stronger) else None),
        "odd_final_stronger": (_odd_from_prob(p_final_stronger) if math.isfinite(p_final_stronger) else None),
        "base_floor": float(base_floor),
        "final_floor": float(final_floor),
        "compression_budget_pp": float(budget_pp),
        "compression_pp": float(compression_pp) if math.isfinite(compression_pp) else None,
        "problem_class": str(problem_class),
        "status": ("review" if reasons else "ok"),
        "reasons": list(reasons),
        "top_adverse_moves": list(top_moves),
    }
