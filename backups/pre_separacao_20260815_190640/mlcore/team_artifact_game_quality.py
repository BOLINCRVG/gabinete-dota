"""Game-quality signal computation for team state artifacts.

Extracted from state_team_artifact to keep the main module focused on
state management and Bayes updates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from mlcore.utils import safe_float as _f, clip as _clip


def _norm_margin(v: Any, scale: float) -> float:
    x = _f(v, float("nan"))
    if not math.isfinite(x) or scale <= 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, x / scale)))


def _row_blue_red_diff(row: Any, stem: str) -> float:
    bv = _f(getattr(row, f"blue_{stem}", float("nan")))
    rv = _f(getattr(row, f"red_{stem}", float("nan")))
    if math.isfinite(bv) and math.isfinite(rv):
        return float(bv - rv)
    if math.isfinite(bv):
        return float(bv)
    if math.isfinite(rv):
        return float(-rv)
    return float("nan")


def _data_completeness_state(v: Any) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return "unknown"
    if s in {"complete", "full", "1", "true", "yes"}:
        return "complete"
    if s in {"partial", "partially_complete", "incomplete", "0", "false", "no"}:
        return "partial"
    return "unknown"


def _data_completeness_weight(state: str) -> float:
    st = str(state or "").strip().lower()
    if st == "complete":
        return 1.00
    if st == "partial":
        return 0.82
    return 0.92


def _row_data_reliability(row: Any) -> Tuple[float, str, str]:
    blue_state = _data_completeness_state(getattr(row, "blue_datacompleteness", None))
    red_state = _data_completeness_state(getattr(row, "red_datacompleteness", None))
    bw = _data_completeness_weight(blue_state)
    rw = _data_completeness_weight(red_state)
    reliability = float(min(bw, rw))
    return float(_clip(reliability, 0.70, 1.00)), str(blue_state), str(red_state)


def _game_quality_blue(
    row: Any,
    *,
    tempo_weight: float = 0.08,
    conversion_weight: float = 0.12,
    throw_weight: float = 1.20,
    comeback_weight: float = 0.65,
    phase_dynamic_strength: float = 0.35,
) -> Tuple[float, float, float, Dict[str, Any]]:
    # Quality signal in [-1, +1] from blue perspective, using only post-game
    # stats of already played matches (causal for future predictions).
    #
    # Returns (q_total, q_tempo_term, q_conversion_term) to keep trace auditable.
    # - q_tempo_term: fast/slow closeout quality conditioned on winner side.
    # - q_conversion_term: convert-lead / throw / comeback quality from early edge.
    kill_diff = _row_blue_red_diff(row, "teamkills")
    death_diff = _row_blue_red_diff(row, "teamdeaths")
    tower_diff = _row_blue_red_diff(row, "towers")
    dragon_diff = _row_blue_red_diff(row, "dragons")
    elemental_diff = _row_blue_red_diff(row, "elementaldrakes")
    baron_diff = _row_blue_red_diff(row, "barons")
    herald_diff = _row_blue_red_diff(row, "heralds")
    grub_diff = _row_blue_red_diff(row, "voidgrubs")
    atakhan_diff = _row_blue_red_diff(row, "atakhans")
    plate_diff = _row_blue_red_diff(row, "turretplates")
    inhib_diff = _row_blue_red_diff(row, "inhibitors")

    gd10_diff = _row_blue_red_diff(row, "golddiffat10")
    gd15_diff = _row_blue_red_diff(row, "golddiffat15")
    gd20_diff = _row_blue_red_diff(row, "golddiffat20")
    xp10_diff = _row_blue_red_diff(row, "xpdiffat10")
    xp15_diff = _row_blue_red_diff(row, "xpdiffat15")
    xp20_diff = _row_blue_red_diff(row, "xpdiffat20")
    cs10_diff = _row_blue_red_diff(row, "csdiffat10")
    cs15_diff = _row_blue_red_diff(row, "csdiffat15")
    cs20_diff = _row_blue_red_diff(row, "csdiffat20")

    fb = _row_blue_red_diff(row, "firstblood")
    fd = _row_blue_red_diff(row, "firstdragon")
    fbr = _row_blue_red_diff(row, "firstbaron")
    fh = _row_blue_red_diff(row, "firstherald")
    ft = _row_blue_red_diff(row, "firsttower")
    fmt = _row_blue_red_diff(row, "firstmidtower")
    f3t = _row_blue_red_diff(row, "firsttothreetowers")
    first_obj = 0.0
    first_obj += 0.12 * (1.0 if fb > 0 else (-1.0 if fb < 0 else 0.0))
    first_obj += 0.16 * (1.0 if fd > 0 else (-1.0 if fd < 0 else 0.0))
    first_obj += 0.22 * (1.0 if fbr > 0 else (-1.0 if fbr < 0 else 0.0))
    first_obj += 0.08 * (1.0 if fh > 0 else (-1.0 if fh < 0 else 0.0))
    first_obj += 0.16 * (1.0 if ft > 0 else (-1.0 if ft < 0 else 0.0))
    first_obj += 0.10 * (1.0 if fmt > 0 else (-1.0 if fmt < 0 else 0.0))
    first_obj += 0.16 * (1.0 if f3t > 0 else (-1.0 if f3t < 0 else 0.0))
    first_obj = float(_clip(first_obj, -1.0, 1.0))

    early_vals: List[Tuple[float, float]] = []
    if math.isfinite(gd10_diff):
        early_vals.append((_norm_margin(gd10_diff, 3000.0), 0.18))
    if math.isfinite(gd15_diff):
        early_vals.append((_norm_margin(gd15_diff, 4500.0), 0.24))
    if math.isfinite(gd20_diff):
        early_vals.append((_norm_margin(gd20_diff, 6000.0), 0.20))
    if math.isfinite(xp10_diff):
        early_vals.append((_norm_margin(xp10_diff, 2500.0), 0.10))
    if math.isfinite(xp15_diff):
        early_vals.append((_norm_margin(xp15_diff, 3500.0), 0.13))
    if math.isfinite(xp20_diff):
        early_vals.append((_norm_margin(xp20_diff, 4500.0), 0.07))
    if math.isfinite(cs10_diff):
        early_vals.append((_norm_margin(cs10_diff, 35.0), 0.05))
    if math.isfinite(cs15_diff):
        early_vals.append((_norm_margin(cs15_diff, 55.0), 0.07))
    if math.isfinite(cs20_diff):
        early_vals.append((_norm_margin(cs20_diff, 75.0), 0.04))
    if early_vals:
        _w_sum = float(sum(w for _, w in early_vals))
        q_early = float(sum(v * w for v, w in early_vals) / max(1e-9, _w_sum))
    else:
        q_early = 0.0

    dragon_ref = float("nan")
    dragon_source = "none"
    if math.isfinite(dragon_diff):
        dragon_ref = float(dragon_diff)
        dragon_source = "dragons"
    elif math.isfinite(elemental_diff):
        dragon_ref = float(elemental_diff)
        dragon_source = "elementaldrakes"

    q_objective = 0.0
    q_objective += 0.29 * _norm_margin(tower_diff, 9.0)
    q_objective += 0.20 * _norm_margin(dragon_ref, 4.0)
    q_objective += 0.17 * _norm_margin(baron_diff, 2.0)
    q_objective += 0.10 * _norm_margin(herald_diff, 2.0)
    q_objective += 0.05 * _norm_margin(grub_diff, 6.0)
    q_objective += 0.03 * _norm_margin(atakhan_diff, 1.0)
    q_objective += 0.08 * _norm_margin(plate_diff, 10.0)
    q_objective += 0.08 * _norm_margin(inhib_diff, 3.0)
    q_objective = float(_clip(q_objective, -1.0, 1.0))

    # Combined pressure from kills and deaths (if deaths unavailable, it gracefully degrades).
    q_kill = float(0.70 * _norm_margin(kill_diff, 16.0) + 0.30 * _norm_margin(-death_diff, 16.0))
    q_kill = float(_clip(q_kill, -1.0, 1.0))

    # Base control/margin quality (pre tempo/conversion terms).
    # Dynamic phase weighting:
    # - shorter games increase early/first-objective contribution
    # - longer games increase kill/objective conversion contribution
    gl = _f(getattr(row, "gamelength", float("nan")))
    late_phase = 0.5
    if math.isfinite(gl) and gl > 0.0:
        # ~25min -> early-lean, ~45min+ -> late-lean
        late_phase = float(_clip((gl - 1500.0) / 1200.0, 0.0, 1.0))
    phase_dyn = float(_clip(float(phase_dynamic_strength), 0.0, 0.80))
    w_early = 0.34 * (1.0 + phase_dyn * (0.5 - late_phase))
    w_objective = 0.28 * (1.0 + phase_dyn * (abs(late_phase - 0.5) * 0.30))
    w_kill = 0.24 * (1.0 + phase_dyn * (late_phase - 0.5))
    w_first = 0.14 * (1.0 + phase_dyn * (0.5 - late_phase))
    w_sum = float(max(1e-9, w_early + w_objective + w_kill + w_first))
    q = 0.0
    q += float((w_early / w_sum) * q_early)
    q += float((w_objective / w_sum) * q_objective)
    q += float((w_kill / w_sum) * q_kill)
    q += float((w_first / w_sum) * float(first_obj))

    # Winner sign from blue perspective: +1 if blue won, -1 otherwise.
    try:
        y_blue = 1.0 if int(_f(getattr(row, "blue_win", 0.0), 0.0)) == 1 else -1.0
    except (TypeError, ValueError, OverflowError):
        y_blue = 1.0 if kill_diff >= 0.0 else -1.0

    # Tempo/closeout quality: fast wins and long losses (from blue perspective) are less noisy.
    q_tempo = 0.0
    if math.isfinite(gl) and gl > 300.0:
        tempo_raw = _norm_margin(2100.0 - gl, 1200.0)  # ~35m baseline
        q_tempo = float(y_blue * tempo_raw)

    # Conversion quality from early edge:
    # - winning with lead: strong positive
    # - losing with lead: throw penalty
    # - winning from deficit: comeback bonus (smaller than lead conversion)
    q_conv = 0.0
    q_throw = 0.0
    q_comeback = 0.0
    lead = float("nan")
    lead_source = ""
    if math.isfinite(gd20_diff):
        lead = _norm_margin(gd20_diff, 6000.0)
        lead_source = "gold20"
    elif math.isfinite(gd15_diff):
        lead = _norm_margin(gd15_diff, 4500.0)
        lead_source = "gold15"
    elif math.isfinite(gd10_diff):
        lead = _norm_margin(gd10_diff, 3000.0)
        lead_source = "gold10"
    elif math.isfinite(xp20_diff):
        lead = _norm_margin(xp20_diff, 4500.0)
        lead_source = "xp20"
    elif math.isfinite(xp15_diff):
        lead = _norm_margin(xp15_diff, 3500.0)
        lead_source = "xp15"
    elif math.isfinite(xp10_diff):
        lead = _norm_margin(xp10_diff, 2500.0)
        lead_source = "xp10"
    elif math.isfinite(cs20_diff):
        lead = _norm_margin(cs20_diff, 75.0)
        lead_source = "cs20"
    elif math.isfinite(cs15_diff):
        lead = _norm_margin(cs15_diff, 55.0)
        lead_source = "cs15"
    elif math.isfinite(cs10_diff):
        lead = _norm_margin(cs10_diff, 35.0)
        lead_source = "cs10"
    throw_w = float(_clip(float(throw_weight), 0.60, 1.80))
    comeback_w = float(_clip(float(comeback_weight), 0.30, 1.20))
    if math.isfinite(lead):
        lead_abs = abs(float(lead))
        if lead_abs > 0.0:
            if (lead * y_blue) > 0.0:
                q_conv = float(lead_abs)
            elif y_blue > 0.0:
                q_conv = float(comeback_w * lead_abs)
                q_comeback = float(comeback_w * lead_abs)
            else:
                q_conv = float(-throw_w * lead_abs)
                q_throw = float(-throw_w * lead_abs)
    q_conv = float(_clip(q_conv, -1.0, 1.0))
    q_throw = float(_clip(q_throw, -1.0, 0.0))
    q_comeback = float(_clip(q_comeback, 0.0, 1.0))

    tw = float(_clip(float(tempo_weight), 0.0, 0.25))
    cw = float(_clip(float(conversion_weight), 0.0, 0.30))
    q += float(tw * q_tempo)
    q += float(cw * q_conv)

    q_data_reliability, q_data_blue_state, q_data_red_state = _row_data_reliability(row)
    q *= float(q_data_reliability)
    q_tempo *= float(q_data_reliability)
    q_conv *= float(q_data_reliability)
    q_throw *= float(q_data_reliability)
    q_comeback *= float(q_data_reliability)

    q = float(_clip(q, -1.0, 1.0))
    terms: Dict[str, Any] = {
        "q_early_term": float(q_early),
        "q_objective_term": float(q_objective),
        "q_kill_term": float(q_kill),
        "q_first_obj_term": float(first_obj),
        "q_tempo_term": float(q_tempo),
        "q_conversion_term": float(q_conv),
        "q_throw_term": float(q_throw),
        "q_comeback_term": float(q_comeback),
        "q_lead_ref": float(lead if math.isfinite(lead) else 0.0),
        "q_lead_source": str(lead_source or "none"),
        "q_late_phase": float(late_phase),
        "q_phase_dynamic_strength": float(phase_dyn),
        "q_weight_early": float(w_early / w_sum),
        "q_weight_objective": float(w_objective / w_sum),
        "q_weight_kill": float(w_kill / w_sum),
        "q_weight_first_obj": float(w_first / w_sum),
        "q_throw_weight_eff": float(throw_w),
        "q_comeback_weight_eff": float(comeback_w),
        "q_data_reliability": float(q_data_reliability),
        "q_data_completeness_blue": str(q_data_blue_state),
        "q_data_completeness_red": str(q_data_red_state),
        "q_objective_dragon_ref": float(dragon_ref if math.isfinite(dragon_ref) else 0.0),
        "q_objective_dragon_source": str(dragon_source),
    }
    return float(q), float(q_tempo), float(q_conv), terms


def _expected_quality_for_winner(
    p_winner: float,
    *,
    base: float,
    slope: float,
) -> float:
    pw = float(_clip(float(p_winner), 0.0, 1.0))
    # Favorite wins are expected to be more decisive than underdog wins.
    fav_strength = float(max(0.0, (pw - 0.5) * 2.0))
    exp_q = float(float(base) + float(slope) * fav_strength)
    return float(_clip(exp_q, 0.0, 0.95))


def _qwl_adjusted_signal(
    q_winner_pos: float,
    p_winner: float,
    *,
    expected_base: float,
    expected_slope: float,
    surplus_mix: float,
    opponent_context_mix: float = 0.0,
) -> Tuple[float, float, float, float]:
    q_pos = float(_clip(float(q_winner_pos), 0.0, 1.0))
    mix = float(_clip(float(surplus_mix), 0.0, 1.0))
    q_expected = _expected_quality_for_winner(
        float(p_winner),
        base=float(expected_base),
        slope=float(expected_slope),
    )
    q_surplus = float(max(0.0, q_pos - q_expected))
    ctx_mix = float(_clip(float(opponent_context_mix), 0.0, 1.0))
    pw = float(_clip(float(p_winner), 0.0, 1.0))
    underdog_strength = float(max(0.0, (0.5 - pw) * 2.0))
    favorite_strength = float(max(0.0, (pw - 0.5) * 2.0))
    # Underdog decisive wins carry more information than expected favorite wins.
    ctx_mult = float(1.0 + (ctx_mix * underdog_strength) - (0.5 * ctx_mix * favorite_strength))
    ctx_mult = float(_clip(ctx_mult, 0.60, 1.40))
    q_surplus_ctx = float(_clip(q_surplus * ctx_mult, 0.0, 1.0))
    q_adjusted = float((1.0 - mix) * q_pos + mix * q_surplus_ctx)
    return float(q_adjusted), float(q_expected), float(q_surplus_ctx), float(ctx_mult)
