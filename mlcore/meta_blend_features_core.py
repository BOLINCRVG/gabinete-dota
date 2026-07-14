"""Meta-blend C layer — feature building.

Shared utilities, gate, constants and feature builder used by both
C-model (trained logistic) and C-lite (heuristic) paths.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from mlcore.utils import clip_prob as _clip_prob
from mlcore.utils import safe_float as _f
from mlcore.utils import logit as _safe_logit
from mlcore.league_context import (
    is_explicit_interregional_league_label as shared_is_explicit_interregional_league_label,
    league_macro_key,
)
from mlcore.tiers import league_strength_band, league_strength_score


def _isfinite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _norm_key(x: Any) -> str:
    return str(x or "").strip().upper()


def _is_explicit_interregional_league(v: Any) -> bool:
    return bool(shared_is_explicit_interregional_league_label(v))


def _macro_key(league: Any) -> str:
    return str(league_macro_key(league, unknown="OTHER"))


def _interregional_strength_profile(league_a: Any, league_b: Any) -> Dict[str, Any]:
    la = str(league_a or "").strip()
    lb = str(league_b or "").strip()
    out: Dict[str, Any] = {
        "league_a": la,
        "league_b": lb,
        "score_a": None,
        "score_b": None,
        "gap": 0.0,
        "stronger_side": "",
        "stronger_band": "",
        "weaker_band": "",
        "cross_strength": False,
        "very_strong_cross_strength": False,
    }
    if not la or not lb:
        return out
    sa = max(0.0, min(1.0, float(league_strength_score(la))))
    sb = max(0.0, min(1.0, float(league_strength_score(lb))))
    gap = abs(sa - sb)
    stronger_side = "a" if sa > sb else ("b" if sb > sa else "")
    stronger_band = str(league_strength_band(la if stronger_side == "a" else lb) or "")
    weaker_band = str(league_strength_band(lb if stronger_side == "a" else la) or "")
    cross_strength = bool(
        stronger_side
        and stronger_band in {"elite", "major"}
        and weaker_band in {"upper", "regional", "development", "local"}
        and gap >= 0.10
    )
    very_strong = bool(cross_strength and gap >= 0.16)
    out.update(
        {
            "score_a": float(sa),
            "score_b": float(sb),
            "gap": float(gap),
            "stronger_side": str(stronger_side),
            "stronger_band": str(stronger_band),
            "weaker_band": str(weaker_band),
            "cross_strength": bool(cross_strength),
            "very_strong_cross_strength": bool(very_strong),
        }
    )
    return out


def _patch_bucket_from_status(status: Any) -> str:
    s = str(status or "").strip().lower()
    if s in {"refit", "refit_needed", "changed"}:
        return "refit"
    if s in {"warn", "warning", "attention"}:
        return "warn"
    return "stable"


def _phase_bucket(*, bo: int, is_playoffs: bool) -> str:
    if bool(is_playoffs) or int(bo) >= 5:
        return "late"
    if int(bo) >= 3:
        return "mid"
    return "early"


def should_apply_meta_blend_c_model_gate(
    *,
    bo: Any = None,
    recent_evidence: Any = None,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    is_playoffs: Any = None,
    patch_status: str = "",
    interregional: bool = False,
) -> Dict[str, Any]:
    """Context gate for C-trained model (v3).

    Uses a deterministic evidence score so C can run beyond BO5-only freeze,
    while remaining conservative on low-signal BO1 spots.
    """
    try:
        bo_i = int(float(bo)) if bo is not None else 1
    except (TypeError, ValueError):
        bo_i = 1
    try:
        re = float(recent_evidence) if recent_evidence is not None else 0.0
    except (TypeError, ValueError):
        re = 0.0
    try:
        nt = float(n_team_eff) if n_team_eff is not None else 0.0
    except (TypeError, ValueError):
        nt = 0.0
    try:
        np_ = float(n_players_eff) if n_players_eff is not None else 0.0
    except (TypeError, ValueError):
        np_ = 0.0
    ps = str(patch_status or "").strip().lower()
    playoffs_b = bool(is_playoffs)
    inter_b = bool(interregional)

    # Evidence score in [0,1]: recent evidence + sample support + format context.
    e_recent = max(0.0, min(1.0, re))
    e_team = nt / (nt + 14.0) if nt > 0 else 0.0
    e_players = np_ / (np_ + 9.0) if np_ > 0 else 0.0
    score = 0.48 * e_recent + 0.32 * e_team + 0.14 * e_players + (0.06 if playoffs_b else 0.0)
    if inter_b:
        score *= 0.92
    if ps in {"refit", "refit_needed", "changed"}:
        score *= 0.88
    score = float(max(0.0, min(1.0, score)))

    if bo_i >= 5:
        thr = 0.24
    elif bo_i >= 3:
        thr = 0.34
    else:
        thr = 0.58
    if inter_b:
        if nt < 6.0 and np_ < 1.0:
            thr = max(thr, 0.58)
        elif nt < 10.0 and np_ < 2.0:
            thr = max(thr, 0.48)
        elif nt < 14.0:
            thr = max(thr, 0.40)
        if not playoffs_b:
            thr = max(thr, 0.44)
    allow = bool(score >= thr)
    if allow:
        reason = "allow_v3"
    elif inter_b and nt < 6.0 and np_ < 1.0:
        reason = "blocked_interregional_low_team_evidence_v3"
    else:
        reason = "blocked_low_evidence_v3"
    return {
        "allow": bool(allow),
        "reason": str(reason),
        "bo": int(bo_i),
        "score": float(score),
        "threshold": float(thr),
        "inputs": {
            "recent_evidence": float(e_recent),
            "n_team_eff": float(nt),
            "n_players_eff": float(np_),
            "playoffs": bool(playoffs_b),
            "interregional": bool(inter_b),
            "patch_status": str(ps),
        },
    }


META_BLEND_C_MODEL_FEATURES: List[str] = [
    "z_anchor",
    "z_adjusted",
    "abs_z_delta",
    "sign_z_delta",
    "log1p_n_team",
    "log1p_n_players",
    "recent_evidence",
    "rd_conf",
    "hybrid_agree",
    "bo_is_3",
    "bo_is_5",
    "is_playoffs",
    "interregional",
    "unknown_macro",
    "roster_changed",
    "patch_refit",
    "cons_guard",
    "cons_drift_scaled",
    "rank_sos_applied",
    "rank_sos_weight",
    "rank_sos_diff_scaled",
    "rank_sos_p_centered",
    "roster_cont_applied",
    "roster_cont_weight",
    "roster_cont_diff_scaled",
    "roster_cont_p_centered",
]


def build_meta_blend_c_model_features(
    *,
    p_anchor: Any,
    p_adjusted: Any,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    recent_evidence: Any = None,
    rd_conf: Any = None,
    hybrid_agree: Any = None,
    bo: Any = None,
    is_playoffs: Any = None,
    interregional: bool = False,
    unknown_macro: bool = False,
    roster_changed: bool = False,
    patch_status: str = "",
    consistency_guard_triggered: bool = False,
    consistency_guard_drift_pp: Any = None,
    rank_sos_applied: bool = False,
    rank_sos_weight: Any = None,
    rank_sos_diff_pp: Any = None,
    rank_sos_p: Any = None,
    roster_cont_applied: bool = False,
    roster_cont_weight: Any = None,
    roster_cont_diff_pp: Any = None,
    roster_cont_target_p: Any = None,
) -> Dict[str, Any]:
    if not (_isfinite(p_anchor) and _isfinite(p_adjusted)):
        return {"ok": False, "reason": "invalid_prob_inputs", "features": {}, "vector": []}

    pa = _clip_prob(_f(p_anchor))
    pb = _clip_prob(_f(p_adjusted))
    za = _safe_logit(pa)
    zb = _safe_logit(pb)
    dz = float(zb - za)

    ev_team = max(0.0, _f(n_team_eff, 0.0))
    ev_players = max(0.0, _f(n_players_eff, 0.0))
    ev_recent = _f(recent_evidence, 0.0)
    if not math.isfinite(ev_recent):
        ev_recent = 0.0
    ev_recent = max(0.0, min(1.0, ev_recent))

    rd_c = _f(rd_conf, 0.5)
    if not math.isfinite(rd_c):
        rd_c = 0.5
    rd_c = max(0.10, min(1.0, rd_c))

    h_agree = _f(hybrid_agree, 0.5)
    if not math.isfinite(h_agree):
        h_agree = 0.5
    h_agree = max(0.0, min(1.0, h_agree))

    try:
        bo_i = int(float(bo))
    except (TypeError, ValueError):
        bo_i = 1
    bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
    patch_s = str(patch_status or "").strip().lower()
    patch_refit = patch_s in {"refit", "refit_needed", "changed"}
    drift_pp = abs(_f(consistency_guard_drift_pp, 0.0))
    if not math.isfinite(drift_pp):
        drift_pp = 0.0
    cons_drift_scaled = max(0.0, min(1.0, drift_pp / 25.0))
    rs_w = _f(rank_sos_weight, 0.0)
    if not math.isfinite(rs_w):
        rs_w = 0.0
    rs_w = max(0.0, min(1.0, rs_w))
    rs_diff = abs(_f(rank_sos_diff_pp, 0.0))
    if not math.isfinite(rs_diff):
        rs_diff = 0.0
    rs_diff_scaled = max(0.0, min(1.0, rs_diff / 20.0))
    rs_p = _f(rank_sos_p, 0.5)
    if not math.isfinite(rs_p):
        rs_p = 0.5
    rs_p = max(0.0, min(1.0, rs_p))
    rc_w = _f(roster_cont_weight, 0.0)
    if not math.isfinite(rc_w):
        rc_w = 0.0
    rc_w = max(0.0, min(1.0, rc_w))
    rc_diff = abs(_f(roster_cont_diff_pp, 0.0))
    if not math.isfinite(rc_diff):
        rc_diff = 0.0
    rc_diff_scaled = max(0.0, min(1.0, rc_diff / 20.0))
    rc_p = _f(roster_cont_target_p, 0.5)
    if not math.isfinite(rc_p):
        rc_p = 0.5
    rc_p = max(0.0, min(1.0, rc_p))

    f: Dict[str, float] = {
        "z_anchor": float(za),
        "z_adjusted": float(zb),
        "abs_z_delta": float(abs(dz)),
        "sign_z_delta": float(1.0 if dz >= 0 else -1.0),
        "log1p_n_team": float(math.log1p(ev_team)),
        "log1p_n_players": float(math.log1p(ev_players)),
        "recent_evidence": float(ev_recent),
        "rd_conf": float(rd_c),
        "hybrid_agree": float(h_agree),
        "bo_is_3": float(1.0 if bo_i == 3 else 0.0),
        "bo_is_5": float(1.0 if bo_i == 5 else 0.0),
        "is_playoffs": float(1.0 if bool(is_playoffs) else 0.0),
        "interregional": float(1.0 if bool(interregional) else 0.0),
        "unknown_macro": float(1.0 if bool(unknown_macro) else 0.0),
        "roster_changed": float(1.0 if bool(roster_changed) else 0.0),
        "patch_refit": float(1.0 if bool(patch_refit) else 0.0),
        "cons_guard": float(1.0 if bool(consistency_guard_triggered) else 0.0),
        "cons_drift_scaled": float(cons_drift_scaled),
        "rank_sos_applied": float(1.0 if bool(rank_sos_applied) else 0.0),
        "rank_sos_weight": float(rs_w),
        "rank_sos_diff_scaled": float(rs_diff_scaled),
        "rank_sos_p_centered": float(rs_p - 0.5),
        "roster_cont_applied": float(1.0 if bool(roster_cont_applied) else 0.0),
        "roster_cont_weight": float(rc_w),
        "roster_cont_diff_scaled": float(rc_diff_scaled),
        "roster_cont_p_centered": float(rc_p - 0.5),
    }
    vec = [float(f.get(k, 0.0)) for k in META_BLEND_C_MODEL_FEATURES]
    return {
        "ok": True,
        "reason": "",
        "features": f,
        "vector": vec,
        "meta": {"p_anchor": float(pa), "p_adjusted": float(pb), "z_delta": float(dz), "bo": int(bo_i)},
    }
