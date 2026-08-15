from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict

from mlcore.utils import clip_prob as _clip_prob
from mlcore.utils import safe_float as _f
from mlcore.utils import logit as _logit
from mlcore.utils import sigmoid as _sigmoid
from mlcore.bo import p_series_from_pmap
from mlcore.league_context import league_macro_key

def _series_target_wins(bo: int) -> int:
    b = 5 if int(bo) >= 5 else (3 if int(bo) >= 3 else 1)
    return (b // 2) + 1


def enforce_series_economic_coherence(
    *,
    p_map: Any,
    p_series: Any,
    bo: Any,
    max_dev_pp: float | None = None,
) -> Dict[str, Any]:
    """Enforce economic coherence between map and series probability.

    Guardrails:
    - BO3/BO5 cannot flip the favorite side from BO1 (map-level p_map).
    - BO3/BO5 should be at least as decisive as BO1 toward 50% away.
    - Series probability cannot drift too far from combinatorial baseline for
      the same p_map (bounded by max_dev_pp, default by BO).
    - For BO5, probability should not be less decisive than BO3 baseline.
    """
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "bo": None,
        "p_map": None,
        "p_series_before": None,
        "p_series_after": None,
        "p_target": None,
        "bounds": {"low": None, "high": None, "max_dev_pp": None},
        "diff_pp": None,
    }
    pm = _f(p_map)
    ps = _f(p_series)
    if not math.isfinite(pm) or not math.isfinite(ps):
        out["reason"] = "invalid_inputs"
        return out

    try:
        bo_i = int(float(bo))
    except Exception:
        bo_i = 1
    bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
    out["bo"] = int(bo_i)
    out["p_map"] = float(_clip_prob(pm))
    out["p_series_before"] = float(_clip_prob(ps))

    if bo_i <= 1:
        out["reason"] = "bo1_noop"
        out["p_target"] = float(out["p_map"])
        out["p_series_after"] = float(out["p_series_before"])
        out["diff_pp"] = 0.0
        return out

    p0 = float(out["p_map"])
    p_in = float(out["p_series_before"])
    p_target = float(_clip_prob(p_series_from_pmap(p0, bo_i)))
    p_target_bo3 = float(_clip_prob(p_series_from_pmap(p0, 3)))

    cap_pp = float(max_dev_pp) if max_dev_pp is not None else (16.0 if bo_i == 3 else 20.0)
    cap_pp = max(2.0, min(35.0, float(cap_pp)))
    cap = cap_pp / 100.0

    low = max(1e-6, p_target - cap)
    high = min(1.0 - 1e-6, p_target + cap)

    # Preserve favorite side and decisiveness vs BO1.
    if p0 > 0.5:
        low = max(low, p0)
    elif p0 < 0.5:
        high = min(high, p0)

    # BO5 should not be less decisive than BO3 baseline for same p_map.
    if bo_i >= 5:
        if p0 > 0.5:
            low = max(low, p_target_bo3)
        elif p0 < 0.5:
            high = min(high, p_target_bo3)

    # Safety if bounds cross due strict constraints.
    if low > high:
        mid = float(_clip_prob((low + high) / 2.0))
        low = mid
        high = mid

    p_out = float(max(low, min(high, p_in)))
    out["p_target"] = float(p_target)
    out["bounds"] = {"low": float(low), "high": float(high), "max_dev_pp": float(cap_pp)}
    out["p_series_after"] = float(p_out)
    out["diff_pp"] = float(100.0 * (p_out - p_in))

    if abs(float(out["diff_pp"])) <= 1e-12:
        out["reason"] = "already_coherent"
        return out

    out["applied"] = True
    out["reason"] = "clamped_to_economic_coherence"
    return out


def _series_win_prob_dp(
    *,
    bo: int,
    p_blue: float,
    p_red: float,
    q_blue_base: float,
    pick_shift: float,
) -> float:
    target = _series_target_wins(int(bo))
    max_maps = int(bo)
    pb = _clip_prob(float(p_blue))
    pr = _clip_prob(float(p_red))
    qb = max(0.0, min(1.0, float(q_blue_base)))
    ps = max(0.0, min(0.25, float(pick_shift)))

    @lru_cache(maxsize=None)
    def _rec(a: int, b: int, m: int, last: int) -> float:
        if a >= target:
            return 1.0
        if b >= target:
            return 0.0
        if m >= max_maps:
            return 1.0 if a > b else 0.0
        q = qb
        if last == 1:
            q -= ps
        elif last == -1:
            q += ps
        q = max(0.0, min(1.0, q))
        p_map = _clip_prob(q * pb + (1.0 - q) * pr)
        return float(p_map * _rec(a + 1, b, m + 1, 1) + (1.0 - p_map) * _rec(a, b + 1, m + 1, -1))

    return float(_rec(0, 0, 0, 0))


def _series_win_prob_dp_v5(
    *,
    bo: int,
    p_blue: float,
    p_red: float,
    q_blue_base: float,
    pick_shift: float,
    g1_blue_shift: float,
    momentum_shift: float,
    loser_pick_decay: float,
    decider_variance: float,
    decider_side_revert: float,
    series_corr_sigma: float,
) -> float:
    """Side-aware + adaptation + pressure dynamics (v5)."""
    out = _series_win_prob_dp_v5_with_diag(
        bo=bo,
        p_blue=p_blue,
        p_red=p_red,
        q_blue_base=q_blue_base,
        pick_shift=pick_shift,
        g1_blue_shift=g1_blue_shift,
        momentum_shift=momentum_shift,
        loser_pick_decay=loser_pick_decay,
        decider_variance=decider_variance,
        decider_side_revert=decider_side_revert,
        series_corr_sigma=series_corr_sigma,
    )
    return float(_f(out.get("p_series"), 0.5))


def _dp_v5_map_win_prob(
    *,
    a: int,
    b: int,
    m: int,
    last: int,
    target: int,
    max_maps: int,
    p_blue: float,
    p_red: float,
    q_blue_base: float,
    pick_shift: float,
    g1_blue_shift: float,
    momentum_shift: float,
    loser_pick_decay: float,
    decider_variance: float,
    decider_side_revert: float,
) -> float:
    """Per-map TeamA win probability under v5 dynamics for state (a,b,m,last)."""
    map_frac = float(m) / float(max(1, max_maps - 1))
    ps_m = float(pick_shift) * (1.0 - float(loser_pick_decay) * map_frac)

    if m == 0:
        q = float(q_blue_base) + float(g1_blue_shift)
    else:
        q = float(q_blue_base)
        if last == 1:
            q -= ps_m
        elif last == -1:
            q += ps_m
        # score-state side drift: trailing side tends to improve side access.
        score_gap = float(b - a) / float(max(1, target))
        q += max(-0.06, min(0.06, score_gap * ps_m * 0.40))
    # Decider-side reversion: in decisive map, side priority tends to be less
    # exploitable than mid-series loser-side-pick dynamics.
    if m >= (max_maps - 1):
        dsr = max(0.0, min(1.0, float(decider_side_revert)))
        q = 0.50 + (q - 0.50) * (1.0 - dsr)
    q = max(0.0, min(1.0, q))
    p_map = _clip_prob(q * float(p_blue) + (1.0 - q) * float(p_red))

    # Adaptation: loser tends to adapt draft/tempo for next map (small, conservative).
    if last == 1:
        p_map = _clip_prob(p_map - float(momentum_shift))
    elif last == -1:
        p_map = _clip_prob(p_map + float(momentum_shift))

    # Pressure shrink near elimination / late maps.
    near_elim = 1.0 if (target - a <= 1 or target - b <= 1) else 0.0
    late_mult = 0.35 + 0.65 * map_frac
    shrink = max(0.0, min(0.45, float(decider_variance) * (0.55 * near_elim + 0.45 * late_mult)))
    p_map = _clip_prob(0.5 + (p_map - 0.5) * (1.0 - shrink))
    return float(p_map)


def _series_win_prob_dp_v5_with_diag(
    *,
    bo: int,
    p_blue: float,
    p_red: float,
    q_blue_base: float,
    pick_shift: float,
    g1_blue_shift: float,
    momentum_shift: float,
    loser_pick_decay: float,
    decider_variance: float,
    decider_side_revert: float,
    series_corr_sigma: float,
) -> Dict[str, Any]:
    """Return v5 series probability plus map-by-map diagnostics."""
    target = _series_target_wins(int(bo))
    max_maps = int(bo)
    pb = _clip_prob(float(p_blue))
    pr = _clip_prob(float(p_red))
    qb = max(0.0, min(1.0, float(q_blue_base)))
    ps = max(0.0, min(0.25, float(pick_shift)))
    g1s = max(-0.08, min(0.08, float(g1_blue_shift)))
    moms = max(0.0, min(0.04, float(momentum_shift)))
    lpd = max(0.0, min(0.9, float(loser_pick_decay)))
    dvar = max(0.0, min(0.35, float(decider_variance)))
    dsr = max(0.0, min(1.0, float(decider_side_revert)))
    scs = max(0.0, min(0.35, float(series_corr_sigma)))

    def _run_one_series(pb_i: float, pr_i: float) -> Dict[str, Any]:
        states: Dict[tuple[int, int, int], float] = {(0, 0, 0): 1.0}  # (a_wins, b_wins, last_winner)
        term_score: Dict[str, float] = {}
        map_play_mass = [0.0 for _ in range(max_maps)]
        map_win_mass = [0.0 for _ in range(max_maps)]
        p_series_i = 0.0
        expected_maps_i = 0.0
        decider_reach_prob_i = 0.0

        def _absorb_terminal(a1: int, b1: int, maps_played: int, mass: float) -> None:
            nonlocal p_series_i, expected_maps_i, decider_reach_prob_i
            if mass <= 0.0:
                return
            expected_maps_i += float(mass) * float(maps_played)
            if maps_played >= max_maps:
                decider_reach_prob_i += float(mass)
            if a1 > b1:
                p_series_i += float(mass)
            k = f"{int(a1)}-{int(b1)}"
            term_score[k] = float(term_score.get(k, 0.0) + float(mass))

        for m in range(max_maps):
            if not states:
                break
            nxt: Dict[tuple[int, int, int], float] = {}
            for (a, b, last), mass in list(states.items()):
                if mass <= 0.0:
                    continue
                if a >= target or b >= target:
                    _absorb_terminal(a, b, m, float(mass))
                    continue

                p_map = _dp_v5_map_win_prob(
                    a=int(a),
                    b=int(b),
                    m=int(m),
                    last=int(last),
                    target=int(target),
                    max_maps=int(max_maps),
                    p_blue=float(pb_i),
                    p_red=float(pr_i),
                    q_blue_base=float(qb),
                    pick_shift=float(ps),
                    g1_blue_shift=float(g1s),
                    momentum_shift=float(moms),
                    loser_pick_decay=float(lpd),
                    decider_variance=float(dvar),
                    decider_side_revert=float(dsr),
                )
                map_play_mass[m] += float(mass)
                map_win_mass[m] += float(mass) * float(p_map)

                a1, b1 = int(a + 1), int(b)
                m1 = int(m + 1)
                mass_a = float(mass) * float(p_map)
                if a1 >= target or b1 >= target or m1 >= max_maps:
                    _absorb_terminal(a1, b1, m1, mass_a)
                else:
                    k1 = (a1, b1, 1)
                    nxt[k1] = float(nxt.get(k1, 0.0) + mass_a)

                a2, b2 = int(a), int(b + 1)
                mass_b = float(mass) * float(1.0 - p_map)
                if a2 >= target or b2 >= target or m1 >= max_maps:
                    _absorb_terminal(a2, b2, m1, mass_b)
                else:
                    k2 = (a2, b2, -1)
                    nxt[k2] = float(nxt.get(k2, 0.0) + mass_b)
            states = nxt

        if states:
            for (a, b, _last), mass in list(states.items()):
                _absorb_terminal(int(a), int(b), int(max_maps), float(mass))

        return {
            "p_series": float(_clip_prob(p_series_i)),
            "expected_maps": float(max(float(target), min(float(max_maps), float(expected_maps_i)))),
            "decider_reach_prob": float(max(0.0, min(1.0, decider_reach_prob_i))),
            "map_play_mass": list(map_play_mass),
            "map_win_mass": list(map_win_mass),
            "scoreline_probs": dict(term_score),
        }

    # Optional intra-series correlation in latent logit space. This captures
    # "same-day form" shared across maps and avoids underestimating sweep risk.
    # Gauss-Hermite (n=7) for E[f(eta)] under eta~N(0, sigma^2).
    if scs > 1e-9:
        _x = [-2.6519613568352335, -1.6735516287674714, -0.8162878828589647, 0.0, 0.8162878828589647, 1.6735516287674714, 2.6519613568352335]
        _w = [0.0009717812450995192, 0.05451558281912703, 0.4256072526101278, 0.8102646175568073, 0.4256072526101278, 0.05451558281912703, 0.0009717812450995192]
        zpb = _logit(pb)
        zpr = _logit(pr)
        p_series = 0.0
        expected_maps = 0.0
        decider_reach_prob = 0.0
        map_play_mass = [0.0 for _ in range(max_maps)]
        map_win_mass = [0.0 for _ in range(max_maps)]
        term_score: Dict[str, float] = {}
        integ_nodes = []
        inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
        for xx, ww in zip(_x, _w):
            eta = math.sqrt(2.0) * float(scs) * float(xx)
            pb_i = _clip_prob(_sigmoid(zpb + eta))
            pr_i = _clip_prob(_sigmoid(zpr + eta))
            run_i = _run_one_series(pb_i, pr_i)
            w_i = float(ww) * inv_sqrt_pi
            p_series += w_i * float(run_i.get("p_series", 0.5))
            expected_maps += w_i * float(run_i.get("expected_maps", float(target)))
            decider_reach_prob += w_i * float(run_i.get("decider_reach_prob", 0.0))
            _play_i = list(run_i.get("map_play_mass") or [])
            _win_i = list(run_i.get("map_win_mass") or [])
            for m in range(max_maps):
                map_play_mass[m] += w_i * float(_play_i[m] if m < len(_play_i) else 0.0)
                map_win_mass[m] += w_i * float(_win_i[m] if m < len(_win_i) else 0.0)
            for k, vv in dict(run_i.get("scoreline_probs") or {}).items():
                term_score[str(k)] = float(term_score.get(str(k), 0.0) + w_i * float(vv))
            integ_nodes.append({"eta": float(eta), "w": float(w_i), "p_blue": float(pb_i), "p_red": float(pr_i)})
    else:
        _base = _run_one_series(pb, pr)
        p_series = float(_base.get("p_series", 0.5))
        expected_maps = float(_base.get("expected_maps", float(target)))
        decider_reach_prob = float(_base.get("decider_reach_prob", 0.0))
        map_play_mass = list(_base.get("map_play_mass") or [0.0 for _ in range(max_maps)])
        map_win_mass = list(_base.get("map_win_mass") or [0.0 for _ in range(max_maps)])
        term_score = dict(_base.get("scoreline_probs") or {})
        integ_nodes = []

    p_series = float(_clip_prob(p_series))
    expected_maps = float(max(float(target), min(float(max_maps), float(expected_maps))))
    decider_reach_prob = float(max(0.0, min(1.0, decider_reach_prob)))

    map_play_prob: Dict[str, float] = {}
    map_win_if_played: Dict[str, float] = {}
    for i in range(max_maps):
        key = f"m{i + 1}"
        play = float(max(0.0, min(1.0, map_play_mass[i])))
        map_play_prob[key] = play
        if play > 1e-12:
            map_win_if_played[key] = float(_clip_prob(map_win_mass[i] / play))
        else:
            map_win_if_played[key] = float("nan")

    # Normalize scoreline probs to avoid tiny floating-point drift.
    z = float(sum(float(v) for v in term_score.values()))
    scoreline_probs: Dict[str, float] = {}
    if z > 1e-12:
        for k in sorted(term_score.keys()):
            scoreline_probs[str(k)] = float(max(0.0, min(1.0, float(term_score.get(k, 0.0)) / z)))

    return {
        "p_series": float(p_series),
        "expected_maps": float(expected_maps),
        "decider_reach_prob": float(decider_reach_prob),
        "decider_side_revert": float(dsr),
        "series_corr_sigma": float(scs),
        "integration_nodes": list(integ_nodes),
        "map_play_prob": map_play_prob,
        "map_win_if_played": map_win_if_played,
        "scoreline_probs": scoreline_probs,
    }


def _league_macro_key(league: Any) -> str:
    return str(league_macro_key(league, unknown="OTHER"))


def _patch_bucket(status: Any) -> str:
    s = str(status or "").strip().upper()
    if ("REFIT" in s) or ("CHANGED" in s):
        return "REFIT"
    if ("WARN" in s) or ("ATTEN" in s):
        return "WARN"
    if ("OK" in s) or ("NONE" in s):
        return "OK"
    return "UNK"


def _coerce_bo_tuning_node(node: Any, bo_i: int) -> Dict[str, float]:
    if not isinstance(node, dict):
        return {}
    bo_key = f"bo{int(bo_i)}"
    src = node.get(bo_key) if isinstance(node.get(bo_key), dict) else node
    if not isinstance(src, dict):
        return {}
    out: Dict[str, float] = {}
    aliases = {
        "w_anchor": "anchor_weight",
        "anchor_w": "anchor_weight",
        "cap": "cap_pp",
        "cappp": "cap_pp",
        "decider_side": "decider_side_revert",
        "series_corr": "series_corr_sigma",
    }
    for k, v in src.items():
        kk = aliases.get(str(k), str(k))
        if kk not in {
            "pick_shift",
            "g1_blue_shift",
            "momentum_shift",
            "loser_pick_decay",
            "decider_variance",
            "decider_side_revert",
            "series_corr_sigma",
            "anchor_weight",
            "cap_pp",
        }:
            continue
        fv = _f(v, float("nan"))
        if not math.isfinite(fv):
            continue
        if kk == "pick_shift":
            out[kk] = float(max(0.0, min(0.06, fv)))
        elif kk == "g1_blue_shift":
            out[kk] = float(max(-0.08, min(0.08, fv)))
        elif kk == "momentum_shift":
            out[kk] = float(max(0.0, min(0.04, fv)))
        elif kk == "loser_pick_decay":
            out[kk] = float(max(0.0, min(0.90, fv)))
        elif kk == "decider_variance":
            out[kk] = float(max(0.0, min(0.35, fv)))
        elif kk == "decider_side_revert":
            out[kk] = float(max(0.0, min(1.0, fv)))
        elif kk == "series_corr_sigma":
            out[kk] = float(max(0.0, min(0.35, fv)))
        elif kk == "anchor_weight":
            out[kk] = float(max(0.45, min(0.95, fv)))
        elif kk == "cap_pp":
            out[kk] = float(max(4.0, min(20.0, fv)))
    return out


def _pick_tuning_node(
    *,
    tuning_cfg: Any,
    bo_i: int,
    league_hint: str,
    patch_status: Any,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "applied": False,
        "source": "",
        "league": str(league_hint or "").strip().upper(),
        "macro": "",
        "patch_bucket": _patch_bucket(patch_status),
        "params": {},
    }
    if not isinstance(tuning_cfg, dict):
        return out

    lg = str(out["league"])
    mk = _league_macro_key(lg)
    pb = str(out["patch_bucket"])
    out["macro"] = str(mk)

    probes = []
    by_lp = tuning_cfg.get("by_league_patch") if isinstance(tuning_cfg.get("by_league_patch"), dict) else {}
    by_lg = tuning_cfg.get("by_league") if isinstance(tuning_cfg.get("by_league"), dict) else {}
    by_mp = tuning_cfg.get("by_macro_patch") if isinstance(tuning_cfg.get("by_macro_patch"), dict) else {}
    by_mk = tuning_cfg.get("by_macro") if isinstance(tuning_cfg.get("by_macro"), dict) else {}
    by_ps = tuning_cfg.get("by_patch_status") if isinstance(tuning_cfg.get("by_patch_status"), dict) else {}
    gg = tuning_cfg.get("global") if isinstance(tuning_cfg.get("global"), dict) else {}

    if lg and pb and f"{lg}|{pb}" in by_lp:
        probes.append((f"league_patch:{lg}|{pb}", by_lp.get(f"{lg}|{pb}")))
    if lg and lg in by_lg:
        probes.append((f"league:{lg}", by_lg.get(lg)))
    if mk and pb and f"{mk}|{pb}" in by_mp:
        probes.append((f"macro_patch:{mk}|{pb}", by_mp.get(f"{mk}|{pb}")))
    if mk and mk in by_mk:
        probes.append((f"macro:{mk}", by_mk.get(mk)))
    if pb and pb in by_ps:
        probes.append((f"patch:{pb}", by_ps.get(pb)))
    probes.append(("global", gg))

    for src, node in probes:
        params = _coerce_bo_tuning_node(node, int(bo_i))
        if params:
            out["applied"] = True
            out["source"] = str(src)
            out["params"] = dict(params)
            break
    return out


def apply_series_bo_dplite_live(
    *,
    p_map: Any,
    p_series_in: Any,
    bo: Any,
    same_league: bool = False,
    interregional: bool = False,
    playoffs: bool = False,
    patch_status: Any = None,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    side_bias_pp: Any = None,
    style_volatility: Any = None,
    layer_confidence_score: Any = None,
    league_hint: Any = None,
    tuning_cfg: Any = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "method": "series_bo_v5_live",
        "bo": None,
        "same_league": bool(same_league),
        "interregional": bool(interregional),
        "playoffs": bool(playoffs),
        "patch_status": str(patch_status or ""),
        "league_hint": str(league_hint or ""),
        "p_map": None,
        "p_before": None,
        "p_after": None,
        "diff_pp": None,
        "weight": None,
        "cap_pp": None,
        "n_team_eff": None,
        "n_players_eff": None,
        "side_bias_pp": None,
        "style_volatility": None,
        "layer_confidence_score": None,
        "anchor_mode": None,
        "anchor_weight": None,
        "tuning": {},
        "dp": {},
        "coherence_guard": {},
    }

    pm = _f(p_map)
    ps0 = _f(p_series_in)
    if not math.isfinite(pm) or not math.isfinite(ps0):
        out["reason"] = "invalid_inputs"
        return out
    out["p_map"] = float(pm)
    out["p_before"] = float(ps0)

    try:
        bo_i = int(float(bo))
    except Exception:
        bo_i = 1
    bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
    out["bo"] = int(bo_i)
    if bo_i <= 1:
        out["reason"] = "bo1_noop"
        return out

    _lg_hint = str(league_hint or "").strip().upper()
    _tune = _pick_tuning_node(
        tuning_cfg=tuning_cfg,
        bo_i=int(bo_i),
        league_hint=_lg_hint,
        patch_status=patch_status,
    )
    _tp = dict(_tune.get("params") or {}) if isinstance(_tune, dict) else {}
    out["tuning"] = dict(_tune)

    nte = _f(n_team_eff, 0.0)
    npe = _f(n_players_eff, 0.0)
    out["n_team_eff"] = (float(nte) if math.isfinite(nte) else None)
    out["n_players_eff"] = (float(npe) if math.isfinite(npe) else None)
    _side_bias_pp = _f(side_bias_pp, 0.0)
    _style_vol = _f(style_volatility, 0.0)
    _layer_conf = _f(layer_confidence_score, float("nan"))
    out["side_bias_pp"] = (float(_side_bias_pp) if math.isfinite(_side_bias_pp) else None)
    out["style_volatility"] = (float(_style_vol) if math.isfinite(_style_vol) else None)
    out["layer_confidence_score"] = (float(_layer_conf) if math.isfinite(_layer_conf) else None)

    # DP side-aware anchor: model map-level win with side split + loser-side-pick drift.
    # The goal is to keep it causal and bounded (no hard assumptions about fixed side schedule).
    _bias_abs = abs(float(_side_bias_pp)) if math.isfinite(_side_bias_pp) else 0.0
    _bias_abs = max(0.0, min(8.0, _bias_abs)) / 100.0
    p_blue = _clip_prob(pm + _bias_abs)
    p_red = _clip_prob(pm - _bias_abs)

    # Base blue share: near-even by default, slightly tilted by relative strength.
    q_blue = 0.50 + max(-0.10, min(0.10, (pm - 0.5) * 0.36))
    if bool(playoffs):
        q_blue += 0.01
    if bool(interregional):
        q_blue = 0.50 + (q_blue - 0.50) * 0.60
    if math.isfinite(_style_vol):
        q_blue = 0.50 + (q_blue - 0.50) * (1.0 - max(0.0, min(0.60, abs(float(_style_vol)) * 0.22)))
    if math.isfinite(_layer_conf):
        _conf01 = max(0.0, min(1.0, float(_layer_conf) / 100.0))
        q_blue = 0.50 + (q_blue - 0.50) * (0.82 + 0.18 * _conf01)
    q_blue = max(0.40, min(0.60, q_blue))

    pick_shift = 0.020 if bo_i == 3 else 0.028
    if bool(interregional):
        pick_shift *= 0.85
    if math.isfinite(_layer_conf):
        _conf01 = max(0.0, min(1.0, float(_layer_conf) / 100.0))
        pick_shift *= (0.85 + 0.20 * _conf01)
    if math.isfinite(_style_vol):
        pick_shift *= (1.0 - max(0.0, min(0.40, abs(float(_style_vol)) * 0.20)))
    pick_shift = max(0.0, min(0.06, pick_shift))
    if "pick_shift" in _tp:
        pick_shift = float(_tp.get("pick_shift"))

    # v5 anchor: side-aware per map + adaptation + pressure-aware dynamics.
    g1_blue_shift = 0.0
    if bo_i >= 5 and bool(playoffs):
        g1_blue_shift += 0.010
    elif bo_i >= 3:
        g1_blue_shift += 0.004
    if bool(interregional):
        g1_blue_shift *= 0.65
    if math.isfinite(_layer_conf):
        _conf01 = max(0.0, min(1.0, float(_layer_conf) / 100.0))
        g1_blue_shift *= (0.85 + 0.20 * _conf01)
    if "g1_blue_shift" in _tp:
        g1_blue_shift = float(_tp.get("g1_blue_shift"))

    momentum_shift = 0.009 if bo_i == 3 else 0.013
    if bool(interregional):
        momentum_shift *= 0.82
    if math.isfinite(_style_vol):
        momentum_shift *= (0.85 + 0.25 * max(0.0, min(1.0, abs(float(_style_vol)))))
    if math.isfinite(_layer_conf):
        _conf01 = max(0.0, min(1.0, float(_layer_conf) / 100.0))
        momentum_shift *= (0.92 + 0.16 * _conf01)
    momentum_shift = max(0.0, min(0.02, momentum_shift))
    if "momentum_shift" in _tp:
        momentum_shift = float(_tp.get("momentum_shift"))

    loser_pick_decay = 0.30 if bo_i == 3 else 0.42
    if bool(playoffs):
        loser_pick_decay *= 1.06
    if bool(interregional):
        loser_pick_decay *= 0.88
    if math.isfinite(_style_vol):
        loser_pick_decay *= (0.92 + 0.20 * max(0.0, min(1.0, abs(float(_style_vol)))))
    loser_pick_decay = max(0.0, min(0.75, loser_pick_decay))
    if "loser_pick_decay" in _tp:
        loser_pick_decay = float(_tp.get("loser_pick_decay"))

    decider_variance = 0.10 if bo_i == 3 else 0.14
    if bool(interregional):
        decider_variance *= 0.90
    if math.isfinite(_layer_conf):
        _conf01 = max(0.0, min(1.0, float(_layer_conf) / 100.0))
        decider_variance *= (1.08 - 0.22 * _conf01)
    if math.isfinite(_style_vol):
        decider_variance *= (0.92 + 0.22 * max(0.0, min(1.0, abs(float(_style_vol)))))
    decider_variance = max(0.0, min(0.22, decider_variance))
    if "decider_variance" in _tp:
        decider_variance = float(_tp.get("decider_variance"))

    decider_side_revert = 0.14 if bo_i == 3 else 0.22
    if bool(interregional):
        decider_side_revert *= 1.05
    if bool(playoffs):
        decider_side_revert *= 1.08
    if math.isfinite(_layer_conf):
        _conf01 = max(0.0, min(1.0, float(_layer_conf) / 100.0))
        decider_side_revert *= (1.06 - 0.18 * _conf01)
    decider_side_revert = max(0.0, min(0.65, decider_side_revert))
    if "decider_side_revert" in _tp:
        decider_side_revert = float(_tp.get("decider_side_revert"))

    series_corr_sigma = 0.07 if bo_i == 3 else 0.10
    if bool(interregional):
        series_corr_sigma += 0.01
    if bool(playoffs):
        series_corr_sigma += 0.01
    if math.isfinite(_style_vol):
        series_corr_sigma += max(0.0, min(0.06, abs(float(_style_vol)) * 0.08))
    if math.isfinite(_layer_conf):
        _conf01 = max(0.0, min(1.0, float(_layer_conf) / 100.0))
        series_corr_sigma *= (1.08 - 0.25 * _conf01)
    series_corr_sigma = max(0.0, min(0.22, series_corr_sigma))
    if "series_corr_sigma" in _tp:
        series_corr_sigma = float(_tp.get("series_corr_sigma"))

    _resolved_tuning = {
        "pick_shift": float(pick_shift),
        "g1_blue_shift": float(g1_blue_shift),
        "momentum_shift": float(momentum_shift),
        "loser_pick_decay": float(loser_pick_decay),
        "decider_variance": float(decider_variance),
        "decider_side_revert": float(decider_side_revert),
        "series_corr_sigma": float(series_corr_sigma),
    }
    if isinstance(out.get("tuning"), dict):
        out["tuning"]["resolved"] = dict(_resolved_tuning)

    _dp_diag = _series_win_prob_dp_v5_with_diag(
        bo=bo_i,
        p_blue=p_blue,
        p_red=p_red,
        q_blue_base=q_blue,
        pick_shift=pick_shift,
        g1_blue_shift=g1_blue_shift,
        momentum_shift=momentum_shift,
        loser_pick_decay=loser_pick_decay,
        decider_variance=decider_variance,
        decider_side_revert=decider_side_revert,
        series_corr_sigma=series_corr_sigma,
    )
    p_anchor = float(_f(_dp_diag.get("p_series"), 0.5))
    out["anchor_mode"] = "dp_sideaware_v5"
    out["dp"] = {
        "p_blue": float(p_blue),
        "p_red": float(p_red),
        "q_blue_base": float(q_blue),
        "pick_shift": float(pick_shift),
        "g1_blue_shift": float(g1_blue_shift),
        "momentum_shift": float(momentum_shift),
        "loser_pick_decay": float(loser_pick_decay),
        "decider_variance": float(decider_variance),
        "decider_side_revert": float(decider_side_revert),
        "series_corr_sigma": float(series_corr_sigma),
        "p_series_anchor": float(p_anchor),
        "expected_maps": (float(_dp_diag.get("expected_maps")) if _dp_diag.get("expected_maps") is not None else None),
        "decider_reach_prob": (float(_dp_diag.get("decider_reach_prob")) if _dp_diag.get("decider_reach_prob") is not None else None),
        "map_play_prob": dict(_dp_diag.get("map_play_prob") or {}),
        "map_win_if_played": dict(_dp_diag.get("map_win_if_played") or {}),
        "scoreline_probs": dict(_dp_diag.get("scoreline_probs") or {}),
    }

    evidence = max(0.0, min(1.0, (0.75 * max(0.0, nte) + 0.25 * max(0.0, npe)) / 70.0))
    # v3: dynamic DP anchor is primary source for BO3/BO5.
    # We keep a bounded pull toward current p_series_in only for stability.
    w_anchor = 0.60 + 0.30 * math.sqrt(evidence)
    if bo_i == 5:
        w_anchor += 0.04
    if bool(playoffs):
        w_anchor += 0.02
    if bool(interregional):
        w_anchor -= 0.08
    if bool(same_league):
        w_anchor += 0.02
    ps = str(patch_status or "").upper()
    if "REFIT" in ps:
        w_anchor -= 0.10
    elif "WARN" in ps:
        w_anchor -= 0.05
    if math.isfinite(_style_vol):
        w_anchor -= max(0.0, min(0.08, abs(float(_style_vol)) * 0.06))
    if math.isfinite(_layer_conf):
        _conf01 = max(0.0, min(1.0, float(_layer_conf) / 100.0))
        w_anchor += 0.06 * (_conf01 - 0.50)
    w_anchor = float(max(0.45, min(0.92, w_anchor)))
    if "anchor_weight" in _tp:
        w_anchor = float(max(0.45, min(0.95, float(_tp.get("anchor_weight")))))
    out["weight"] = float(w_anchor)
    out["anchor_weight"] = float(w_anchor)

    z0 = _logit(ps0)
    za = _logit(p_anchor)
    z1 = (1.0 - w_anchor) * z0 + w_anchor * za
    p1 = _sigmoid(z1)
    diff_pp = float(100.0 * (p1 - ps0))
    cap_pp = 8.0 if bo_i == 3 else 12.0
    if bool(interregional):
        cap_pp *= 0.85
    if math.isfinite(_layer_conf):
        _conf01 = max(0.0, min(1.0, float(_layer_conf) / 100.0))
        cap_pp *= (0.90 + 0.20 * _conf01)
    if math.isfinite(_style_vol):
        cap_pp *= (1.0 - max(0.0, min(0.20, abs(float(_style_vol)) * 0.08)))
    cap_pp = float(max(4.0, min(16.0, cap_pp)))
    if "cap_pp" in _tp:
        cap_pp = float(max(4.0, min(20.0, float(_tp.get("cap_pp")))))
    out["cap_pp"] = float(cap_pp)
    if abs(diff_pp) > cap_pp:
        p1 = _clip_prob(ps0 + math.copysign(cap_pp / 100.0, diff_pp))
        diff_pp = float(100.0 * (p1 - ps0))

    # Full economic coherence is enforced later in the pipeline after
    # series_gamma and again after post-live layers. Running the same guard
    # here only duplicates bounds/caps without protecting a later state.
    out["coherence_guard"] = {
        "enabled": False,
        "applied": False,
        "reason": "deferred_to_pipeline",
        "stage": "series_bo_dplite_live",
        "bo": int(bo_i),
        "p_map": float(pm),
        "p_series_before": float(p1),
        "p_series_after": float(p1),
    }

    if abs(diff_pp) < 0.02:
        out["reason"] = "tiny_move"
        out["p_after"] = float(p1)
        out["diff_pp"] = float(diff_pp)
        return out

    out["applied"] = True
    out["reason"] = "ok"
    out["p_after"] = float(p1)
    out["diff_pp"] = float(diff_pp)
    return out
