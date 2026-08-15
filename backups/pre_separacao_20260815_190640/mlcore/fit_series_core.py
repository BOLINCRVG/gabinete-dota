from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from .bo import prob_win_series
from .series_bo_core import apply_series_bo_dplite_live
from .utils import brier, clip_prob, logit, logloss, sigmoid
from .fit_samples_core import Sample, _fuse_probs_by_precision, _league_macro_key
from .fit_form_core import _apply_form_blend_on_sample
def _series_prob_with_gamma(p_map: float, bo: int, gamma: float) -> float:
    p0 = float(clip_prob(float(prob_win_series(float(p_map), int(bo)))))
    g = max(0.25, float(gamma))
    return float(clip_prob(float(sigmoid(float(logit(p0)) * g))))


def _fit_series_gamma(
    samples: List[Sample],
    *,
    team_scale: float,
    players_scale: float,
    form_cfg: Dict[str, Any],
    league_filter: str | None = None,
    macro_filter: str | None = None,
) -> Dict[str, Any]:
    src = samples
    if league_filter:
        src = [s for s in src if str(s.league) == str(league_filter)]
    if macro_filter:
        src = [s for s in src if str(s.macro) == str(macro_filter)]
    if not src:
        return {"error": "no_samples"}

    # Build series from map rows by day + league + unordered teams.
    groups: Dict[str, List[Sample]] = defaultdict(list)
    for s in src:
        ta = str(s.blue_team or "").strip()
        tb = str(s.red_team or "").strip()
        if not ta or not tb:
            continue
        u1, u2 = sorted([ta, tb])
        key = f"{s.league}|{s.date_day}|{u1}|{u2}"
        groups[key].append(s)

    rows: List[Dict[str, Any]] = []
    for _, maps in groups.items():
        maps = sorted(maps, key=lambda x: int(x.game_no))
        if len(maps) < 2:
            continue
        bo = int(max(int(m.game_no or 1) for m in maps))
        if bo not in (3, 5):
            continue
        m1 = maps[0]
        first_blue = str(m1.blue_team)
        wins: Dict[str, int] = defaultdict(int)
        for m in maps:
            wb = int(m.y)
            if wb == 1:
                wins[str(m.blue_team)] += 1
            else:
                wins[str(m.red_team)] += 1
        if not wins:
            continue
        winner = max(wins.items(), key=lambda kv: kv[1])[0]
        y_series = 1 if str(winner) == first_blue else 0
        p_pre = _fuse_probs_by_precision(
            p_team=float(m1.p_team),
            p_players=float(m1.p_players),
            n_team=float(m1.n_team),
            n_players=float(m1.n_players),
            team_scale=float(team_scale),
            players_scale=float(players_scale),
            coverage=float(m1.coverage_ratio),
            transfer_signal=float(m1.transfer_signal),
            lineup_continuity=float(m1.lineup_continuity),
            roster_changed=bool(m1.roster_changed),
            season_phase=("playoffs" if str(m1.phase).strip().lower() == "playoffs" else "auto"),
            players_freshness=float(m1.freshness_avg),
        )
        p_map = _apply_form_blend_on_sample(
            p_model=float(p_pre),
            sample=m1,
            enabled=bool((form_cfg or {}).get("enabled", False)),
            max_weight=float((form_cfg or {}).get("max_weight", 0.75) or 0.75),
            knee_games=float((form_cfg or {}).get("knee_games", 6.0) or 6.0),
            beta_prior=float((form_cfg or {}).get("beta_prior", 2.0) or 2.0),
            signal_power=float((form_cfg or {}).get("signal_power", 1.0) or 1.0),
            elite_same_league_parity_strength=float(
                (form_cfg or {}).get("elite_same_league_parity_strength", 0.35) or 0.35
            ),
        )
        rows.append({"bo": int(bo), "y": int(y_series), "p_map": float(p_map)})

    if not rows:
        return {"error": "no_series_rows"}

    gammas = [0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00]
    out = {"rows": len(rows)}
    for bo in (3, 5):
        rb = [r for r in rows if int(r["bo"]) == bo]
        if len(rb) < 40:
            continue
        y = [int(r["y"]) for r in rb]
        best = {"gamma": 1.0, "logloss": float("inf"), "n": len(rb)}
        for g in gammas:
            p = [_series_prob_with_gamma(float(r["p_map"]), int(bo), float(g)) for r in rb]
            ll = float(logloss(y, p))
            if ll < float(best["logloss"]):
                best = {"gamma": float(g), "logloss": float(ll), "n": len(rb)}
        base_ll = float(logloss(y, [_series_prob_with_gamma(float(r["p_map"]), int(bo), 1.0) for r in rb]))
        best["improvement_logloss_vs_gamma1"] = float(base_ll - float(best["logloss"]))
        out[f"bo{bo}"] = best
    if "bo3" not in out and "bo5" not in out:
        return {"error": "insufficient_series_rows", "rows": len(rows)}
    return out


def _patch_status_from_seen(prev_seen: int) -> str:
    pv = int(max(0, int(prev_seen)))
    if pv < 20:
        return "REFIT"
    if pv < 60:
        return "WARN"
    return "OK"


def _build_series_rows_for_bo_dplite(
    samples: List[Sample],
    *,
    team_scale: float,
    players_scale: float,
    form_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not samples:
        return []

    seen_patch: Dict[Tuple[str, str], int] = defaultdict(int)
    patch_status_by_gid: Dict[str, str] = {}
    for s in samples:
        gid = str(getattr(s, "gameid", "") or "").strip()
        lg = str(getattr(s, "league", "") or "").strip().upper()
        pt = str(getattr(s, "patch", "") or "").strip().upper()
        if not gid:
            continue
        key = (lg, pt)
        prev = int(seen_patch.get(key, 0))
        patch_status_by_gid[gid] = _patch_status_from_seen(prev)
        seen_patch[key] = prev + 1

    groups: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        ta = str(s.blue_team or "").strip()
        tb = str(s.red_team or "").strip()
        lg = str(s.league or "").strip().upper()
        sd = str(s.date_day or "").strip()
        sp = str(s.split or "").strip().upper()
        ph = str(s.phase or "").strip().lower()
        if not ta or not tb or not lg or not sd:
            continue
        u1, u2 = sorted([ta, tb])
        key = f"{lg}|{sp}|{ph}|{sd}|{u1}|{u2}"
        groups[key].append(s)

    out: List[Dict[str, Any]] = []
    for maps in groups.values():
        if not maps:
            continue
        maps = sorted(maps, key=lambda x: int(getattr(x, "game_no", 1) or 1))
        bo = int(max(int(getattr(m, "game_no", 1) or 1) for m in maps))
        if bo not in (3, 5):
            continue

        wins: Dict[str, int] = defaultdict(int)
        for m in maps:
            wb = int(getattr(m, "y", 0) or 0)
            if wb == 1:
                wins[str(getattr(m, "blue_team", "") or "")] += 1
            else:
                wins[str(getattr(m, "red_team", "") or "")] += 1
        if not wins:
            continue
        m1 = maps[0]
        first_blue = str(getattr(m1, "blue_team", "") or "")
        winner = max(wins.items(), key=lambda kv: kv[1])[0]
        y_series = 1 if str(winner) == first_blue else 0

        p_pre = _fuse_probs_by_precision(
            p_team=float(getattr(m1, "p_team", 0.5)),
            p_players=float(getattr(m1, "p_players", 0.5)),
            n_team=float(getattr(m1, "n_team", 0.0)),
            n_players=float(getattr(m1, "n_players", 0.0)),
            team_scale=float(team_scale),
            players_scale=float(players_scale),
            coverage=float(getattr(m1, "coverage_ratio", 0.0) or 0.0),
            transfer_signal=float(getattr(m1, "transfer_signal", 0.0) or 0.0),
            lineup_continuity=float(getattr(m1, "lineup_continuity", 1.0) or 1.0),
            roster_changed=bool(getattr(m1, "roster_changed", False)),
            season_phase=("playoffs" if str(getattr(m1, "phase", "")).strip().lower() == "playoffs" else "auto"),
            players_freshness=float(getattr(m1, "freshness_avg", 0.0) or 0.0),
        )
        p_map = _apply_form_blend_on_sample(
            p_model=float(p_pre),
            sample=m1,
            enabled=bool((form_cfg or {}).get("enabled", False)),
            max_weight=float((form_cfg or {}).get("max_weight", 0.75) or 0.75),
            knee_games=float((form_cfg or {}).get("knee_games", 6.0) or 6.0),
            beta_prior=float((form_cfg or {}).get("beta_prior", 2.0) or 2.0),
            signal_power=float((form_cfg or {}).get("signal_power", 1.0) or 1.0),
            elite_same_league_parity_strength=float(
                (form_cfg or {}).get("elite_same_league_parity_strength", 0.35) or 0.35
            ),
        )
        p_series_in = float(prob_win_series(float(p_map), int(bo)))

        lane_sum = (
            abs(float(getattr(m1, "lane_top", 0.0) or 0.0))
            + abs(float(getattr(m1, "lane_jungle", 0.0) or 0.0))
            + abs(float(getattr(m1, "lane_mid", 0.0) or 0.0))
            + abs(float(getattr(m1, "lane_bot", 0.0) or 0.0))
            + abs(float(getattr(m1, "lane_support", 0.0) or 0.0))
        )
        style_vol = float(max(0.0, min(1.0, lane_sum / 8.0)))
        conf_score = float(
            max(
                5.0,
                min(
                    100.0,
                    12.0 + 1.15 * float(getattr(m1, "n_team", 0.0) or 0.0) + 0.65 * float(getattr(m1, "n_players", 0.0) or 0.0),
                ),
            )
        )

        gid1 = str(getattr(m1, "gameid", "") or "").strip()
        ps = str(patch_status_by_gid.get(gid1, "OK"))
        lg = str(getattr(m1, "league", "") or "").strip().upper()
        mk = _league_macro_key(lg)
        out.append(
            {
                "bo": int(bo),
                "y": int(y_series),
                "p_map": float(clip_prob(float(p_map))),
                "p_series_in": float(clip_prob(float(p_series_in))),
                "league": str(lg),
                "macro": str(mk),
                "patch_status": str(ps),
                "playoffs": bool(str(getattr(m1, "phase", "") or "").strip().lower() == "playoffs"),
                "same_league": bool(getattr(m1, "same_league", True)),
                "interregional": not bool(getattr(m1, "same_league", True)),
                "n_team_eff": float(max(0.0, float(getattr(m1, "n_team", 0.0) or 0.0))),
                "n_players_eff": float(max(0.0, float(getattr(m1, "n_players", 0.0) or 0.0))),
                "style_volatility": float(style_vol),
                "layer_confidence_score": float(conf_score),
            }
        )
    return out


def _series_dplite_candidates(bo: int, speed: str) -> List[Dict[str, float]]:
    b = int(bo)
    sp = str(speed or "full").strip().lower()
    if sp == "fast":
        if b == 3:
            pick = [0.017, 0.024]
            g1 = [0.001, 0.006]
            mom = [0.007, 0.012]
            lpd = [0.22, 0.38]
            dvar = [0.08, 0.13]
        else:
            pick = [0.024, 0.033]
            g1 = [0.008, 0.014]
            mom = [0.011, 0.016]
            lpd = [0.34, 0.52]
            dvar = [0.12, 0.18]
    else:
        if b == 3:
            pick = [0.014, 0.020, 0.026]
            g1 = [0.000, 0.004, 0.008]
            mom = [0.005, 0.009, 0.013]
            lpd = [0.16, 0.30, 0.44]
            dvar = [0.06, 0.10, 0.14]
        else:
            pick = [0.021, 0.028, 0.036]
            g1 = [0.005, 0.010, 0.016]
            mom = [0.009, 0.013, 0.018]
            lpd = [0.28, 0.42, 0.58]
            dvar = [0.10, 0.14, 0.20]
    out: List[Dict[str, float]] = []
    for a in pick:
        for b1 in g1:
            for c in mom:
                for d in lpd:
                    for e in dvar:
                        out.append(
                            {
                                "pick_shift": float(a),
                                "g1_blue_shift": float(b1),
                                "momentum_shift": float(c),
                                "loser_pick_decay": float(d),
                                "decider_variance": float(e),
                            }
                        )
    return out


def _series_v5_default_extras(bo: int) -> Dict[str, float]:
    b = int(bo)
    if b >= 5:
        return {"decider_side_revert": 0.22, "series_corr_sigma": 0.10}
    return {"decider_side_revert": 0.14, "series_corr_sigma": 0.07}


def _series_v5_extra_grid(bo: int, speed: str) -> List[Dict[str, float]]:
    b = int(bo)
    sp = str(speed or "full").strip().lower()
    if sp == "fast":
        if b >= 5:
            dsr = [0.18, 0.28]
            scs = [0.08, 0.13]
            aw = [0.68, 0.78]
            cap = [10.0, 12.0]
        else:
            dsr = [0.11, 0.17]
            scs = [0.05, 0.09]
            aw = [0.66, 0.74]
            cap = [7.0, 9.0]
    else:
        if b >= 5:
            dsr = [0.16, 0.22, 0.30]
            scs = [0.06, 0.10, 0.15]
            aw = [0.62, 0.70, 0.80]
            cap = [9.0, 11.0, 13.0]
        else:
            dsr = [0.10, 0.14, 0.20]
            scs = [0.04, 0.07, 0.11]
            aw = [0.60, 0.68, 0.76]
            cap = [6.0, 8.0, 10.0]
    out: List[Dict[str, float]] = []
    for a in dsr:
        for c in scs:
            for w in aw:
                for cp in cap:
                    out.append(
                        {
                            "decider_side_revert": float(a),
                            "series_corr_sigma": float(c),
                            "anchor_weight": float(w),
                            "cap_pp": float(cp),
                        }
                    )
    return out


def _merge_series_v5_params(bo: int, base: Dict[str, Any] | None, extra: Dict[str, Any] | None = None) -> Dict[str, float]:
    out: Dict[str, float] = dict(_series_v5_default_extras(int(bo)))
    for src in (base or {}, extra or {}):
        if not isinstance(src, dict):
            continue
        for k in (
            "pick_shift",
            "g1_blue_shift",
            "momentum_shift",
            "loser_pick_decay",
            "decider_variance",
            "decider_side_revert",
            "series_corr_sigma",
            "anchor_weight",
            "cap_pp",
        ):
            if k in src:
                try:
                    fv = float(src.get(k))
                except Exception:
                    continue
                if math.isfinite(fv):
                    out[k] = float(fv)
    return out


def _refine_series_v5_extras(
    rows: List[Dict[str, Any]],
    *,
    bo: int,
    base_params: Dict[str, Any],
    speed: str,
) -> Dict[str, Any]:
    rb = [r for r in rows if int(r.get("bo", 0)) == int(bo)]
    if not rb:
        return {"error": "no_rows"}
    p0 = _merge_series_v5_params(int(bo), base_params, None)
    m0 = _eval_series_dplite_rows(rb, bo=int(bo), cand=p0)
    if m0.get("error"):
        return {
            "params": dict(p0),
            "n": int(m0.get("n", 0)),
            "logloss": float("inf"),
            "brier": float("inf"),
            "gain_vs_base": 0.0,
            "grid_evals": 0,
        }

    best = {
        "params": dict(p0),
        "n": int(m0.get("n", 0)),
        "logloss": float(m0.get("logloss", 9.0)),
        "brier": float(m0.get("brier", 9.0)),
    }
    evals = 0
    for ex in _series_v5_extra_grid(int(bo), str(speed)):
        cand = _merge_series_v5_params(int(bo), base_params, ex)
        m = _eval_series_dplite_rows(rb, bo=int(bo), cand=cand)
        if m.get("error"):
            continue
        evals += 1
        ll = float(m.get("logloss", 9.0))
        br = float(m.get("brier", 9.0))
        if (ll < float(best.get("logloss", 9.0))) or (
            abs(ll - float(best.get("logloss", 9.0))) <= 1e-12 and br < float(best.get("brier", 9.0))
        ):
            best = {
                "params": dict(cand),
                "n": int(m.get("n", 0)),
                "logloss": float(ll),
                "brier": float(br),
            }
    gain = float(m0.get("logloss", float("nan"))) - float(best.get("logloss", float("nan")))
    best["gain_vs_base"] = float(gain if math.isfinite(gain) else 0.0)
    best["grid_evals"] = int(evals)
    return best


def _eval_series_dplite_rows(
    rows: List[Dict[str, Any]],
    *,
    bo: int,
    cand: Dict[str, float] | None,
) -> Dict[str, Any]:
    rb = [r for r in rows if int(r.get("bo", 0)) == int(bo)]
    if not rb:
        return {"error": "no_rows", "n": 0}
    y: List[int] = []
    p: List[float] = []
    _cfg = {}
    if isinstance(cand, dict) and cand:
        _cfg = {"global": {f"bo{int(bo)}": dict(cand)}}

    def _to_f(v: Any, default: float = float("nan")) -> float:
        try:
            x = float(v)
            return x if math.isfinite(x) else float(default)
        except Exception:
            return float(default)

    for r in rb:
        out = apply_series_bo_dplite_live(
            p_map=float(r.get("p_map", 0.5)),
            p_series_in=float(r.get("p_series_in", 0.5)),
            bo=int(bo),
            same_league=bool(r.get("same_league", True)),
            interregional=bool(r.get("interregional", False)),
            playoffs=bool(r.get("playoffs", False)),
            patch_status=str(r.get("patch_status", "")),
            n_team_eff=float(r.get("n_team_eff", 0.0)),
            n_players_eff=float(r.get("n_players_eff", 0.0)),
            side_bias_pp=0.0,
            style_volatility=float(r.get("style_volatility", 0.0)),
            layer_confidence_score=float(r.get("layer_confidence_score", 50.0)),
            league_hint=str(r.get("league", "")),
            tuning_cfg=_cfg,
        )
        p1 = _to_f(out.get("p_after"), float("nan"))
        if not math.isfinite(p1):
            p1 = _to_f(r.get("p_series_in"), 0.5)
        y.append(int(r.get("y", 0)))
        p.append(float(clip_prob(float(p1))))
    return {
        "n": len(y),
        "logloss": float(logloss(y, p)),
        "brier": float(brier(y, p)),
    }


def _fit_series_bo_dplite_tuning(
    samples: List[Sample],
    *,
    team_scale: float,
    players_scale: float,
    form_cfg: Dict[str, Any],
    speed_profile: str = "full",
) -> Dict[str, Any]:
    def _gain(a: float, b: float) -> float:
        if not (math.isfinite(float(a)) and math.isfinite(float(b))):
            return float("nan")
        return float(float(a) - float(b))

    rows = _build_series_rows_for_bo_dplite(
        samples,
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        form_cfg=dict(form_cfg or {}),
    )
    if not rows:
        return {"error": "no_series_rows"}

    sp = str(speed_profile or "full").strip().lower()
    if sp == "fast":
        min_n_league = 80
        min_n_patch = 120
        min_n_league_patch = 100
        min_gain = 0.0003
        top_k = 8
    else:
        min_n_league = 120
        min_n_patch = 180
        min_n_league_patch = 220
        min_gain = 0.0002
        top_k = 12

    out: Dict[str, Any] = {
        "version": "series_bo_dplite_tuning_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "speed_profile": str(sp),
        "meta": {
            "rows_total": len(rows),
            "rows_bo3": int(sum(1 for r in rows if int(r.get("bo", 0)) == 3)),
            "rows_bo5": int(sum(1 for r in rows if int(r.get("bo", 0)) == 5)),
            "min_n_league": int(min_n_league),
            "min_n_patch_status": int(min_n_patch),
            "min_n_league_patch": int(min_n_league_patch),
            "min_gain_logloss": float(min_gain),
            "top_k_for_segments": int(top_k),
        },
        "global": {},
        "by_league": {},
        "by_patch_status": {},
        "by_league_patch": {},
        "diagnostics": {},
    }

    for bo in (3, 5):
        rb = [r for r in rows if int(r.get("bo", 0)) == int(bo)]
        if len(rb) < (60 if sp == "fast" else 120):
            continue
        baseline = _eval_series_dplite_rows(rows, bo=int(bo), cand=None)
        cands = _series_dplite_candidates(int(bo), sp)
        scored: List[Dict[str, Any]] = []
        for c in cands:
            m = _eval_series_dplite_rows(rows, bo=int(bo), cand=c)
            if m.get("error"):
                continue
            scored.append(
                {
                    "params": dict(c),
                    "n": int(m.get("n", 0)),
                    "logloss": float(m.get("logloss", 9.0)),
                    "brier": float(m.get("brier", 9.0)),
                }
            )
        if not scored:
            continue
        scored = sorted(scored, key=lambda x: (float(x.get("logloss", 9.0)), float(x.get("brier", 9.0))))
        best = dict(scored[0])
        _ref = _refine_series_v5_extras(
            rb,
            bo=int(bo),
            base_params=dict(best.get("params") or {}),
            speed=str(sp),
        )
        if not _ref.get("error"):
            best["params"] = dict(_ref.get("params") or {})
            best["logloss"] = float(_ref.get("logloss", best.get("logloss", 9.0)))
            best["brier"] = float(_ref.get("brier", best.get("brier", 9.0)))
            best["n"] = int(_ref.get("n", best.get("n", 0)))
            best["extra_refine_gain"] = float(_ref.get("gain_vs_base", 0.0))
            best["extra_refine_grid_evals"] = int(_ref.get("grid_evals", 0))
        gain_vs_base = float(_gain(float(baseline.get("logloss", float("nan"))), float(best.get("logloss", float("nan")))))
        out["global"][f"bo{int(bo)}"] = {
            **dict(best.get("params") or {}),
            "n": int(best.get("n", 0)),
            "logloss": float(best.get("logloss", 9.0)),
            "brier": float(best.get("brier", 9.0)),
            "baseline_logloss_no_tuning": float(baseline.get("logloss", 9.0)),
            "improvement_logloss_vs_no_tuning": (float(gain_vs_base) if math.isfinite(gain_vs_base) else None),
            "extra_refine_gain": float(best.get("extra_refine_gain", 0.0)),
            "extra_refine_grid_evals": int(best.get("extra_refine_grid_evals", 0)),
        }

        shortlist = scored[: int(max(1, top_k))]
        seg_diag: Dict[str, Any] = {"league": {}, "patch_status": {}, "league_patch": {}}

        def _best_for_segment(seg_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not seg_rows:
                return {}
            # Evaluate global best inside segment as baseline.
            g_params = dict(best.get("params") or {})
            g_metrics = _eval_series_dplite_rows(seg_rows, bo=int(bo), cand=g_params)
            g_ll = float(g_metrics.get("logloss", float("nan")))
            g_extra = {
                "decider_side_revert": float(g_params.get("decider_side_revert"))
                if math.isfinite(float(g_params.get("decider_side_revert", float("nan"))))
                else float(_series_v5_default_extras(int(bo)).get("decider_side_revert", 0.22)),
                "series_corr_sigma": float(g_params.get("series_corr_sigma"))
                if math.isfinite(float(g_params.get("series_corr_sigma", float("nan"))))
                else float(_series_v5_default_extras(int(bo)).get("series_corr_sigma", 0.10)),
            }
            local_best = {
                "params": dict(_merge_series_v5_params(int(bo), g_params, None)),
                "logloss": float(g_ll) if math.isfinite(g_ll) else float("inf"),
                "brier": float(g_metrics.get("brier", float("nan"))) if g_metrics.get("brier") is not None else float("inf"),
                "n": int(g_metrics.get("n", 0)),
                "gain_vs_global": 0.0,
            }
            for cand_node in shortlist:
                cand_params = _merge_series_v5_params(int(bo), cand_node.get("params") or {}, g_extra)
                m = _eval_series_dplite_rows(seg_rows, bo=int(bo), cand=cand_params)
                if m.get("error"):
                    continue
                ll = float(m.get("logloss", 9.0))
                if ll < float(local_best.get("logloss", float("inf"))):
                    gain = _gain(float(g_ll), float(ll))
                    local_best = {
                        "params": cand_params,
                        "logloss": float(ll),
                        "brier": float(m.get("brier", 9.0)),
                        "n": int(m.get("n", 0)),
                        "gain_vs_global": (float(gain) if math.isfinite(gain) else 0.0),
                    }
            return local_best

        # by league
        leagues = sorted({str(r.get("league", "")).strip().upper() for r in rb if str(r.get("league", "")).strip()})
        for lg in leagues:
            seg = [r for r in rb if str(r.get("league", "")).strip().upper() == lg]
            nseg = len(seg)
            if nseg < int(min_n_league):
                seg_diag["league"][str(lg)] = {"n": int(nseg), "selected": False, "reason": "low_n"}
                continue
            bseg = _best_for_segment(seg)
            gain = float(bseg.get("gain_vs_global", 0.0))
            selected = bool(gain >= float(min_gain))
            seg_diag["league"][str(lg)] = {
                "n": int(nseg),
                "selected": bool(selected),
                "gain_vs_global": float(gain),
                "logloss": float(bseg.get("logloss", 9.0)),
                "reason": ("ok" if selected else "low_gain"),
            }
            if selected:
                node = out["by_league"].setdefault(str(lg), {})
                node[f"bo{int(bo)}"] = {
                    **dict(bseg.get("params") or {}),
                    "n": int(nseg),
                    "logloss": float(bseg.get("logloss", 9.0)),
                    "gain_vs_global": float(gain),
                }

        # by patch status
        pstats = sorted({str(r.get("patch_status", "")).strip().upper() for r in rb if str(r.get("patch_status", "")).strip()})
        for ps in pstats:
            seg = [r for r in rb if str(r.get("patch_status", "")).strip().upper() == ps]
            nseg = len(seg)
            if nseg < int(min_n_patch):
                seg_diag["patch_status"][str(ps)] = {"n": int(nseg), "selected": False, "reason": "low_n"}
                continue
            bseg = _best_for_segment(seg)
            gain = float(bseg.get("gain_vs_global", 0.0))
            selected = bool(gain >= float(min_gain))
            seg_diag["patch_status"][str(ps)] = {
                "n": int(nseg),
                "selected": bool(selected),
                "gain_vs_global": float(gain),
                "logloss": float(bseg.get("logloss", 9.0)),
                "reason": ("ok" if selected else "low_gain"),
            }
            if selected:
                node = out["by_patch_status"].setdefault(str(ps), {})
                node[f"bo{int(bo)}"] = {
                    **dict(bseg.get("params") or {}),
                    "n": int(nseg),
                    "logloss": float(bseg.get("logloss", 9.0)),
                    "gain_vs_global": float(gain),
                }

        # by league + patch status (only when high evidence)
        combo_keys = sorted(
            {
                f"{str(r.get('league', '')).strip().upper()}|{str(r.get('patch_status', '')).strip().upper()}"
                for r in rb
                if str(r.get("league", "")).strip() and str(r.get("patch_status", "")).strip()
            }
        )
        for ck in combo_keys:
            lg, ps = ck.split("|", 1)
            seg = [
                r
                for r in rb
                if str(r.get("league", "")).strip().upper() == lg
                and str(r.get("patch_status", "")).strip().upper() == ps
            ]
            nseg = len(seg)
            if nseg < int(min_n_league_patch):
                seg_diag["league_patch"][str(ck)] = {"n": int(nseg), "selected": False, "reason": "low_n"}
                continue
            bseg = _best_for_segment(seg)
            gain = float(bseg.get("gain_vs_global", 0.0))
            selected = bool(gain >= float(min_gain))
            seg_diag["league_patch"][str(ck)] = {
                "n": int(nseg),
                "selected": bool(selected),
                "gain_vs_global": float(gain),
                "logloss": float(bseg.get("logloss", 9.0)),
                "reason": ("ok" if selected else "low_gain"),
            }
            if selected:
                node = out["by_league_patch"].setdefault(str(ck), {})
                node[f"bo{int(bo)}"] = {
                    **dict(bseg.get("params") or {}),
                    "n": int(nseg),
                    "logloss": float(bseg.get("logloss", 9.0)),
                    "gain_vs_global": float(gain),
                }

        out["diagnostics"][f"bo{int(bo)}"] = {
            "rows": len(rb),
            "candidate_count": len(cands),
            "shortlist_count": len(shortlist),
            "global_best_logloss": float(best.get("logloss", 9.0)),
            "global_baseline_logloss_no_tuning": float(baseline.get("logloss", 9.0)),
            "global_gain_vs_no_tuning": (float(gain_vs_base) if math.isfinite(gain_vs_base) else None),
            "segments": seg_diag,
        }

    _gg = out.get("global") if isinstance(out.get("global"), dict) else {}
    if isinstance(_gg, dict):
        if ("bo3" in _gg) and ("bo5" not in _gg):
            _b3 = dict(_gg.get("bo3") or {})
            _gg["bo5"] = {
                "pick_shift": float(_b3.get("pick_shift", 0.028) or 0.028),
                "g1_blue_shift": float(_b3.get("g1_blue_shift", 0.010) or 0.010),
                "momentum_shift": float(_b3.get("momentum_shift", 0.013) or 0.013),
                "loser_pick_decay": float(_b3.get("loser_pick_decay", 0.42) or 0.42),
                "decider_variance": float(_b3.get("decider_variance", 0.14) or 0.14),
                "decider_side_revert": float(_b3.get("decider_side_revert", 0.22) or 0.22),
                "series_corr_sigma": float(_b3.get("series_corr_sigma", 0.10) or 0.10),
                "n": 0,
                "fallback_from": "bo3",
            }
        elif ("bo5" in _gg) and ("bo3" not in _gg):
            _b5 = dict(_gg.get("bo5") or {})
            _gg["bo3"] = {
                "pick_shift": float(_b5.get("pick_shift", 0.020) or 0.020),
                "g1_blue_shift": float(_b5.get("g1_blue_shift", 0.004) or 0.004),
                "momentum_shift": float(_b5.get("momentum_shift", 0.009) or 0.009),
                "loser_pick_decay": float(_b5.get("loser_pick_decay", 0.30) or 0.30),
                "decider_variance": float(_b5.get("decider_variance", 0.10) or 0.10),
                "decider_side_revert": float(_b5.get("decider_side_revert", 0.14) or 0.14),
                "series_corr_sigma": float(_b5.get("series_corr_sigma", 0.07) or 0.07),
                "n": 0,
                "fallback_from": "bo5",
            }
    if not _gg:
        return {"error": "insufficient_series_rows", "meta": out.get("meta", {})}
    return out


