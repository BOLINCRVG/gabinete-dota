from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .players_context_core import compute_players_context_quality_signal
from .utils import brier, clip_prob, logit, logloss, safe_float as _f, sigmoid
from .fit_samples_core import Sample, _ece_metric, _fuse_probs_by_precision, _is_interregional_league
from .fit_form_core import _apply_form_blend_on_sample
from .fit_players_weights_core import _fit_players_lane_weights, _fit_players_league_weights  # noqa: F401


def _apply_logit_delta(p_in: float, delta: float) -> float:
    try:
        p0 = float(clip_prob(float(p_in)))
        d = float(delta)
        if (not math.isfinite(d)) or abs(d) <= 1e-12:
            return float(p0)
        return float(clip_prob(float(sigmoid(float(logit(p0)) + float(d)))))
    except Exception:
        return float(clip_prob(float(p_in)))


def _players_context_settings_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    c = cfg if isinstance(cfg, dict) else {}
    return {
        "ml_players_context_enabled": bool(c.get("enabled", True)),
        "ml_players_context_lookback_games": int(c.get("lookback_games", 24) or 24),
        "ml_players_context_half_life_days": float(c.get("half_life_days", 45.0) or 45.0),
        "ml_players_context_min_games": int(c.get("min_games", 6) or 6),
        "ml_players_context_strength": float(c.get("strength", 0.35) or 0.35),
        "ml_players_context_cap": float(c.get("cap", 0.22) or 0.22),
        "ml_players_context_min_reliability": float(c.get("min_reliability", 0.28) or 0.28),
        "ml_players_context_inter_mult": float(c.get("interregional_mult", 0.88) or 0.88),
    }


def _players_context_info_for_sample(
    sample: Sample,
    *,
    team_games: pd.DataFrame,
    cfg: Dict[str, Any],
    cache: Dict[Tuple[Any, ...], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    cc = cache if isinstance(cache, dict) else {}
    lookback_games = int(max(4, min(80, int((cfg or {}).get("lookback_games", 24) or 24))))
    half_life_days = float(max(5.0, min(180.0, float((cfg or {}).get("half_life_days", 45.0) or 45.0))))
    min_games = int(max(2, min(24, int((cfg or {}).get("min_games", 6) or 6))))
    strength = float(max(0.0, min(2.0, float((cfg or {}).get("strength", 0.35) or 0.35))))
    cap = float(max(0.02, min(0.75, float((cfg or {}).get("cap", 0.22) or 0.22))))
    min_rel = float(max(0.05, min(0.95, float((cfg or {}).get("min_reliability", 0.28) or 0.28))))
    inter_mult = float(max(0.60, min(1.00, float((cfg or {}).get("interregional_mult", 0.88) or 0.88))))
    enabled = bool((cfg or {}).get("enabled", False))

    cache_key = (
        str(sample.gameid or ""),
        str(sample.blue_team or ""),
        str(sample.red_team or ""),
        str(sample.league or ""),
        str(sample.split or ""),
        str(sample.phase or ""),
        str(sample.date_day or ""),
        bool(enabled),
        int(lookback_games),
        round(float(half_life_days), 4),
        int(min_games),
        round(float(strength), 4),
        round(float(cap), 4),
        round(float(min_rel), 4),
        round(float(inter_mult), 4),
    )
    if cache_key in cc:
        return dict(cc.get(cache_key) or {})

    if (team_games is None) or team_games.empty:
        out = {"enabled": bool(enabled), "applied": False, "reason": "no_team_games", "delta": 0.0}
        cc[cache_key] = out
        return out

    year_hint: Any = None
    try:
        _month = str(sample.month or "")
        year_hint = int(_month.split("-", 1)[0])
    except Exception:
        year_hint = None
    playoffs_hint = 1 if str(sample.phase or "").strip().lower() == "playoffs" else 0
    split_hint = str(sample.split or "").strip()
    league_hint = str(sample.league or "").strip()
    interregional = bool((not bool(sample.same_league)) or _is_interregional_league(league_hint))

    filters = {
        "year": year_hint,
        "split": (split_hint or None),
        "playoffs": int(playoffs_hint),
        "league": (league_hint or None),
    }
    try:
        ctx = compute_players_context_quality_signal(
            team_games=team_games,
            team_a=str(sample.blue_team or ""),
            team_b=str(sample.red_team or ""),
            filters=filters,
            as_of_date=(str(sample.date_day) if str(sample.date_day or "").strip() else str(sample.month or "")),
            enabled=bool(enabled),
            lookback_games=int(lookback_games),
            half_life_days=float(half_life_days),
            min_games=int(min_games),
            strength=float(strength),
            cap=float(cap),
            min_reliability=float(min_rel),
            interregional=bool(interregional),
            interregional_mult=float(inter_mult),
            league_hint_a=str(league_hint),
            league_hint_b=str(league_hint),
        )
        out = dict(ctx) if isinstance(ctx, dict) else {"enabled": bool(enabled), "applied": False, "reason": "invalid_ctx", "delta": 0.0}
    except Exception:
        out = {"enabled": bool(enabled), "applied": False, "reason": "ctx_exception", "delta": 0.0}
    cc[cache_key] = out
    return dict(out)


def _evaluate_walkforward_players_context(
    samples: List[Sample],
    *,
    team_scale: float,
    players_scale: float,
    min_train_games: int,
    team_games: pd.DataFrame,
    players_context_cfg: Dict[str, Any] | None = None,
    league_filter: str | None = None,
    macro_filter: str | None = None,
    form_cfg: Dict[str, Any] | None = None,
    return_rows: bool = False,
    ctx_cache: Dict[Tuple[Any, ...], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if league_filter:
        lg = str(league_filter)
        samples = [s for s in samples if str(s.league) == lg]
    if macro_filter:
        mk = str(macro_filter)
        samples = [s for s in samples if str(s.macro) == mk]
    if not samples:
        return {
            "folds": [],
            "logloss": None,
            "brier": None,
            "ece": None,
            "n_total": 0,
            "context": {
                "enabled": bool((players_context_cfg or {}).get("enabled", False)),
                "has_team_games": bool(isinstance(team_games, pd.DataFrame) and not team_games.empty),
                "ctx_total": 0,
                "ctx_applied": 0,
                "ctx_applied_rate": 0.0,
                "ctx_mean_abs_delta": 0.0,
                "reason_counts": {},
            },
        }

    by_month: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_month[str(sample.month)].append(sample)
    months = sorted(by_month.keys())
    fold_sets: List[Tuple[str, int, List[Sample]]] = []
    train_n_running = 0
    for idx, month_key in enumerate(months):
        month_samples = list(by_month.get(month_key) or [])
        if idx > 0:
            fold_sets.append((str(month_key), int(train_n_running), month_samples))
        train_n_running += len(month_samples)
    folds: List[Dict[str, Any]] = []
    all_y: List[int] = []
    all_p: List[float] = []
    pred_rows: List[Dict[str, Any]] = []

    cfg = dict(players_context_cfg or {})
    ctx_enabled = bool(cfg.get("enabled", False))
    has_team_games = bool(isinstance(team_games, pd.DataFrame) and not team_games.empty)
    cache = ctx_cache if isinstance(ctx_cache, dict) else {}
    ctx_total = 0
    ctx_applied = 0
    ctx_abs_sum = 0.0
    reason_counts: Dict[str, int] = defaultdict(int)

    for m, train_n, test in fold_sets:
        if train_n < int(min_train_games):
            continue
        if not test:
            continue
        y = [int(s.y) for s in test]
        p = []
        for s in test:
            p_pre = _fuse_probs_by_precision(
                p_team=float(s.p_team),
                p_players=float(s.p_players),
                n_team=float(s.n_team),
                n_players=float(s.n_players),
                team_scale=float(team_scale),
                players_scale=float(players_scale),
                coverage=float(s.coverage_ratio),
                transfer_signal=float(s.transfer_signal),
                lineup_continuity=float(s.lineup_continuity),
                roster_changed=bool(s.roster_changed),
                season_phase=("playoffs" if str(s.phase).strip().lower() == "playoffs" else "auto"),
                players_freshness=float(s.freshness_avg),
            )
            p_post = _apply_form_blend_on_sample(
                p_model=float(p_pre),
                sample=s,
                enabled=bool((form_cfg or {}).get("enabled", False)),
                max_weight=float((form_cfg or {}).get("max_weight", 0.75) or 0.75),
                knee_games=float((form_cfg or {}).get("knee_games", 6.0) or 6.0),
                beta_prior=float((form_cfg or {}).get("beta_prior", 2.0) or 2.0),
                signal_power=float((form_cfg or {}).get("signal_power", 1.0) or 1.0),
                elite_same_league_parity_strength=float(
                    (form_cfg or {}).get("elite_same_league_parity_strength", 0.35) or 0.35
                ),
            )
            p_used = float(p_post)
            ctx_delta = 0.0
            ctx_applied_i = False
            if bool(ctx_enabled) and bool(has_team_games):
                ctx_total += 1
                ctx_info = _players_context_info_for_sample(
                    s,
                    team_games=team_games,
                    cfg=cfg,
                    cache=cache,
                )
                ctx_delta = _f((ctx_info or {}).get("delta"), 0.0)
                if math.isfinite(ctx_delta):
                    ctx_abs_sum += float(abs(ctx_delta))
                rs = str((ctx_info or {}).get("reason") or "").strip().lower()
                if rs:
                    reason_counts[rs] += 1
                ctx_applied_i = bool((ctx_info or {}).get("applied", False)) and math.isfinite(ctx_delta) and abs(ctx_delta) > 1e-12
                if ctx_applied_i:
                    p_used = _apply_logit_delta(float(p_post), float(ctx_delta))
                    ctx_applied += 1

            p.append(float(clip_prob(float(p_used))))
            if bool(return_rows):
                pred_rows.append(
                    {
                        "month": str(m),
                        "league": str(s.league),
                        "split": str(s.split),
                        "side": str(s.side),
                        "bo": str(s.bo_bucket),
                        "phase": str(s.phase),
                        "y": int(s.y),
                        "p": float(p_used),
                        "ctx_delta": float(ctx_delta),
                        "ctx_applied": bool(ctx_applied_i),
                    }
                )

        ll = float(logloss(y, p))
        br = float(brier(y, p))
        folds.append({"month": m, "n": len(y), "logloss": ll, "brier": br})
        all_y.extend(y)
        all_p.extend(p)

    ctx_summary = {
        "enabled": bool(ctx_enabled),
        "has_team_games": bool(has_team_games),
        "ctx_total": int(ctx_total),
        "ctx_applied": int(ctx_applied),
        "ctx_applied_rate": (float(ctx_applied) / float(max(1, ctx_total))),
        "ctx_mean_abs_delta": (float(ctx_abs_sum) / float(max(1, ctx_total))),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "cache_entries": len(cache),
    }

    if not all_y:
        rep0 = {"folds": folds, "logloss": None, "brier": None, "ece": None, "n_total": 0, "context": ctx_summary}
        if bool(return_rows):
            rep0["pred_rows"] = pred_rows
        return rep0
    rep = {
        "folds": folds,
        "logloss": float(logloss(all_y, all_p)),
        "brier": float(brier(all_y, all_p)),
        "ece": _ece_metric(all_y, all_p, bins=10),
        "n_total": len(all_y),
        "context": ctx_summary,
    }
    if bool(return_rows):
        rep["pred_rows"] = pred_rows
    return rep


def _fit_players_context_tuning(
    samples: List[Sample],
    *,
    team_games: pd.DataFrame,
    team_scale: float,
    players_scale: float,
    min_train_games: int,
    form_cfg: Dict[str, Any],
    speed_profile: str = "full",
    budget_profile: Optional[str] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    checkpoint_state: Optional[Dict[str, Any]] = None,
    checkpoint_save_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    base_cfg = {
        "enabled": True,
        "lookback_games": 24,
        "half_life_days": 45.0,
        "min_games": 6,
        "strength": 0.35,
        "cap": 0.22,
        "min_reliability": 0.28,
        "interregional_mult": 0.88,
    }
    out_base = {
        "enabled": False,
        "version": "ml_players_context_fit_v1",
        "source": "walkforward_oos_ablation",
        "settings": _players_context_settings_from_cfg(base_cfg),
        "promoted_settings": {},
        "settings_by_league": {},
        "settings_by_macro": {},
        "promoted_settings_by_league": {},
        "promoted_settings_by_macro": {},
        "gate": {"decision": "shadow_only", "reasons": ["not_evaluated"]},
    }
    if not samples:
        return {**out_base, "reason": "no_samples"}
    if not isinstance(team_games, pd.DataFrame) or team_games.empty:
        return {**out_base, "reason": "no_team_games"}

    speed = str(speed_profile or "full").strip().lower()
    if speed not in {"full", "fast"}:
        speed = "full"
    budget = str(budget_profile or speed).strip().lower()
    if budget not in {"full", "fast", "runtime_full"}:
        budget = str(speed)
    if budget == "fast" and len(samples) > 1600:
        # Fast profile: keep a tight recent window to avoid quadratic-ish
        # runtime in context lookups while preserving current-meta signal.
        samples = list(samples[-1600:])
    elif budget == "runtime_full" and len(samples) > 2400:
        # Auto maintenance still needs a stronger profile than `fast`, but the
        # full sample history is too expensive for daily runs.
        samples = list(samples[-2400:])

    def _pulse(payload: Optional[Dict[str, Any]] = None) -> None:
        if not callable(progress_cb):
            return
        try:
            progress_cb(dict(payload or {}))
        except Exception:
            pass

    def _save_partial(payload: Optional[Dict[str, Any]] = None) -> None:
        if not callable(checkpoint_save_cb):
            return
        try:
            checkpoint_save_cb(dict(payload or {}))
        except Exception:
            pass

    def _eta_payload(*, started_at: float, index: int, total: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = dict(extra or {})
        out["index"] = int(max(0, index))
        out["total"] = int(max(1, total))
        elapsed_s = float(max(0.0, time.perf_counter() - float(started_at)))
        out["elapsed_s"] = float(elapsed_s)
        if total > 0 and index > 0:
            eta_s = float(max(0.0, (elapsed_s / float(index)) * float(max(0, total - index))))
            out["eta_s"] = float(eta_s)
        return out
    samples_by_league: Dict[str, List[Sample]] = defaultdict(list)
    samples_by_macro: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        lg = str(sample.league or "").strip()
        mk = str(sample.macro or "").strip()
        if lg:
            samples_by_league[lg].append(sample)
        if mk:
            samples_by_macro[mk].append(sample)

    candidates: List[Dict[str, Any]] = [
        dict(base_cfg),
        {"enabled": True, "lookback_games": 16, "half_life_days": 35.0, "min_games": 5, "strength": 0.30, "cap": 0.18, "min_reliability": 0.24, "interregional_mult": 0.88},
        {"enabled": True, "lookback_games": 16, "half_life_days": 35.0, "min_games": 5, "strength": 0.45, "cap": 0.18, "min_reliability": 0.28, "interregional_mult": 0.88},
        {"enabled": True, "lookback_games": 24, "half_life_days": 45.0, "min_games": 6, "strength": 0.25, "cap": 0.18, "min_reliability": 0.32, "interregional_mult": 0.82},
        {"enabled": True, "lookback_games": 24, "half_life_days": 45.0, "min_games": 6, "strength": 0.55, "cap": 0.28, "min_reliability": 0.24, "interregional_mult": 0.90},
        {"enabled": True, "lookback_games": 32, "half_life_days": 60.0, "min_games": 8, "strength": 0.35, "cap": 0.22, "min_reliability": 0.30, "interregional_mult": 0.85},
        {"enabled": True, "lookback_games": 12, "half_life_days": 28.0, "min_games": 4, "strength": 0.42, "cap": 0.16, "min_reliability": 0.22, "interregional_mult": 0.90},
        {"enabled": True, "lookback_games": 20, "half_life_days": 45.0, "min_games": 6, "strength": 0.35, "cap": 0.14, "min_reliability": 0.28, "interregional_mult": 0.88},
        {"enabled": True, "lookback_games": 28, "half_life_days": 55.0, "min_games": 7, "strength": 0.40, "cap": 0.30, "min_reliability": 0.28, "interregional_mult": 0.85},
        {"enabled": True, "lookback_games": 24, "half_life_days": 45.0, "min_games": 6, "strength": 0.35, "cap": 0.22, "min_reliability": 0.28, "interregional_mult": 0.75},
        {"enabled": True, "lookback_games": 24, "half_life_days": 45.0, "min_games": 6, "strength": 0.35, "cap": 0.22, "min_reliability": 0.28, "interregional_mult": 0.95},
        {"enabled": True, "lookback_games": 36, "half_life_days": 75.0, "min_games": 10, "strength": 0.33, "cap": 0.20, "min_reliability": 0.34, "interregional_mult": 0.82},
        {"enabled": True, "lookback_games": 28, "half_life_days": 60.0, "min_games": 8, "strength": 0.18, "cap": 0.12, "min_reliability": 0.36, "interregional_mult": 0.80},
        {"enabled": True, "lookback_games": 32, "half_life_days": 70.0, "min_games": 9, "strength": 0.12, "cap": 0.10, "min_reliability": 0.40, "interregional_mult": 0.78},
        {"enabled": True, "lookback_games": 36, "half_life_days": 85.0, "min_games": 10, "strength": 0.08, "cap": 0.08, "min_reliability": 0.44, "interregional_mult": 0.75},
        {"enabled": True, "lookback_games": 40, "half_life_days": 100.0, "min_games": 12, "strength": 0.05, "cap": 0.06, "min_reliability": 0.48, "interregional_mult": 0.70},
    ]
    if budget == "fast":
        # Keep a representative envelope in fast mode:
        # - core defaults/aggressive candidates for lift
        # - conservative tail to protect calibration (ECE) regressions.
        head = candidates[:6]
        tail = candidates[-4:]
        candidates = head + tail
    elif budget == "runtime_full":
        # Runtime auto-full keeps a broader envelope than `fast`, but trims the
        # long conservative tail that dominates wall-clock time in daily runs.
        candidates = candidates[:6] + candidates[-2:]

    # Deduplicate candidate tuples.
    dedup: List[Dict[str, Any]] = []
    seen_cfg: set[Tuple[Any, ...]] = set()
    for c in candidates:
        key = (
            int(c.get("lookback_games", 24) or 24),
            round(float(c.get("half_life_days", 45.0) or 45.0), 6),
            int(c.get("min_games", 6) or 6),
            round(float(c.get("strength", 0.35) or 0.35), 6),
            round(float(c.get("cap", 0.22) or 0.22), 6),
            round(float(c.get("min_reliability", 0.28) or 0.28), 6),
            round(float(c.get("interregional_mult", 0.88) or 0.88), 6),
        )
        if key in seen_cfg:
            continue
        seen_cfg.add(key)
        dedup.append(dict(c))
    candidates = dedup

    ctx_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    partial_state = dict(checkpoint_state or {}) if isinstance(checkpoint_state, dict) else {}
    partial_phase = str(partial_state.get("phase") or "").strip().lower()
    if partial_phase == "complete" and isinstance(partial_state.get("final_output"), dict):
        return dict(partial_state.get("final_output") or {})
    partial_meta = dict(partial_state.get("meta") or {}) if isinstance(partial_state.get("meta"), dict) else {}
    rep_off = dict(partial_state.get("baseline_off") or {}) if partial_phase in {"global_running", "global_done", "segments_running", "complete"} else {}
    if not rep_off:
        rep_off = _evaluate_walkforward_players_context(
            samples,
            team_scale=float(team_scale),
            players_scale=float(players_scale),
            min_train_games=int(min_train_games),
            team_games=team_games,
            players_context_cfg={"enabled": False},
            form_cfg=form_cfg,
            ctx_cache=ctx_cache,
        )
    off_ll = _f(rep_off.get("logloss"), float("nan"))
    off_br = _f(rep_off.get("brier"), float("nan"))
    off_ece = _f(rep_off.get("ece"), float("nan"))
    _pulse(_eta_payload(started_at=time.perf_counter(), index=0, total=len(candidates), extra={"scope": "global", "phase": "baseline_off"}))
    if not math.isfinite(off_ll):
        return {
            **out_base,
            "reason": "off_eval_invalid",
            "baseline_off": rep_off,
            "meta": {"cache_entries": len(ctx_cache), "candidates_tested": 0},
        }

    grid_rows: List[Dict[str, Any]] = list(partial_state.get("grid_rows") or []) if partial_phase in {"global_running", "global_done", "segments_running", "complete"} else []
    cand_total = len(candidates)
    cand_done = int(max(0, min(cand_total, int(partial_meta.get("candidates_completed") or 0))))
    if (partial_phase not in {"global_running", "global_done", "segments_running", "complete"}) or (cand_done < cand_total):
        _cand_started = time.perf_counter()
        for cand_idx, cfg in enumerate(candidates[cand_done:], start=(cand_done + 1)):
            _pulse(
                _eta_payload(
                    started_at=_cand_started,
                    index=int(cand_idx),
                    total=int(cand_total),
                    extra={
                        "scope": "global",
                        "phase": "candidate_eval",
                        "lookback_games": int(cfg.get("lookback_games", 24) or 24),
                        "strength": float(cfg.get("strength", 0.35) or 0.35),
                        "cap": float(cfg.get("cap", 0.22) or 0.22),
                    },
                )
            )
            rep_on = _evaluate_walkforward_players_context(
                samples,
                team_scale=float(team_scale),
                players_scale=float(players_scale),
                min_train_games=int(min_train_games),
                team_games=team_games,
                players_context_cfg=dict(cfg),
                form_cfg=form_cfg,
                ctx_cache=ctx_cache,
            )
            ll = _f(rep_on.get("logloss"), float("nan"))
            if not math.isfinite(ll):
                continue
            br = _f(rep_on.get("brier"), float("nan"))
            ece = _f(rep_on.get("ece"), float("nan"))
            row = {
                "cfg": dict(cfg),
                "settings": _players_context_settings_from_cfg(cfg),
                "n_total": int(rep_on.get("n_total") or 0),
                "logloss": float(ll),
                "brier": (float(br) if math.isfinite(br) else None),
                "ece": (float(ece) if math.isfinite(ece) else None),
                "improvement_logloss_vs_off": (float(off_ll - ll) if math.isfinite(off_ll) else None),
                "improvement_brier_vs_off": (float(off_br - br) if (math.isfinite(off_br) and math.isfinite(br)) else None),
                "improvement_ece_vs_off": (float(off_ece - ece) if (math.isfinite(off_ece) and math.isfinite(ece)) else None),
                "context": dict(rep_on.get("context") or {}),
            }
            grid_rows.append(row)
            _save_partial(
                {
                        "phase": "global_running",
                        "baseline_off": rep_off,
                        "grid_rows": list(grid_rows),
                        "meta": {"candidates_total": int(cand_total), "candidates_completed": int(cand_idx)},
                    }
                )

    if not grid_rows:
        return {
            **out_base,
            "reason": "no_valid_on_candidates",
            "baseline_off": rep_off,
            "meta": {"cache_entries": len(ctx_cache), "candidates_tested": len(candidates)},
        }

    grid_sorted = sorted(
        grid_rows,
        key=lambda r: (
            float(r.get("logloss", 9e9)),
            float(r.get("brier")) if r.get("brier") is not None else 9e9,
            float(r.get("ece")) if r.get("ece") is not None else 9e9,
        ),
    )
    # Prefer candidates that are calibration-safe; fallback to pure best logloss.
    _safe_rows = [
        r
        for r in grid_sorted
        if (
            (r.get("improvement_brier_vs_off") is None or float(r.get("improvement_brier_vs_off")) >= -0.0006)
            and (r.get("improvement_ece_vs_off") is None or float(r.get("improvement_ece_vs_off")) >= -0.0015)
        )
    ]
    _selected_pool = _safe_rows if _safe_rows else grid_sorted
    best = dict(_selected_pool[0])
    best_cfg = dict(best.get("cfg") or {})
    on_ll = _f(best.get("logloss"), float("nan"))
    on_br = _f(best.get("brier"), float("nan"))
    on_ece = _f(best.get("ece"), float("nan"))
    gain_ll = float(off_ll - on_ll) if (math.isfinite(off_ll) and math.isfinite(on_ll)) else float("nan")
    gain_br = float(off_br - on_br) if (math.isfinite(off_br) and math.isfinite(on_br)) else float("nan")
    gain_ece = float(off_ece - on_ece) if (math.isfinite(off_ece) and math.isfinite(on_ece)) else float("nan")
    _save_partial(
        {
            "phase": "global_done",
            "baseline_off": rep_off,
            "grid_rows": list(grid_rows),
            "meta": {
                "candidates_total": int(cand_total),
                "candidates_completed": int(cand_total),
                "best_logloss": (float(on_ll) if math.isfinite(on_ll) else None),
            },
        }
    )

    gate_decision = "promote"
    gate_reasons: List[str] = []
    if budget == "fast":
        min_n_gate = int(max(100, int(min_train_games * 0.12)))
    elif budget == "runtime_full":
        min_n_gate = int(max(140, int(min_train_games * 0.15)))
    else:
        min_n_gate = int(max(180, int(min_train_games * 0.18)))
    n_best = int(best.get("n_total") or 0)
    if n_best < min_n_gate:
        gate_decision = "shadow_only"
        gate_reasons.append("low_sample")
    if (not math.isfinite(gain_ll)) or (gain_ll < 0.0004):
        gate_decision = "shadow_only"
        gate_reasons.append("logloss_gain_small")
    if math.isfinite(off_ll) and math.isfinite(on_ll) and (on_ll > off_ll + 0.0010):
        gate_decision = "shadow_only"
        gate_reasons.append("logloss_regression")
    if math.isfinite(gain_br) and (gain_br < -0.0006):
        gate_decision = "shadow_only"
        gate_reasons.append("brier_regression")
    if math.isfinite(gain_ece) and (gain_ece < -0.0015):
        gate_decision = "shadow_only"
        gate_reasons.append("ece_regression")
    if not gate_reasons:
        gate_reasons = ["pass"]

    # Optional segment-level overrides from global top candidates.
    seg_pool = [dict(r.get("cfg") or {}) for r in grid_sorted[:4] if isinstance(r.get("cfg"), dict)]
    if best_cfg not in seg_pool:
        seg_pool = [dict(best_cfg), *seg_pool]
    if budget == "fast":
        min_n_lg = 140
        min_n_mk = 160
        seg_pool = []
    elif budget == "runtime_full":
        min_n_lg = 180
        min_n_mk = 210
        seg_pool = [dict(r.get("cfg") or {}) for r in grid_sorted[:3] if isinstance(r.get("cfg"), dict)]
        if best_cfg not in seg_pool:
            seg_pool = [dict(best_cfg), *seg_pool]
    else:
        min_n_lg = 220
        min_n_mk = 260
    min_gain_seg = float(0.0006)

    by_league: Dict[str, Any] = dict(partial_state.get("by_league") or {}) if partial_phase in {"segments_running", "complete"} else {}
    by_macro: Dict[str, Any] = dict(partial_state.get("by_macro") or {}) if partial_phase in {"segments_running", "complete"} else {}
    seg_diag_league: Dict[str, Any] = dict(partial_state.get("seg_diag_league") or {}) if partial_phase in {"segments_running", "complete"} else {}
    seg_diag_macro: Dict[str, Any] = dict(partial_state.get("seg_diag_macro") or {}) if partial_phase in {"segments_running", "complete"} else {}

    def _fit_segment(segment_samples: List[Sample], *, league_filter: str | None = None, macro_filter: str | None = None) -> Dict[str, Any]:
        rep_off_seg = _evaluate_walkforward_players_context(
            segment_samples,
            team_scale=float(team_scale),
            players_scale=float(players_scale),
            min_train_games=int(max(80, min_train_games * 0.35)),
            team_games=team_games,
            players_context_cfg={"enabled": False},
            form_cfg=form_cfg,
            ctx_cache=ctx_cache,
        )
        n_seg = int(rep_off_seg.get("n_total") or 0)
        min_n_req = int(min_n_lg if league_filter else min_n_mk)
        if n_seg < min_n_req:
            return {
                "selected": False,
                "reason": "low_n",
                "n_total": int(n_seg),
                "min_n_required": int(min_n_req),
                "baseline_off": {
                    "logloss": rep_off_seg.get("logloss"),
                    "brier": rep_off_seg.get("brier"),
                    "ece": rep_off_seg.get("ece"),
                },
            }
        ll_off_seg = _f(rep_off_seg.get("logloss"), float("nan"))
        best_seg: Dict[str, Any] | None = None
        seg_total = len(seg_pool)
        _seg_started = time.perf_counter()
        for seg_idx, cfg_seg in enumerate(seg_pool, start=1):
            _pulse(
                _eta_payload(
                    started_at=_seg_started,
                    index=int(seg_idx),
                    total=int(seg_total),
                    extra={
                        "scope": ("league" if league_filter else "macro"),
                        "phase": "segment_candidate_eval",
                        "segment": str(league_filter or macro_filter or ""),
                        "lookback_games": int(cfg_seg.get("lookback_games", 24) or 24),
                        "strength": float(cfg_seg.get("strength", 0.35) or 0.35),
                    },
                )
            )
            rep_on_seg = _evaluate_walkforward_players_context(
                segment_samples,
                team_scale=float(team_scale),
                players_scale=float(players_scale),
                min_train_games=int(max(80, min_train_games * 0.35)),
                team_games=team_games,
                players_context_cfg=dict(cfg_seg),
                form_cfg=form_cfg,
                ctx_cache=ctx_cache,
            )
            ll_seg = _f(rep_on_seg.get("logloss"), float("nan"))
            if not math.isfinite(ll_seg):
                continue
            br_seg = _f(rep_on_seg.get("brier"), float("nan"))
            ece_seg = _f(rep_on_seg.get("ece"), float("nan"))
            row_seg = {
                "cfg": dict(cfg_seg),
                "settings": _players_context_settings_from_cfg(cfg_seg),
                "n_total": int(rep_on_seg.get("n_total") or 0),
                "logloss": float(ll_seg),
                "brier": (float(br_seg) if math.isfinite(br_seg) else None),
                "ece": (float(ece_seg) if math.isfinite(ece_seg) else None),
                "context": dict(rep_on_seg.get("context") or {}),
                "gain_logloss_vs_off": (
                    float(ll_off_seg - ll_seg) if math.isfinite(ll_off_seg) and math.isfinite(ll_seg) else None
                ),
            }
            if best_seg is None:
                best_seg = row_seg
                continue
            if (
                float(row_seg.get("logloss", 9e9)) < float(best_seg.get("logloss", 9e9))
                or (
                    float(row_seg.get("logloss", 9e9)) == float(best_seg.get("logloss", 9e9))
                    and float(row_seg.get("brier") if row_seg.get("brier") is not None else 9e9)
                    < float(best_seg.get("brier") if best_seg.get("brier") is not None else 9e9)
                )
            ):
                best_seg = row_seg

        if not isinstance(best_seg, dict):
            return {
                "selected": False,
                "reason": "no_valid_candidate",
                "n_total": int(n_seg),
                "min_n_required": int(min_n_req),
            }
        gain_seg = _f(best_seg.get("gain_logloss_vs_off"), float("nan"))
        selected = bool(math.isfinite(gain_seg) and gain_seg >= float(min_gain_seg))
        return {
            "selected": bool(selected),
            "reason": ("ok" if selected else "gain_small"),
            "n_total": int(n_seg),
            "min_n_required": int(min_n_req),
            "gain_logloss_vs_off": (float(gain_seg) if math.isfinite(gain_seg) else None),
            "baseline_off": {
                "logloss": rep_off_seg.get("logloss"),
                "brier": rep_off_seg.get("brier"),
                "ece": rep_off_seg.get("ece"),
            },
            "best_on": best_seg,
        }

    if budget == "fast":
        seg_diag_league["_meta"] = {"skipped": True, "reason": "speed_fast"}
        seg_diag_macro["_meta"] = {"skipped": True, "reason": "speed_fast"}
    else:
        league_items = sorted(samples_by_league.items(), key=lambda item: (-len(item[1]), str(item[0])))
        macro_items = sorted(samples_by_macro.items(), key=lambda item: (-len(item[1]), str(item[0])))
        if budget == "runtime_full":
            league_keys = [str(key) for key, _ in league_items[:4]]
            macro_keys = [str(key) for key, _ in macro_items[:3]]
            seg_diag_league["_meta"] = {"restricted": True, "reason": "runtime_auto_full", "top_n": len(league_keys)}
            seg_diag_macro["_meta"] = {"restricted": True, "reason": "runtime_auto_full", "top_n": len(macro_keys)}
        else:
            league_keys = [str(key) for key, _ in league_items]
            macro_keys = [str(key) for key, _ in macro_items]
        _league_started = time.perf_counter()
        for lg_idx, lg in enumerate(league_keys, start=1):
            if str(lg) in seg_diag_league:
                continue
            _pulse(
                _eta_payload(
                    started_at=_league_started,
                    index=int(lg_idx),
                    total=len(league_keys),
                    extra={"scope": "league", "phase": "segment_start", "segment": str(lg)},
                )
            )
            seg = _fit_segment(list(samples_by_league.get(str(lg)) or []), league_filter=str(lg), macro_filter=None)
            seg_diag_league[str(lg)] = seg
            if bool(seg.get("selected")) and isinstance((seg.get("best_on") or {}).get("settings"), dict):
                best_on_seg = dict(seg.get("best_on") or {})
                by_league[str(lg)] = {
                    "settings": dict(best_on_seg.get("settings") or {}),
                    "n_total": int(best_on_seg.get("n_total") or 0),
                    "logloss": best_on_seg.get("logloss"),
                    "brier": best_on_seg.get("brier"),
                    "ece": best_on_seg.get("ece"),
                    "gain_logloss_vs_off": seg.get("gain_logloss_vs_off"),
                    "source": "walkforward_oos_segment",
                }
            _save_partial(
                {
                    "phase": "segments_running",
                    "baseline_off": rep_off,
                    "grid_rows": list(grid_rows),
                    "seg_diag_league": dict(seg_diag_league),
                    "seg_diag_macro": dict(seg_diag_macro),
                    "by_league": dict(by_league),
                    "by_macro": dict(by_macro),
                    "meta": {
                        "league_total": len(league_keys),
                        "league_completed": len([k for k in seg_diag_league.keys() if not str(k).startswith("_")]),
                        "macro_total": len(macro_keys),
                        "macro_completed": len([k for k in seg_diag_macro.keys() if not str(k).startswith("_")]),
                    },
                }
            )

        _macro_started = time.perf_counter()
        for mk_idx, mk in enumerate(macro_keys, start=1):
            if str(mk) in seg_diag_macro:
                continue
            _pulse(
                _eta_payload(
                    started_at=_macro_started,
                    index=int(mk_idx),
                    total=len(macro_keys),
                    extra={"scope": "macro", "phase": "segment_start", "segment": str(mk)},
                )
            )
            seg = _fit_segment(list(samples_by_macro.get(str(mk)) or []), league_filter=None, macro_filter=str(mk))
            seg_diag_macro[str(mk)] = seg
            if bool(seg.get("selected")) and isinstance((seg.get("best_on") or {}).get("settings"), dict):
                best_on_seg = dict(seg.get("best_on") or {})
                by_macro[str(mk)] = {
                    "settings": dict(best_on_seg.get("settings") or {}),
                    "n_total": int(best_on_seg.get("n_total") or 0),
                    "logloss": best_on_seg.get("logloss"),
                    "brier": best_on_seg.get("brier"),
                    "ece": best_on_seg.get("ece"),
                    "gain_logloss_vs_off": seg.get("gain_logloss_vs_off"),
                    "source": "walkforward_oos_segment",
                }
            _save_partial(
                {
                    "phase": "segments_running",
                    "baseline_off": rep_off,
                    "grid_rows": list(grid_rows),
                    "seg_diag_league": dict(seg_diag_league),
                    "seg_diag_macro": dict(seg_diag_macro),
                    "by_league": dict(by_league),
                    "by_macro": dict(by_macro),
                    "meta": {
                        "league_total": len(league_keys),
                        "league_completed": len([k for k in seg_diag_league.keys() if not str(k).startswith("_")]),
                        "macro_total": len(macro_keys),
                        "macro_completed": len([k for k in seg_diag_macro.keys() if not str(k).startswith("_")]),
                    },
                }
            )

    settings_global = _players_context_settings_from_cfg(best_cfg)
    promoted_settings = dict(settings_global) if str(gate_decision) == "promote" else {}
    promoted_by_league = {k: dict(v.get("settings") or {}) for k, v in by_league.items()} if str(gate_decision) == "promote" else {}
    promoted_by_macro = {k: dict(v.get("settings") or {}) for k, v in by_macro.items()} if str(gate_decision) == "promote" else {}

    final_payload = {
        "enabled": True,
        "version": "ml_players_context_fit_v1",
        "source": "walkforward_oos_ablation",
        "baseline_off": {
            "logloss": (float(off_ll) if math.isfinite(off_ll) else None),
            "brier": (float(off_br) if math.isfinite(off_br) else None),
            "ece": (float(off_ece) if math.isfinite(off_ece) else None),
            "n_total": int(rep_off.get("n_total") or 0),
            "folds": len(rep_off.get("folds") or []),
            "context": dict(rep_off.get("context") or {}),
        },
        "best_on": {
            "cfg": dict(best_cfg),
            "settings": dict(settings_global),
            "logloss": (float(on_ll) if math.isfinite(on_ll) else None),
            "brier": (float(on_br) if math.isfinite(on_br) else None),
            "ece": (float(on_ece) if math.isfinite(on_ece) else None),
            "n_total": int(best.get("n_total") or 0),
            "context": dict(best.get("context") or {}),
        },
        "ablation": {
            "improvement_logloss_on_vs_off": (float(gain_ll) if math.isfinite(gain_ll) else None),
            "improvement_brier_on_vs_off": (float(gain_br) if math.isfinite(gain_br) else None),
            "improvement_ece_on_vs_off": (float(gain_ece) if math.isfinite(gain_ece) else None),
        },
        "gate": {
            "decision": str(gate_decision),
            "reasons": list(gate_reasons),
            "thresholds": {
                "min_n_total": int(min_n_gate),
                "min_improvement_logloss": 0.0004,
                "max_brier_regression": 0.0006,
                "max_ece_regression": 0.0015,
                "max_logloss_regression": 0.0010,
            },
        },
        "settings": dict(settings_global),
        "promoted_settings": dict(promoted_settings),
        "settings_by_league": {k: dict(v.get("settings") or {}) for k, v in by_league.items()},
        "settings_by_macro": {k: dict(v.get("settings") or {}) for k, v in by_macro.items()},
        "promoted_settings_by_league": promoted_by_league,
        "promoted_settings_by_macro": promoted_by_macro,
        "by_league": dict(by_league),
        "by_macro": dict(by_macro),
        "segment_diagnostics": {
            "by_league": seg_diag_league,
            "by_macro": seg_diag_macro,
        },
        "grid_top": [
            {
                "cfg": dict(r.get("cfg") or {}),
                "settings": dict(r.get("settings") or {}),
                "n_total": int(r.get("n_total") or 0),
                "logloss": r.get("logloss"),
                "brier": r.get("brier"),
                "ece": r.get("ece"),
                "improvement_logloss_vs_off": r.get("improvement_logloss_vs_off"),
                "improvement_brier_vs_off": r.get("improvement_brier_vs_off"),
                "improvement_ece_vs_off": r.get("improvement_ece_vs_off"),
                "context": dict(r.get("context") or {}),
            }
            for r in grid_sorted[:20]
        ],
        "meta": {
            "samples_total": len(samples),
            "team_games_rows": len(team_games),
            "candidates_tested": len(candidates),
            "cache_entries": len(ctx_cache),
            "speed_profile": str(speed),
            "budget_profile": str(budget),
            "safe_candidates": len(_safe_rows),
            "selected_pool": ("safe" if _selected_pool is _safe_rows else "all"),
            "segment_fit_skipped": bool(budget == "fast"),
            "segment_fit_restricted": bool(budget == "runtime_full"),
        },
    }
    _save_partial(
        {
            "phase": "complete",
            "baseline_off": rep_off,
            "grid_rows": list(grid_rows),
            "seg_diag_league": dict(seg_diag_league),
            "seg_diag_macro": dict(seg_diag_macro),
            "by_league": dict(by_league),
            "by_macro": dict(by_macro),
            "final_output": dict(final_payload),
            "meta": {
                "league_completed": len([k for k in seg_diag_league.keys() if not str(k).startswith("_")]),
                "macro_completed": len([k for k in seg_diag_macro.keys() if not str(k).startswith("_")]),
            },
        }
    )
    return final_payload


def _blind_backtest_by_league(
    samples: List[Sample],
    *,
    team_scale: float,
    players_scale: float,
    form_cfg: Dict[str, Any],
    min_holdout_n: int = 30,
) -> Dict[str, Any]:
    """One-step blind holdout by league (last month unseen for each league)."""
    out_rows: List[Dict[str, Any]] = []
    by_league: Dict[str, Any] = {}

    leagues = sorted({str(s.league) for s in samples if str(s.league).strip()})
    for lg in leagues:
        rows_lg = [s for s in samples if str(s.league) == str(lg)]
        if not rows_lg:
            continue
        months = sorted({str(s.month) for s in rows_lg if str(s.month).strip()})
        if len(months) < 2:
            continue
        holdout_month = str(months[-1])
        holdout = [s for s in rows_lg if str(s.month) == holdout_month]
        if len(holdout) < int(min_holdout_n):
            continue

        y = [int(s.y) for s in holdout]
        p_team = [float(clip_prob(float(s.p_team))) for s in holdout]
        p_players = [float(clip_prob(float(s.p_players))) for s in holdout]
        p_fused: List[float] = []
        for s in holdout:
            p_pre = _fuse_probs_by_precision(
                p_team=float(s.p_team),
                p_players=float(s.p_players),
                n_team=float(s.n_team),
                n_players=float(s.n_players),
                team_scale=float(team_scale),
                players_scale=float(players_scale),
                coverage=float(s.coverage_ratio),
                transfer_signal=float(s.transfer_signal),
                lineup_continuity=float(s.lineup_continuity),
                roster_changed=bool(s.roster_changed),
                season_phase=("playoffs" if str(s.phase).strip().lower() == "playoffs" else "auto"),
                players_freshness=float(s.freshness_avg),
            )
            p_post = _apply_form_blend_on_sample(
                p_model=float(p_pre),
                sample=s,
                enabled=bool((form_cfg or {}).get("enabled", False)),
                max_weight=float((form_cfg or {}).get("max_weight", 0.75) or 0.75),
                knee_games=float((form_cfg or {}).get("knee_games", 6.0) or 6.0),
                beta_prior=float((form_cfg or {}).get("beta_prior", 2.0) or 2.0),
                signal_power=float((form_cfg or {}).get("signal_power", 1.0) or 1.0),
                elite_same_league_parity_strength=float(
                    (form_cfg or {}).get("elite_same_league_parity_strength", 0.35) or 0.35
                ),
            )
            p_fused.append(float(clip_prob(float(p_post))))

        ll_team = float(logloss(y, p_team))
        ll_players = float(logloss(y, p_players))
        ll_fused = float(logloss(y, p_fused))
        br_team = float(brier(y, p_team))
        br_players = float(brier(y, p_players))
        br_fused = float(brier(y, p_fused))

        row = {
            "league": str(lg),
            "holdout_month": str(holdout_month),
            "n_total": len(y),
            "logloss_team_only": float(ll_team),
            "logloss_players_only": float(ll_players),
            "logloss_fused": float(ll_fused),
            "gain_fused_vs_team": float(ll_team - ll_fused),
            "gain_fused_vs_players": float(ll_players - ll_fused),
            "brier_team_only": float(br_team),
            "brier_players_only": float(br_players),
            "brier_fused": float(br_fused),
        }
        out_rows.append(row)
        by_league[str(lg)] = dict(row)

    out_rows = sorted(out_rows, key=lambda r: (float(r.get("gain_fused_vs_team", -999.0)), int(r.get("n_total", 0))), reverse=True)
    return {
        "rows": out_rows,
        "by_league": by_league,
        "meta": {
            "min_holdout_n": int(min_holdout_n),
            "leagues_tested": len(out_rows),
        },
    }


def _resolve_players_context_budget_profile(speed_profile: str, run_label: Optional[str] = None) -> str:
    speed = str(speed_profile or "full").strip().lower()
    if speed == "fast":
        return "fast"
    run_label_key = str(run_label or "").strip().lower()
    if run_label_key.startswith("runtime_auto_"):
        return "runtime_full"
    return "full"
