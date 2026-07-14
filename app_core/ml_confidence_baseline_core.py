"""Build and write the ML confidence baseline from maintenance reports."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlcore.utils import safe_float_or_none as _safe_float
from mlcore.league_context import league_macro_key as _league_macro_key
from app_core.ml_league_runtime_core import norm_league_key as _norm_league_key
from app_core.interleague_baseline_core import (
    build_interleague_pairs_table as core_build_interleague_pairs_table,
    infer_league_class_from_node as core_infer_league_class_from_node,
)


# Replace _canon_league inner call to use _norm_league_key
# (the inner function _canon_league inside build_ml_confidence_baseline calls core_norm_league_key)


def build_ml_confidence_baseline(
    lv_report: Optional[Dict[str, Any]],
    wf_report: Optional[Dict[str, Any]],
    params_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a league-aware confidence baseline from maintenance reports."""
    if not isinstance(lv_report, dict) or bool(lv_report.get("error")):
        return {"error": "lv_report_invalido"}

    params_fit = params_fit if isinstance(params_fit, dict) else {}
    wf_valid = isinstance(wf_report, dict) and (not bool((wf_report or {}).get("error")))
    warnings_out: List[str] = []
    if not wf_valid:
        wf_report = {}
        warnings_out.append("wf_report_invalido_fallback_lv_only")

    lv_models = lv_report.get("models") if isinstance(lv_report.get("models"), dict) else {}
    lv_by_league = lv_report.get("by_league") if isinstance(lv_report.get("by_league"), dict) else {}
    wf_overall = wf_report.get("overall") if isinstance(wf_report.get("overall"), dict) else {}
    wf_by_league = wf_report.get("by_league") if isinstance(wf_report.get("by_league"), dict) else {}

    def _inum(v: Any) -> int:
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    def _canon_league(v: Any) -> str:
        return _norm_league_key(v)

    def _lv_metrics(row: Any) -> Dict[str, Any]:
        fused = row.get("fused_active") if isinstance(row, dict) and isinstance(row.get("fused_active"), dict) else {}
        return {
            "n_total": _inum(fused.get("n_total")),
            "logloss": _safe_float(fused.get("logloss")),
            "gain_vs_team_logloss": _safe_float(fused.get("gain_vs_team_logloss")),
        }

    def _wf_metrics(row: Any) -> Dict[str, Any]:
        row = row if isinstance(row, dict) else {}
        return {
            "n_total": _inum(row.get("n_total")),
            "months": _inum(row.get("months")),
            "logloss_mean": _safe_float(row.get("logloss_mean")),
            "drift_mean": _safe_float(row.get("drift_mean")),
            "roi_proxy_mean": _safe_float(row.get("roi_proxy_mean")),
        }

    def _score_bias(lv_m: Dict[str, Any], wf_m: Dict[str, Any]) -> float:
        bias = 0.0
        lv_n = int(lv_m.get("n_total") or 0)
        wf_n = int(wf_m.get("n_total") or 0)
        wf_months = int(wf_m.get("months") or 0)
        lv_ll = _safe_float(lv_m.get("logloss"))
        lv_gain = _safe_float(lv_m.get("gain_vs_team_logloss"))
        wf_ll = _safe_float(wf_m.get("logloss_mean"))
        wf_drift = _safe_float(wf_m.get("drift_mean"))
        wf_roi = _safe_float(wf_m.get("roi_proxy_mean"))

        if lv_gain is not None:
            if lv_gain >= 0.020:
                bias += 6.0
            elif lv_gain >= 0.010:
                bias += 3.0
            elif lv_gain >= 0.0:
                bias += 1.0
            else:
                bias -= 6.0
        if lv_n >= 1500:
            bias += 4.0
        elif lv_n >= 700:
            bias += 2.0
        elif 0 < lv_n < 300:
            bias -= 2.5
        if lv_ll is not None:
            if lv_ll <= 0.62:
                bias += 2.5
            elif lv_ll <= 0.66:
                bias += 1.0
            elif lv_ll >= 0.72:
                bias -= 2.0

        if wf_n >= 80:
            if wf_roi is not None:
                if wf_roi > 0.03:
                    bias += 2.0
                elif wf_roi > 0.0:
                    bias += 1.0
                elif wf_roi < -0.02:
                    bias -= 2.5
                elif wf_roi < 0.0:
                    bias -= 1.0
            if wf_drift is not None:
                ad = abs(float(wf_drift))
                if ad <= 0.005:
                    bias += 1.0
                elif ad <= 0.015:
                    bias += 0.4
                elif ad > 0.03:
                    bias -= 2.0
                elif ad > 0.02:
                    bias -= 1.0
            if wf_ll is not None:
                if wf_ll <= 0.66:
                    bias += 1.0
                elif wf_ll >= 0.73:
                    bias -= 1.5

        ev_lv = min(1.0, max(0.0, lv_n / 1500.0))
        ev_wf = min(1.0, max(0.0, wf_n / 400.0))
        ev_wf *= min(1.0, max(0.25, wf_months / 4.0 if wf_months > 0 else 0.25))
        ev = (0.65 * ev_lv) + (0.35 * ev_wf)
        bias *= float(max(0.15, min(1.0, ev)))
        return float(max(-15.0, min(15.0, bias)))

    def _agg_lv_metrics(by_lg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for _lg_raw, _row in (by_lg.items() if isinstance(by_lg, dict) else []):
            _lg = _canon_league(_lg_raw)
            if not _lg:
                continue
            _m = _lv_metrics(_row)
            _n = int(max(0, int(_m.get("n_total") or 0)))
            _acc = out.setdefault(
                str(_lg),
                {
                    "n_total": 0,
                    "_sum_logloss": 0.0,
                    "_sum_gain": 0.0,
                },
            )
            _acc["n_total"] = int(_acc.get("n_total", 0) + _n)
            _ll = _safe_float(_m.get("logloss"))
            if _ll is not None:
                _acc["_sum_logloss"] = float(_acc.get("_sum_logloss", 0.0) + (float(_ll) * float(max(1, _n))))
            _g = _safe_float(_m.get("gain_vs_team_logloss"))
            if _g is not None:
                _acc["_sum_gain"] = float(_acc.get("_sum_gain", 0.0) + (float(_g) * float(max(1, _n))))
        for _lg, _acc in out.items():
            _n = int(max(0, int(_acc.get("n_total") or 0)))
            _w = float(max(1, _n))
            _acc["logloss"] = float(_acc.get("_sum_logloss", 0.0) / _w) if _n > 0 else None
            _acc["gain_vs_team_logloss"] = float(_acc.get("_sum_gain", 0.0) / _w) if _n > 0 else None
            _acc.pop("_sum_logloss", None)
            _acc.pop("_sum_gain", None)
        return out

    def _agg_wf_metrics(by_lg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for _lg_raw, _row in (by_lg.items() if isinstance(by_lg, dict) else []):
            _lg = _canon_league(_lg_raw)
            if not _lg:
                continue
            _m = _wf_metrics(_row)
            _n = int(max(0, int(_m.get("n_total") or 0)))
            _months = int(max(0, int(_m.get("months") or 0)))
            _acc = out.setdefault(
                str(_lg),
                {
                    "n_total": 0,
                    "months": 0,
                    "_sum_logloss": 0.0,
                    "_sum_drift": 0.0,
                    "_sum_roi": 0.0,
                },
            )
            _acc["n_total"] = int(_acc.get("n_total", 0) + _n)
            _acc["months"] = int(max(int(_acc.get("months") or 0), _months))
            _ll = _safe_float(_m.get("logloss_mean"))
            if _ll is not None:
                _acc["_sum_logloss"] = float(_acc.get("_sum_logloss", 0.0) + (float(_ll) * float(max(1, _n))))
            _dr = _safe_float(_m.get("drift_mean"))
            if _dr is not None:
                _acc["_sum_drift"] = float(_acc.get("_sum_drift", 0.0) + (float(_dr) * float(max(1, _n))))
            _roi = _safe_float(_m.get("roi_proxy_mean"))
            if _roi is not None:
                _acc["_sum_roi"] = float(_acc.get("_sum_roi", 0.0) + (float(_roi) * float(max(1, _n))))
        for _lg, _acc in out.items():
            _n = int(max(0, int(_acc.get("n_total") or 0)))
            _w = float(max(1, _n))
            _acc["logloss_mean"] = float(_acc.get("_sum_logloss", 0.0) / _w) if _n > 0 else None
            _acc["drift_mean"] = float(_acc.get("_sum_drift", 0.0) / _w) if _n > 0 else None
            _acc["roi_proxy_mean"] = float(_acc.get("_sum_roi", 0.0) / _w) if _n > 0 else None
            _acc.pop("_sum_logloss", None)
            _acc.pop("_sum_drift", None)
            _acc.pop("_sum_roi", None)
        return out

    lv_g = _lv_metrics({"fused_active": lv_models.get("fused_active") if isinstance(lv_models, dict) else {}})
    wf_g = _wf_metrics(wf_overall)
    global_bias = _score_bias(lv_g, wf_g)

    lv_by_league_agg = _agg_lv_metrics(lv_by_league)
    wf_by_league_agg = _agg_wf_metrics(wf_by_league)
    leagues = sorted({
        str(k).strip()
        for k in list(lv_by_league_agg.keys()) + list(wf_by_league_agg.keys())
        if str(k).strip()
    })
    by_league_out: Dict[str, Any] = {}
    for lg in leagues:
        lv_m = dict(lv_by_league_agg.get(lg) or {})
        wf_m = dict(wf_by_league_agg.get(lg) or {})
        score_bias = _score_bias(lv_m, wf_m)
        evidence = int(max(lv_m.get("n_total") or 0, wf_m.get("n_total") or 0))
        if evidence >= 1200 and score_bias >= 4.0:
            band = "strong"
        elif evidence >= 700 and score_bias >= 1.0:
            band = "good"
        elif evidence >= 300:
            band = "ok"
        else:
            band = "low_sample"
        by_league_out[str(lg)] = {
            "league": str(lg),
            "lv_n": int(lv_m.get("n_total") or 0),
            "lv_logloss": lv_m.get("logloss"),
            "lv_gain_vs_team_logloss": lv_m.get("gain_vs_team_logloss"),
            "wf_n": int(wf_m.get("n_total") or 0),
            "wf_months": int(wf_m.get("months") or 0),
            "wf_logloss_mean": wf_m.get("logloss_mean"),
            "wf_drift_mean": wf_m.get("drift_mean"),
            "wf_roi_proxy_mean": wf_m.get("roi_proxy_mean"),
            "score_bias": float(score_bias),
            "quality_band": band,
        }

    def _macro_group_for_class(_league: Any) -> str:
        z = _canon_league(_league)
        if not z:
            return ""
        return _league_macro_key(z)

    def _quality_band_from_n(_n_eff: int) -> str:
        _n = int(max(0, _n_eff))
        if _n >= 1200:
            return "strong"
        if _n >= 500:
            return "good"
        if _n >= 200:
            return "ok"
        return "low_sample"

    def _degrade_quality_band(_band: str, _steps: int) -> str:
        order = ["strong", "good", "ok", "low_sample"]
        b = str(_band or "").strip().lower()
        idx = order.index(b) if b in order else 2
        return str(order[min(len(order) - 1, idx + max(0, int(_steps)))])

    def _league_class_from_league_node(_league: str, _node: Dict[str, Any]) -> str:
        return str(core_infer_league_class_from_node(_league, _node))

    league_class_out: Dict[str, str] = {}
    for _lg, _node in (by_league_out.items() if isinstance(by_league_out, dict) else []):
        if not isinstance(_node, dict):
            continue
        _lg_key = _canon_league(_lg)
        if not _lg_key:
            continue
        league_class_out[_lg_key] = str(_league_class_from_league_node(_lg_key, _node))

    # Optional league/segment table from fitted params (data-driven BO behavior).
    by_segment_out: Dict[str, Any] = {}
    try:
        _sg_global = (params_fit.get("series_gamma_global") or {}) if isinstance(params_fit, dict) else {}
        _sg_by_lg = (params_fit.get("series_gamma_by_league") or {}) if isinstance(params_fit, dict) else {}
        _g_bo3 = (((_sg_global.get("bo3") or {}).get("gamma")) if isinstance(_sg_global, dict) else None)
        _g_bo5 = (((_sg_global.get("bo5") or {}).get("gamma")) if isinstance(_sg_global, dict) else None)
        _g_bo3 = _safe_float(_g_bo3)
        _g_bo5 = _safe_float(_g_bo5)

        def _seg_bias_from_gamma(g_lg: Optional[float], g_glb: Optional[float], imp: Optional[float], n: int) -> float:
            if g_lg is None or g_glb is None:
                return 0.0
            ev = float(max(0.0, min(1.0, int(max(0, n)) / 220.0)))
            # data-driven signal:
            # - gamma delta captures BO-specific series behavior vs global prior
            # - improvement vs gamma=1 captures utility of BO-adjustment for that league
            raw = (float(g_lg) - float(g_glb)) * 8.0 + float((_safe_float(imp) or 0.0)) * 120.0
            return float(max(-6.0, min(6.0, raw * ev)))

        for _lg, _row in (_sg_by_lg.items() if isinstance(_sg_by_lg, dict) else []):
            if not isinstance(_row, dict):
                continue
            _lg_key = _canon_league(_lg)
            if not _lg_key:
                continue
            _rows = int(_row.get("rows") or 0)
            for _bo_i, _bo_name, _g_glb in ((3, "bo3", _g_bo3), (5, "bo5", _g_bo5)):
                _bo = _row.get(_bo_name) if isinstance(_row.get(_bo_name), dict) else {}
                _g_lg = _safe_float(_bo.get("gamma"))
                _imp = _safe_float(_bo.get("improvement_logloss_vs_gamma1"))
                _n = int(_bo.get("n") or _rows or 0)
                _sb = _seg_bias_from_gamma(_g_lg, _g_glb, _imp, _n)
                if abs(float(_sb)) < 1e-9 and _n <= 0:
                    continue
                # same-league segment entries; po/inter copied with same bias (runtime will still shrink).
                for _po in (0, 1):
                    _k = f"league:{_lg_key}|bo:{int(_bo_i)}|po:{int(_po)}|inter:0"
                    by_segment_out[_k] = {
                        "score_bias": float(_sb),
                        "wf_n": int(max(0, _n)),
                        "quality_band": ("good" if _n >= 180 else ("ok" if _n >= 80 else "low_sample")),
                        "source": "series_gamma_by_league",
                    }
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        by_segment_out = {}

    # Fallback: ensure hierarchical runtime has segment coverage even when
    # series_gamma_by_league is unavailable/insufficient.
    # This is data-driven from by_league score_bias (no directional override).
    if not isinstance(by_segment_out, dict):
        by_segment_out = {}
    if len(by_segment_out) == 0 and isinstance(by_league_out, dict) and by_league_out:
        try:
            for _lg, _node in by_league_out.items():
                if not isinstance(_node, dict):
                    continue
                _lg_key = _canon_league(_lg)
                if not _lg_key:
                    continue
                _sb = _safe_float(_node.get("score_bias"))
                _n = int(max(0, int(_node.get("wf_n") or 0), int(_node.get("lv_n") or 0)))
                _qb = str(_node.get("quality_band") or ("ok" if _n >= 300 else "low_sample"))
                if _sb is None:
                    continue
                for _bo_i in (1, 3, 5):
                    for _po in (0, 1):
                        _k = f"league:{_lg_key}|bo:{int(_bo_i)}|po:{int(_po)}|inter:0"
                        by_segment_out[_k] = {
                            "score_bias": float(_sb),
                            "wf_n": int(_n),
                            "quality_band": _qb,
                            "source": "fallback_by_league_copy",
                        }
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            by_segment_out = by_segment_out if isinstance(by_segment_out, dict) else {}

    def _segment_node_from_entries(_entries: Any, *, _source: str) -> Dict[str, Any]:
        rows = [r for r in (_entries if isinstance(_entries, list) else []) if isinstance(r, dict)]
        if not rows:
            return {}
        sw = 0.0
        sb = 0.0
        n_sum = 0.0
        for r in rows:
            _b = _safe_float(r.get("score_bias"))
            if _b is None:
                continue
            _n = float(max(1.0, float(_safe_float(r.get("n_eff")) or 1.0)))
            _w = float(max(0.50, min(8.00, math.sqrt(_n / 80.0))))
            sb += float(_b) * _w
            sw += _w
            n_sum += _n
        if sw <= 0.0:
            return {}
        n_eff = int(max(1, min(3600, round(n_sum))))
        return {
            "score_bias": float(max(-8.0, min(8.0, sb / sw))),
            "wf_n": int(n_eff),
            "n_eff": int(n_eff),
            "quality_band": str(_quality_band_from_n(n_eff)),
            "source": str(_source),
        }

    # Segment families (same/inter + major/minor) built from league-level OOS confidence.
    # This gives runtime contextual coverage even when league-specific segment is missing.
    try:
        _same_all: List[Dict[str, Any]] = []
        _same_major: List[Dict[str, Any]] = []
        _same_minor: List[Dict[str, Any]] = []
        _lg_rows: List[Tuple[str, Dict[str, Any], float, int, str]] = []
        for _lg, _node in (by_league_out.items() if isinstance(by_league_out, dict) else []):
            if not isinstance(_node, dict):
                continue
            _lg_key = _canon_league(_lg)
            if not _lg_key:
                continue
            _sb = _safe_float(_node.get("score_bias"))
            if _sb is None:
                continue
            _n = int(max(0, int(_node.get("n_eff") or 0), int(_node.get("wf_n") or 0), int(_node.get("lv_n") or 0)))
            if _n <= 0:
                continue
            _cls = str(league_class_out.get(_lg_key) or "minor")
            _entry = {"score_bias": float(_sb), "n_eff": int(_n)}
            _same_all.append(_entry)
            if _cls == "major":
                _same_major.append(_entry)
            else:
                _same_minor.append(_entry)
            _lg_rows.append((_lg_key, _node, float(_sb), int(_n), _cls))

        _inter_all: List[Dict[str, Any]] = []
        _inter_mm: List[Dict[str, Any]] = []
        _inter_mix: List[Dict[str, Any]] = []
        _inter_nn: List[Dict[str, Any]] = []
        for _i in range(len(_lg_rows)):
            _lga, _na, _sb_a, _n_a, _cls_a = _lg_rows[_i]
            for _j in range(_i + 1, len(_lg_rows)):
                _lgb, _nb, _sb_b, _n_b, _cls_b = _lg_rows[_j]
                _n_pair = int(max(1, min(_n_a, _n_b)))
                # Inter context confidence: average OOS confidence of both leagues.
                _pair_bias = float(max(-8.0, min(8.0, 0.5 * (_sb_a + _sb_b))))
                _entry = {"score_bias": float(_pair_bias), "n_eff": int(_n_pair)}
                _inter_all.append(_entry)
                if _cls_a == "major" and _cls_b == "major":
                    _inter_mm.append(_entry)
                elif _cls_a == "minor" and _cls_b == "minor":
                    _inter_nn.append(_entry)
                else:
                    _inter_mix.append(_entry)

        def _put_seg(_prefix: str, _inter_i: int, _entries: Any, _source: str) -> None:
            if not _prefix:
                return
            _node = _segment_node_from_entries(_entries, _source=_source)
            if not _node:
                return
            for _bo_i in (1, 3, 5):
                for _po in (0, 1):
                    _k = f"{_prefix}|bo:{int(_bo_i)}|po:{int(_po)}|inter:{int(_inter_i)}"
                    if _k not in by_segment_out:
                        by_segment_out[_k] = dict(_node)

        _put_seg("class:same_major", 0, _same_major, "segment_class_same_major_v1")
        _put_seg("class:same_minor", 0, _same_minor, "segment_class_same_minor_v1")
        _put_seg("class:inter_major_major", 1, _inter_mm, "segment_class_inter_major_major_v1")
        _put_seg("class:inter_minor_minor", 1, _inter_nn, "segment_class_inter_minor_minor_v1")
        _put_seg("class:inter_mixed", 1, _inter_mix, "segment_class_inter_mixed_v1")
        _put_seg("global", 0, _same_all, "segment_global_same_v1")
        _put_seg("global", 1, _inter_all, "segment_global_inter_v1")
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        pass

    # Patch-aware variants: "warn/refit" should shrink confidence toward neutral.
    # These are deterministic transforms over OOS-derived segments (risk-aware, monotonic).
    try:
        _patch_bias_mult = {"ok": 1.00, "warn": 0.78, "refit": 0.55}
        _patch_n_mult = {"ok": 1.00, "warn": 0.72, "refit": 0.50}
        _base_items = list(by_segment_out.items()) if isinstance(by_segment_out, dict) else []
        for _k, _node in _base_items:
            if (not isinstance(_node, dict)) or ("|bo:" not in str(_k)) or ("|patch:" in str(_k)):
                continue
            for _pt in ("ok", "warn", "refit"):
                _pk = str(_k).replace("|bo:", f"|patch:{_pt}|bo:", 1)
                if _pk in by_segment_out:
                    continue
                _m_b = float(_patch_bias_mult.get(_pt, 1.0))
                _m_n = float(_patch_n_mult.get(_pt, 1.0))
                _sb = float(_safe_float(_node.get("score_bias")) or 0.0) * _m_b
                _n0 = int(max(1, int(_node.get("n_eff") or 0), int(_node.get("wf_n") or 0)))
                _n1 = int(max(1, round(float(_n0) * _m_n)))
                _qb0 = str(_node.get("quality_band") or _quality_band_from_n(_n0))
                _degrade = 0 if _pt == "ok" else (1 if _pt == "warn" else 2)
                _qb1 = _degrade_quality_band(_qb0, _degrade)
                _nd = dict(_node)
                _nd["score_bias"] = float(max(-8.0, min(8.0, _sb)))
                _nd["n_eff"] = int(_n1)
                _nd["wf_n"] = int(_n1)
                _nd["quality_band"] = str(_qb1)
                _src = str(_node.get("source") or "")
                _nd["source"] = (f"{_src}+patch_{_pt}_v1" if _src else f"patch_{_pt}_v1")
                by_segment_out[_pk] = _nd
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        pass

    def _sigma0_from_nodes(_nodes: Any, default_sigma: float = 4.0) -> float:
        _vals: List[float] = []
        for _node in (_nodes if isinstance(_nodes, list) else []):
            if not isinstance(_node, dict):
                continue
            _sb = _safe_float(_node.get("score_bias"))
            if _sb is None:
                continue
            _vals.append(float(_sb))
        if len(_vals) >= 4:
            _m = float(sum(_vals) / max(1, len(_vals)))
            _var = float(sum((float(_x) - _m) ** 2 for _x in _vals) / max(1, len(_vals) - 1))
            return float(max(2.0, min(12.0, math.sqrt(max(1e-9, _var)))))
        return float(max(2.0, min(12.0, float(default_sigma))))

    def _attach_stat_fields(_node: Dict[str, Any], sigma0: float) -> Dict[str, Any]:
        if not isinstance(_node, dict):
            return {}
        _bias = float(_safe_float(_node.get("score_bias")) or 0.0)
        _n = int(max(0, int(_node.get("n_eff") or 0), int(_node.get("wf_n") or 0), int(_node.get("lv_n") or 0)))
        _n_eff = float(max(1.0, float(_n)))
        _se = float(max(1e-6, float(sigma0) / math.sqrt(_n_eff)))
        _z = float(_bias / max(1e-9, _se))
        _p = float(math.erfc(abs(_z) / math.sqrt(2.0)))
        _node["n_eff"] = int(max(1, round(_n_eff)))
        _node["score_se"] = float(_se)
        _node["score_var"] = float(max(1e-9, _se * _se))
        _node["score_z"] = float(_z)
        _node["score_p"] = float(max(0.0, min(1.0, _p)))
        return _node

    def _apply_fdr_to_table(_tbl: Dict[str, Any], *, q_alpha: float, z_soft: float) -> Dict[str, Any]:
        if not isinstance(_tbl, dict) or (len(_tbl) == 0):
            return _tbl if isinstance(_tbl, dict) else {}
        _cand: List[Tuple[str, float]] = []
        for _k, _v in _tbl.items():
            if not isinstance(_v, dict):
                continue
            _p = _safe_float(_v.get("score_p"))
            if _p is None or (not math.isfinite(float(_p))):
                continue
            _cand.append((str(_k), float(max(0.0, min(1.0, _p)))))
        _m = len(_cand)
        if _m <= 0:
            return _tbl
        _srt = sorted(_cand, key=lambda kv: float(kv[1]))
        _p_cut = -1.0
        _cut_rank = 0
        for _i, (_k, _p) in enumerate(_srt, start=1):
            _thr = float((float(q_alpha) * float(_i)) / float(max(1, _m)))
            if float(_p) <= _thr:
                _p_cut = float(_p)
                _cut_rank = int(_i)
        _q_map: Dict[str, float] = {}
        _q_prev = 1.0
        for _i in range(_m, 0, -1):
            _k, _p = _srt[_i - 1]
            _q_i = float((_p * float(_m)) / float(max(1, _i)))
            _q_prev = float(min(_q_prev, _q_i))
            _q_map[str(_k)] = float(max(0.0, min(1.0, _q_prev)))
        for _k, _p in _cand:
            _node = _tbl.get(_k) if isinstance(_tbl.get(_k), dict) else {}
            if not isinstance(_node, dict):
                continue
            _qv = float(_q_map.get(str(_k), 1.0))
            _reject = bool(_cut_rank > 0 and float(_p) <= float(_p_cut))
            _node["score_q"] = float(_qv)
            _node["fdr_q_alpha"] = float(q_alpha)
            _node["fdr_reject"] = bool(_reject)
            if not _reject:
                _b0 = float(_safe_float(_node.get("score_bias")) or 0.0)
                _z0 = float(abs(float(_safe_float(_node.get("score_z")) or 0.0)))
                _sh = float(max(0.10, min(0.85, _z0 / max(1.0, float(z_soft)))))
                _node["score_bias_pre_fdr"] = float(_b0)
                _node["fdr_shrink"] = float(_sh)
                _node["score_bias"] = float(_b0 * _sh)
            else:
                _node["fdr_shrink"] = 1.0
            _tbl[_k] = _node
        return _tbl

    _same_sigma0 = _sigma0_from_nodes(list(by_league_out.values()), default_sigma=4.0)
    _same_fdr_q_alpha = 0.10
    _same_fdr_soft_z = 2.20
    for _k, _node in list(by_league_out.items()):
        if not isinstance(_node, dict):
            continue
        by_league_out[_k] = _attach_stat_fields(dict(_node), _same_sigma0)
    by_league_out = _apply_fdr_to_table(by_league_out, q_alpha=_same_fdr_q_alpha, z_soft=_same_fdr_soft_z)
    for _k, _node in list(by_segment_out.items()):
        if not isinstance(_node, dict):
            continue
        by_segment_out[_k] = _attach_stat_fields(dict(_node), _same_sigma0)
    by_segment_out = _apply_fdr_to_table(by_segment_out, q_alpha=_same_fdr_q_alpha, z_soft=_same_fdr_soft_z)

    try:
        interleague_pairs_out = core_build_interleague_pairs_table(
            by_league_out,
            league_class_out=league_class_out,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        interleague_pairs_out = {}

    # Runtime fine profiles (context-specific).
    fine_profiles_out: Dict[str, Any] = {
        "default": {
            "fine_z_hi": 1.20,
            "fine_z_xhi": 1.85,
            "fine_shrink_hi": 0.05,
            "fine_shrink_xhi": 0.09,
        },
        "same_league": {
            "fine_z_hi": 1.25,
            "fine_z_xhi": 1.95,
            "fine_shrink_hi": 0.045,
            "fine_shrink_xhi": 0.080,
        },
        "interregional": {
            "fine_z_hi": 1.10,
            "fine_z_xhi": 1.70,
            "fine_shrink_hi": 0.070,
            "fine_shrink_xhi": 0.120,
        },
        "strict_low_sample": {
            "fine_z_hi": 1.00,
            "fine_z_xhi": 1.55,
            "fine_shrink_hi": 0.085,
            "fine_shrink_xhi": 0.145,
        },
        "major_stable": {
            "fine_z_hi": 1.35,
            "fine_z_xhi": 2.05,
            "fine_shrink_hi": 0.035,
            "fine_shrink_xhi": 0.065,
        },
    }
    fine_profile_by_league_same: Dict[str, str] = {}
    fine_profile_by_league_inter: Dict[str, str] = {}
    try:
        for _lg, _node in (by_league_out.items() if isinstance(by_league_out, dict) else []):
            if not isinstance(_node, dict):
                continue
            _lg_key = _canon_league(_lg)
            if not _lg_key:
                continue
            _qb = str(_node.get("quality_band") or "").strip().lower()
            _n = int(max(0, int(_node.get("wf_n") or 0), int(_node.get("lv_n") or 0)))
            _macro = _macro_group_for_class(_lg_key)
            _same_prof = "same_league"
            _inter_prof = "interregional"
            if _qb in {"low_sample", "low", "warn"} or _n < 180:
                _same_prof = "strict_low_sample"
                _inter_prof = "strict_low_sample"
            elif _macro in {"CN", "KR"} and _n >= 800:
                _same_prof = "major_stable"
            fine_profile_by_league_same[_lg_key] = str(_same_prof)
            fine_profile_by_league_inter[_lg_key] = str(_inter_prof)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        fine_profile_by_league_same = {}
        fine_profile_by_league_inter = {}

    # OOS digest for promotion gating (stored in baseline for paired comparisons).
    oos_eval_out: Dict[str, Any] = {}
    try:
        _m_team = (lv_models.get("team_only") or {}) if isinstance(lv_models.get("team_only"), dict) else {}
        _m_fused = (lv_models.get("fused_active") or {}) if isinstance(lv_models.get("fused_active"), dict) else {}
        _lv_team_ll = _safe_float(_m_team.get("logloss"))
        _lv_fused_ll = _safe_float(_m_fused.get("logloss"))
        _lv_gain = _safe_float(_m_fused.get("gain_vs_team_logloss"))
        _wf_ll = _safe_float(wf_overall.get("logloss_mean"))
        _wf_br = _safe_float(wf_overall.get("brier_mean"))
        _wf_n = _inum(wf_overall.get("n_total"))
        _wf_folds = _inum(wf_overall.get("folds"))
        _wf_folds_rows: List[Dict[str, Any]] = []
        for _r in (wf_report.get("folds") if isinstance(wf_report.get("folds"), list) else []):
            if not isinstance(_r, dict):
                continue
            _mth = str(_r.get("month") or "").strip()
            _ll = _safe_float(_r.get("logloss"))
            _br = _safe_float(_r.get("brier"))
            if (not _mth) or (_ll is None and _br is None):
                continue
            _wf_folds_rows.append(
                {
                    "month": str(_mth),
                    "logloss": (_ll if _ll is not None else None),
                    "brier": (_br if _br is not None else None),
                    "n": int(_inum(_r.get("n"))),
                }
            )
        oos_eval_out = {
            "lv_team_logloss": (_lv_team_ll if _lv_team_ll is not None else None),
            "lv_fused_logloss": (_lv_fused_ll if _lv_fused_ll is not None else None),
            "lv_gain_vs_team_logloss": (_lv_gain if _lv_gain is not None else None),
            "wf_logloss_mean": (_wf_ll if _wf_ll is not None else None),
            "wf_brier_mean": (_wf_br if _wf_br is not None else None),
            "wf_n_total": int(max(0, _wf_n)),
            "wf_folds": int(max(0, _wf_folds)),
            "wf_fold_metrics": _wf_folds_rows[:120],
        }
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        oos_eval_out = {}

    out: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": "ml_confidence_baseline_v3",
        "source": ("layer_validation+walkforward" if wf_valid else "layer_validation_only_fallback"),
        "global": {
            "lv_n": int(lv_g.get("n_total") or 0),
            "lv_logloss": lv_g.get("logloss"),
            "lv_gain_vs_team_logloss": lv_g.get("gain_vs_team_logloss"),
            "wf_n": int(wf_g.get("n_total") or 0),
            "wf_months": int(wf_g.get("months") or 0),
            "wf_logloss_mean": wf_g.get("logloss_mean"),
            "wf_drift_mean": wf_g.get("drift_mean"),
            "wf_roi_proxy_mean": wf_g.get("roi_proxy_mean"),
            "score_bias": float(global_bias),
        },
        "by_league": by_league_out,
        "league_class": league_class_out,
        "by_segment": by_segment_out,
        "interleague_pairs": interleague_pairs_out,
        "oos_eval": oos_eval_out,
        "thresholds": {
            "low_sample_min_n": 300,
            "bias_clip_abs": 15.0,
            "seg_shrink_k": 160.0,
            "league_shrink_k": 260.0,
            "global_shrink_k": 400.0,
            "hier_decay": 0.78,
            "same_sigma0": float(_same_sigma0),
            "same_z_gate": 1.96,
            "same_fdr_q_alpha": float(_same_fdr_q_alpha),
            "same_fdr_soft_z": float(_same_fdr_soft_z),
            "inter_shrink_k": 320.0,
            "inter_pair_mix_k": 260.0,
            "inter_sigma0": 4.0,
            "inter_z_gate": 1.96,
            "inter_z_full": 2.58,
            "inter_fdr_q_alpha": 0.10,
            "inter_fdr_soft_z": 2.20,
            "oos_gate_enabled": 1,
            "oos_bootstrap_samples": 1000,
            "oos_ci_level": 0.95,
            "oos_logloss_tol": 0.0015,
            "oos_brier_tol": 0.0010,
            "oos_min_overlap_folds": 6,
            "fine_z_hi": 1.20,
            "fine_z_xhi": 1.85,
            "fine_shrink_hi": 0.05,
            "fine_shrink_xhi": 0.09,
            "fine_profiles": fine_profiles_out,
            "fine_profile_by_league": fine_profile_by_league_same,  # backward-compatible key
            "fine_profile_by_league_same": fine_profile_by_league_same,
            "fine_profile_by_league_inter": fine_profile_by_league_inter,
            "segment_patch_bias_mult": {"ok": 1.00, "warn": 0.78, "refit": 0.55},
            "segment_patch_n_mult": {"ok": 1.00, "warn": 0.72, "refit": 0.50},
        },
    }
    if warnings_out:
        out["warnings"] = warnings_out
    if isinstance(params_fit, dict):
        out["params_generated_at"] = str(params_fit.get("generated_at") or "")
        _best = (params_fit.get("best") or {}) if isinstance(params_fit.get("best"), dict) else {}
        out["params_active_scales"] = {
            "team_scale": float(_safe_float(_best.get("team_scale")) or 1.0),
            "players_scale": float(_safe_float(_best.get("players_scale")) or 1.0),
        }
    return out


def write_ml_confidence_baseline(path_like: str | Path, payload: Dict[str, Any]) -> None:
    Path(path_like).expanduser().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")




__all__ = ["build_ml_confidence_baseline", "write_ml_confidence_baseline"]
