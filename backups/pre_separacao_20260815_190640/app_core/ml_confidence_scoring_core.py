"""Compute per-matchup ML confidence score from validation/pipeline layers."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from mlcore.utils import safe_float_or_none as _safe_float


def ml_confidence_from_layer_validation(
    report: Optional[Dict[str, Any]],
    *,
    league_hint: str = "",
    n_team_eff: float = float("nan"),
    ml_trace: Optional[Dict[str, Any]] = None,
    confidence_baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    score = 50.0
    source = "global"
    n_total = 0
    ll = float("nan")
    gain = float("nan")
    pipeline_adj = 0.0
    pipeline_notes: List[str] = []
    pipeline_meta_method = ""
    baseline_adj = 0.0
    baseline_source = ""
    baseline_band = ""

    def _nan_or_float(value: Any) -> float:
        fv = _safe_float(value)
        return float(fv) if fv is not None else float("nan")

    if isinstance(report, dict) and not report.get("error"):
        row = None
        by_lg = report.get("by_league") if isinstance(report.get("by_league"), dict) else {}
        lg = str(league_hint or "").strip()
        if lg and lg in by_lg and isinstance(by_lg.get(lg), dict):
            source = f"liga:{lg}"
            row = by_lg.get(lg)
        if row is None:
            source = "global"
            row = {"fused_active": report.get("models", {}).get("fused_active", {})}

        fused = row.get("fused_active") if isinstance(row, dict) else {}
        n_total = int((fused or {}).get("n_total") or 0)
        ll = _nan_or_float((fused or {}).get("logloss"))
        gain = _nan_or_float((fused or {}).get("gain_vs_team_logloss"))

        if math.isfinite(gain):
            if gain >= 0.015:
                score += 25
            elif gain >= 0.008:
                score += 15
            elif gain >= 0.0:
                score += 5
            else:
                score -= 20
        else:
            score -= 8

        if n_total >= 1500:
            score += 20
        elif n_total >= 700:
            score += 12
        elif n_total >= 300:
            score += 6
        else:
            score -= 8

        if math.isfinite(ll):
            if ll <= 0.62:
                score += 10
            elif ll <= 0.66:
                score += 5
            else:
                score -= 5
    else:
        score -= 15

    nte = _nan_or_float(n_team_eff)
    if math.isfinite(nte):
        if nte >= 25:
            score += 10
        elif nte >= 12:
            score += 5
        else:
            score -= 5

    # Ajuste complementar pela arquitetura atual (A-real/A-lite/B(matchup/style)/C + guards).
    try:
        _tr = ml_trace if isinstance(ml_trace, dict) else {}
        if "ml_trace" in _tr and isinstance(_tr.get("ml_trace"), dict) and not isinstance(_tr.get("guards"), dict):
            _tr = _tr.get("ml_trace") or {}
        _guards = _tr.get("guards") if isinstance(_tr.get("guards"), dict) else {}
        _state_real = _tr.get("state_areal_live") if isinstance(_tr.get("state_areal_live"), dict) else {}
        _state_real_shadow = _tr.get("shadow_state_areal") if isinstance(_tr.get("shadow_state_areal"), dict) else {}
        _state = _tr.get("state_alite_live") if isinstance(_tr.get("state_alite_live"), dict) else {}
        _style = _tr.get("style_b_lite_live") if isinstance(_tr.get("style_b_lite_live"), dict) else {}
        _rank_sos = _tr.get("league_rank_sos_live") if isinstance(_tr.get("league_rank_sos_live"), dict) else {}
        _roster_cont = _tr.get("roster_continuity_live") if isinstance(_tr.get("roster_continuity_live"), dict) else {}
        _hier_cal = _tr.get("hier_calibration_live") if isinstance(_tr.get("hier_calibration_live"), dict) else {}
        _series_bo = _tr.get("series_bo_dplite_live") if isinstance(_tr.get("series_bo_dplite_live"), dict) else {}
        _meta = _tr.get("meta_c_lite_live") if isinstance(_tr.get("meta_c_lite_live"), dict) else {}
        _layer_unc = _tr.get("layer_uncertainty_live") if isinstance(_tr.get("layer_uncertainty_live"), dict) else {}
        _fusion = _tr.get("fusion") if isinstance(_tr.get("fusion"), dict) else {}
        _padj = _tr.get("players_adjustment") if isinstance(_tr.get("players_adjustment"), dict) else {}

        # A-real (estado persistente mu/sigma)
        if bool(_guards.get("state_areal_live_on", False)):
            if bool(_guards.get("state_areal_live_applied", False)) or bool(_state_real.get("applied", False)):
                pipeline_adj += 8.0
                pipeline_notes.append("A-real ativo")
            else:
                pipeline_adj -= 8.0
                pipeline_notes.append("A-real sem aplicar")
        elif bool(_state_real_shadow.get("enabled", False)) and bool(_state_real_shadow.get("applied", False)):
            # Keep the UI honest: A-real is contributing in shadow even if not promoted to live.
            pipeline_notes.append("A-real sombra")

        # A-lite (estado dinamico simplificado)
        if bool(_guards.get("state_alite_live_on", False)):
            if bool(_guards.get("state_alite_live_applied", False)) or bool(_state.get("applied", False)):
                pipeline_adj += 6.0
                pipeline_notes.append("A-lite ativo")
            else:
                pipeline_adj -= 6.0
                pipeline_notes.append("A-lite sem aplicar")

        # B (matchup/style; treinado com fallback B-lite)
        if bool(_guards.get("style_b_lite_live_on", False)):
            if bool(_guards.get("style_b_lite_live_applied", False)) or bool(_style.get("applied", False)):
                _style_method = str(_style.get("method", "") or "")
                _b_label = "B-real" if _style_method.startswith("style_matchup_b_model") else "B-lite"
                _d_style = abs(_nan_or_float(_style.get("diff_pp")))
                if math.isfinite(_d_style):
                    if _d_style <= 3.0:
                        pipeline_adj += 3.0
                    elif _d_style <= 8.0:
                        pipeline_adj += 2.0
                    elif _d_style <= 15.0:
                        pipeline_adj += 0.5
                    else:
                        pipeline_adj -= 2.0
                    pipeline_notes.append(f"{_b_label} Δ={_d_style:.1f}pp")
                else:
                    pipeline_adj += 1.0
                    pipeline_notes.append(f"{_b_label} ativo")
            else:
                pipeline_adj -= 2.0
                pipeline_notes.append("B sem aplicar")

        # Rank-SOS intra-liga (prior de força de calendario)
        if bool(_guards.get("league_rank_sos_live_on", False)):
            if bool(_guards.get("league_rank_sos_live_applied", False)) or bool(_rank_sos.get("applied", False)):
                _d_rank = abs(_nan_or_float(_rank_sos.get("diff_pp")))
                if math.isfinite(_d_rank):
                    if _d_rank <= 2.5:
                        pipeline_adj += 2.0
                    elif _d_rank <= 6.0:
                        pipeline_adj += 1.0
                    elif _d_rank <= 12.0:
                        pipeline_adj -= 0.5
                    else:
                        pipeline_adj -= 2.0
                    pipeline_notes.append(f"Rank-SOS Δ={_d_rank:.1f}pp")
                else:
                    pipeline_adj += 0.5
                    pipeline_notes.append("Rank-SOS ativo")
            else:
                pipeline_notes.append("Rank-SOS sem aplicar")

        # Roster Continuity / identidade do time (ajuste por continuidade de lineup)
        if bool(_guards.get("roster_continuity_live_on", False)):
            if bool(_guards.get("roster_continuity_live_applied", False)) or bool(_roster_cont.get("applied", False)):
                _d_rc = abs(_nan_or_float(_roster_cont.get("diff_pp")))
                if math.isfinite(_d_rc):
                    if _d_rc <= 2.0:
                        pipeline_adj += 1.0
                    elif _d_rc <= 5.0:
                        pipeline_adj += 0.5
                    elif _d_rc <= 10.0:
                        pipeline_adj -= 0.5
                    else:
                        pipeline_adj -= 1.5
                    pipeline_notes.append(f"RosterCont Δ={_d_rc:.1f}pp")
                else:
                    pipeline_notes.append("RosterCont ativo")
            else:
                pipeline_notes.append("RosterCont sem aplicar")

        # Calibração hierárquica por liga/segmento (leve, conservadora)
        if bool(_guards.get("hier_calibration_live_on", False)):
            if bool(_guards.get("hier_calibration_live_applied", False)) or bool(_hier_cal.get("applied", False)):
                _d_hc = abs(_nan_or_float(_hier_cal.get("diff_pp")))
                if math.isfinite(_d_hc):
                    if _d_hc <= 1.0:
                        pipeline_adj += 2.0
                    elif _d_hc <= 2.0:
                        pipeline_adj += 1.0
                    else:
                        pipeline_adj += 0.25
                    pipeline_notes.append(f"HierCal Δ={_d_hc:.1f}pp")
                else:
                    pipeline_notes.append("HierCal ativo")
            else:
                pipeline_notes.append("HierCal sem aplicar")

        # SeriesBO (DP-lite): ajuste conservador de p_serie em BO3/BO5
        if bool(_guards.get("series_bo_dplite_live_on", False)):
            if bool(_guards.get("series_bo_dplite_live_applied", False)) or bool(_series_bo.get("applied", False)):
                _d_sbd = abs(_nan_or_float(_series_bo.get("diff_pp")))
                if math.isfinite(_d_sbd):
                    if _d_sbd <= 1.5:
                        pipeline_adj += 1.0
                    elif _d_sbd <= 4.0:
                        pipeline_adj += 0.5
                    elif _d_sbd <= 8.0:
                        pipeline_adj -= 0.5
                    else:
                        pipeline_adj -= 1.5
                    pipeline_notes.append(f"SeriesBO Δ={_d_sbd:.1f}pp")
                else:
                    pipeline_notes.append("SeriesBO ativo")
            else:
                pipeline_notes.append("SeriesBO sem aplicar")

        # C (meta-blend treinado / fallback)
        if bool(_guards.get("meta_c_lite_live_on", False)):
            _meta_applied = bool(_guards.get("meta_c_lite_live_applied", False)) or bool(_meta.get("applied", False))
            if _meta_applied:
                pipeline_meta_method = str(_meta.get("method", "") or "")
                _m = pipeline_meta_method.lower()
                if _m.startswith("meta_blend_c_model"):
                    pipeline_adj += 10.0
                    pipeline_notes.append("C treinado")
                else:
                    pipeline_adj += 4.0
                    pipeline_notes.append("C fallback")
                _d_meta = abs(_nan_or_float(_meta.get("diff_pp")))
                if math.isfinite(_d_meta) and _d_meta > 18.0:
                    pipeline_adj -= 2.0
                    pipeline_notes.append("C muito agressivo")
            else:
                pipeline_adj -= 5.0
                pipeline_notes.append("C sem aplicar")

        # LayerUnc (incerteza formal por camada; shrink conservador em logit)
        if bool(_guards.get("layer_uncertainty_live_on", False)):
            _lu_applied = bool(_guards.get("layer_uncertainty_live_applied", False)) or bool(_layer_unc.get("applied", False))
            if _lu_applied:
                _d_lu = abs(_nan_or_float(_layer_unc.get("diff_pp")))
                _u_score = _nan_or_float(_layer_unc.get("uncertainty_score"))
                _c_score = _nan_or_float(_layer_unc.get("confidence_score"))
                if math.isfinite(_d_lu):
                    if _d_lu <= 1.0:
                        pipeline_adj += 2.0
                    elif _d_lu <= 2.5:
                        pipeline_adj += 1.0
                    elif _d_lu <= 4.0:
                        pipeline_adj += 0.25
                    else:
                        pipeline_adj -= 1.0
                    pipeline_notes.append(f"LayerUnc Δ={_d_lu:.1f}pp")
                else:
                    pipeline_notes.append("LayerUnc ativo")
                if math.isfinite(_u_score):
                    if _u_score >= 0.70:
                        pipeline_adj -= 3.0
                    elif _u_score >= 0.55:
                        pipeline_adj -= 1.5
                    elif _u_score <= 0.30:
                        pipeline_adj += 1.0
                if math.isfinite(_c_score) and _c_score >= 0.80:
                    pipeline_adj += 1.0
                _zone = str(_layer_unc.get("actionability_zone", "") or "").strip().lower()
                if _zone == "low":
                    pipeline_adj -= 3.0
                    pipeline_notes.append("acionabilidade baixa")
                elif _zone == "medium":
                    pipeline_adj -= 1.0
                    pipeline_notes.append("acionabilidade media")
                elif _zone == "high":
                    pipeline_adj += 0.5
                    pipeline_notes.append("acionabilidade alta")
            else:
                pipeline_notes.append("LayerUnc sem aplicar")

        # Qualidade de fusao/players (lineup)
        _cov = _nan_or_float(_fusion.get("coverage_eff"))
        if math.isfinite(_cov):
            if _cov >= 0.80:
                pipeline_adj += 3.0
            elif _cov >= 0.50:
                pipeline_adj += 1.0
            elif _cov < 0.30:
                pipeline_adj -= 3.0

        _npe = _nan_or_float(_padj.get("n_players_eff"))
        if bool(_padj.get("enabled", False)) and math.isfinite(_npe):
            if _npe >= 20:
                pipeline_adj += 4.0
            elif _npe >= 10:
                pipeline_adj += 2.0
            elif _npe < 5:
                pipeline_adj -= 3.0

        # Guards que reduzem confiança quando precisaram agir
        if bool(_guards.get("consistency_guard_triggered", False)):
            pipeline_adj -= 4.0
            pipeline_notes.append("consistency guard")
        if bool(_guards.get("patch_guard_applied", False)):
            _pgs = str(_guards.get("patch_guard_status", "") or "").upper()
            if "REFIT" in _pgs:
                pipeline_adj -= 10.0
                pipeline_notes.append("patch REFIT")
            elif ("WARN" in _pgs) or ("ATEN" in _pgs):
                pipeline_adj -= 5.0
                pipeline_notes.append("patch WARN")
        if bool(_guards.get("wf_correction_applied", False)):
            _wf_sh = abs(_nan_or_float(_guards.get("wf_shrink")))
            pipeline_adj += (2.0 if (math.isfinite(_wf_sh) and _wf_sh <= 0.08) else 0.5)
            pipeline_notes.append("WF correction")
    except (AttributeError, KeyError, TypeError, ValueError):
        pipeline_adj += 0.0

    # Ajuste de baseline por liga (construído a partir de layer_validation + walk-forward).
    try:
        _cb = confidence_baseline if isinstance(confidence_baseline, dict) else {}
        if _cb and not _cb.get("error"):
            _by = _cb.get("by_league") if isinstance(_cb.get("by_league"), dict) else {}
            _g = _cb.get("global") if isinstance(_cb.get("global"), dict) else {}
            _lg = str(league_hint or "").strip()
            _rowb = _by.get(_lg) if (_lg and isinstance(_by.get(_lg), dict)) else None
            if isinstance(_rowb, dict):
                baseline_source = f"liga:{_lg}"
                baseline_band = str(_rowb.get("quality_band") or "")
                baseline_adj = float(_safe_float(_rowb.get("score_bias")) or 0.0)
            else:
                baseline_source = "global"
                baseline_band = "global"
                baseline_adj = float(_safe_float(_g.get("score_bias")) or 0.0)
            baseline_adj = float(max(-15.0, min(15.0, baseline_adj)))
        else:
            baseline_source = ""
            baseline_adj = 0.0
    except (AttributeError, KeyError, TypeError, ValueError):
        baseline_source = ""
        baseline_band = ""
        baseline_adj = 0.0

    score += float(pipeline_adj)
    score += float(baseline_adj)

    score = float(max(0.0, min(100.0, score)))
    if score >= 80.0:
        grade = "A (Alta)"
    elif score >= 65.0:
        grade = "B (Media)"
    else:
        grade = "C (Baixa)"

    return {
        "grade": grade,
        "score": score,
        "source": source,
        "n_total": n_total,
        "logloss": ll if math.isfinite(ll) else None,
        "gain_vs_team_logloss": gain if math.isfinite(gain) else None,
        "pipeline_adj": float(pipeline_adj),
        "pipeline_meta_method": (pipeline_meta_method or None),
        "pipeline_summary": " | ".join([x for x in pipeline_notes if str(x).strip()])[:280],
        "baseline_adj": float(baseline_adj),
        "baseline_source": (baseline_source or None),
        "baseline_band": (baseline_band or None),
    }




__all__ = ["ml_confidence_from_layer_validation"]
