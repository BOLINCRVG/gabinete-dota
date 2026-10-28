"""ML pre-live pipeline orchestration extracted from plays_app."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping

import pandas as pd

from app_core.pipeline_trace_core import append_pipeline_step
from mlcore.league_context import is_explicit_interregional_league_label as shared_is_explicit_interregional_league_label
from mlcore.tiers import league_strength_band, league_strength_score


def _f(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return float(x) if math.isfinite(x) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _norm_token(v: Any) -> str:
    txt = str(v or "").strip().lower()
    if not txt:
        return ""
    try:
        norm = unicodedata.normalize("NFKD", txt)
        return "".join(ch for ch in norm if ord(ch) < 128)
    except TypeError:
        return txt


def _league_hint(*, league_mode: str, filters: Any, dfA_used: Any, dfB_used: Any) -> str:
    hint = ""
    try:
        if str(league_mode) == "fixed" and getattr(filters, "league", None):
            hint = str(filters.league or "").strip()
        if not hint and isinstance(dfA_used, pd.DataFrame) and (not dfA_used.empty) and ("league" in dfA_used.columns):
            v = dfA_used["league"].dropna().astype(str).str.strip()
            if not v.empty:
                hint = str(v.value_counts().index[0]).strip()
        if not hint and isinstance(dfB_used, pd.DataFrame) and (not dfB_used.empty) and ("league" in dfB_used.columns):
            v = dfB_used["league"].dropna().astype(str).str.strip()
            if not v.empty:
                hint = str(v.value_counts().index[0]).strip()
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        hint = ""
    return str(hint or "")


def _dominant_league(df: Any) -> str:
    try:
        if isinstance(df, pd.DataFrame) and (not df.empty) and ("league" in df.columns):
            v = df["league"].dropna().astype(str).str.strip()
            if not v.empty:
                return str(v.value_counts().index[0]).strip()
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return ""
    return ""


def _league_pair_context(
    *,
    league_mode: str,
    filters: Any,
    dfA_used: Any,
    dfB_used: Any,
    lg_a_shadow: Any = "",
    lg_b_shadow: Any = "",
) -> Dict[str, Any]:
    lg_a = str(lg_a_shadow or "").strip()
    lg_b = str(lg_b_shadow or "").strip()
    try:
        if str(league_mode) == "fixed" and getattr(filters, "league", None):
            fixed = str(filters.league or "").strip()
            lg_a = fixed
            lg_b = fixed
        else:
            lg_a = lg_a or _dominant_league(dfA_used)
            lg_b = lg_b or _dominant_league(dfB_used)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        pass
    same = bool(lg_a and lg_b and _norm_token(lg_a) == _norm_token(lg_b))
    explicit_inter = bool(_is_explicit_interregional_league(lg_a) or _is_explicit_interregional_league(lg_b))
    interregional = bool((lg_a and lg_b and not same) or explicit_inter)
    return {
        "league_a": str(lg_a or ""),
        "league_b": str(lg_b or ""),
        "same_league": bool(same and not explicit_inter),
        "interregional": bool(interregional),
        "explicit_interregional": bool(explicit_inter),
    }


def _safe_prob_win_series(prob_win_series_fn: Callable[[float, int], float], p_map: float, bo: int) -> float:
    try:
        return float(prob_win_series_fn(float(p_map), int(bo)))
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _is_explicit_interregional_league(v: Any) -> bool:
    return bool(shared_is_explicit_interregional_league_label(v))


def _combined_confidence_reference(
    *,
    n_team_eff: Any,
    n_players_eff: Any,
    interregional: bool,
    players_weight: float,
    players_cap: float,
) -> Dict[str, float]:
    team_ref = _f(n_team_eff, float("nan"))
    players_ref_raw = _f(n_players_eff, float("nan"))
    players_ref = float("nan")
    if bool(interregional) and math.isfinite(players_ref_raw) and players_ref_raw > 0.0:
        pw = max(0.0, float(players_weight))
        pc = max(0.0, float(players_cap))
        players_ref = min(pc, max(0.0, players_ref_raw) * pw) if pc > 0.0 else max(0.0, players_ref_raw) * pw
    merged = team_ref
    if math.isfinite(players_ref):
        merged = players_ref if (not math.isfinite(merged)) else max(float(merged), float(players_ref))
    return {
        "team_ref": float(team_ref) if math.isfinite(team_ref) else float("nan"),
        "players_ref": float(players_ref) if math.isfinite(players_ref) else float("nan"),
        "merged_ref": float(merged) if math.isfinite(merged) else float("nan"),
    }


def _clip_prob_local(p: Any) -> float:
    return float(max(1e-6, min(1.0 - 1e-6, _f(p, 0.5))))


_PSCORE_CACHE: Dict[str, Any] = {}


def _carregar_pscore(artifact_path: str) -> Dict[str, Any] | None:
    """Le players_pscore.json ao lado do artefato. Devolve None se nao houver.

    AUSENCIA NAO E ERRO: modelo treinado antes desta camada, ou com
    --no-pscore, simplesmente nao tem o arquivo, e o estagio fica inerte.
    O cache e por (caminho, mtime, tamanho) para pegar retreino sem reiniciar.
    """
    import json as _json
    import os as _os

    try:
        alvo = Path(str(artifact_path)).expanduser().with_name("players_pscore.json")
    except (TypeError, ValueError):
        return None
    try:
        st = _os.stat(str(alvo))
    except OSError:
        return None
    chave = "{}|{}|{}".format(alvo, int(st.st_mtime), int(st.st_size))
    if chave in _PSCORE_CACHE:
        return _PSCORE_CACHE[chave]
    try:
        with open(str(alvo), "r", encoding="utf-8") as fh:
            pac = _json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(pac, dict) or not isinstance(pac.get("notas"), dict):
        return None
    _PSCORE_CACHE.clear()
    _PSCORE_CACHE[chave] = pac
    return pac


def _aplicar_pscore(
    *,
    p_map: float,
    artifact_path: str,
    blue_lineup: Mapping[str, Any],
    red_lineup: Mapping[str, Any],
    session_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Desloca o logito pela diferenca de nota de desempenho das escalacoes."""
    from mlcore.players_pscore_core import delta_da_escalacao as _delta_ps

    if not bool(session_state.get("ml_pscore_enabled", True)):
        return {"applied": False, "reason": "desligado"}
    if not math.isfinite(float(p_map)):
        return {"applied": False, "reason": "p_map invalido"}

    pac = _carregar_pscore(str(artifact_path))
    if not pac:
        return {"applied": False, "reason": "sem players_pscore.json"}

    w = float(pac.get("w", 0.0) or 0.0)
    if not math.isfinite(w) or abs(w) <= 1e-12:
        return {"applied": False, "reason": "peso zero"}

    # `apelidos` traduz NOME -> id do jogador.
    #
    # ESTE CAMINHO NAO PRECISAVA DELE: a escalacao daqui vem de `lineup_ml_*`,
    # que guarda a CHAVE do jogador (o editor mostra o nome e grava a chave).
    # Conferido no app dirigido em 14/09/2026 com o artefato de producao:
    # applied=True, cobertura 5/5. Quem precisa e o Radar, que monta o draft
    # por `playername`.
    #
    # Fica aqui mesmo assim porque custa nada e cobre quem chamar por nome —
    # e porque em 13/09 eu concluí, ERRADO, que esta chamada recebia nomes.
    # Artefato antigo nao tem a chave; `.get` devolve None e nada muda.
    d, cobertura = _delta_ps(
        pac.get("notas") or {},
        blue_lineup={str(k): str(v) for k, v in (blue_lineup or {}).items() if v},
        red_lineup={str(k): str(v) for k, v in (red_lineup or {}).items() if v},
        apelidos=pac.get("apelidos") or {},
    )
    # PRINCIPIO IV: escalacao incompleta nao vira um delta menor disfarcado de
    # delta certo. Sem os 5 de cada lado, o estagio nao age.
    minimo = int(session_state.get("ml_pscore_min_cobertura", 5) or 5)
    if int(cobertura) < minimo:
        return {"applied": False, "reason": "cobertura {}/{}".format(cobertura, minimo),
                "delta": float(d), "cobertura": int(cobertura)}
    if not math.isfinite(float(d)) or abs(float(d)) <= 1e-12:
        return {"applied": False, "reason": "delta zero",
                "delta": float(d), "cobertura": int(cobertura)}

    pc = min(max(float(p_map), 1e-6), 1.0 - 1e-6)
    z = math.log(pc / (1.0 - pc)) + w * float(d)
    p_out = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    return {
        "applied": True,
        "reason": "ok",
        "p_out": float(p_out),
        "delta": float(d),
        "cobertura": int(cobertura),
        "w": float(w),
    }


def _interregional_strength_prior(
    *,
    p_map: Any,
    bo: int,
    league_pair_ctx: Dict[str, Any],
    n_team_eff: Any,
    n_players_eff: Any,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "p_in": (_f(p_map) if math.isfinite(_f(p_map)) else None),
        "p_out": (_f(p_map) if math.isfinite(_f(p_map)) else None),
        "diff_pp": 0.0,
        "league_a": str(league_pair_ctx.get("league_a", "") or ""),
        "league_b": str(league_pair_ctx.get("league_b", "") or ""),
        "strength_score_a": None,
        "strength_score_b": None,
        "strength_gap": None,
        "stronger_side": "",
        "stronger_band": "",
        "weaker_band": "",
        "target_floor": None,
        "move_cap_pp": None,
        "n_team_eff": (float(_f(n_team_eff)) if math.isfinite(_f(n_team_eff)) else None),
        "n_players_eff": (float(_f(n_players_eff)) if math.isfinite(_f(n_players_eff)) else None),
    }
    p0 = _f(p_map)
    if not math.isfinite(p0):
        out["reason"] = "invalid_p_in"
        return out
    if not bool(league_pair_ctx.get("interregional", False)):
        out["reason"] = "same_league"
        return out
    la = str(league_pair_ctx.get("league_a", "") or "").strip()
    lb = str(league_pair_ctx.get("league_b", "") or "").strip()
    if not la or not lb:
        out["reason"] = "missing_leagues"
        return out
    sa = float(max(0.0, min(1.0, league_strength_score(la))))
    sb = float(max(0.0, min(1.0, league_strength_score(lb))))
    gap = float(abs(sa - sb))
    out["strength_score_a"] = float(sa)
    out["strength_score_b"] = float(sb)
    out["strength_gap"] = float(gap)
    if gap < 0.16:
        out["reason"] = "gap_too_small"
        return out
    stronger_side = "a" if sa >= sb else "b"
    stronger_band = str(league_strength_band(la if stronger_side == "a" else lb) or "local")
    weaker_band = str(league_strength_band(lb if stronger_side == "a" else la) or "local")
    out["stronger_side"] = str(stronger_side)
    out["stronger_band"] = str(stronger_band)
    out["weaker_band"] = str(weaker_band)
    if stronger_band not in {"elite", "major"}:
        out["reason"] = "stronger_band_not_high"
        return out
    if weaker_band not in {"upper", "regional", "development", "local"}:
        out["reason"] = "weaker_band_not_low_enough"
        return out
    nt = max(0.0, _f(n_team_eff, 0.0))
    np_ = max(0.0, _f(n_players_eff, 0.0))
    stronger_p = float(p0 if stronger_side == "a" else (1.0 - p0))
    floor = 0.62
    if stronger_band == "elite":
        floor = 0.67
    elif stronger_band == "major" and weaker_band in {"regional", "development", "local"}:
        floor = 0.64
    floor += float(min(0.03, max(0.0, gap - 0.16) * 0.30))
    if int(bo or 1) >= 5:
        floor += 0.01
    floor = float(max(0.60, min(0.72, floor)))
    out["target_floor"] = float(floor)
    if stronger_p >= floor:
        out["reason"] = "already_above_floor"
        return out
    evidence = float(max(0.0, min(1.0, (0.75 * nt + 0.25 * np_) / 36.0)))
    move_cap_pp = float(2.5 + (5.0 * (1.0 - evidence)))
    if stronger_band == "elite":
        move_cap_pp += 1.0
    if gap >= 0.19:
        move_cap_pp += 0.8
    move_cap_pp = float(max(2.5, min(9.0, move_cap_pp)))
    out["move_cap_pp"] = float(move_cap_pp)
    target_p = float(floor if stronger_side == "a" else (1.0 - floor))
    requested_pp = abs(target_p - p0) * 100.0
    if requested_pp <= 0.02:
        out["reason"] = "tiny_move"
        return out
    applied_pp = float(min(move_cap_pp, requested_pp))
    delta = applied_pp / 100.0
    p1 = float(_clip_prob_local(p0 + (delta if target_p > p0 else -delta)))
    out["applied"] = True
    out["reason"] = "interregional_strength_floor"
    out["p_out"] = float(p1)
    out["diff_pp"] = float(100.0 * (p1 - p0))
    out["requested_pp"] = float(requested_pp)
    out["applied_pp"] = float(applied_pp)
    return out


_STRONG_BANDS = {"elite", "major"}
_WEAK_BANDS = {"upper", "regional", "development", "local"}


def _elite_vs_weak_mismatch(strength_info: Any) -> bool:
    """Descasamento forte: banda elite/major contra upper..local com gap grande.

    Mesma condição que os estágios de forma, híbrido e confidence-guard usam
    para amortecer ainda mais o ajuste interregional (a ação/multiplicador é
    específica de cada estágio; só o gatilho é comum)."""
    info = strength_info or {}
    sg = float(_f(info.get("strength_gap"), 0.0))
    sb = str(info.get("stronger_band") or "")
    wb = str(info.get("weaker_band") or "")
    return sg >= 0.16 and sb in _STRONG_BANDS and wb in _WEAK_BANDS


def _stage_series_gamma(
    *, p_map: float, p_series: float, bo_i: int, ss: Mapping[str, Any], steps: list,
    league_mode: str, filters: Any, dfA_used: Any, dfB_used: Any,
    series_gamma_for_league_fn: Callable, apply_series_gamma_fn: Callable,
    prob_win_series_fn: Callable,
) -> tuple:
    """Estágio series_gamma. Retorna (p_series, trace_series_gamma_info)."""
    trace_series_gamma_info: Dict[str, Any] = {}
    try:
        ps_before = float(p_series) if math.isfinite(float(p_series)) else float("nan")
        use_sg = bool(ss.get("ml_series_gamma_use_fitted", True))
        sg = {"enabled": bool(use_sg), "applied": False, "gamma": 1.0, "source": ""}
        if use_sg and int(bo_i) in (3, 5) and math.isfinite(float(p_map)):
            lg_hint = _league_hint(league_mode=str(league_mode), filters=filters, dfA_used=dfA_used, dfB_used=dfB_used)
            gamma, src = series_gamma_for_league_fn(str(lg_hint), int(bo_i))
            ps_base = float(prob_win_series_fn(float(p_map), int(bo_i)))
            ps_new = float(apply_series_gamma_fn(float(ps_base), float(gamma)))
            p_series = float(ps_new)
            sg.update({"applied": bool(abs(float(gamma) - 1.0) > 1e-12), "gamma": float(gamma), "source": str(src or ""), "bo": int(bo_i), "p_base": float(ps_base), "p_out": float(ps_new), "league_hint": str(lg_hint or "")})
        trace_series_gamma_info = dict(sg)
        ps_after = float(p_series) if math.isfinite(float(p_series)) else float("nan")
        applied = bool(trace_series_gamma_info.get("applied", False)) or (
            math.isfinite(float(ps_before)) and math.isfinite(float(ps_after)) and abs(float(ps_after) - float(ps_before)) > 1e-12
        )
        append_pipeline_step(
            steps=steps,
            stage="series_gamma",
            before=ps_before,
            after=ps_after,
            applied=bool(applied),
            enabled=bool(ss.get("ml_series_gamma_use_fitted", True)),
            source=str(trace_series_gamma_info.get("source", "")),
            gamma=(
                float(trace_series_gamma_info.get("gamma"))
                if trace_series_gamma_info.get("gamma") is not None
                else None
            ),
        )
    except Exception as exc:
        trace_series_gamma_info = {"enabled": bool(ss.get("ml_series_gamma_use_fitted", True)), "applied": False}
        append_pipeline_step(
            steps=steps,
            stage="series_gamma",
            before=p_series,
            after=p_series,
            applied=False,
            enabled=bool(ss.get("ml_series_gamma_use_fitted", True)),
            error=f"{type(exc).__name__}: {exc}",
        )
    return p_series, trace_series_gamma_info


def _stage_series_economic_coherence(
    *, p_map: float, p_series: float, bo_i: int, steps: list,
    enforce_series_economic_coherence_fn: Callable, default_trace: Dict[str, Any],
) -> tuple:
    """Estágio series_economic_coherence. Retorna (p_series, trace)."""
    trace_series_coherence_info = default_trace
    try:
        ps_before = float(p_series) if math.isfinite(float(p_series)) else float("nan")
        pm = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        if int(bo_i) in (3, 5) and math.isfinite(pm) and math.isfinite(ps_before):
            coh = enforce_series_economic_coherence_fn(p_map=float(pm), p_series=float(ps_before), bo=int(bo_i))
            if isinstance(coh, dict):
                ps_after = float(coh.get("p_series_after", float("nan")) or float("nan"))
                if bool(coh.get("applied", False)) and math.isfinite(ps_after):
                    p_series = float(ps_after)
                trace_series_coherence_info = dict(coh)
                append_pipeline_step(
                    steps=steps,
                    stage="series_economic_coherence",
                    before=ps_before,
                    after=p_series,
                    applied=bool(coh.get("applied", False)),
                    enabled=True,
                    source=str(coh.get("reason", "")),
                )
    except Exception as exc:
        trace_series_coherence_info = {
            "enabled": True,
            "applied": False,
            "reason": f"error:{type(exc).__name__}",
            "error": str(exc),
            "p_map": (float(p_map) if math.isfinite(float(p_map)) else None),
            "p_series_before": (float(p_series) if math.isfinite(float(p_series)) else None),
            "p_series_after": None,
        }
        append_pipeline_step(
            steps=steps,
            stage="series_economic_coherence",
            before=p_series,
            after=p_series,
            applied=False,
            enabled=True,
            error=f"{type(exc).__name__}: {exc}",
        )
    return p_series, trace_series_coherence_info


def _stage_series_bo_dplite_live(
    *, p_map: float, p_series: float, bo_i: int, ss: Mapping[str, Any], ssw: Any, steps: list,
    league_mode: str, filters: Any, dfA_used: Any, dfB_used: Any,
    lg_a_shadow: Any, lg_b_shadow: Any, league_pair_ctx: Dict[str, Any],
    trace_patch: Dict[str, Any], trace_style_b_lite_live: Any, ml_conf_score_snapshot: Any,
    trace_n_team_eff: Any, trace_n_players_eff: Any, playoffs_opt: Any,
    feature_flag_enabled_fn: Callable, apply_series_bo_dplite_live_fn: Callable,
) -> tuple:
    """Estágio series_bo_dplite_live. Retorna (p_series, ff_series_bo, trace)."""
    ff_series_bo = bool(feature_flag_enabled_fn("ff_ml_series_bo_dplite_live", False))
    trace_series_bo_dplite_live: Dict[str, Any] = {
        "enabled": bool(ff_series_bo),
        "applied": False,
        "reason": "disabled_by_flag" if not bool(ff_series_bo) else "",
        "p_map": (float(p_map) if math.isfinite(float(p_map)) else None),
        "p_series_before": (float(p_series) if math.isfinite(float(p_series)) else None),
        "p_series_after": None,
    }
    try:
        if bool(ff_series_bo):
            ps_before = float(p_series) if math.isfinite(float(p_series)) else float("nan")
            pm = float(p_map) if math.isfinite(float(p_map)) else float("nan")
            _pair_series = _league_pair_context(
                league_mode=str(league_mode),
                filters=filters,
                dfA_used=dfA_used,
                dfB_used=dfB_used,
                lg_a_shadow=lg_a_shadow,
                lg_b_shadow=lg_b_shadow,
            )
            lg_a = str((_pair_series or {}).get("league_a") or "").strip().upper()
            lg_b = str((_pair_series or {}).get("league_b") or "").strip().upper()
            if (not lg_a) or (not lg_b):
                _ml_blue_series = str((league_pair_ctx or {}).get("league_a") or lg_a_shadow or "").strip().upper()
                _ml_red_series = str((league_pair_ctx or {}).get("league_b") or lg_b_shadow or "").strip().upper()
                if _ml_blue_series:
                    lg_a = lg_a or _ml_blue_series
                if _ml_red_series:
                    lg_b = lg_b or _ml_red_series
            if (not lg_a) or (not lg_b):
                lg_hint = str((trace_patch or {}).get("league_hint", "") or "").strip().upper()
                if lg_hint:
                    lg_a = lg_a or lg_hint
                    lg_b = lg_b or lg_hint
            explicit_inter = bool(_is_explicit_interregional_league(lg_a) or _is_explicit_interregional_league(lg_b))
            same_lg = bool(lg_a) and bool(lg_b) and (lg_a == lg_b) and (not explicit_inter)
            side_bias = None
            try:
                sbp = (trace_patch or {}).get("side_adv_pp", None)
                if sbp is not None and math.isfinite(float(sbp)):
                    side_bias = float(sbp)
            except (TypeError, ValueError):
                side_bias = None
            style_vol = None
            try:
                if isinstance(trace_style_b_lite_live, dict):
                    sv = trace_style_b_lite_live.get("style_volatility", None)
                    if sv is not None and math.isfinite(float(sv)):
                        style_vol = float(sv)
            except (TypeError, ValueError):
                style_vol = None
            layer_conf = None
            try:
                if ml_conf_score_snapshot is not None and math.isfinite(float(ml_conf_score_snapshot)):
                    layer_conf = float(ml_conf_score_snapshot)
            except (TypeError, ValueError):
                layer_conf = None
            sbd = apply_series_bo_dplite_live_fn(
                p_map=(pm if math.isfinite(pm) else None),
                p_series_in=(ps_before if math.isfinite(ps_before) else None),
                bo=int(bo_i),
                same_league=bool(same_lg),
                interregional=(not bool(same_lg)) or bool(explicit_inter),
                playoffs=playoffs_opt,
                patch_status=str((trace_patch or {}).get("status", "") or ""),
                n_team_eff=(float(trace_n_team_eff) if math.isfinite(_f(trace_n_team_eff)) else None),
                n_players_eff=(float(trace_n_players_eff) if math.isfinite(_f(trace_n_players_eff)) else None),
                side_bias_pp=side_bias,
                style_volatility=style_vol,
                layer_confidence_score=layer_conf,
                league_hint=(lg_a if lg_a else lg_b),
                tuning_cfg=(ss.get("_fit_series_bo_dplite_cfg") if isinstance(ss.get("_fit_series_bo_dplite_cfg"), dict) else {}),
            )
            if isinstance(sbd, dict):
                sbd.setdefault("enabled", True)
                ps_after = float(sbd.get("p_after", float("nan")) or float("nan"))
                if bool(sbd.get("applied", False)) and math.isfinite(ps_after):
                    p_series = float(ps_after)
                    if isinstance(ssw, MutableMapping):
                        ssw["_p_series_used"] = float(p_series)
                    sbd["p_series_before"] = (float(ps_before) if math.isfinite(ps_before) else None)
                    sbd["p_series_after"] = (float(p_series) if math.isfinite(float(p_series)) else None)
                    append_pipeline_step(
                        steps=steps,
                        stage="series_bo_dplite_live",
                        before=ps_before,
                        after=p_series,
                        applied=True,
                        enabled=True,
                        source=str(sbd.get("method") or "series_bo_dplite"),
                    )
                trace_series_bo_dplite_live = dict(sbd)
    except Exception as exc:
        trace_series_bo_dplite_live = {
            "enabled": bool(ff_series_bo),
            "applied": False,
            "reason": f"error:{type(exc).__name__}",
            "error": str(exc),
            "p_map": (float(p_map) if math.isfinite(float(p_map)) else None),
            "p_series_before": (float(p_series) if math.isfinite(float(p_series)) else None),
            "p_series_after": None,
        }
        append_pipeline_step(
            steps=steps,
            stage="series_bo_dplite_live",
            before=p_series,
            after=p_series,
            applied=False,
            enabled=bool(ff_series_bo),
            error=f"{type(exc).__name__}: {exc}",
        )
    return p_series, ff_series_bo, trace_series_bo_dplite_live


def _stage_season_form_blend(
    *, p_map: float, p_series: float, bo_i: int, ss: Mapping[str, Any], ssw: Any, steps: list,
    league_mode: str, filters: Any, dfA_used: Any, dfB_used: Any,
    interregional_pre: bool, league_pair_ctx: Dict[str, Any],
    trace_interregional_strength_info: Dict[str, Any],
    season_form_blend_fn: Callable, current_season_override_fn: Callable,
    prob_win_series_fn: Callable, fit_form_for_league_fn: Any,
) -> tuple:
    """Estágio season_form_blend. Retorna (p_map, p_series, trace_form_info)."""
    trace_form_info: Dict[str, Any] = {}
    try:
        p_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        if math.isfinite(float(p_map)):
            fit_cfg = ss.get("_fit_form_cfg") if isinstance(ss.get("_fit_form_cfg"), dict) else {}
            fit_lg = ""
            try:
                lg_hint = _league_hint(league_mode=str(league_mode), filters=filters, dfA_used=dfA_used, dfB_used=dfB_used)
                if callable(fit_form_for_league_fn):
                    fit_ret = fit_form_for_league_fn(str(lg_hint))
                    if isinstance(fit_ret, tuple) and len(fit_ret) >= 2:
                        fit_cfg = fit_ret[0] if isinstance(fit_ret[0], dict) else {}
                        fit_lg = str(fit_ret[1] or "")
                    elif isinstance(fit_ret, dict):
                        fit_cfg = fit_ret
                if isinstance(ssw, MutableMapping):
                    ssw["_fit_form_cfg_used"] = dict(fit_cfg) if isinstance(fit_cfg, dict) else {}
                    ssw["_fit_form_cfg_league_used"] = str(fit_lg or "")
            except Exception:
                pass
            use_fit = bool(ss.get("ml_form_use_fitted", True))
            sf_enabled = bool(ss.get("ml_form_blend_enabled", True))
            sf_w = float(ss.get("ml_form_blend_max_weight", 0.75) or 0.75)
            sf_knee = float(ss.get("ml_form_blend_knee_games", 6.0) or 6.0)
            sf_beta = float(ss.get("ml_form_blend_beta", 2.0) or 2.0)
            sf_pow = float(ss.get("ml_form_blend_signal_power", 1.0) or 1.0)
            sf_par = float(ss.get("ml_form_blend_parity_strength", 0.35) or 0.35)
            if use_fit and isinstance(fit_cfg, dict) and fit_cfg:
                sf_enabled = bool(fit_cfg.get("ml_form_blend_enabled", sf_enabled))
                sf_w = float(fit_cfg.get("ml_form_blend_max_weight", sf_w) or sf_w)
                sf_knee = float(fit_cfg.get("ml_form_blend_knee_games", sf_knee) or sf_knee)
                sf_beta = float(fit_cfg.get("ml_form_blend_beta", sf_beta) or sf_beta)
                sf_pow = float(fit_cfg.get("ml_form_blend_signal_power", sf_pow) or sf_pow)
                sf_par = float(fit_cfg.get("ml_form_blend_parity_strength", sf_par) or sf_par)
            form_damp = {
                "interregional": bool(interregional_pre),
                "league_a": str(league_pair_ctx.get("league_a", "") or ""),
                "league_b": str(league_pair_ctx.get("league_b", "") or ""),
                "max_weight_mult": 1.0,
                "knee_mult": 1.0,
                "override_disabled": False,
            }
            if bool(interregional_pre):
                w_mult = float(ss.get("ml_interregional_form_max_weight_mult", 0.18) or 0.18)
                knee_mult = float(ss.get("ml_interregional_form_knee_mult", 2.0) or 2.0)
                if _elite_vs_weak_mismatch(trace_interregional_strength_info):
                    w_mult *= 0.55
                    knee_mult *= 1.20
                sf_w *= max(0.0, min(1.0, w_mult))
                sf_knee *= max(1.0, knee_mult)
                form_damp["max_weight_mult"] = float(max(0.0, min(1.0, w_mult)))
                form_damp["knee_mult"] = float(max(1.0, knee_mult))
                form_damp["override_disabled"] = bool(
                    ss.get("ml_interregional_disable_form_override", True)
                )
            form = season_form_blend_fn(
                p_model=float(p_map),
                df_team_a=dfA_used,
                df_team_b=dfB_used,
                enabled=bool(sf_enabled),
                max_weight=float(sf_w),
                knee_games=float(sf_knee),
                beta_prior=float(sf_beta),
                signal_power=float(sf_pow),
                elite_same_league_parity_strength=float(sf_par),
            )
            if isinstance(form, dict):
                trace_form_info = dict(form)
                trace_form_info["source"] = (f"fitted:{fit_lg}" if (use_fit and fit_cfg and fit_lg) else ("fitted:global" if (use_fit and fit_cfg) else "manual"))
                trace_form_info["league_pair"] = dict(league_pair_ctx)
                trace_form_info["interregional_damp"] = dict(form_damp)
                p_bl = form.get("p_blended")
                if form.get("applied") and p_bl is not None and math.isfinite(float(p_bl)):
                    allow_override = not bool(form_damp.get("override_disabled", False))
                    if bool(allow_override):
                        ov = current_season_override_fn(
                            p_model=float(form.get("p_model", p_map)),
                            p_form=float(form.get("p_form", float("nan"))),
                            w_form=float(form.get("w_form", 0.0) or 0.0),
                            n_a=int(form.get("n_a", 0) or 0),
                            n_b=int(form.get("n_b", 0) or 0),
                            min_games_gate=int(ss.get("ml_form_override_min_games", 8) or 8),
                            divergence_gate_pp=float(ss.get("ml_form_override_div_pp", 8.0) or 8.0),
                            max_weight=float(ss.get("ml_form_override_max_w", 0.88) or 0.88),
                        )
                        if isinstance(ov, dict) and bool(ov.get("applied")):
                            p_bl = float(ov.get("p_out", p_bl))
                            trace_form_info["override"] = dict(ov)
                    else:
                        trace_form_info["override"] = {
                            "applied": False,
                            "reason": "disabled_interregional",
                        }
                    p_map = float(p_bl)
                    p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
        p_after = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        applied = bool(trace_form_info.get("applied", False)) or (
            math.isfinite(float(p_before)) and math.isfinite(float(p_after)) and abs(float(p_after) - float(p_before)) > 1e-12
        )
        append_pipeline_step(
            steps=steps,
            stage="season_form_blend",
            before=p_before,
            after=p_after,
            applied=bool(applied),
            source=str(trace_form_info.get("source", "")),
        )
    except Exception as exc:
        append_pipeline_step(
            steps=steps,
            stage="season_form_blend",
            before=p_map,
            after=p_map,
            applied=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return p_map, p_series, trace_form_info


def _stage_nota_jogadores_lol(
    *, p_map: float, p_series: float, bo_i: int, ssw: Any, steps: list, teamA: Any, teamB: Any,
    league_pair_ctx: Dict[str, Any], prob_win_series_fn: Callable,
) -> tuple:
    """Estagio nota_jogadores (so LoL). Retorna (p_map, p_series); nunca derruba a previsao."""
    try:
        from app_core.nota_jogadores_lol_core import p_com_jogadores
    except ImportError:
        return p_map, p_series
    try:
        entre = bool(league_pair_ctx.get("interregional") or league_pair_ctx.get("explicit_interregional"))
        nj = p_com_jogadores(str(teamA), str(teamB), float(p_map), entre_regioes=entre)
        antes = float(p_map)
        if nj.get("aplicado") and math.isfinite(float(nj.get("p_final", float("nan")))):
            p_map = float(nj["p_final"])
            p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
        append_pipeline_step(steps=steps, stage="nota_jogadores", before=antes, after=float(p_map),
                             applied=bool(nj.get("aplicado")), engine="nota_jogadores_lol",
                             reason=str(nj.get("motivo") or ""))
        if ssw is not None:
            ssw["_nota_jogadores_info"] = dict(nj)
    except Exception as exc:  # nunca derruba a previsao
        append_pipeline_step(steps=steps, stage="nota_jogadores", before=p_map, after=p_map, applied=False,
                             engine="nota_jogadores_lol", error=f"{type(exc).__name__}: {exc}")
    return p_map, p_series


def _stage_hybrid_blend(
    *, p_map: float, p_series: float, bo_i: int, ss: Mapping[str, Any], steps: list,
    dfA_used: Any, dfB_used: Any, interregional_pre: bool, league_pair_ctx: Dict[str, Any],
    trace_interregional_strength_info: Dict[str, Any],
    hybrid_momentum_blend_fn: Callable, prob_win_series_fn: Callable,
) -> tuple:
    """Estágio hybrid_blend. Retorna (p_map, p_series, trace_hybrid_info)."""
    trace_hybrid_info: Dict[str, Any] = {}
    try:
        p_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        if math.isfinite(float(p_map)):
            hy_max_weight = float(ss.get("ml_hybrid_blend_max_weight", 0.70) or 0.70)
            hy_knee = float(ss.get("ml_hybrid_blend_knee_games", 12.0) or 12.0)
            hy_damp = {
                "interregional": bool(interregional_pre),
                "league_a": str(league_pair_ctx.get("league_a", "") or ""),
                "league_b": str(league_pair_ctx.get("league_b", "") or ""),
                "max_weight_mult": 1.0,
                "knee_mult": 1.0,
            }
            if bool(interregional_pre):
                w_mult = float(ss.get("ml_interregional_hybrid_max_weight_mult", 0.18) or 0.18)
                knee_mult = float(ss.get("ml_interregional_hybrid_knee_mult", 1.8) or 1.8)
                if _elite_vs_weak_mismatch(trace_interregional_strength_info):
                    w_mult *= 0.55
                    knee_mult *= 1.15
                hy_max_weight *= max(0.0, min(1.0, w_mult))
                hy_knee *= max(1.0, knee_mult)
                hy_damp["max_weight_mult"] = float(max(0.0, min(1.0, w_mult)))
                hy_damp["knee_mult"] = float(max(1.0, knee_mult))
            hy = hybrid_momentum_blend_fn(
                p_model=float(p_map),
                df_team_a=dfA_used,
                df_team_b=dfB_used,
                enabled=bool(ss.get("ml_hybrid_blend_enabled", True)),
                max_weight=float(hy_max_weight),
                knee_games=float(hy_knee),
                momentum_games=int(ss.get("ml_hybrid_momentum_games", 10) or 10),
                beta_prior=float(ss.get("ml_hybrid_beta", 2.0) or 2.0),
                # 1.0 = o comportamento de sempre. Ver a explicacao em
                # mlcore.fusion_core.hybrid_momentum_blend: a forma so
                # acrescenta onde DISCORDA do modelo, e e justo ali que a
                # formula de hoje a cala. O botao existe para medir isso.
                agree_damp=float(ss.get("ml_hybrid_agree_damp", 1.0)
                                 if ss.get("ml_hybrid_agree_damp") is not None else 1.0),
            )
            if isinstance(hy, dict):
                trace_hybrid_info = dict(hy)
                trace_hybrid_info["league_pair"] = dict(league_pair_ctx)
                trace_hybrid_info["interregional_damp"] = dict(hy_damp)
                p_hy = hy.get("p_blended")
                if hy.get("applied") and p_hy is not None and math.isfinite(float(p_hy)):
                    p_map = float(p_hy)
                    p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
        p_after = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        applied = bool(trace_hybrid_info.get("applied", False)) or (
            math.isfinite(float(p_before)) and math.isfinite(float(p_after)) and abs(float(p_after) - float(p_before)) > 1e-12
        )
        append_pipeline_step(
            steps=steps,
            stage="hybrid_blend",
            before=p_before,
            after=p_after,
            applied=bool(applied),
            enabled=bool(trace_hybrid_info.get("enabled", False)),
        )
    except Exception as exc:
        append_pipeline_step(
            steps=steps,
            stage="hybrid_blend",
            before=p_map,
            after=p_map,
            applied=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return p_map, p_series, trace_hybrid_info


def _stage_confidence_guard(
    *, p_map: float, p_series: float, bo_i: int, ss: Mapping[str, Any], ssw: Any, steps: list,
    trace_n_team_eff: Any, trace_n_players_eff: Any, interregional_pre: bool,
    trace_interregional_strength_info: Dict[str, Any], gov_enabled: bool, gov_budget_pp: float,
    clip_prob_fn: Callable, apply_confidence_governor_fn: Any, prob_win_series_fn: Callable,
) -> tuple:
    """Estágio confidence_guard. Retorna (p_map, p_series, trace_conf_guard_info)."""
    trace_conf_guard_info: Dict[str, Any] = {}
    try:
        p_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        cg_on = bool(ss.get("ml_conf_guard_enabled", True))
        cg_min = float(ss.get("ml_conf_guard_min_games", 12.0) or 12.0)
        cg_max = float(ss.get("ml_conf_guard_max_shrink", 0.35) or 0.35)
        conf_ref = _combined_confidence_reference(
            n_team_eff=trace_n_team_eff,
            n_players_eff=trace_n_players_eff,
            interregional=bool(interregional_pre),
            players_weight=float(ss.get("ml_interregional_conf_guard_players_eff_weight", 0.12) or 0.12),
            players_cap=float(ss.get("ml_interregional_conf_guard_players_eff_cap", 18.0) or 18.0),
        )
        n_ref = float(conf_ref.get("merged_ref", float("nan")))
        if (not math.isfinite(n_ref)) or n_ref < 0.0:
            n_ref = float("nan")
        cg_damp = {
            "interregional": bool(interregional_pre),
            "max_shrink_mult": 1.0,
            "team_ref": (float(conf_ref.get("team_ref")) if math.isfinite(_f(conf_ref.get("team_ref"))) else None),
            "players_ref": (float(conf_ref.get("players_ref")) if math.isfinite(_f(conf_ref.get("players_ref"))) else None),
            "merged_ref": (float(conf_ref.get("merged_ref")) if math.isfinite(_f(conf_ref.get("merged_ref"))) else None),
        }
        if bool(interregional_pre):
            cg_max_mult = float(ss.get("ml_interregional_conf_guard_max_shrink_mult", 0.35) or 0.35)
            if _elite_vs_weak_mismatch(trace_interregional_strength_info):
                cg_max_mult *= 0.65
            cg_max *= max(0.0, min(1.0, cg_max_mult))
            cg_damp["max_shrink_mult"] = float(max(0.0, min(1.0, cg_max_mult)))
        if cg_on and math.isfinite(float(p_map)) and cg_min > 0.0 and math.isfinite(n_ref):
            cov = max(0.0, min(1.0, float(n_ref) / float(cg_min)))
            shrink = max(0.0, min(1.0, float(cg_max) * (1.0 - cov)))
            p0 = float(p_map)
            p1 = float(clip_prob_fn(0.5 + (float(p0) - 0.5) * (1.0 - shrink)))
            gov_conf: Dict[str, Any] = {}
            if callable(apply_confidence_governor_fn):
                try:
                    gov_conf = apply_confidence_governor_fn(
                        stage="confidence_guard",
                        p_before=float(p0),
                        p_candidate=float(p1),
                        session_state=ssw,
                        enabled=bool(gov_enabled),
                        max_total_shrink_pp=float(gov_budget_pp),
                        reset=False,
                    )
                except Exception:
                    gov_conf = {}
            if isinstance(gov_conf, dict) and math.isfinite(_f(gov_conf.get("p_out"))):
                p1 = float(gov_conf.get("p_out"))
            trace_conf_guard_info = {
                "enabled": True,
                "n_ref": float(n_ref),
                "min_games": float(cg_min),
                "coverage": float(cov),
                "shrink": float(shrink),
                "p_in": float(p0),
                "p_out": float(p1),
                "applied": bool(abs(p1 - p0) > 1e-12),
                "interregional_damp": dict(cg_damp),
            }
            if isinstance(gov_conf, dict) and gov_conf:
                trace_conf_guard_info["governor"] = {
                    "enabled": bool(gov_conf.get("enabled", False)),
                    "limited": bool(gov_conf.get("limited", False)),
                    "requested_shrink_pp": gov_conf.get("requested_shrink_pp"),
                    "applied_shrink_pp": gov_conf.get("applied_shrink_pp"),
                    "used_shrink_pp": gov_conf.get("used_shrink_pp"),
                    "remaining_shrink_pp": gov_conf.get("remaining_shrink_pp"),
                }
            p_map = float(p1)
            p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
        else:
            trace_conf_guard_info = {
                "enabled": bool(cg_on),
                "n_ref": (float(n_ref) if math.isfinite(n_ref) else None),
                "min_games": float(cg_min),
                "coverage": None,
                "shrink": 0.0,
                "applied": False,
                "interregional_damp": dict(cg_damp),
            }
        p_after = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        applied = bool(trace_conf_guard_info.get("applied", False)) or (
            math.isfinite(float(p_before)) and math.isfinite(float(p_after)) and abs(float(p_after) - float(p_before)) > 1e-12
        )
        append_pipeline_step(
            steps=steps,
            stage="confidence_guard",
            before=p_before,
            after=p_after,
            applied=bool(applied),
            enabled=bool(trace_conf_guard_info.get("enabled", False)),
            governor_limited=bool(((trace_conf_guard_info or {}).get("governor") or {}).get("limited", False)),
            governor_applied_shrink_pp=(
                float(((trace_conf_guard_info or {}).get("governor") or {}).get("applied_shrink_pp"))
                if ((trace_conf_guard_info or {}).get("governor") or {}).get("applied_shrink_pp") is not None
                else None
            ),
        )
    except Exception as exc:
        append_pipeline_step(
            steps=steps,
            stage="confidence_guard",
            before=p_map,
            after=p_map,
            applied=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return p_map, p_series, trace_conf_guard_info


def _stage_patch_meta_guard(
    *, p_map: float, p_series: float, bo_i: int, ss: Mapping[str, Any], ssw: Any, steps: list,
    league_mode: str, filters: Any, dfA_used: Any, dfB_used: Any,
    csv_path: Any, fit_params_path: Any, gov_enabled: bool, gov_budget_pp: float,
    clip_prob_fn: Callable, apply_confidence_governor_fn: Any, prob_win_series_fn: Callable,
    file_sig_fn: Callable, patch_meta_guard_payload_fn: Callable, patch_changed_leagues_set_fn: Callable,
) -> tuple:
    """Estágio patch_meta_guard. Retorna (p_map, p_series, trace_patch)."""
    trace_patch: Dict[str, Any] = {}
    try:
        p_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        pg_on = bool(ss.get("ml_patch_guard_enabled", True))
        pg_warn = float(ss.get("ml_patch_guard_warn_shrink", 0.08) or 0.08)
        pg_refit = float(ss.get("ml_patch_guard_refit_shrink", 0.16) or 0.16)
        pg_focus = float(ss.get("ml_patch_guard_focus_bonus", 0.06) or 0.06)
        pg_applied = False
        if pg_on and math.isfinite(float(p_map)):
            lg_hint = _league_hint(league_mode=str(league_mode), filters=filters, dfA_used=dfA_used, dfB_used=dfB_used)
            csv_sig = file_sig_fn(str(csv_path)) if str(csv_path or "").strip() and Path(str(csv_path)).exists() else ("", 0, 0)
            fit_path = Path(str(fit_params_path)) if str(fit_params_path or "").strip() else Path("")
            fit_sig = file_sig_fn(str(fit_path)) if fit_path.exists() else ("", 0, 0)
            pg = patch_meta_guard_payload_fn(csv_sig, fit_sig)
            status = _norm_token((pg or {}).get("status", "warn"))
            changed = list((pg or {}).get("changed_leagues") or [])
            changed_set = patch_changed_leagues_set_fn(changed)
            shrink = 0.0
            if status == "refit":
                shrink = float(pg_refit)
            elif status in {"warn", "warning", "atencao"}:
                shrink = float(pg_warn)
            if str(lg_hint or "").strip().upper() in changed_set:
                shrink += float(pg_focus)
            shrink = max(0.0, min(0.95, float(shrink)))
            p0 = float(p_map)
            p1 = float(clip_prob_fn(0.5 + (float(p0) - 0.5) * (1.0 - shrink)))
            gov_patch: Dict[str, Any] = {}
            if callable(apply_confidence_governor_fn):
                try:
                    gov_patch = apply_confidence_governor_fn(
                        stage="patch_meta_guard",
                        p_before=float(p0),
                        p_candidate=float(p1),
                        session_state=ssw,
                        enabled=bool(gov_enabled),
                        max_total_shrink_pp=float(gov_budget_pp),
                        reset=False,
                    )
                except Exception:
                    gov_patch = {}
            if isinstance(gov_patch, dict) and math.isfinite(_f(gov_patch.get("p_out"))):
                p1 = float(gov_patch.get("p_out"))
            if abs(p1 - p0) > 1e-12:
                pg_applied = True
                p_map = float(p1)
                p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
            trace_patch = {
                "enabled": bool(pg_on),
                "applied": bool(pg_applied),
                "status": str(status),
                "league_hint": str(lg_hint or ""),
                "changed_leagues": list(changed_set),
                "age_days": (pg or {}).get("age_days"),
                "shrink": float(shrink),
                "p_in": float(p0),
                "p_out": float(p1),
            }
            if isinstance(gov_patch, dict) and gov_patch:
                trace_patch["governor"] = {
                    "enabled": bool(gov_patch.get("enabled", False)),
                    "limited": bool(gov_patch.get("limited", False)),
                    "requested_shrink_pp": gov_patch.get("requested_shrink_pp"),
                    "applied_shrink_pp": gov_patch.get("applied_shrink_pp"),
                    "used_shrink_pp": gov_patch.get("used_shrink_pp"),
                    "remaining_shrink_pp": gov_patch.get("remaining_shrink_pp"),
                }
        else:
            trace_patch = {
                "enabled": bool(pg_on),
                "applied": False,
                "status": "off",
                "league_hint": "",
                "changed_leagues": [],
                "age_days": None,
                "shrink": 0.0,
            }
        p_after = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        applied = bool(trace_patch.get("applied", False)) or (
            math.isfinite(float(p_before)) and math.isfinite(float(p_after)) and abs(float(p_after) - float(p_before)) > 1e-12
        )
        append_pipeline_step(
            steps=steps,
            stage="patch_meta_guard",
            before=p_before,
            after=p_after,
            applied=bool(applied),
            enabled=bool(trace_patch.get("enabled", False)),
            status=str(trace_patch.get("status", "")),
            league_hint=str(trace_patch.get("league_hint", "")),
            governor_limited=bool(((trace_patch or {}).get("governor") or {}).get("limited", False)),
            governor_applied_shrink_pp=(
                float(((trace_patch or {}).get("governor") or {}).get("applied_shrink_pp"))
                if ((trace_patch or {}).get("governor") or {}).get("applied_shrink_pp") is not None
                else None
            ),
        )
    except Exception as exc:
        append_pipeline_step(
            steps=steps,
            stage="patch_meta_guard",
            before=p_map,
            after=p_map,
            applied=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return p_map, p_series, trace_patch


def _stage_consistency_guard(
    *, p_map: float, p_series: float, diff_cur: float, delta_cur: float, bo_i: int,
    ss: Mapping[str, Any], ssw: Any, steps: list,
    season_label: str, teamA: str, teamB: str, ml_engine: str, year_opt: Any, split_opt: Any,
    playoffs_opt: Any, league_mode: str, fixed_league: Any, as_of_date: Any, artifact_path: str,
    lineup_mode_sig: str, lineupA_sig: Dict[str, Any], lineupB_sig: Dict[str, Any],
    ml_consistency_signature_fn: Callable, caption_fn: Any,
) -> tuple:
    """Estágio consistency_guard. Retorna
    (p_map, p_series, diff_cur, delta_cur, guard_triggered, guard_dpp)."""
    guard_triggered = False
    guard_dpp = float("nan")
    try:
        p_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        sig_ml = ml_consistency_signature_fn(
            season_label=str(season_label),
            teamA=str(teamA),
            teamB=str(teamB),
            bo=int(bo_i),
            ml_engine=str(ml_engine),
            calc_model_mode="Padrao",
            year_opt=year_opt,
            split_opt=split_opt,
            playoffs_opt=playoffs_opt,
            league_mode=str(league_mode),
            fixed_league=fixed_league,
            as_of_date=as_of_date,
            artifact_path=str(artifact_path),
            use_lineup_adjust=bool(ss.get("use_lineup_adjust", False)),
            lineup_mode=str(lineup_mode_sig),
            lineupA=lineupA_sig,
            lineupB=lineupB_sig,
        )
        tol_pp = float(ss.get("ml_consistency_tol_pp", 0.05) or 0.05)
        guard_on = bool(ss.get("ml_consistency_guard", True))
        prev = ss.get("_ml_consistency_prev")
        if isinstance(prev, dict) and int(prev.get("sig", -1)) == int(sig_ml):
            prev_p = float(prev.get("p_map", float("nan")))
            cur_p = float(p_map)
            if math.isfinite(prev_p) and math.isfinite(cur_p):
                dpp = abs(cur_p - prev_p) * 100.0
                guard_dpp = float(dpp)
                if float(dpp) > float(tol_pp):
                    if guard_on:
                        guard_triggered = True
                        p_map = float(prev_p)
                        p_series = float(prev.get("p_series", p_series))
                        diff_cur = float(prev.get("diff_used", diff_cur))
                        delta_cur = float(prev.get("delta_players", delta_cur))
                        if callable(caption_fn):
                            caption_fn(
                                f"Guardiao ML ativo: drift {float(dpp):.3f} pp com assinatura igual; mantendo odd anterior."
                            )
                    elif callable(caption_fn):
                        caption_fn(f"Drift ML detectado com assinatura igual: {float(dpp):.3f} pp (guardiao desligado).")
        if isinstance(ssw, MutableMapping):
            ssw["_ml_consistency_prev"] = {
                "sig": int(sig_ml),
                "p_map": float(p_map) if p_map is not None else float("nan"),
                "p_series": float(p_series) if p_series is not None else float("nan"),
                "diff_used": float(diff_cur) if diff_cur is not None else float("nan"),
                "delta_players": float(delta_cur) if delta_cur is not None else 0.0,
            }
        p_after = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        applied = bool(guard_triggered) or (
            math.isfinite(float(p_before)) and math.isfinite(float(p_after)) and abs(float(p_after) - float(p_before)) > 1e-12
        )
        append_pipeline_step(
            steps=steps,
            stage="consistency_guard",
            before=p_before,
            after=p_after,
            applied=bool(applied),
            enabled=bool(ss.get("ml_consistency_guard", True)),
            triggered=bool(guard_triggered),
            drift_pp=(float(guard_dpp) if math.isfinite(float(guard_dpp)) else None),
        )
    except Exception as exc:
        append_pipeline_step(
            steps=steps,
            stage="consistency_guard",
            before=p_map,
            after=p_map,
            applied=False,
            enabled=bool(ss.get("ml_consistency_guard", True)),
            error=f"{type(exc).__name__}: {exc}",
        )
    return p_map, p_series, diff_cur, delta_cur, guard_triggered, guard_dpp


def run_ml_pre_live_pipeline(
    *,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    season_label: str,
    teamA: str,
    teamB: str,
    ml_engine: str,
    year_opt: Any,
    split_opt: Any,
    playoffs_opt: Any,
    league_mode: str,
    fixed_league: Any,
    as_of_date: Any,
    artifact_path: str,
    lineup_mode: str,
    lineupA_sel: Dict[str, Any] | None,
    lineupB_sel: Dict[str, Any] | None,
    diff_used: Any,
    delta_players: Any,
    trace_n_team_eff: Any,
    trace_n_players_eff: Any,
    trace_patch_guard_info: Dict[str, Any] | None,
    trace_style_b_lite_live: Dict[str, Any] | None,
    ml_conf_score_snapshot: Any,
    lg_a_shadow: str,
    lg_b_shadow: str,
    csv_path: Any,
    fit_params_path: Any,
    filters: Any,
    dfA_used: Any,
    dfB_used: Any,
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None,
    trace_pipeline_steps: List[Dict[str, Any]] | None,
    season_form_blend_fn: Callable[..., Dict[str, Any]],
    current_season_override_fn: Callable[..., Dict[str, Any]],
    hybrid_momentum_blend_fn: Callable[..., Dict[str, Any]],
    clip_prob_fn: Callable[[float], float],
    ml_consistency_signature_fn: Callable[..., int],
    feature_flag_enabled_fn: Callable[[str, bool], bool],
    apply_series_bo_dplite_live_fn: Callable[..., Dict[str, Any]],
    enforce_series_economic_coherence_fn: Callable[..., Dict[str, Any]],
    series_gamma_for_league_fn: Callable[[str, int], Any],
    apply_series_gamma_fn: Callable[[float, float], float],
    patch_meta_guard_payload_fn: Callable[..., Dict[str, Any]],
    patch_changed_leagues_set_fn: Callable[[List[str]], Any],
    file_sig_fn: Callable[[str], Any],
    prob_win_series_fn: Callable[[float, int], float],
    fit_form_for_league_fn: Callable[[str], Any] | None = None,
    apply_confidence_governor_fn: Callable[..., Dict[str, Any]] | None = None,
    caption_fn: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    steps = trace_pipeline_steps if isinstance(trace_pipeline_steps, list) else []
    ss = session_state if isinstance(session_state, Mapping) else {}
    ssw = session_state if isinstance(session_state, MutableMapping) else None

    p_map = _f(p_map_used)
    p_series = _f(p_series_used)
    bo_i = int(bo)
    diff_cur = _f(diff_used)
    delta_cur = _f(delta_players, 0.0)

    trace_form_info: Dict[str, Any] = {}
    trace_hybrid_info: Dict[str, Any] = {}
    trace_conf_guard_info: Dict[str, Any] = {}
    trace_patch = dict(trace_patch_guard_info) if isinstance(trace_patch_guard_info, dict) else {}
    trace_series_gamma_info: Dict[str, Any] = {}
    trace_series_coherence_info: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "p_map": (float(p_map) if math.isfinite(float(p_map)) else None),
        "p_series_before": (float(p_series) if math.isfinite(float(p_series)) else None),
        "p_series_after": None,
    }

    league_pair_ctx = _league_pair_context(
        league_mode=str(league_mode),
        filters=filters,
        dfA_used=dfA_used,
        dfB_used=dfB_used,
        lg_a_shadow=lg_a_shadow,
        lg_b_shadow=lg_b_shadow,
    )
    interregional_pre = bool(league_pair_ctx.get("interregional", False))
    trace_interregional_strength_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # CAMADA DE JOGADORES POR DESEMPENHO
    #
    # POR QUE AQUI, e nao no ramo `use_lineup` que ja existe. Medido em
    # 11/09/2026 na regua honesta (8 dobras, n=8.494, pareado, contra a base
    # REAL servida):
    #   producao hoje (camada bloqueada) ....... 7,09
    #   + este estagio ......................... 7,20   +0,00388
    #                                            IC [+0,00270, +0,00506]
    #
    # O ramo `use_lineup` faria o deslocamento E a fusao por precisao. A fusao
    # sozinha CUSTA -0,00183, com o intervalo inteiro abaixo de zero — e por
    # isso que `use_lineup_adjust` estar desligado esta CERTO. Abrir aquele
    # portao traria o estagio ruim junto. Este estagio faz so o que paga.
    #
    # POSICAO NA FILA: logo apos a probabilidade calibrada e ANTES de
    # season_form_blend, que e exatamente onde a regua mediu.
    #
    # DESLIGAR: `ml_pscore_enabled = False`, ou apagar players_pscore.json.
    # Sem o arquivo o estagio e inerte, nao quebra nada.
    # ------------------------------------------------------------------
    trace_pscore_info: Dict[str, Any] = {}
    try:
        _ps_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        trace_pscore_info = _aplicar_pscore(
            p_map=float(p_map),
            artifact_path=str(artifact_path),
            blue_lineup=(lineupA_sel if isinstance(lineupA_sel, dict) else {}),
            red_lineup=(lineupB_sel if isinstance(lineupB_sel, dict) else {}),
            session_state=ss,
        )
        if bool(trace_pscore_info.get("applied", False)) and math.isfinite(
            _f(trace_pscore_info.get("p_out"))
        ):
            p_map = float(trace_pscore_info.get("p_out"))
            p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
        append_pipeline_step(
            steps=steps,
            stage="players_pscore",
            before=_ps_before,
            after=(float(p_map) if math.isfinite(float(p_map)) else float("nan")),
            applied=bool(trace_pscore_info.get("applied", False)),
            reason=str(trace_pscore_info.get("reason", "") or ""),
            delta=(float(trace_pscore_info["delta"])
                   if trace_pscore_info.get("delta") is not None else None),
            cobertura=(int(trace_pscore_info["cobertura"])
                       if trace_pscore_info.get("cobertura") is not None else None),
        )
    except Exception as exc:
        append_pipeline_step(
            steps=steps,
            stage="players_pscore",
            before=p_map,
            after=p_map,
            applied=False,
            error="{}: {}".format(type(exc).__name__, exc),
        )

    try:
        p_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        trace_interregional_strength_info = _interregional_strength_prior(
            p_map=float(p_map),
            bo=int(bo_i),
            league_pair_ctx=league_pair_ctx,
            n_team_eff=trace_n_team_eff,
            n_players_eff=trace_n_players_eff,
        )
        if bool(trace_interregional_strength_info.get("applied", False)) and math.isfinite(
            _f(trace_interregional_strength_info.get("p_out"))
        ):
            p_map = float(trace_interregional_strength_info.get("p_out"))
            p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
        p_after = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        append_pipeline_step(
            steps=steps,
            stage="interregional_strength_prior",
            before=p_before,
            after=p_after,
            applied=bool(trace_interregional_strength_info.get("applied", False)),
            reason=str(trace_interregional_strength_info.get("reason", "") or ""),
            stronger_side=str(trace_interregional_strength_info.get("stronger_side", "") or ""),
            strength_gap=(
                float(trace_interregional_strength_info.get("strength_gap"))
                if trace_interregional_strength_info.get("strength_gap") is not None
                else None
            ),
        )
    except Exception as exc:
        append_pipeline_step(
            steps=steps,
            stage="interregional_strength_prior",
            before=p_map,
            after=p_map,
            applied=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    guard_triggered = False
    guard_dpp = float("nan")
    gov_enabled = bool(ss.get("ml_confidence_governor_enabled", True))
    gov_budget_pp = float(
        ss.get(
            "_ml_confidence_governor_budget_runtime_pp",
            ss.get("ml_confidence_governor_max_total_shrink_pp", 8.0),
        )
        or 8.0
    )

    lineup_mode_sig = str(lineup_mode or "")
    lineupA_sig = dict(lineupA_sel) if isinstance(lineupA_sel, dict) else {}
    lineupB_sig = dict(lineupB_sel) if isinstance(lineupB_sel, dict) else {}

    p_map, p_series, trace_form_info = _stage_season_form_blend(
        p_map=p_map, p_series=p_series, bo_i=bo_i, ss=ss, ssw=ssw, steps=steps,
        league_mode=league_mode, filters=filters, dfA_used=dfA_used, dfB_used=dfB_used,
        interregional_pre=interregional_pre, league_pair_ctx=league_pair_ctx,
        trace_interregional_strength_info=trace_interregional_strength_info,
        season_form_blend_fn=season_form_blend_fn,
        current_season_override_fn=current_season_override_fn,
        prob_win_series_fn=prob_win_series_fn, fit_form_for_league_fn=fit_form_for_league_fn,
    )

    p_map, p_series, trace_hybrid_info = _stage_hybrid_blend(
        p_map=p_map, p_series=p_series, bo_i=bo_i, ss=ss, steps=steps,
        dfA_used=dfA_used, dfB_used=dfB_used, interregional_pre=interregional_pre,
        league_pair_ctx=league_pair_ctx,
        trace_interregional_strength_info=trace_interregional_strength_info,
        hybrid_momentum_blend_fn=hybrid_momentum_blend_fn, prob_win_series_fn=prob_win_series_fn,
    )

    # NOTA POR JOGADOR do LoL (28/10/2026, aprovada por ele): logo DEPOIS do hibrido,
    # o ponto da regua em que foi provada. So liga nacional. Ver
    # app_core/nota_jogadores_lol_core.py. No Dota o modulo nao existe: nada acontece
    # (la a nota por jogador entra no pre-engine, ver ml_pre_engine_core.py).
    p_map, p_series = _stage_nota_jogadores_lol(
        p_map=p_map, p_series=p_series, bo_i=bo_i, ssw=ssw, steps=steps, teamA=teamA, teamB=teamB,
        league_pair_ctx=league_pair_ctx, prob_win_series_fn=prob_win_series_fn,
    )

    p_map, p_series, trace_conf_guard_info = _stage_confidence_guard(
        p_map=p_map, p_series=p_series, bo_i=bo_i, ss=ss, ssw=ssw, steps=steps,
        trace_n_team_eff=trace_n_team_eff, trace_n_players_eff=trace_n_players_eff,
        interregional_pre=interregional_pre,
        trace_interregional_strength_info=trace_interregional_strength_info,
        gov_enabled=gov_enabled, gov_budget_pp=gov_budget_pp,
        clip_prob_fn=clip_prob_fn, apply_confidence_governor_fn=apply_confidence_governor_fn,
        prob_win_series_fn=prob_win_series_fn,
    )

    p_map, p_series, trace_patch = _stage_patch_meta_guard(
        p_map=p_map, p_series=p_series, bo_i=bo_i, ss=ss, ssw=ssw, steps=steps,
        league_mode=league_mode, filters=filters, dfA_used=dfA_used, dfB_used=dfB_used,
        csv_path=csv_path, fit_params_path=fit_params_path,
        gov_enabled=gov_enabled, gov_budget_pp=gov_budget_pp,
        clip_prob_fn=clip_prob_fn, apply_confidence_governor_fn=apply_confidence_governor_fn,
        prob_win_series_fn=prob_win_series_fn, file_sig_fn=file_sig_fn,
        patch_meta_guard_payload_fn=patch_meta_guard_payload_fn,
        patch_changed_leagues_set_fn=patch_changed_leagues_set_fn,
    )

    p_map, p_series, diff_cur, delta_cur, guard_triggered, guard_dpp = _stage_consistency_guard(
        p_map=p_map, p_series=p_series, diff_cur=diff_cur, delta_cur=delta_cur, bo_i=bo_i,
        ss=ss, ssw=ssw, steps=steps,
        season_label=season_label, teamA=teamA, teamB=teamB, ml_engine=ml_engine,
        year_opt=year_opt, split_opt=split_opt, playoffs_opt=playoffs_opt,
        league_mode=league_mode, fixed_league=fixed_league, as_of_date=as_of_date,
        artifact_path=artifact_path, lineup_mode_sig=lineup_mode_sig,
        lineupA_sig=lineupA_sig, lineupB_sig=lineupB_sig,
        ml_consistency_signature_fn=ml_consistency_signature_fn, caption_fn=caption_fn,
    )

    p_series, ff_series_bo, trace_series_bo_dplite_live = _stage_series_bo_dplite_live(
        p_map=p_map, p_series=p_series, bo_i=bo_i, ss=ss, ssw=ssw, steps=steps,
        league_mode=league_mode, filters=filters, dfA_used=dfA_used, dfB_used=dfB_used,
        lg_a_shadow=lg_a_shadow, lg_b_shadow=lg_b_shadow, league_pair_ctx=league_pair_ctx,
        trace_patch=trace_patch, trace_style_b_lite_live=trace_style_b_lite_live,
        ml_conf_score_snapshot=ml_conf_score_snapshot,
        trace_n_team_eff=trace_n_team_eff, trace_n_players_eff=trace_n_players_eff,
        playoffs_opt=playoffs_opt,
        feature_flag_enabled_fn=feature_flag_enabled_fn,
        apply_series_bo_dplite_live_fn=apply_series_bo_dplite_live_fn,
    )

    p_series, trace_series_gamma_info = _stage_series_gamma(
        p_map=p_map, p_series=p_series, bo_i=bo_i, ss=ss, steps=steps,
        league_mode=league_mode, filters=filters, dfA_used=dfA_used, dfB_used=dfB_used,
        series_gamma_for_league_fn=series_gamma_for_league_fn,
        apply_series_gamma_fn=apply_series_gamma_fn,
        prob_win_series_fn=prob_win_series_fn,
    )

    p_series, trace_series_coherence_info = _stage_series_economic_coherence(
        p_map=p_map, p_series=p_series, bo_i=bo_i, steps=steps,
        enforce_series_economic_coherence_fn=enforce_series_economic_coherence_fn,
        default_trace=trace_series_coherence_info,
    )

    return {
        "p_map_used": float(p_map) if math.isfinite(float(p_map)) else float("nan"),
        "p_series_used": float(p_series) if math.isfinite(float(p_series)) else float("nan"),
        "diff_used": float(diff_cur) if math.isfinite(float(diff_cur)) else float("nan"),
        "delta_players": float(delta_cur) if math.isfinite(float(delta_cur)) else 0.0,
        "trace_form_info": trace_form_info,
        "trace_hybrid_info": trace_hybrid_info,
        "trace_conf_guard_info": trace_conf_guard_info,
        "trace_patch_guard_info": trace_patch,
        "trace_series_gamma_info": trace_series_gamma_info,
        "trace_series_coherence_info": trace_series_coherence_info,
        "_guard_triggered": bool(guard_triggered),
        "_guard_dpp": float(guard_dpp) if math.isfinite(float(guard_dpp)) else float("nan"),
        "_ff_ml_series_bo_dplite_live": bool(ff_series_bo),
        "trace_series_bo_dplite_live": trace_series_bo_dplite_live,
        "_lineup_mode_sig": str(lineup_mode_sig),
        "_lineupA_sig": lineupA_sig,
        "_lineupB_sig": lineupB_sig,
        "trace_pipeline_steps": steps,
    }


