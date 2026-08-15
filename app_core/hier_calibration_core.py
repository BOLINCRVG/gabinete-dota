from __future__ import annotations

import math
from typing import Any, Dict

from app_core.interleague_baseline_core import infer_league_class_from_node as core_infer_league_class_from_node
from app_core.league_transfer_features_core import build_interleague_transfer_adjustment
from app_core.prob_math_core import clip_prob as _clip_prob
from app_core.prob_math_core import safe_float as _f
from app_core.prob_math_core import safe_logit as _logit
from app_core.prob_math_core import safe_sigmoid as _sigmoid
from mlcore.league_context import league_lookup_keys, league_macro_key
from mlcore.tiers import canonical_league, tier_for_league


def _norm_playoffs(v: Any) -> bool:
    s = str(v or "").strip().lower()
    return s in {"1", "true", "playoffs", "sim", "yes"}


def _canon_league(league: Any) -> str:
    return str(canonical_league(str(league or "")))


def _league_keys(league: Any) -> list[str]:
    return list(league_lookup_keys(league))


def _lookup_first_dict(table: Dict[str, Any], keys: list[str]) -> tuple[Dict[str, Any], str]:
    if not isinstance(table, dict):
        return {}, ""
    for key in keys:
        if key and isinstance(table.get(key), dict):
            return dict(table.get(key) or {}), str(key)
    return {}, ""


def _lookup_first_str(table: Dict[str, Any], keys: list[str]) -> tuple[str, str]:
    if not isinstance(table, dict):
        return "", ""
    for key in keys:
        val = str(table.get(key) or "").strip().lower() if key else ""
        if val:
            return val, str(key)
    return "", ""


def _append_unique(out: list[str], seen: set[str], key: str) -> None:
    if not key or key in seen:
        return
    out.append(str(key))
    seen.add(str(key))


def _pair_candidate_keys(
    prefix: str,
    left_keys: list[str],
    right_keys: list[str],
    *,
    bo_i: int,
    is_po: bool,
    allow_same_directed: bool = False,
    include_undirected: bool = True,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    left = [str(v) for v in left_keys if str(v or "").strip()]
    right = [str(v) for v in right_keys if str(v or "").strip()]
    for la in left:
        for lb in right:
            if (not allow_same_directed) and la == lb:
                continue
            _append_unique(out, seen, f"{prefix}:{la}->{lb}|bo:{int(bo_i)}|po:{1 if bool(is_po) else 0}|inter:1")
            _append_unique(out, seen, f"{prefix}:{la}->{lb}|bo:{int(bo_i)}|inter:1")
            _append_unique(out, seen, f"{prefix}:{la}->{lb}|inter:1")
            _append_unique(out, seen, f"{prefix}:{la}->{lb}")
    if not include_undirected:
        return out
    und_seen: set[str] = set()
    for la in left:
        for lb in right:
            if la == lb:
                continue
            und = "->".join(sorted([la, lb]))
            if und in und_seen:
                continue
            und_seen.add(und)
            _append_unique(out, seen, f"{prefix}:{und}|bo:{int(bo_i)}|po:{1 if bool(is_po) else 0}|inter:1")
            _append_unique(out, seen, f"{prefix}:{und}|bo:{int(bo_i)}|inter:1")
            _append_unique(out, seen, f"{prefix}:{und}|inter:1")
            _append_unique(out, seen, f"{prefix}:{und}")
    return out


def _macro_key(league: str) -> str:
    return str(league_macro_key(_canon_league(league), unknown="OTHER"))


def _norm_patch_status(v: Any) -> str:
    s = str(v or "").strip().upper()
    if not s:
        return "ok"
    if "REFIT" in s:
        return "refit"
    if "WARN" in s:
        return "warn"
    return "ok"


def _infer_league_class_from_node(league: str, node: Dict[str, Any]) -> str:
    return str(core_infer_league_class_from_node(league, node))


def _league_class_from_baseline(bp: Dict[str, Any], by_league: Dict[str, Any], league: str) -> str:
    lg = _canon_league(league)
    if not lg:
        return "minor"
    cls_tbl = bp.get("league_class") if isinstance(bp.get("league_class"), dict) else {}
    cls_v, _ = _lookup_first_str(cls_tbl, _league_keys(lg))
    if cls_v in {"major", "minor"}:
        return cls_v
    node, _ = _lookup_first_dict(by_league, _league_keys(lg))
    return _infer_league_class_from_node(lg, node)


def _node_n(node: Dict[str, Any]) -> float:
    """Amostra efetiva de um nó de calibração (wf_n/lv_n/n_total/n, 1º > 0).

    Extraído de apply_hier_calibration_live (era nested; 21/07, refactor guiado
    por hiercal_golden). Puro."""
    if not isinstance(node, dict):
        return 0.0
    for k in ("wf_n", "lv_n", "n_total", "n"):
        v = _f(node.get(k))
        if math.isfinite(v) and v > 0:
            return float(v)
    return 0.0


def _node_se(node: Dict[str, Any], sigma0: float) -> float:
    """Erro-padrão do viés do nó: score_se direto, ou de score_var, ou
    sigma0/sqrt(n). Puro (extraído do nested)."""
    if isinstance(node, dict):
        se_v = _f(node.get("score_se"))
        if math.isfinite(se_v) and se_v > 0.0:
            return float(se_v)
        var_v = _f(node.get("score_var"))
        if math.isfinite(var_v) and var_v > 0.0:
            return float(math.sqrt(max(1e-9, float(var_v))))
    n_v = _node_n(node)
    if n_v <= 0.0:
        return float(max(0.8, float(sigma0)))
    return float(max(1e-6, float(sigma0) / math.sqrt(max(1.0, float(n_v)))))


def _sh(n: float, k: float) -> float:
    """Fator de shrink por amostra: n/(n+k). Puro (extraído do nested)."""
    nn = max(0.0, float(n))
    return float(nn / (nn + max(1.0, float(k))))


def _node_base_stats(node: Dict[str, Any], *, sigma0: float, fdr_q_alpha: float) -> Dict[str, float]:
    """Estatísticas-base de um nó: viés, amostra, erro-padrão, z/p/q e rejeição
    FDR. É a parte comum entre o nó hierárquico (_node_stats) e o par
    interregional (_pair_stats), que antes duplicavam este cálculo.

    Consolidado 21/07 (passo 1 do mapa, tirar duplicação). Puro."""
    _bias = _f(node.get("score_bias"), 0.0) if isinstance(node, dict) else 0.0
    _n = _node_n(node)
    _se = _node_se(node, sigma0)
    _var = float(max(1e-9, _se * _se))
    _z_in = _f(node.get("score_z")) if isinstance(node, dict) else float("nan")
    _z = float(_z_in if math.isfinite(_z_in) else (_bias / max(1e-9, _se)))
    _p_in = _f(node.get("score_p")) if isinstance(node, dict) else float("nan")
    _p = float(_p_in if math.isfinite(_p_in) else math.erfc(abs(_z) / math.sqrt(2.0)))
    _q_in = _f(node.get("score_q")) if isinstance(node, dict) else float("nan")
    _q = float(_q_in if math.isfinite(_q_in) else _p)
    _fdr_reject = bool(node.get("fdr_reject", _q <= fdr_q_alpha)) if isinstance(node, dict) else bool(_q <= fdr_q_alpha)
    return {
        "bias": float(_bias),
        "n": float(max(0.0, _n)),
        "se": float(max(1e-6, _se)),
        "var": float(max(1e-9, _var)),
        "z": float(_z),
        "p": float(max(0.0, min(1.0, _p))),
        "q": float(max(0.0, min(1.0, _q))),
        "fdr_reject": bool(_fdr_reject),
    }


def _node_stats(
    node: Dict[str, Any],
    *,
    k_shrink: float,
    sigma0: float,
    z_gate: float,
    fdr_q_alpha: float,
) -> Dict[str, float]:
    """Estatísticas de um nó: viés ajustado por significância (z/q gates) e peso
    por inverso-da-variância × shrink. O CORE do peso hierárquico.

    Base comum em _node_base_stats; aqui adiciona os multiplicadores de
    significância e o peso. Puro; hiercal_golden prova a saída idêntica."""
    base = _node_base_stats(node, sigma0=sigma0, fdr_q_alpha=fdr_q_alpha)
    _bias, _n, _var = base["bias"], base["n"], base["var"]
    _z, _q, _fdr_reject = base["z"], base["q"], base["fdr_reject"]
    _z_mult = float(max(0.0, min(1.0, abs(_z) / max(1e-9, z_gate))))
    _q_mult = 1.0
    if _q > fdr_q_alpha:
        _q_mult = float(max(0.05, min(1.0, fdr_q_alpha / max(1e-9, _q))))
    if not _fdr_reject:
        _q_mult = float(min(_q_mult, 0.75))
    _sig_mult = float(max(0.0, min(1.0, _z_mult * _q_mult)))
    _bias_adj = float(_bias * _sig_mult)
    _ev = float(_sh(_n, k_shrink))
    _w = float((1.0 / max(1e-9, _var)) * _ev * max(0.05, _sig_mult))
    return {
        **base,
        "bias_adj": float(_bias_adj),
        "z_mult": float(_z_mult),
        "q_mult": float(_q_mult),
        "sig_mult": float(_sig_mult),
        "ev": float(_ev),
        "weight": float(_w),
    }


def apply_hier_calibration_live(
    *,
    p_map_in: Any,
    league: Any,
    league_a: Any = None,
    league_b: Any = None,
    team_a: Any = None,
    team_b: Any = None,
    team_aliases_a: list[str] | None = None,
    team_aliases_b: list[str] | None = None,
    bo: Any = 1,
    playoffs: Any = None,
    interregional: bool = False,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    patch_status: Any = None,
    baseline_payload: Dict[str, Any] | None = None,
    league_transfer_features: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "method": "hier_calibration_v3",
        "scope_used": None,
        "p_before": None,
        "p_after": None,
        "diff_pp": None,
        "weight": None,
        "bias_raw": None,
        "bias_adj": None,
        "n_team_eff": None,
        "n_players_eff": None,
        "bo": None,
        "playoffs": None,
        "interregional": bool(interregional),
        "league_a": None,
        "league_b": None,
        "patch_status": str(patch_status or ""),
        "interleague": {},
        "runtime_fine": {},
        "hier": {},
    }
    p0 = _f(p_map_in)
    if not math.isfinite(p0):
        out["reason"] = "invalid_p_in"
        return out
    out["p_before"] = float(p0)

    bp = baseline_payload if isinstance(baseline_payload, dict) else {}
    by_league = bp.get("by_league") if isinstance(bp.get("by_league"), dict) else {}
    global_node = bp.get("global") if isinstance(bp.get("global"), dict) else {}
    lg = _canon_league(league)
    lga = _canon_league(league_a)
    lgb = _canon_league(league_b)
    if not lga and lg and lg != "INTER":
        lga = lg
    lg_scope = lga or (lg if lg != "INTER" else "")
    lg_scope_keys = _league_keys(lg_scope) if lg_scope else []
    lg_node, lg_scope_lookup = _lookup_first_dict(by_league, lg_scope_keys)
    bo_i = 1
    try:
        bo_i = int(float(bo))
    except Exception:
        bo_i = 1
    bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
    is_po = _norm_playoffs(playoffs)
    out["bo"] = int(bo_i)
    out["playoffs"] = bool(is_po)
    patch_tag = _norm_patch_status(patch_status)
    pair_strength_band_a = ""
    pair_strength_band_b = ""
    strong_cross_strength = False
    very_strong_cross_strength = False

    if not bp:
        out["reason"] = "missing_baseline"
        return out

    seg_table = bp.get("by_segment") if isinstance(bp.get("by_segment"), dict) else {}
    if not seg_table:
        # Optional alternative format:
        # {"by_league_segment": {"LCK": {"bo:5|po:1|inter:0": {...}}}}
        _bls = bp.get("by_league_segment") if isinstance(bp.get("by_league_segment"), dict) else {}
        if isinstance(_bls.get(lg), dict):
            for _sk, _sv in (_bls.get(lg) or {}).items():
                seg_table[f"league:{lg}|{_sk!s}"] = _sv

    def _seg_key(prefix: str) -> str:
        return f"{prefix}|bo:{int(bo_i)}|po:{1 if bool(is_po) else 0}|inter:{1 if bool(interregional) else 0}"

    cls_a = _league_class_from_baseline(bp, by_league, lga)
    cls_b = _league_class_from_baseline(bp, by_league, lgb)
    seg_class = ""
    if bool(interregional):
        if cls_a == "major" and cls_b == "major":
            seg_class = "inter_major_major"
        elif cls_a == "minor" and cls_b == "minor":
            seg_class = "inter_minor_minor"
        else:
            seg_class = "inter_mixed"
    else:
        seg_class = "same_major" if cls_a == "major" else "same_minor"

    seg_candidates: list[str] = []
    seg_seen: set[str] = set()
    for lg_key in lg_scope_keys:
        _append_unique(seg_candidates, seg_seen, _seg_key(f"league:{lg_key}|patch:{patch_tag}"))
    _append_unique(seg_candidates, seg_seen, (_seg_key(f"class:{seg_class}|patch:{patch_tag}") if seg_class else ""))
    _append_unique(seg_candidates, seg_seen, _seg_key(f"global|patch:{patch_tag}"))
    for lg_key in lg_scope_keys:
        _append_unique(seg_candidates, seg_seen, _seg_key(f"league:{lg_key}"))
    _append_unique(seg_candidates, seg_seen, (_seg_key(f"class:{seg_class}") if seg_class else ""))
    _append_unique(seg_candidates, seg_seen, _seg_key("global"))
    for lg_key in lg_scope_keys:
        _append_unique(seg_candidates, seg_seen, f"league:{lg_key}|bo:{int(bo_i)}")
        _append_unique(seg_candidates, seg_seen, f"league:{lg_key}|inter:{1 if bool(interregional) else 0}")
        _append_unique(seg_candidates, seg_seen, f"league:{lg_key}|po:{1 if bool(is_po) else 0}")
    _append_unique(seg_candidates, seg_seen, f"global|bo:{int(bo_i)}")
    seg_node = {}
    seg_scope = ""
    for kk in seg_candidates:
        if kk and isinstance(seg_table.get(kk), dict):
            seg_node = dict(seg_table.get(kk) or {})
            seg_scope = str(kk)
            break

    th = bp.get("thresholds") if isinstance(bp.get("thresholds"), dict) else {}
    seg_shrink_k = max(30.0, min(800.0, _f(th.get("seg_shrink_k"), 160.0)))
    league_shrink_k = max(30.0, min(1200.0, _f(th.get("league_shrink_k"), 260.0)))
    global_shrink_k = max(30.0, min(2000.0, _f(th.get("global_shrink_k"), 400.0)))
    hier_decay = max(0.35, min(1.0, _f(th.get("hier_decay"), 0.78)))
    same_sigma0 = max(0.8, min(10.0, _f(th.get("same_sigma0"), _f(th.get("inter_sigma0"), 4.0))))
    same_z_gate = max(0.80, min(4.00, _f(th.get("same_z_gate"), _f(th.get("inter_z_gate"), 1.96))))
    same_fdr_q_alpha = max(0.01, min(0.30, _f(th.get("same_fdr_q_alpha"), _f(th.get("inter_fdr_q_alpha"), 0.10))))

    def _node_stats_same(node: Dict[str, Any], *, k_shrink: float) -> Dict[str, float]:
        # thin closure: fixa os thresholds same_* do escopo atual no helper puro.
        return _node_stats(node, k_shrink=k_shrink, sigma0=same_sigma0,
                           z_gate=same_z_gate, fdr_q_alpha=same_fdr_q_alpha)

    seg_st = _node_stats_same(seg_node, k_shrink=seg_shrink_k) if isinstance(seg_node, dict) and seg_node else {
        "bias": 0.0, "bias_adj": 0.0, "n": 0.0, "se": same_sigma0, "var": max(1e-9, same_sigma0 * same_sigma0),
        "z": 0.0, "p": 1.0, "q": 1.0, "fdr_reject": False, "z_mult": 0.0, "q_mult": 1.0, "sig_mult": 0.0, "ev": 0.0, "weight": 0.0
    }
    lg_st = _node_stats_same(lg_node, k_shrink=league_shrink_k) if isinstance(lg_node, dict) and lg_node else {
        "bias": 0.0, "bias_adj": 0.0, "n": 0.0, "se": same_sigma0, "var": max(1e-9, same_sigma0 * same_sigma0),
        "z": 0.0, "p": 1.0, "q": 1.0, "fdr_reject": False, "z_mult": 0.0, "q_mult": 1.0, "sig_mult": 0.0, "ev": 0.0, "weight": 0.0
    }
    gl_st = _node_stats_same(global_node, k_shrink=global_shrink_k) if isinstance(global_node, dict) and global_node else {
        "bias": 0.0, "bias_adj": 0.0, "n": 0.0, "se": same_sigma0, "var": max(1e-9, same_sigma0 * same_sigma0),
        "z": 0.0, "p": 1.0, "q": 1.0, "fdr_reject": False, "z_mult": 0.0, "q_mult": 1.0, "sig_mult": 0.0, "ev": 0.0, "weight": 0.0
    }

    seg_bias = float(seg_st["bias_adj"])
    lg_bias = float(lg_st["bias_adj"])
    gl_bias = float(gl_st["bias_adj"])
    seg_n = float(seg_st["n"])
    lg_n = float(lg_st["n"])
    gl_n = float(gl_st["n"])

    # Precision/evidence based weights (global never disappears to avoid overfit local segments).
    w_seg = float(seg_st["weight"])
    w_lg = float(lg_st["weight"]) * 0.85
    w_gl = float(gl_st["weight"]) * 0.55

    # Hierarchical decay if upper levels are weak/unavailable.
    if w_seg <= 0.0:
        w_lg *= float(hier_decay)
    if w_seg <= 0.0 and w_lg <= 0.0:
        w_gl *= float(hier_decay)

    denom = float(max(1e-9, w_seg + w_lg + w_gl))
    bias_raw = float(
        ((float(seg_bias) * w_seg) if math.isfinite(seg_bias) else 0.0)
        + ((float(lg_bias) * w_lg) if math.isfinite(lg_bias) else 0.0)
        + ((float(gl_bias) * w_gl) if math.isfinite(gl_bias) else 0.0)
    ) / denom
    scope_used = seg_scope or (f"league:{lg_scope_lookup or lg_scope}" if lg_node and (lg_scope_lookup or lg_scope) else "global")
    quality_band = str((seg_node.get("quality_band") if isinstance(seg_node, dict) else "") or (lg_node.get("quality_band") if isinstance(lg_node, dict) else "") or "")
    wf_n = float(max(seg_n, lg_n, gl_n))

    out["scope_used"] = scope_used
    out["bias_raw"] = (float(bias_raw) if math.isfinite(bias_raw) else None)
    z0 = _logit(p0)
    out["hier"] = {
        "segment_scope": (str(seg_scope) if seg_scope else None),
        "segment_bias": (float(seg_st["bias"]) if math.isfinite(float(seg_st["bias"])) else None),
        "segment_bias_adj": (float(seg_bias) if math.isfinite(seg_bias) else None),
        "segment_n": float(seg_n),
        "league_bias": (float(lg_st["bias"]) if math.isfinite(float(lg_st["bias"])) else None),
        "league_bias_adj": (float(lg_bias) if math.isfinite(lg_bias) else None),
        "league_n": float(lg_n),
        "global_bias": (float(gl_st["bias"]) if math.isfinite(float(gl_st["bias"])) else None),
        "global_bias_adj": (float(gl_bias) if math.isfinite(gl_bias) else None),
        "global_n": float(gl_n),
        "weights": {
            "segment": float(w_seg),
            "league": float(w_lg),
            "global": float(w_gl),
        },
        "stats": {
            "segment_z": float(seg_st["z"]),
            "segment_q": float(seg_st["q"]),
            "segment_sig_mult": float(seg_st["sig_mult"]),
            "segment_fdr_reject": bool(seg_st["fdr_reject"]),
            "league_z": float(lg_st["z"]),
            "league_q": float(lg_st["q"]),
            "league_sig_mult": float(lg_st["sig_mult"]),
            "league_fdr_reject": bool(lg_st["fdr_reject"]),
            "global_z": float(gl_st["z"]),
            "global_q": float(gl_st["q"]),
            "global_sig_mult": float(gl_st["sig_mult"]),
            "global_fdr_reject": bool(gl_st["fdr_reject"]),
        },
        "params": {
            "seg_shrink_k": float(seg_shrink_k),
            "league_shrink_k": float(league_shrink_k),
            "global_shrink_k": float(global_shrink_k),
            "hier_decay": float(hier_decay),
            "same_sigma0": float(same_sigma0),
            "same_z_gate": float(same_z_gate),
            "same_fdr_q_alpha": float(same_fdr_q_alpha),
        },
        "segment_class": str(seg_class),
        "league_class_a": str(cls_a),
        "league_class_b": str(cls_b),
        "patch_tag": str(patch_tag),
    }
    out["league_a"] = (str(lga) if lga else None)
    out["league_b"] = (str(lgb) if lgb else None)

    # Inter-league transfer calibration: uncertainty-aware shrink + optional pair delta.
    inter_dir = 0.0
    inter_shrink = 0.0
    inter_pair_scope = ""
    inter_band = "na"
    inter_ev = float("nan")
    if bool(interregional):
        macro_a = _macro_key(lga)
        macro_b = _macro_key(lgb)
        tier_a = str(tier_for_league(lga) or "OTHER").upper()
        tier_b = str(tier_for_league(lgb) or "OTHER").upper()
        by_lg = bp.get("by_league") if isinstance(bp.get("by_league"), dict) else {}
        node_a, _ = _lookup_first_dict(by_lg, _league_keys(lga))
        node_b, _ = _lookup_first_dict(by_lg, _league_keys(lgb))
        n_a = _node_n(node_a)
        n_b = _node_n(node_b)
        inter_n = float(max(0.0, min(n_a, n_b)))
        pair_tbl = bp.get("interleague_pairs") if isinstance(bp.get("interleague_pairs"), dict) else {}
        pair_node: Dict[str, Any] = {}

        def _pick_pair_node(keys: list[str]) -> tuple[Dict[str, Any], str]:
            for _kk in keys:
                if _kk and isinstance(pair_tbl.get(_kk), dict):
                    return dict(pair_tbl.get(_kk) or {}), str(_kk)
            return {}, ""

        lg_pair_keys: list[str] = []
        if lga and lgb and (lga != lgb):
            lg_pair_keys = _pair_candidate_keys(
                "league_pair",
                _league_keys(lga),
                _league_keys(lgb),
                bo_i=int(bo_i),
                is_po=bool(is_po),
            )
        macro_pair_keys = _pair_candidate_keys(
            "inter",
            [macro_a],
            [macro_b],
            bo_i=int(bo_i),
            is_po=bool(is_po),
            allow_same_directed=True,
            include_undirected=bool(macro_a and macro_b and macro_a != "OTHER" and macro_b != "OTHER" and macro_a != macro_b),
        )
        tier_pair_keys = _pair_candidate_keys(
            "tier_pair",
            [tier_a],
            [tier_b],
            bo_i=int(bo_i),
            is_po=bool(is_po),
            allow_same_directed=True,
            include_undirected=bool(tier_a and tier_b and tier_a != "OTHER" and tier_b != "OTHER" and tier_a != tier_b),
        )

        inter_sigma0 = max(0.8, min(10.0, _f(th.get("inter_sigma0"), 4.0)))
        mix_k = max(40.0, min(2400.0, _f(th.get("inter_pair_mix_k"), 260.0)))
        inter_shrink_k = max(60.0, min(2200.0, _f(th.get("inter_shrink_k"), 320.0)))
        z_gate = max(0.80, min(4.00, _f(th.get("inter_z_gate"), 1.96)))
        z_full = max(z_gate, min(5.00, _f(th.get("inter_z_full"), 2.58)))
        fdr_q_alpha = max(0.01, min(0.30, _f(th.get("inter_fdr_q_alpha"), 0.10)))

        def _pair_stats(node: Dict[str, Any]) -> Dict[str, Any]:
            # mesma base do nó hierárquico, com o sigma0 interregional.
            return _node_base_stats(node, sigma0=inter_sigma0, fdr_q_alpha=fdr_q_alpha)

        pair_node_lg, pair_scope_lg = _pick_pair_node(lg_pair_keys)
        pair_node_tier, pair_scope_tier = _pick_pair_node(tier_pair_keys)
        pair_node_macro, pair_scope_macro = _pick_pair_node(macro_pair_keys)
        st_lg = _pair_stats(pair_node_lg) if pair_node_lg else {}
        st_tp = _pair_stats(pair_node_tier) if pair_node_tier else {}
        st_mc = _pair_stats(pair_node_macro) if pair_node_macro else {}
        blend_info: Dict[str, Any] = {}
        def _pair_rel(node: Dict[str, Any]) -> float:
            return float(max(0.0, min(1.0, _f((node or {}).get("pair_reliability"), 0.0))))

        def _blend_pair_nodes(
            *,
            node_a: Dict[str, Any],
            scope_a: str,
            st_a: Dict[str, Any],
            node_b: Dict[str, Any],
            scope_b: str,
            st_b: Dict[str, Any],
            shrink_k_a: float,
            shrink_k_b: float,
            source: str,
            source_a_label: str,
            source_b_label: str,
        ) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
            rel_a = _pair_rel(node_a)
            rel_b = _pair_rel(node_b)
            gap_a = float(max(0.0, min(1.0, _f((node_a or {}).get("strength_gap"), 0.0))))
            gap_b = float(max(0.0, min(1.0, _f((node_b or {}).get("strength_gap"), 0.0))))
            shrink_a = float(max(0.0, min(1.0, st_a["n"] / (st_a["n"] + shrink_k_a))))
            shrink_b = float(max(0.0, min(1.0, st_b["n"] / (st_b["n"] + shrink_k_b))))
            w_a = float((1.0 / max(1e-9, st_a["var"])) * max(0.20, shrink_a) * (0.82 + (0.30 * rel_a)))
            w_b = float((1.0 / max(1e-9, st_b["var"])) * max(0.20, shrink_b) * (0.82 + (0.30 * rel_b)))
            if (w_a + w_b) <= 0.0:
                return {}, "", {}
            b_mix = float(((st_a["bias"] * w_a) + (st_b["bias"] * w_b)) / max(1e-9, (w_a + w_b)))
            se_mix = float(math.sqrt(1.0 / max(1e-9, (w_a + w_b))))
            z_mix = float(b_mix / max(1e-9, se_mix))
            p_mix = float(math.erfc(abs(z_mix) / math.sqrt(2.0)))
            n_mix = float(max(1.0, (0.85 * st_a["n"]) + (0.65 * st_b["n"])))
            if n_mix >= 1200.0 and abs(z_mix) >= 2.58:
                q_mix = "strong"
            elif n_mix >= 500.0 and abs(z_mix) >= 1.96:
                q_mix = "good"
            elif n_mix >= 200.0 and abs(z_mix) >= 1.20:
                q_mix = "ok"
            else:
                q_mix = "low_sample"
            node = {
                "score_bias": float(b_mix),
                "score_se": float(se_mix),
                "score_var": float(max(1e-9, se_mix * se_mix)),
                "score_z": float(z_mix),
                "score_p": float(p_mix),
                "score_q": float(p_mix),
                "fdr_reject": bool(p_mix <= fdr_q_alpha),
                "wf_n": round(n_mix),
                "n_eff": round(n_mix),
                "quality_band": str(q_mix),
                "source": str(source),
                "pair_reliability": float(max(0.0, min(1.0, ((rel_a * w_a) + (rel_b * w_b)) / max(1e-9, (w_a + w_b))))),
                "class_relation": str((node_a or {}).get("class_relation") or (node_b or {}).get("class_relation") or ""),
                "strength_gap": float(max(0.0, min(1.0, ((gap_a * w_a) + (gap_b * w_b)) / max(1e-9, (w_a + w_b))))),
                "strength_band_a": str((node_a or {}).get("strength_band_a") or (node_b or {}).get("strength_band_a") or ""),
                "strength_band_b": str((node_a or {}).get("strength_band_b") or (node_b or {}).get("strength_band_b") or ""),
            }
            blend = {
                f"{source_a_label}_scope": str(scope_a),
                f"{source_b_label}_scope": str(scope_b),
                f"w_{source_a_label}": float(w_a),
                f"w_{source_b_label}": float(w_b),
                f"n_{source_a_label}": float(st_a["n"]),
                f"n_{source_b_label}": float(st_b["n"]),
                f"se_{source_a_label}": float(st_a["se"]),
                f"se_{source_b_label}": float(st_b["se"]),
                f"shrink_{source_a_label}": float(shrink_a),
                f"shrink_{source_b_label}": float(shrink_b),
            }
            node["blend"] = dict(blend)
            return node, f"blend_ivw:{scope_a}|{scope_b}", blend

        pair_node_mid: Dict[str, Any] = {}
        pair_scope_mid = ""
        mid_blend_info: Dict[str, Any] = {}
        tier_mix_k = max(30.0, min(1800.0, _f(th.get("inter_tier_mix_k"), 220.0)))

        if pair_node_tier and pair_node_macro:
            pair_node_mid, pair_scope_mid, mid_blend_info = _blend_pair_nodes(
                node_a=pair_node_tier,
                scope_a=pair_scope_tier,
                st_a=st_tp,
                node_b=pair_node_macro,
                scope_b=pair_scope_macro,
                st_b=st_mc,
                shrink_k_a=tier_mix_k,
                shrink_k_b=inter_shrink_k,
                source="tier_pair+macro_ivw_bayes_v1",
                source_a_label="tier",
                source_b_label="macro",
            )
        elif pair_node_tier:
            pair_node_mid = dict(pair_node_tier)
            pair_scope_mid = str(pair_scope_tier)
        elif pair_node_macro:
            pair_node_mid = dict(pair_node_macro)
            pair_scope_mid = str(pair_scope_macro)

        st_mid = _pair_stats(pair_node_mid) if pair_node_mid else {}

        if pair_node_lg and pair_node_mid:
            pair_node, inter_pair_scope, blend_info = _blend_pair_nodes(
                node_a=pair_node_lg,
                scope_a=pair_scope_lg,
                st_a=st_lg,
                node_b=pair_node_mid,
                scope_b=pair_scope_mid,
                st_b=st_mid,
                shrink_k_a=mix_k,
                shrink_k_b=max(80.0, 0.85 * mix_k),
                source="league_pair+mid_ivw_bayes_v1",
                source_a_label="league",
                source_b_label="mid",
            )
            if pair_node and mid_blend_info:
                pair_node["blend"]["mid_blend"] = dict(mid_blend_info)
            if not pair_node:
                pair_node = dict(pair_node_lg)
                inter_pair_scope = str(pair_scope_lg)
        elif pair_node_lg:
            pair_node = dict(pair_node_lg)
            inter_pair_scope = str(pair_scope_lg)
        elif pair_node_mid:
            pair_node = dict(pair_node_mid)
            inter_pair_scope = str(pair_scope_mid)

        st_pair = _pair_stats(pair_node) if isinstance(pair_node, dict) and pair_node else {
            "bias": 0.0,
            "n": float(max(0.0, inter_n)),
            "se": float(_node_se({}, inter_sigma0)),
            "var": float(max(1e-9, _node_se({}, inter_sigma0) ** 2)),
            "z": 0.0,
            "p": 1.0,
            "q": 1.0,
            "fdr_reject": False,
        }
        pair_n = float(max(0.0, st_pair["n"] if st_pair["n"] > 0.0 else inter_n))
        pair_bias_raw = float(st_pair["bias"])
        pair_se = float(max(1e-6, st_pair["se"]))
        pair_z_raw = float(pair_bias_raw / max(1e-9, pair_se))
        z_mult = float(max(0.0, min(1.0, abs(pair_z_raw) / max(1e-9, z_gate))))
        pair_q = float(max(0.0, min(1.0, float(st_pair.get("q", 1.0)))))
        pair_reliability = 0.0
        pair_class_relation = ""
        pair_strength_gap = 0.0
        pair_strength_band_a = ""
        pair_strength_band_b = ""
        if isinstance(pair_node, dict):
            pair_reliability = float(max(0.0, min(1.0, _f(pair_node.get("pair_reliability"), 0.0))))
            pair_class_relation = str(pair_node.get("class_relation") or "")
            pair_strength_gap = float(max(0.0, min(1.0, _f(pair_node.get("strength_gap"), 0.0))))
            pair_strength_band_a = str(pair_node.get("strength_band_a") or "")
            pair_strength_band_b = str(pair_node.get("strength_band_b") or "")
        q_mult = 1.0
        if pair_q > fdr_q_alpha:
            q_mult = float(max(0.05, min(1.0, fdr_q_alpha / max(1e-9, pair_q))))
        if not bool(st_pair.get("fdr_reject", False)):
            q_mult = float(min(q_mult, 0.75))
        sig_mult = float(max(0.0, min(1.0, z_mult * q_mult)))
        pair_bias = float(pair_bias_raw * sig_mult)
        pair_z = float(pair_bias / max(1e-9, pair_se))
        pair_p = float(math.erfc(abs(pair_z) / math.sqrt(2.0)))

        inter_ev_n = _sh(pair_n, inter_shrink_k)
        pair_scope_kind = "none"
        if inter_pair_scope.startswith("blend_ivw:"):
            pair_scope_kind = "blend"
        elif inter_pair_scope.startswith("league_pair:"):
            pair_scope_kind = "league_pair"
        elif inter_pair_scope.startswith("tier_pair:"):
            pair_scope_kind = "tier_pair"
        elif inter_pair_scope.startswith("inter:"):
            pair_scope_kind = "macro_pair"
        league_pair_support = float(_sh(float(st_lg.get("n", 0.0)), mix_k)) if st_lg else 0.0
        tier_pair_support = float(_sh(float(st_tp.get("n", 0.0)), tier_mix_k)) if st_tp else 0.0
        macro_pair_support = float(_sh(float(st_mc.get("n", 0.0)), inter_shrink_k)) if st_mc else 0.0
        pair_support_ev = float(inter_ev_n)
        if pair_scope_kind == "blend":
            if pair_scope_mid.startswith("tier_pair:") or "tier_pair:" in pair_scope_mid:
                pair_support_ev = float(
                    max(
                        pair_support_ev,
                        (0.58 * league_pair_support) + (0.24 * tier_pair_support) + (0.18 * macro_pair_support),
                    )
                )
            else:
                pair_support_ev = float(max(pair_support_ev, (0.70 * league_pair_support) + (0.30 * macro_pair_support)))
        elif pair_scope_kind == "league_pair":
            pair_support_ev = float(max(pair_support_ev, league_pair_support))
        elif pair_scope_kind == "tier_pair":
            pair_support_ev = float(max(pair_support_ev, tier_pair_support))
        elif pair_scope_kind == "macro_pair":
            pair_support_ev = float(max(pair_support_ev, macro_pair_support))
        if pair_reliability > 0.0:
            if pair_scope_kind in {"blend", "league_pair"}:
                pair_support_ev = float(
                    max(
                        pair_support_ev,
                        min(1.0, pair_support_ev * (0.86 + (0.22 * pair_reliability))),
                    )
                )
            elif pair_scope_kind == "macro_pair":
                pair_support_ev = float(
                    max(
                        pair_support_ev,
                        min(1.0, pair_support_ev * (0.90 + (0.12 * pair_reliability))),
                    )
                )
        if pair_strength_gap >= 0.12 and pair_scope_kind in {"blend", "league_pair", "tier_pair"}:
            pair_support_ev = float(min(1.0, pair_support_ev * (1.00 + min(0.06, 0.18 * pair_strength_gap))))
        ev_floor = 0.35
        if pair_scope_kind == "blend":
            ev_floor = 0.44
        elif pair_scope_kind == "league_pair":
            ev_floor = 0.40
        elif pair_scope_kind == "tier_pair":
            ev_floor = 0.38
        if pair_strength_gap >= 0.18:
            ev_floor += 0.02
        inter_ev = float(pair_support_ev * (ev_floor + ((1.0 - ev_floor) * sig_mult)))
        if pair_reliability > 0.0 and pair_scope_kind in {"blend", "league_pair"}:
            inter_ev = float(min(1.0, inter_ev * (0.94 + (0.10 * pair_reliability))))
        if pair_scope_kind in {"blend", "league_pair"} and pair_q <= min(0.20, 2.0 * fdr_q_alpha):
            inter_ev += float(0.04 * league_pair_support)
        elif pair_scope_kind == "tier_pair" and pair_q <= min(0.20, 2.0 * fdr_q_alpha):
            inter_ev += float(0.025 * tier_pair_support)
        if not bool(st_pair.get("fdr_reject", False)) and pair_q > min(0.25, 2.0 * fdr_q_alpha):
            inter_ev *= 0.85
        if macro_a == "OTHER" or macro_b == "OTHER" or not macro_a or not macro_b:
            inter_ev *= 0.55
        inter_ev = float(max(0.0, min(1.0, inter_ev)))
        # Reliability band combines sample evidence and statistical significance.
        inter_band = "low"
        if abs(pair_z_raw) >= z_full and pair_q <= fdr_q_alpha and inter_ev >= 0.40:
            inter_band = "high"
        elif abs(pair_z_raw) >= z_gate and pair_q <= min(0.25, 2.0 * fdr_q_alpha) and inter_ev >= 0.20:
            inter_band = "mid"
        inter_unc = float(max(0.0, min(1.0, 1.0 - inter_ev)))
        inter_shrink = 0.05 + 0.12 * inter_unc
        if inter_band == "low":
            inter_shrink += 0.015
        elif inter_band == "high":
            inter_shrink -= 0.012
        if pair_scope_kind in {"blend", "league_pair"} and league_pair_support >= 0.35 and pair_q <= min(0.20, 2.0 * fdr_q_alpha):
            inter_shrink -= 0.010
        elif pair_scope_kind == "tier_pair" and tier_pair_support >= 0.30 and pair_q <= min(0.20, 2.0 * fdr_q_alpha):
            inter_shrink -= 0.006
        if pair_reliability > 0.0:
            if pair_scope_kind in {"blend", "league_pair"}:
                inter_shrink -= float(0.010 * pair_reliability)
                if pair_class_relation == "major_major":
                    inter_shrink -= 0.004
                elif pair_class_relation == "minor_minor":
                    inter_shrink += 0.003
            elif pair_scope_kind == "tier_pair":
                inter_shrink -= float(0.008 * pair_reliability)
            elif pair_scope_kind == "macro_pair":
                inter_shrink -= float(0.006 * pair_reliability)
        if pair_strength_gap >= 0.16 and pair_scope_kind in {"blend", "league_pair"}:
            inter_shrink -= float(min(0.007, 0.030 * pair_strength_gap))
        elif pair_strength_gap <= 0.05 and pair_class_relation == "major_major":
            inter_shrink += 0.003
        strong_cross_strength = bool(
            pair_strength_gap >= 0.16
            and pair_scope_kind in {"blend", "league_pair", "tier_pair"}
            and pair_strength_band_a in {"elite", "major"}
            and pair_strength_band_b in {"upper", "regional", "development", "local"}
        )
        very_strong_cross_strength = bool(strong_cross_strength and pair_strength_gap >= 0.19 and inter_ev >= 0.85)
        if strong_cross_strength:
            inter_shrink *= 0.88
        if very_strong_cross_strength:
            inter_shrink *= 0.78
        if bo_i == 1:
            inter_shrink += 0.02
        elif bo_i >= 5:
            inter_shrink -= 0.01
        if str(patch_status or "").upper() in {"REFIT", "WARN"}:
            inter_shrink += 0.01
        inter_shrink = float(max(0.02, min(0.26, inter_shrink)))
        # Directional nudge with significance gate and reliability-dependent clipping.
        dir_cap = float(0.007 + (0.040 * inter_ev))
        if pair_scope_kind in {"blend", "league_pair"}:
            dir_cap += float(0.004 * max(0.0, min(1.0, league_pair_support)))
            dir_cap += float(0.0035 * pair_reliability)
        elif pair_scope_kind == "tier_pair":
            dir_cap += float(0.0030 * max(0.0, min(1.0, tier_pair_support)))
            dir_cap += float(0.0025 * pair_reliability)
        elif pair_scope_kind == "macro_pair":
            dir_cap += float(0.0020 * pair_reliability)
        if pair_strength_gap >= 0.16:
            dir_cap += float(min(0.004, 0.018 * pair_strength_gap))
        elif pair_strength_gap <= 0.05 and pair_class_relation == "major_major":
            dir_cap -= 0.0015
        dir_cap = float(max(0.006, min(0.050, dir_cap)))
        if inter_band == "low":
            dir_cap *= 0.62
        elif inter_band == "mid":
            dir_cap *= 0.90
        elif inter_band == "high" and pair_scope_kind in {"blend", "league_pair"}:
            dir_cap *= 1.05
        inter_dir_raw = float((pair_bias / 230.0) * (0.40 + (0.60 * inter_ev)))
        inter_dir = float(max(-dir_cap, min(dir_cap, inter_dir_raw)))
        if strong_cross_strength and (z0 * inter_dir) < 0.0:
            inter_dir *= 0.68
        if very_strong_cross_strength and (z0 * inter_dir) < 0.0:
            inter_dir *= 0.72
        transfer_adj = build_interleague_transfer_adjustment(
            payload=(league_transfer_features if isinstance(league_transfer_features, dict) else {}),
            interregional=True,
            league_a=lga,
            league_b=lgb,
            team_a=team_a,
            team_b=team_b,
            team_aliases_a=list(team_aliases_a or []),
            team_aliases_b=list(team_aliases_b or []),
        )
        inter_shrink_before_transfer = float(inter_shrink)
        dir_cap_before_transfer = float(dir_cap)
        inter_dir_before_transfer = float(inter_dir)
        if bool(transfer_adj.get("applied", False)):
            inter_shrink = float(max(0.02, min(0.26, inter_shrink + _f(transfer_adj.get("shrink_adj"), 0.0))))
            dir_cap = float(max(0.006, min(0.050, dir_cap + _f(transfer_adj.get("dir_cap_bonus"), 0.0))))
            inter_dir = float(
                max(
                    -dir_cap,
                    min(
                        dir_cap,
                        inter_dir + _f(transfer_adj.get("dir_bonus"), 0.0),
                    ),
                )
            )
        out["interleague"] = {
            "macro_a": str(macro_a),
            "macro_b": str(macro_b),
            "pair_scope": (str(inter_pair_scope) if inter_pair_scope else None),
            "pair_scope_kind": str(pair_scope_kind),
            "pair_n": float(pair_n),
            "pair_bias": float(pair_bias),
            "pair_bias_raw": float(pair_bias_raw),
            "pair_se": float(pair_se),
            "pair_z": float(pair_z),
            "pair_z_raw": float(pair_z_raw),
            "pair_p": float(pair_p),
            "pair_q": float(pair_q),
            "pair_reliability": float(pair_reliability),
            "class_relation": (str(pair_class_relation) if pair_class_relation else None),
            "strength_gap": float(pair_strength_gap),
            "strength_band_a": (str(pair_strength_band_a) if pair_strength_band_a else None),
            "strength_band_b": (str(pair_strength_band_b) if pair_strength_band_b else None),
            "significance_mult": float(sig_mult),
            "z_mult": float(z_mult),
            "q_mult": float(q_mult),
            "fdr_reject": bool(st_pair.get("fdr_reject", False)),
            "fdr_q_alpha": float(fdr_q_alpha),
            "z_gate": float(z_gate),
            "z_full": float(z_full),
            "inter_ev": float(inter_ev),
            "pair_support_ev": float(pair_support_ev),
            "league_pair_support_ev": float(league_pair_support),
            "tier_pair_support_ev": float(tier_pair_support),
            "macro_pair_support_ev": float(macro_pair_support),
            "inter_unc": float(inter_unc),
            "inter_shrink": float(inter_shrink),
            "inter_dir": float(inter_dir),
            "inter_dir_raw": float(inter_dir_raw),
            "reliability_band": str(inter_band),
            "dir_cap": float(dir_cap),
            "dir_cap_before_transfer": float(dir_cap_before_transfer),
            "inter_shrink_before_transfer": float(inter_shrink_before_transfer),
            "inter_dir_before_transfer": float(inter_dir_before_transfer),
            "blend": (blend_info if blend_info else None),
            "transfer_features": transfer_adj,
        }

    if (not math.isfinite(bias_raw)) or abs(float(bias_raw)) < 1e-9:
        out["reason"] = "zero_or_missing_bias"
        return out

    nte = _f(n_team_eff, 0.0)
    npe = _f(n_players_eff, 0.0)
    out["n_team_eff"] = (float(nte) if math.isfinite(nte) else None)
    out["n_players_eff"] = (float(npe) if math.isfinite(npe) else None)
    evidence = max(0.0, min(1.0, (0.7 * nte + 0.3 * npe) / 60.0))
    wf_conf = max(0.0, min(1.0, (_f(wf_n, 0.0) / 120.0))) if math.isfinite(wf_n) else 0.4

    # Map score_bias (points on confidence scale) into a tiny logit calibration delta.
    # Positive bias => allow a bit more confidence; negative => damp.
    base_mag = min(0.16, abs(float(bias_raw)) / 120.0)
    w = 0.03 + 0.14 * math.sqrt(evidence) * (0.55 + 0.45 * wf_conf)
    if bo_i == 1:
        w *= 0.75
    elif bo_i >= 5:
        w *= 1.10
    if bool(interregional):
        w *= 0.80
    ps = str(patch_status or "").upper()
    if "REFIT" in ps:
        w *= 0.75
    elif "WARN" in ps:
        w *= 0.88
    if quality_band.lower() in {"low", "warn"}:
        w *= 0.85
    if is_po and bo_i >= 5:
        w *= 1.06
    w = float(max(0.0, min(0.16, w)))

    z_adj = z0 * (1.0 + math.copysign(base_mag * w, z0) * (1.0 if float(bias_raw) >= 0 else -1.0))
    if bool(interregional):
        z_adj = (z_adj + float(inter_dir)) * (1.0 - float(inter_shrink))

    # Runtime fine calibration (odds bands): prevents unstable extremes under weak evidence.
    # Supports contextual profiles in thresholds:
    # thresholds.fine_profiles.default / same_league / interregional
    # thresholds.fine_profile_by_league_same["LCK"] -> profile for same-league
    # thresholds.fine_profile_by_league_inter["LCK"] -> profile for interregional
    # (fallback-safe: fine_profile_by_league remains supported for backwards compatibility).
    fine_profiles = th.get("fine_profiles") if isinstance(th.get("fine_profiles"), dict) else {}
    fine_profile_by_league = th.get("fine_profile_by_league") if isinstance(th.get("fine_profile_by_league"), dict) else {}
    fine_profile_by_league_same = (
        th.get("fine_profile_by_league_same") if isinstance(th.get("fine_profile_by_league_same"), dict) else {}
    )
    fine_profile_by_league_inter = (
        th.get("fine_profile_by_league_inter") if isinstance(th.get("fine_profile_by_league_inter"), dict) else {}
    )
    profile_used = "default"
    profile_source = "base"
    if bool(interregional):
        profile_used = "interregional"
    elif lga and lgb and (str(lga) == str(lgb)):
        profile_used = "same_league"
    if bool(interregional):
        lg_pref = str(
            fine_profile_by_league_inter.get(lga)
            or fine_profile_by_league_inter.get(lg_scope)
            or fine_profile_by_league.get(lga)
            or fine_profile_by_league.get(lg_scope)
            or ""
        ).strip()
    else:
        lg_pref = str(
            fine_profile_by_league_same.get(lga)
            or fine_profile_by_league_same.get(lg_scope)
            or fine_profile_by_league.get(lga)
            or fine_profile_by_league.get(lg_scope)
            or ""
        ).strip()
    if lg_pref and isinstance(fine_profiles.get(lg_pref), dict):
        profile_used = lg_pref
        profile_source = "league_override"
    prof_node = fine_profiles.get(profile_used) if isinstance(fine_profiles.get(profile_used), dict) else {}

    def _pf(name: str, default: float) -> float:
        if isinstance(prof_node, dict) and (name in prof_node):
            return _f(prof_node.get(name), default)
        return _f(th.get(name), default)

    z_abs = abs(float(z_adj))
    z_hi = max(0.9, min(2.4, _pf("fine_z_hi", 1.20)))
    z_xhi = max(z_hi + 0.2, min(3.2, _pf("fine_z_xhi", 1.85)))
    s_hi = max(0.0, min(0.18, _pf("fine_shrink_hi", 0.05)))
    s_xhi = max(0.0, min(0.25, _pf("fine_shrink_xhi", 0.09)))
    fine = 0.0
    band = "mid"
    if z_abs >= z_hi:
        fine += s_hi
        band = "high"
    if z_abs >= z_xhi:
        fine += s_xhi
        band = "extreme"
    if quality_band.lower() in {"low", "warn", "low_sample"}:
        fine += 0.03
        if bool(interregional):
            fine += 0.02
            # Inter-league reliability feedback: low pair evidence gets stricter runtime shrink.
            if str(inter_band) == "low":
                fine += 0.025
            elif str(inter_band) == "mid":
                fine += 0.010
            elif str(inter_band) == "high":
                fine -= 0.005
            if strong_cross_strength:
                fine *= 0.88
            if very_strong_cross_strength:
                fine *= 0.80
    # Less evidence => more shrink.
    fine *= (0.55 + 0.45 * (1.0 - max(0.0, min(1.0, evidence))))
    fine = float(max(0.0, min(0.22, fine)))
    z_adj = z_adj * (1.0 - fine)
    out["runtime_fine"] = {
        "band": str(band),
        "z_abs_before": float(z_abs),
        "fine_shrink": float(fine),
        "z_hi": float(z_hi),
        "z_xhi": float(z_xhi),
        "profile_used": str(profile_used),
        "profile_source": str(profile_source),
        "inter_reliability_band": str(inter_band),
        "inter_ev": (float(inter_ev) if math.isfinite(float(inter_ev)) else None),
    }

    p1 = float(_sigmoid(z_adj))
    diff_pp = float(100.0 * (p1 - p0))
    # Cap move strictly.
    cap_pp = 1.8 if bo_i == 1 else (2.4 if bo_i == 3 else 3.0)
    if bool(interregional):
        cap_pp *= 0.88
    if abs(diff_pp) > cap_pp:
        p1 = float(_clip_prob(p0 + math.copysign(cap_pp / 100.0, diff_pp)))
        diff_pp = float(100.0 * (p1 - p0))
    if bool(interregional) and strong_cross_strength:
        stronger_side = "a" if pair_strength_band_a in {"elite", "major"} and pair_strength_band_b in {"upper", "regional", "development", "local"} else (
            "b" if pair_strength_band_b in {"elite", "major"} and pair_strength_band_a in {"upper", "regional", "development", "local"} else ""
        )
        if stronger_side:
            before_stronger = float(p0 if stronger_side == "a" else (1.0 - p0))
            after_stronger = float(p1 if stronger_side == "a" else (1.0 - p1))
            adverse_pp = float(max(0.0, (before_stronger - after_stronger) * 100.0))
            adverse_cap_pp = 1.4 if not very_strong_cross_strength else 1.0
            if adverse_pp > adverse_cap_pp:
                clipped_stronger = float(before_stronger - (adverse_cap_pp / 100.0))
                clipped_stronger = max(1e-6, min(1.0 - 1e-6, clipped_stronger))
                p1 = float(clipped_stronger if stronger_side == "a" else (1.0 - clipped_stronger))
                diff_pp = float(100.0 * (p1 - p0))

    if abs(diff_pp) < 0.02:
        out["reason"] = "tiny_move"
        out["weight"] = float(w)
        out["bias_adj"] = float(base_mag * w)
        return out

    out["applied"] = True
    out["reason"] = "ok"
    out["weight"] = float(w)
    out["bias_adj"] = float(base_mag * w)
    out["p_after"] = float(p1)
    out["diff_pp"] = float(diff_pp)
    return out
