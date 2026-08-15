from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from mlcore.league_context import league_macro_key
from mlcore.tiers import (
    canonical_league,
    league_strength_band,
    league_strength_score,
    tier_for_league,
)
from mlcore.utils import safe_float_or_none as _fnum


def _canon_league(v: Any) -> str:
    return str(canonical_league(str(v or "")))


def _macro_group(league: Any) -> str:
    lg = _canon_league(league)
    if not lg:
        return ""
    return str(league_macro_key(lg, unknown="OTHER"))


def _node_n_eff(node: Dict[str, Any]) -> int:
    try:
        return int(
            max(
                0,
                int((node or {}).get("n_eff") or 0),
                int((node or {}).get("wf_n") or 0),
                int((node or {}).get("lv_n") or 0),
                int((node or {}).get("n_total") or 0),
            )
        )
    except Exception:
        return 0


def _quality_band_from_n(n_eff: int) -> str:
    n = int(max(0, n_eff))
    if n >= 1200:
        return "strong"
    if n >= 500:
        return "good"
    if n >= 200:
        return "ok"
    return "low_sample"


def _quality_band_from_stats(n_eff: int, z_abs: float) -> str:
    n = int(max(0, n_eff))
    z = float(max(0.0, z_abs))
    if n >= 1200 and z >= 2.58:
        return "strong"
    if n >= 500 and z >= 1.96:
        return "good"
    if n >= 200 and z >= 1.20:
        return "ok"
    return "low_sample"


def infer_league_class_from_node(league: str, node: Dict[str, Any]) -> str:
    lg = _canon_league(league)
    if not lg:
        return "minor"
    tier = str(tier_for_league(lg) or "OTHER").upper()
    strength_band = str(league_strength_band(lg) or "local")
    n_eff = _node_n_eff(node)
    qb = str((node or {}).get("quality_band") or "").strip().lower()
    strong_q = qb in {"strong", "good"}
    ok_q = qb in {"strong", "good", "ok"}

    if strength_band in {"elite", "major"} and n_eff >= 220 and ok_q:
        return "major"
    if strength_band == "upper" and n_eff >= 280 and ok_q:
        return "major"
    if tier == "ERL" and n_eff >= 1300 and strong_q:
        return "major"
    if strength_band == "development" and n_eff >= 2400 and strong_q:
        return "major"
    return "minor"


def _source_reliability(league: str, node: Dict[str, Any], league_class_out: Dict[str, str]) -> float:
    lg = _canon_league(league)
    tier = str(tier_for_league(lg) or "OTHER").upper()
    strength_band = str(league_strength_band(lg) or "local")
    strength_score = float(max(0.0, min(1.0, league_strength_score(lg))))
    qb = str((node or {}).get("quality_band") or "").strip().lower()
    q = float(max(0.0, min(1.0, float(_fnum((node or {}).get("score_q")) or 1.0))))
    n_eff = _node_n_eff(node)
    cls = str(league_class_out.get(lg) or infer_league_class_from_node(lg, node))
    fdr_reject = bool((node or {}).get("fdr_reject", False))

    n_mult = float(0.55 + (0.45 * max(0.0, min(1.0, float(n_eff) / 900.0))))
    qb_mult = {
        "strong": 1.05,
        "good": 0.97,
        "ok": 0.88,
        "low_sample": 0.70,
    }.get(qb, 0.82)
    if q <= 0.10:
        q_mult = 1.00
    elif q <= 0.25:
        q_mult = 0.96
    elif q <= 0.50:
        q_mult = 0.88
    else:
        q_mult = 0.78
    tier_mult = {
        "T1": 1.00,
        "T2": 0.97,
        "ERL": 0.88,
        "MINOR": 0.78,
        "OTHER": 0.80,
    }.get(tier, 0.80)
    band_mult = {
        "elite": 1.03,
        "major": 1.00,
        "upper": 0.97,
        "regional": 0.92,
        "development": 0.87,
        "local": 0.82,
    }.get(strength_band, 0.84)
    strength_mult = float(0.80 + (0.24 * strength_score))
    cls_mult = 1.00 if cls == "major" else 0.92
    fdr_mult = 1.00 if fdr_reject else 0.92
    rel = float(n_mult * qb_mult * q_mult * tier_mult * band_mult * strength_mult * cls_mult * fdr_mult)
    return float(max(0.20, min(1.00, rel)))


def _class_relation(class_a: str, class_b: str) -> str:
    ca = str(class_a or "minor")
    cb = str(class_b or "minor")
    if ca == "major" and cb == "major":
        return "major_major"
    if ca == "minor" and cb == "minor":
        return "minor_minor"
    return "mixed"


def _relation_mult(relation: str) -> float:
    rel = str(relation or "")
    if rel == "major_major":
        return 1.05
    if rel == "minor_minor":
        return 0.82
    return 0.92


def _macro_mult(macro_a: str, macro_b: str) -> float:
    ma = str(macro_a or "")
    mb = str(macro_b or "")
    if not ma or not mb or ma == "OTHER" or mb == "OTHER":
        return 0.75
    if {ma, mb} == {"KR", "CN"}:
        return 1.04
    return 1.00


def _meta_stats_from_contrib(rows: Any, n_cap: int, inter_sigma2_base: float) -> Dict[str, float]:
    items: List[Tuple[float, float, float]] = []
    for row in (rows if isinstance(rows, list) else []):
        if not isinstance(row, dict):
            continue
        b = _fnum(row.get("dir_bias"))
        if b is None:
            continue
        n_pair = float(_fnum(row.get("n_pair_adj")) or _fnum(row.get("n_pair")) or 0.0)
        n_pair = float(max(1.0, min(float(max(1, n_cap)), n_pair)))
        v_i = float(max(1e-6, float(inter_sigma2_base) / n_pair))
        items.append((float(b), float(n_pair), float(v_i)))
    if not items:
        return {
            "bias": 0.0,
            "se": math.sqrt(float(inter_sigma2_base)),
            "var": float(inter_sigma2_base),
            "z": 0.0,
            "p": 1.0,
            "tau2": 0.0,
            "i2": 0.0,
            "n_eff": 1.0,
            "k": 0.0,
        }
    w_fe = [1.0 / max(1e-9, v) for _, _, v in items]
    sum_w_fe = float(sum(w_fe))
    mu_fe = float(sum((w_fe[i] * items[i][0]) for i in range(len(items))) / max(1e-9, sum_w_fe))
    q = float(sum((w_fe[i] * (items[i][0] - mu_fe) ** 2) for i in range(len(items))))
    tau2 = 0.0
    if len(items) > 1:
        sum_w2 = float(sum((w * w) for w in w_fe))
        c = float(sum_w_fe - (sum_w2 / max(1e-9, sum_w_fe)))
        if c > 1e-9:
            tau2 = float(max(0.0, (q - float(len(items) - 1)) / c))
    w_re = [1.0 / max(1e-9, (items[i][2] + tau2)) for i in range(len(items))]
    sum_w_re = float(sum(w_re))
    mu = float(sum((w_re[i] * items[i][0]) for i in range(len(items))) / max(1e-9, sum_w_re))
    se = float(math.sqrt(1.0 / max(1e-9, sum_w_re)))
    z = float(mu / max(1e-9, se))
    p = float(math.erfc(abs(z) / math.sqrt(2.0)))
    i2 = 0.0
    if len(items) > 1 and q > 1e-9:
        i2 = float(max(0.0, min(1.0, (q - float(len(items) - 1)) / q)))
    n_eff = float(sum(n for _, n, _ in items))
    return {
        "bias": float(max(-8.0, min(8.0, mu))),
        "se": float(max(1e-6, se)),
        "var": float(max(1e-9, se * se)),
        "z": float(z),
        "p": float(max(0.0, min(1.0, p))),
        "tau2": float(max(0.0, tau2)),
        "i2": float(max(0.0, min(1.0, i2))),
        "n_eff": float(max(1.0, min(float(max(1, n_cap)), n_eff))),
        "k": float(len(items)),
    }


def _summarize_contrib_meta(contrib: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in contrib if isinstance(r, dict)]
    if not rows:
        return {
            "pair_reliability": 0.0,
            "source_reliability_a": 0.0,
            "source_reliability_b": 0.0,
            "class_relation": "mixed",
            "class_mix": {},
            "dominant_tier_pair": "",
        }
    total_w = 0.0
    rel_sum = 0.0
    rel_a_sum = 0.0
    rel_b_sum = 0.0
    rel_classes: Dict[str, float] = {}
    tier_pairs: Dict[str, float] = {}
    band_pairs: Dict[str, float] = {}
    strength_gap_sum = 0.0
    for row in rows:
        w = float(max(1.0, float(_fnum(row.get("n_pair_adj")) or _fnum(row.get("n_pair")) or 1.0)))
        total_w += w
        rel_sum += float(_fnum(row.get("pair_reliability")) or 0.0) * w
        rel_a_sum += float(_fnum(row.get("source_reliability_a")) or 0.0) * w
        rel_b_sum += float(_fnum(row.get("source_reliability_b")) or 0.0) * w
        strength_gap_sum += float(_fnum(row.get("strength_gap")) or 0.0) * w
        relation = str(row.get("class_relation") or "mixed")
        rel_classes[relation] = float(rel_classes.get(relation, 0.0) + w)
        tier_pair = f"{row.get('tier_a') or ''!s}->{row.get('tier_b') or ''!s}"
        tier_pairs[tier_pair] = float(tier_pairs.get(tier_pair, 0.0) + w)
        band_pair = f"{row.get('strength_band_a') or ''!s}->{row.get('strength_band_b') or ''!s}"
        band_pairs[band_pair] = float(band_pairs.get(band_pair, 0.0) + w)
    class_relation = max(rel_classes.items(), key=lambda kv: float(kv[1]))[0] if rel_classes else "mixed"
    class_mix = {
        str(k): float(v / max(1.0, total_w))
        for k, v in sorted(rel_classes.items(), key=lambda kv: str(kv[0]))
    }
    dominant_tier_pair = max(tier_pairs.items(), key=lambda kv: float(kv[1]))[0] if tier_pairs else ""
    dominant_strength_pair = max(band_pairs.items(), key=lambda kv: float(kv[1]))[0] if band_pairs else ""
    return {
        "pair_reliability": float(max(0.0, min(1.0, rel_sum / max(1.0, total_w)))),
        "source_reliability_a": float(max(0.0, min(1.0, rel_a_sum / max(1.0, total_w)))),
        "source_reliability_b": float(max(0.0, min(1.0, rel_b_sum / max(1.0, total_w)))),
        "strength_gap": float(max(0.0, min(1.0, strength_gap_sum / max(1.0, total_w)))),
        "class_relation": str(class_relation),
        "class_mix": class_mix,
        "dominant_tier_pair": str(dominant_tier_pair),
        "dominant_strength_pair": str(dominant_strength_pair),
    }


def build_interleague_pairs_table(
    by_league_out: Dict[str, Any],
    *,
    league_class_out: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    if not isinstance(by_league_out, dict) or (not by_league_out):
        return {}

    classes = dict(league_class_out) if isinstance(league_class_out, dict) else {}
    lg_items = [
        (_canon_league(lg), node)
        for lg, node in by_league_out.items()
        if _canon_league(lg) and isinstance(node, dict)
    ]
    for lg, node in lg_items:
        classes[lg] = str(classes.get(lg) or infer_league_class_from_node(lg, node))

    bias_vals: List[float] = []
    for node in by_league_out.values():
        if not isinstance(node, dict):
            continue
        sb = _fnum(node.get("score_bias"))
        if sb is None:
            continue
        bias_vals.append(float(sb))
    if len(bias_vals) >= 4:
        mean_bias = float(sum(bias_vals) / max(1, len(bias_vals)))
        var_bias = float(sum((float(x) - mean_bias) ** 2 for x in bias_vals) / max(1, len(bias_vals) - 1))
        inter_sigma2_base = float(max(4.0, min(36.0, var_bias)))
    else:
        inter_sigma2_base = 16.0

    macro_acc: Dict[str, Any] = {}
    macro_und_n: Dict[str, float] = {}
    tier_acc: Dict[str, Any] = {}
    tier_und_n: Dict[str, float] = {}
    league_acc: Dict[str, Any] = {}
    league_und_n: Dict[str, float] = {}

    def acc_add_macro(ma: str, mb: str, dir_bias: float, row: Dict[str, Any]) -> None:
        key = f"{ma}->{mb}"
        pair_rel = float(_fnum(row.get("pair_reliability")) or 0.0)
        n_pair_adj = float(max(1.0, float(_fnum(row.get("n_pair_adj")) or _fnum(row.get("n_pair")) or 1.0)))
        w = float(max(0.25, min(2.20, math.sqrt(n_pair_adj / 120.0) * (0.85 + (0.30 * pair_rel)))))
        acc = macro_acc.setdefault(
            key,
            {
                "macro_a": str(ma),
                "macro_b": str(mb),
                "contributors": [],
            },
        )
        rows = acc.get("contributors") if isinstance(acc.get("contributors"), list) else []
        item = dict(row)
        item["dir_bias"] = float(dir_bias)
        item["w"] = float(w)
        rows.append(item)
        acc["contributors"] = rows

    def acc_add_league(la: str, lb: str, ma: str, mb: str, dir_bias: float, row: Dict[str, Any]) -> None:
        key = f"{la}->{lb}"
        pair_rel = float(_fnum(row.get("pair_reliability")) or 0.0)
        n_pair_adj = float(max(1.0, float(_fnum(row.get("n_pair_adj")) or _fnum(row.get("n_pair")) or 1.0)))
        w = float(max(0.20, min(2.50, math.sqrt(n_pair_adj / 90.0) * (0.84 + (0.36 * pair_rel)))))
        acc = league_acc.setdefault(
            key,
            {
                "league_a": str(la),
                "league_b": str(lb),
                "macro_a": str(ma),
                "macro_b": str(mb),
                "contributors": [],
            },
        )
        rows = acc.get("contributors") if isinstance(acc.get("contributors"), list) else []
        item = dict(row)
        item["dir_bias"] = float(dir_bias)
        item["w"] = float(w)
        rows.append(item)
        acc["contributors"] = rows

    def acc_add_tier(ta: str, tb: str, dir_bias: float, row: Dict[str, Any]) -> None:
        key = f"{ta}->{tb}"
        pair_rel = float(_fnum(row.get("pair_reliability")) or 0.0)
        n_pair_adj = float(max(1.0, float(_fnum(row.get("n_pair_adj")) or _fnum(row.get("n_pair")) or 1.0)))
        w = float(max(0.22, min(2.35, math.sqrt(n_pair_adj / 100.0) * (0.84 + (0.34 * pair_rel)))))
        acc = tier_acc.setdefault(
            key,
            {
                "tier_a": str(ta),
                "tier_b": str(tb),
                "contributors": [],
            },
        )
        rows = acc.get("contributors") if isinstance(acc.get("contributors"), list) else []
        item = dict(row)
        item["dir_bias"] = float(dir_bias)
        item["w"] = float(w)
        rows.append(item)
        acc["contributors"] = rows

    for i in range(len(lg_items)):
        lga, na = lg_items[i]
        ma = _macro_group(lga)
        if not ma:
            continue
        for j in range(i + 1, len(lg_items)):
            lgb, nb = lg_items[j]
            mb = _macro_group(lgb)
            if (not mb) or (ma == mb):
                continue
            n_a = _node_n_eff(na)
            n_b = _node_n_eff(nb)
            pair_n = int(max(0, min(n_a, n_b)))
            if pair_n <= 0:
                continue
            sb_a = float(_fnum(na.get("score_bias")) or 0.0)
            sb_b = float(_fnum(nb.get("score_bias")) or 0.0)
            cls_a = str(classes.get(lga) or "minor")
            cls_b = str(classes.get(lgb) or "minor")
            tier_a = str(tier_for_league(lga) or "OTHER").upper()
            tier_b = str(tier_for_league(lgb) or "OTHER").upper()
            strength_score_a = float(max(0.0, min(1.0, league_strength_score(lga))))
            strength_score_b = float(max(0.0, min(1.0, league_strength_score(lgb))))
            strength_band_a = str(league_strength_band(lga) or "local")
            strength_band_b = str(league_strength_band(lgb) or "local")
            strength_gap = float(abs(strength_score_a - strength_score_b))
            structural_edge = float(max(-1.5, min(1.5, (strength_score_a - strength_score_b) * 5.0)))
            src_rel_a = _source_reliability(lga, na, classes)
            src_rel_b = _source_reliability(lgb, nb, classes)
            relation = _class_relation(cls_a, cls_b)
            pair_rel = float(math.sqrt(max(1e-9, src_rel_a * src_rel_b)))
            pair_rel *= float(_relation_mult(relation) * _macro_mult(ma, mb))
            if strength_gap >= 0.14:
                pair_rel *= float(1.01 + min(0.03, 0.08 * strength_gap))
            elif strength_gap <= 0.05 and relation == "major_major":
                pair_rel *= 1.02
            pair_rel = float(max(0.20, min(1.00, pair_rel)))
            pair_n_adj = int(max(1, round(float(pair_n) * (0.45 + (0.55 * pair_rel)))))
            dir_bias_raw = float(max(-8.0, min(8.0, ((sb_a - sb_b) * 0.55) + structural_edge)))
            dir_bias = float(max(-8.0, min(8.0, dir_bias_raw * (0.78 + (0.22 * pair_rel)))))
            row_ab = {
                "league_a": str(lga),
                "league_b": str(lgb),
                "macro_a": str(ma),
                "macro_b": str(mb),
                "tier_a": str(tier_a),
                "tier_b": str(tier_b),
                "strength_band_a": str(strength_band_a),
                "strength_band_b": str(strength_band_b),
                "strength_score_a": float(strength_score_a),
                "strength_score_b": float(strength_score_b),
                "strength_gap": float(strength_gap),
                "structural_edge": float(structural_edge),
                "class_a": str(cls_a),
                "class_b": str(cls_b),
                "class_relation": str(relation),
                "n_pair": int(pair_n),
                "n_pair_adj": int(pair_n_adj),
                "source_reliability_a": float(src_rel_a),
                "source_reliability_b": float(src_rel_b),
                "pair_reliability": float(pair_rel),
            }
            row_ba = dict(row_ab)
            row_ba.update(
                {
                    "league_a": str(lgb),
                    "league_b": str(lga),
                    "macro_a": str(mb),
                    "macro_b": str(ma),
                    "tier_a": str(tier_b),
                    "tier_b": str(tier_a),
                    "strength_band_a": str(strength_band_b),
                    "strength_band_b": str(strength_band_a),
                    "strength_score_a": float(strength_score_b),
                    "strength_score_b": float(strength_score_a),
                    "strength_gap": float(strength_gap),
                    "structural_edge": float(-structural_edge),
                    "class_a": str(cls_b),
                    "class_b": str(cls_a),
                }
            )
            acc_add_macro(ma, mb, dir_bias, row_ab)
            acc_add_macro(mb, ma, -dir_bias, row_ba)
            und = "->".join(sorted([ma, mb]))
            macro_und_n[und] = float(macro_und_n.get(und, 0.0) + float(max(1, min(3200, pair_n_adj))))
            acc_add_tier(tier_a, tier_b, dir_bias, row_ab)
            acc_add_tier(tier_b, tier_a, -dir_bias, row_ba)
            und_tier = "->".join(sorted([tier_a, tier_b]))
            tier_und_n[und_tier] = float(tier_und_n.get(und_tier, 0.0) + float(max(1, min(3400, pair_n_adj))))
            acc_add_league(lga, lgb, ma, mb, dir_bias, row_ab)
            acc_add_league(lgb, lga, mb, ma, -dir_bias, row_ba)
            und_lg = "->".join(sorted([lga, lgb]))
            league_und_n[und_lg] = float(league_und_n.get(und_lg, 0.0) + float(max(1, min(3600, pair_n_adj))))

    out: Dict[str, Any] = {}

    for _dir_key, acc in league_acc.items():
        la = str(acc.get("league_a") or "")
        lb = str(acc.get("league_b") or "")
        ma = str(acc.get("macro_a") or "")
        mb = str(acc.get("macro_b") or "")
        if (not la) or (not lb):
            continue
        contrib = acc.get("contributors") if isinstance(acc.get("contributors"), list) else []
        stats = _meta_stats_from_contrib(contrib, 3600, inter_sigma2_base)
        bias = float(stats.get("bias") or 0.0)
        pair_n_eff = int(max(1, round(float(stats.get("n_eff") or 1.0))))
        band = _quality_band_from_stats(pair_n_eff, abs(float(stats.get("z") or 0.0)))
        meta = _summarize_contrib_meta(contrib)
        top_contrib = sorted(contrib, key=lambda x: int(x.get("n_pair_adj") or x.get("n_pair") or 0), reverse=True)[:8]

        def put_lg(key: str) -> None:
            if not key:
                return
            out[key] = {
                "score_bias": float(bias),
                "score_se": float(stats.get("se") or 0.0),
                "score_var": float(stats.get("var") or 0.0),
                "score_z": float(stats.get("z") or 0.0),
                "score_p": float(stats.get("p") or 1.0),
                "heterogeneity_tau2": float(stats.get("tau2") or 0.0),
                "heterogeneity_i2": float(stats.get("i2") or 0.0),
                "contributors_count": round(float(stats.get("k") or 0.0)),
                "wf_n": int(pair_n_eff),
                "n_eff": int(pair_n_eff),
                "quality_band": str(band),
                "source": "interleague_league_pair_meta_v3",
                "league_a": str(la),
                "league_b": str(lb),
                "macro_a": str(ma),
                "macro_b": str(mb),
                "tier_a": str(top_contrib[0].get("tier_a") or ""),
                "tier_b": str(top_contrib[0].get("tier_b") or ""),
                "strength_band_a": str(top_contrib[0].get("strength_band_a") or ""),
                "strength_band_b": str(top_contrib[0].get("strength_band_b") or ""),
                "strength_score_a": float(_fnum(top_contrib[0].get("strength_score_a")) or 0.0),
                "strength_score_b": float(_fnum(top_contrib[0].get("strength_score_b")) or 0.0),
                "strength_gap": float(meta.get("strength_gap") or 0.0),
                "structural_edge": float(_fnum(top_contrib[0].get("structural_edge")) or 0.0),
                "class_relation": str(meta.get("class_relation") or "mixed"),
                "class_mix": dict(meta.get("class_mix") or {}),
                "pair_reliability": float(meta.get("pair_reliability") or 0.0),
                "source_reliability_a": float(meta.get("source_reliability_a") or 0.0),
                "source_reliability_b": float(meta.get("source_reliability_b") or 0.0),
                "contributors": top_contrib,
            }

        for bo_i in (1, 3, 5):
            for po in (0, 1):
                put_lg(f"league_pair:{la}->{lb}|bo:{int(bo_i)}|po:{int(po)}|inter:1")
            put_lg(f"league_pair:{la}->{lb}|bo:{int(bo_i)}|inter:1")
        put_lg(f"league_pair:{la}->{lb}|inter:1")
        put_lg(f"league_pair:{la}->{lb}")

    for _dir_key, acc in tier_acc.items():
        ta = str(acc.get("tier_a") or "")
        tb = str(acc.get("tier_b") or "")
        if (not ta) or (not tb):
            continue
        contrib = acc.get("contributors") if isinstance(acc.get("contributors"), list) else []
        stats = _meta_stats_from_contrib(contrib, 3400, inter_sigma2_base)
        bias = float(stats.get("bias") or 0.0)
        pair_n_eff = int(max(1, round(float(stats.get("n_eff") or 1.0))))
        band = _quality_band_from_stats(pair_n_eff, abs(float(stats.get("z") or 0.0)))
        meta = _summarize_contrib_meta(contrib)
        top_contrib = sorted(contrib, key=lambda x: int(x.get("n_pair_adj") or x.get("n_pair") or 0), reverse=True)[:8]

        def put_tier(key: str) -> None:
            if not key:
                return
            out[key] = {
                "score_bias": float(bias),
                "score_se": float(stats.get("se") or 0.0),
                "score_var": float(stats.get("var") or 0.0),
                "score_z": float(stats.get("z") or 0.0),
                "score_p": float(stats.get("p") or 1.0),
                "heterogeneity_tau2": float(stats.get("tau2") or 0.0),
                "heterogeneity_i2": float(stats.get("i2") or 0.0),
                "contributors_count": round(float(stats.get("k") or 0.0)),
                "wf_n": int(pair_n_eff),
                "n_eff": int(pair_n_eff),
                "quality_band": str(band),
                "source": "interleague_tier_pair_meta_v1",
                "tier_a": str(ta),
                "tier_b": str(tb),
                "strength_gap": float(meta.get("strength_gap") or 0.0),
                "dominant_strength_pair": str(meta.get("dominant_strength_pair") or ""),
                "class_relation": str(meta.get("class_relation") or "mixed"),
                "class_mix": dict(meta.get("class_mix") or {}),
                "pair_reliability": float(meta.get("pair_reliability") or 0.0),
                "contributors": top_contrib,
            }

        for bo_i in (1, 3, 5):
            for po in (0, 1):
                put_tier(f"tier_pair:{ta}->{tb}|bo:{int(bo_i)}|po:{int(po)}|inter:1")
            put_tier(f"tier_pair:{ta}->{tb}|bo:{int(bo_i)}|inter:1")
        put_tier(f"tier_pair:{ta}->{tb}|inter:1")
        put_tier(f"tier_pair:{ta}->{tb}")

    for _dir_key, acc in macro_acc.items():
        ma = str(acc.get("macro_a") or "")
        mb = str(acc.get("macro_b") or "")
        if (not ma) or (not mb):
            continue
        contrib = acc.get("contributors") if isinstance(acc.get("contributors"), list) else []
        stats = _meta_stats_from_contrib(contrib, 3200, inter_sigma2_base)
        bias = float(stats.get("bias") or 0.0)
        pair_n_eff = int(max(1, round(float(stats.get("n_eff") or 1.0))))
        band = _quality_band_from_stats(pair_n_eff, abs(float(stats.get("z") or 0.0)))
        meta = _summarize_contrib_meta(contrib)
        top_contrib = sorted(contrib, key=lambda x: int(x.get("n_pair_adj") or x.get("n_pair") or 0), reverse=True)[:8]

        def put_mc(key: str) -> None:
            if not key:
                return
            out[key] = {
                "score_bias": float(bias),
                "score_se": float(stats.get("se") or 0.0),
                "score_var": float(stats.get("var") or 0.0),
                "score_z": float(stats.get("z") or 0.0),
                "score_p": float(stats.get("p") or 1.0),
                "heterogeneity_tau2": float(stats.get("tau2") or 0.0),
                "heterogeneity_i2": float(stats.get("i2") or 0.0),
                "contributors_count": round(float(stats.get("k") or 0.0)),
                "wf_n": int(pair_n_eff),
                "n_eff": int(pair_n_eff),
                "quality_band": str(band),
                "source": "interleague_pair_meta_v4",
                "macro_a": str(ma),
                "macro_b": str(mb),
                "strength_gap": float(meta.get("strength_gap") or 0.0),
                "class_relation": str(meta.get("class_relation") or "mixed"),
                "class_mix": dict(meta.get("class_mix") or {}),
                "pair_reliability": float(meta.get("pair_reliability") or 0.0),
                "dominant_tier_pair": str(meta.get("dominant_tier_pair") or ""),
                "dominant_strength_pair": str(meta.get("dominant_strength_pair") or ""),
                "contributors": top_contrib,
            }

        for bo_i in (1, 3, 5):
            for po in (0, 1):
                put_mc(f"inter:{ma}->{mb}|bo:{int(bo_i)}|po:{int(po)}|inter:1")
            put_mc(f"inter:{ma}->{mb}|bo:{int(bo_i)}|inter:1")
        put_mc(f"inter:{ma}->{mb}|inter:1")
        put_mc(f"inter:{ma}->{mb}")

    for und_tier, n_sum in tier_und_n.items():
        pair_n_eff = int(max(1, min(3400, round(float(n_sum)))))
        band = _quality_band_from_n(pair_n_eff)
        se = float(max(1e-6, math.sqrt(float(inter_sigma2_base) / max(1.0, float(pair_n_eff)))))
        var = float(max(1e-9, float(inter_sigma2_base) / max(1.0, float(pair_n_eff))))
        for bo_i in (1, 3, 5):
            for po in (0, 1):
                key = f"tier_pair:{und_tier}|bo:{int(bo_i)}|po:{int(po)}|inter:1"
                if key in out:
                    continue
                out[key] = {
                    "score_bias": 0.0,
                    "score_se": se,
                    "score_var": var,
                    "score_z": 0.0,
                    "score_p": 1.0,
                    "wf_n": int(pair_n_eff),
                    "n_eff": int(pair_n_eff),
                    "quality_band": str(band),
                    "source": "interleague_tier_pair_undirected_v1",
                }
            key_bo = f"tier_pair:{und_tier}|bo:{int(bo_i)}|inter:1"
            if key_bo not in out:
                out[key_bo] = {
                    "score_bias": 0.0,
                    "score_se": se,
                    "score_var": var,
                    "score_z": 0.0,
                    "score_p": 1.0,
                    "wf_n": int(pair_n_eff),
                    "n_eff": int(pair_n_eff),
                    "quality_band": str(band),
                    "source": "interleague_tier_pair_undirected_v1",
                }
        key_inter = f"tier_pair:{und_tier}|inter:1"
        if key_inter not in out:
            out[key_inter] = {
                "score_bias": 0.0,
                "score_se": se,
                "score_var": var,
                "score_z": 0.0,
                "score_p": 1.0,
                "wf_n": int(pair_n_eff),
                "n_eff": int(pair_n_eff),
                "quality_band": str(band),
                "source": "interleague_tier_pair_undirected_v1",
            }
        key_plain = f"tier_pair:{und_tier}"
        if key_plain not in out:
            out[key_plain] = {
                "score_bias": 0.0,
                "score_se": se,
                "score_var": var,
                "score_z": 0.0,
                "score_p": 1.0,
                "wf_n": int(pair_n_eff),
                "n_eff": int(pair_n_eff),
                "quality_band": str(band),
                "source": "interleague_tier_pair_undirected_v1",
            }

    for und, n_sum in macro_und_n.items():
        pair_n_eff = int(max(1, min(3200, round(float(n_sum)))))
        band = _quality_band_from_n(pair_n_eff)
        se = float(max(1e-6, math.sqrt(float(inter_sigma2_base) / max(1.0, float(pair_n_eff)))))
        var = float(max(1e-9, float(inter_sigma2_base) / max(1.0, float(pair_n_eff))))
        for bo_i in (1, 3, 5):
            for po in (0, 1):
                key = f"inter:{und}|bo:{int(bo_i)}|po:{int(po)}|inter:1"
                if key in out:
                    continue
                out[key] = {
                    "score_bias": 0.0,
                    "score_se": se,
                    "score_var": var,
                    "score_z": 0.0,
                    "score_p": 1.0,
                    "wf_n": int(pair_n_eff),
                    "n_eff": int(pair_n_eff),
                    "quality_band": str(band),
                    "source": "interleague_pair_undirected_v2",
                }
            key_bo = f"inter:{und}|bo:{int(bo_i)}|inter:1"
            if key_bo not in out:
                out[key_bo] = {
                    "score_bias": 0.0,
                    "score_se": se,
                    "score_var": var,
                    "score_z": 0.0,
                    "score_p": 1.0,
                    "wf_n": int(pair_n_eff),
                    "n_eff": int(pair_n_eff),
                    "quality_band": str(band),
                    "source": "interleague_pair_undirected_v2",
                }
        key_inter = f"inter:{und}|inter:1"
        if key_inter not in out:
            out[key_inter] = {
                "score_bias": 0.0,
                "score_se": se,
                "score_var": var,
                "score_z": 0.0,
                "score_p": 1.0,
                "wf_n": int(pair_n_eff),
                "n_eff": int(pair_n_eff),
                "quality_band": str(band),
                "source": "interleague_pair_undirected_v2",
            }
        key_plain = f"inter:{und}"
        if key_plain not in out:
            out[key_plain] = {
                "score_bias": 0.0,
                "score_se": se,
                "score_var": var,
                "score_z": 0.0,
                "score_p": 1.0,
                "wf_n": int(pair_n_eff),
                "n_eff": int(pair_n_eff),
                "quality_band": str(band),
                "source": "interleague_pair_undirected_v2",
            }

    for und_lg, n_sum in league_und_n.items():
        pair_n_eff = int(max(1, min(3600, round(float(n_sum)))))
        band = _quality_band_from_n(pair_n_eff)
        se = float(max(1e-6, math.sqrt(float(inter_sigma2_base) / max(1.0, float(pair_n_eff)))))
        var = float(max(1e-9, float(inter_sigma2_base) / max(1.0, float(pair_n_eff))))
        for bo_i in (1, 3, 5):
            for po in (0, 1):
                key = f"league_pair:{und_lg}|bo:{int(bo_i)}|po:{int(po)}|inter:1"
                if key in out:
                    continue
                out[key] = {
                    "score_bias": 0.0,
                    "score_se": se,
                    "score_var": var,
                    "score_z": 0.0,
                    "score_p": 1.0,
                    "wf_n": int(pair_n_eff),
                    "n_eff": int(pair_n_eff),
                    "quality_band": str(band),
                    "source": "interleague_league_pair_undirected_v1",
                }
            key_bo = f"league_pair:{und_lg}|bo:{int(bo_i)}|inter:1"
            if key_bo not in out:
                out[key_bo] = {
                    "score_bias": 0.0,
                    "score_se": se,
                    "score_var": var,
                    "score_z": 0.0,
                    "score_p": 1.0,
                    "wf_n": int(pair_n_eff),
                    "n_eff": int(pair_n_eff),
                    "quality_band": str(band),
                    "source": "interleague_league_pair_undirected_v1",
                }
        key_inter = f"league_pair:{und_lg}|inter:1"
        if key_inter not in out:
            out[key_inter] = {
                "score_bias": 0.0,
                "score_se": se,
                "score_var": var,
                "score_z": 0.0,
                "score_p": 1.0,
                "wf_n": int(pair_n_eff),
                "n_eff": int(pair_n_eff),
                "quality_band": str(band),
                "source": "interleague_league_pair_undirected_v1",
            }
        key_plain = f"league_pair:{und_lg}"
        if key_plain not in out:
            out[key_plain] = {
                "score_bias": 0.0,
                "score_se": se,
                "score_var": var,
                "score_z": 0.0,
                "score_p": 1.0,
                "wf_n": int(pair_n_eff),
                "n_eff": int(pair_n_eff),
                "quality_band": str(band),
                "source": "interleague_league_pair_undirected_v1",
            }

    fdr_q_alpha = 0.10
    fdr_z_soft = 2.20
    try:
        cand: List[Tuple[str, float]] = []
        for key, node in out.items():
            if not isinstance(node, dict):
                continue
            src = str(node.get("source") or "").strip().lower()
            if "undirected" in src:
                continue
            if "|inter:1" not in str(key):
                continue
            if ("->" not in str(key)) or (
                not (
                    str(key).startswith("league_pair:")
                    or str(key).startswith("tier_pair:")
                    or str(key).startswith("inter:")
                )
            ):
                continue
            p = _fnum(node.get("score_p"))
            if p is None or (not math.isfinite(float(p))):
                continue
            cand.append((str(key), float(max(0.0, min(1.0, p)))))
        m = len(cand)
        if m > 0:
            srt = sorted(cand, key=lambda kv: float(kv[1]))
            p_cut = -1.0
            cut_rank = 0
            for i, (_, p) in enumerate(srt, start=1):
                thr = float((fdr_q_alpha * float(i)) / float(max(1, m)))
                if float(p) <= thr:
                    p_cut = float(p)
                    cut_rank = int(i)
            q_map: Dict[str, float] = {}
            q_prev = 1.0
            for i in range(m, 0, -1):
                key, p = srt[i - 1]
                q_i = float((p * float(m)) / float(max(1, i)))
                q_prev = float(min(q_prev, q_i))
                q_map[str(key)] = float(max(0.0, min(1.0, q_prev)))
            for key, p in cand:
                node = out.get(key) if isinstance(out.get(key), dict) else {}
                if not isinstance(node, dict):
                    continue
                qv = float(q_map.get(str(key), 1.0))
                reject = bool(cut_rank > 0 and float(p) <= float(p_cut))
                node["score_q"] = float(qv)
                node["fdr_q_alpha"] = float(fdr_q_alpha)
                node["fdr_reject"] = bool(reject)
                node["fdr_rank"] = int(next((idx for idx, (kk, _) in enumerate(srt, start=1) if str(kk) == str(key)), m))
                if not reject:
                    b0 = float(_fnum(node.get("score_bias")) or 0.0)
                    z0 = float(abs(float(_fnum(node.get("score_z")) or 0.0)))
                    sh = float(max(0.10, min(0.85, z0 / max(1.0, fdr_z_soft))))
                    node["score_bias_pre_fdr"] = float(b0)
                    node["fdr_shrink"] = float(sh)
                    node["score_bias"] = float(b0 * sh)
                else:
                    node["fdr_shrink"] = 1.0
                out[key] = node
    except Exception:
        pass

    try:
        done_sym: set[str] = set()
        for key in list(out.keys()):
            ks = str(key)
            if ("->" not in ks) or (not (ks.startswith(("league_pair:", "tier_pair:", "inter:")))):
                continue
            if "|inter:1" not in ks:
                continue
            node_ab = out.get(ks) if isinstance(out.get(ks), dict) else {}
            if not isinstance(node_ab, dict):
                continue
            src = str(node_ab.get("source") or "").strip().lower()
            if "undirected" in src:
                continue
            prefix, rest_all = ([*ks.split(":", 1), ""])[:2]
            pair_tok, rest = ([*rest_all.split("|", 1), ""])[:2]
            if "->" not in pair_tok:
                continue
            a, b = ([*pair_tok.split("->", 1), ""])[:2]
            if (not a) or (not b):
                continue
            key_ba = f"{prefix}:{b}->{a}" + (f"|{rest}" if rest else "")
            if key_ba not in out:
                continue
            pair_id = "||".join(sorted([ks, key_ba]))
            if pair_id in done_sym:
                continue
            done_sym.add(pair_id)
            node_ba = out.get(key_ba) if isinstance(out.get(key_ba), dict) else {}
            if not isinstance(node_ba, dict):
                continue
            bab = float(_fnum(node_ab.get("score_bias")) or 0.0)
            bba = float(_fnum(node_ba.get("score_bias")) or 0.0)
            anti = float((bab - bba) * 0.5)
            node_ab["score_bias_pre_sym"] = float(bab)
            node_ba["score_bias_pre_sym"] = float(bba)
            node_ab["score_bias"] = float(anti)
            node_ba["score_bias"] = float(-anti)
            node_ab["symmetry_adjusted_v1"] = True
            node_ba["symmetry_adjusted_v1"] = True
            out[ks] = node_ab
            out[key_ba] = node_ba
    except Exception:
        pass

    return out
