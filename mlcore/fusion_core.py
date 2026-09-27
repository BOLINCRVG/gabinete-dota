"""Fusion helpers for Team vs Players probability blending."""

from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd
import numpy as np

from mlcore.ml_fusion import FusionConfig, fuse_team_players_probs
from mlcore.utils import clip_prob, safe_float as _safe_float


def _infer_coverage(n_team: float, n_players: float) -> float:
    nt = max(0.0, _safe_float(n_team, 0.0))
    np_ = max(0.0, _safe_float(n_players, 0.0))
    den = nt + np_
    if den <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, np_ / den)))


def fuse_with_coherence_guard(
    *,
    p_team: float,
    p_players: float,
    n_team: float,
    n_players: float,
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    coverage: float | None = None,
    transfer_signal: float = 0.0,
    season_phase: str = "auto",
    team_weight_early: float = 0.58,
    team_weight_mid: float = 0.75,
    team_weight_playoffs: float = 0.82,
    early_season_team_weight: float = 0.55,
    season_knee_games: float = 18.0,
    coverage_power: float = 1.5,
    transfer_boost: float = 0.35,
    divergence_pp_cap: float = 22.0,
    divergence_low_coverage: float = 0.30,
    divergence_shrink: float = 0.35,
    lineup_continuity: float | None = None,
    roster_changed: bool = False,
) -> Dict[str, Any]:
    nt_raw = max(0.0, float(_safe_float(n_team, 0.0)))
    np_raw = max(0.0, float(_safe_float(n_players, 0.0)))
    cov_raw = _infer_coverage(nt_raw, np_raw)
    cov_actual = _safe_float(coverage, float("nan"))
    if not math.isfinite(cov_actual):
        cov_actual = cov_raw
        cov_source = "inferred_raw"
    else:
        cov_actual = max(0.0, min(1.0, float(cov_actual)))
        cov_source = "explicit"
    # Keep fitted scales from rewriting actual lineup completeness.
    cov = float(max(0.0, min(1.0, min(float(cov_actual), float(cov_raw) if math.isfinite(cov_raw) else float(cov_actual)))))

    ts_raw = max(0.0, float(_safe_float(team_scale, 1.0)))
    ps_raw = max(0.0, float(_safe_float(players_scale, 1.0)))
    # Scales are evidence multipliers, but only the covered portion of the
    # lineup should activate them aggressively.
    team_scale_eff = float(max(0.0, 1.0 + (ts_raw - 1.0) * cov))
    players_scale_eff = float(max(0.0, 1.0 + (ps_raw - 1.0) * cov))

    # Roster continuity adjustment:
    # if continuity is low (new roster), reduce trust in team-history and
    # increase relative trust in players.
    cont = _safe_float(lineup_continuity, float("nan"))
    if not math.isfinite(cont):
        cont = 1.0
    cont = max(0.0, min(1.0, float(cont)))
    nt_eff = float(nt_raw * team_scale_eff)
    np_eff = float(np_raw * players_scale_eff)
    # continuity 1.0 => factor 1.0 ; continuity 0.0 => factor 0.35
    team_hist_factor = 0.35 + 0.65 * cont
    if bool(roster_changed):
        team_hist_factor *= 0.85
    nt_eff *= float(max(0.10, min(1.0, team_hist_factor)))
    # boost players evidence when continuity is low.
    players_boost = 1.0 + (1.0 - cont) * 0.80
    if bool(roster_changed):
        players_boost *= 1.10
    np_eff *= float(max(1.0, players_boost))
    ts_eff = max(0.0, min(1.0, max(float(_safe_float(transfer_signal, 0.0)), 1.0 - cont)))
    # Players amplification should follow actual covered evidence, not create
    # synthetic coverage from a fitted scale.
    players_weight_mult = float(
        max(
            0.25,
            min(
                3.0,
                1.0 + (((players_scale_eff * max(1.0, players_boost)) - 1.0) * cov),
            ),
        )
    )

    cfg = FusionConfig(
        team_weight_early=float(team_weight_early),
        team_weight_mid=float(team_weight_mid),
        team_weight_playoffs=float(team_weight_playoffs),
        early_season_team_weight=float(early_season_team_weight),
        season_knee_games=max(1.0, float(season_knee_games)),
        coverage_power=max(0.0, float(coverage_power)),
        transfer_boost=max(0.0, float(transfer_boost)),
    )

    out = fuse_team_players_probs(
        p_team=float(p_team),
        p_players=float(p_players),
        coverage=float(cov),
        players_weight_mult=float(players_weight_mult),
        team_games_min=float(max(0.0, nt_eff)),
        transfer_signal=float(max(0.0, min(1.0, ts_eff))),
        season_phase=str(season_phase or "auto"),
        cfg=cfg,
    )
    out["coverage_source"] = str(cov_source)
    out["coverage_actual"] = float(cov_actual)
    out["coverage_inferred_raw"] = float(cov_raw)
    out["coverage_used"] = float(cov)
    out["team_scale_raw"] = float(ts_raw)
    out["players_scale_raw"] = float(ps_raw)
    out["team_scale_eff"] = float(team_scale_eff)
    out["players_scale_eff"] = float(players_scale_eff)
    out["lineup_continuity"] = float(cont)
    out["roster_changed"] = bool(roster_changed)
    out["team_hist_factor"] = float(max(0.10, min(1.0, team_hist_factor)))
    out["players_boost"] = float(max(1.0, players_boost))
    out["n_team_eff_adj"] = float(max(0.0, nt_eff))
    out["n_players_eff_adj"] = float(max(0.0, np_eff))

    # Coherence guard: when players and team diverge too much and players coverage is
    # still low, pull fused probability closer to team baseline.
    pt = _safe_float(p_team, float("nan"))
    pp = _safe_float(p_players, float("nan"))
    pf = _safe_float(out.get("p_fused"), float("nan"))
    if (
        math.isfinite(pt)
        and math.isfinite(pp)
        and math.isfinite(pf)
        and 0.0 < pt < 1.0
        and 0.0 < pp < 1.0
        and 0.0 < pf < 1.0
    ):
        dpp = abs(pp - pt) * 100.0
        cap = max(1.0, float(_safe_float(divergence_pp_cap, 22.0)))
        low_cov = max(0.0, min(1.0, float(_safe_float(divergence_low_coverage, 0.30))))
        shr = max(0.0, min(1.0, float(_safe_float(divergence_shrink, 0.35))))
        if dpp > cap and cov < low_cov and shr > 0.0:
            excess = (dpp - cap) / cap
            pull = min(1.0, shr * max(0.0, excess))
            pf2 = pt + (pf - pt) * (1.0 - pull)
            out["p_fused"] = float(clip_prob(float(pf2)))
            out["coherence_guard"] = {
                "applied": True,
                "dpp": float(dpp),
                "cap_pp": float(cap),
                "coverage": float(cov),
                "pull": float(pull),
            }
        else:
            out["coherence_guard"] = {
                "applied": False,
                "dpp": float(dpp),
                "cap_pp": float(cap),
                "coverage": float(cov),
                "pull": 0.0,
            }
    return out


def _win_series(df: pd.DataFrame) -> pd.Series:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    if "win" in df.columns:
        w = pd.to_numeric(df.get("win"), errors="coerce")
        if w.notna().any():
            return w
    return pd.to_numeric(df.get("result"), errors="coerce")


def season_form_blend(
    *,
    p_model: float,
    df_team_a: pd.DataFrame,
    df_team_b: pd.DataFrame,
    enabled: bool = True,
    max_weight: float = 0.75,
    knee_games: float = 6.0,
    beta_prior: float = 2.0,
    signal_power: float = 1.0,
    elite_same_league_parity_strength: float = 0.35,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "applied": False,
        "p_model": float(p_model) if math.isfinite(float(_safe_float(p_model, float("nan")))) else None,
        "p_form": None,
        "p_blended": float(p_model) if math.isfinite(float(_safe_float(p_model, float("nan")))) else None,
        "w_form": 0.0,
        "n_a": 0,
        "n_b": 0,
        "wins_a": 0,
        "wins_b": 0,
        "reason": "disabled",
    }
    pm = _safe_float(p_model, float("nan"))
    if (not bool(enabled)) or (not math.isfinite(pm)) or pm <= 0.0 or pm >= 1.0:
        return out

    wa = _win_series(df_team_a)
    wb = _win_series(df_team_b)
    wa = wa[wa.notna()]
    wb = wb[wb.notna()]
    n_a = int(wa.shape[0])
    n_b = int(wb.shape[0])
    if n_a <= 0 or n_b <= 0:
        out["reason"] = "missing_sample"
        out["n_a"] = n_a
        out["n_b"] = n_b
        return out

    wins_a = int((wa > 0.5).sum())
    wins_b = int((wb > 0.5).sum())
    bp = max(0.5, float(_safe_float(beta_prior, 2.0)))
    p_a = (float(wins_a) + bp) / (float(n_a) + 2.0 * bp)
    p_b = (float(wins_b) + bp) / (float(n_b) + 2.0 * bp)
    p_a = float(clip_prob(p_a))
    p_b = float(clip_prob(p_b))
    logit_diff = math.log(p_a / (1.0 - p_a)) - math.log(p_b / (1.0 - p_b))

    # Elite same-league parity shrink:
    # if both teams are strong in the same dominant league, reduce overconfidence.
    try:
        lg_a = ""
        lg_b = ""
        if isinstance(df_team_a, pd.DataFrame) and (not df_team_a.empty) and ("league" in df_team_a.columns):
            va = df_team_a["league"].dropna().astype(str).str.strip()
            if not va.empty:
                lg_a = str(va.value_counts().index[0]).strip().casefold()
        if isinstance(df_team_b, pd.DataFrame) and (not df_team_b.empty) and ("league" in df_team_b.columns):
            vb = df_team_b["league"].dropna().astype(str).str.strip()
            if not vb.empty:
                lg_b = str(vb.value_counts().index[0]).strip().casefold()
        same_league = bool(lg_a and lg_b and lg_a == lg_b)
    except Exception:
        same_league = False

    if same_league:
        p_weak = min(float(p_a), float(p_b))
        if p_weak >= 0.60:
            parity_signal = max(0.0, min(1.0, (p_weak - 0.60) / 0.25))
            parity_strength = max(0.0, min(0.80, float(_safe_float(elite_same_league_parity_strength, 0.35))))
            shrink = parity_strength * parity_signal
            logit_diff = float(logit_diff) * float(1.0 - shrink)
    p_form = float(1.0 / (1.0 + math.exp(-logit_diff)))

    n_min = float(min(n_a, n_b))
    knee = max(1.0, float(_safe_float(knee_games, 6.0)))
    sample_factor = n_min / (n_min + knee)
    sig = abs(p_form - 0.5) / 0.5
    sig = max(0.0, min(1.0, sig))
    sp = max(0.1, float(_safe_float(signal_power, 1.0)))
    signal_factor = sig ** sp
    w_cap = max(0.0, min(0.95, float(_safe_float(max_weight, 0.75))))
    w_form = w_cap * sample_factor * signal_factor
    w_form = max(0.0, min(0.95, float(w_form)))

    p_blended = float(clip_prob((1.0 - w_form) * pm + w_form * p_form))

    out.update(
        {
            "applied": True,
            "p_form": float(p_form),
            "p_blended": float(p_blended),
            "w_form": float(w_form),
            "n_a": int(n_a),
            "n_b": int(n_b),
            "wins_a": int(wins_a),
            "wins_b": int(wins_b),
            "p_a_smooth": float(p_a),
            "p_b_smooth": float(p_b),
            "same_league": bool(same_league),
            "reason": "ok",
        }
    )
    return out


def current_season_override(
    *,
    p_model: float,
    p_form: float,
    w_form: float,
    n_a: int,
    n_b: int,
    min_games_gate: int = 8,
    divergence_gate_pp: float = 8.0,
    max_weight: float = 0.88,
) -> Dict[str, Any]:
    """Reinforce current-season signal when sample is decent and divergence is high.

    Returns dict with:
    - applied, reason
    - p_out
    - w_used (final weight for p_form)
    - w_extra
    """
    pm = _safe_float(p_model, float("nan"))
    pf = _safe_float(p_form, float("nan"))
    wf = max(0.0, min(0.95, _safe_float(w_form, 0.0)))
    if (not math.isfinite(pm)) or (not math.isfinite(pf)) or pm <= 0.0 or pm >= 1.0 or pf <= 0.0 or pf >= 1.0:
        return {"applied": False, "reason": "invalid_probs", "p_out": pm, "w_used": wf, "w_extra": 0.0}

    n_min = max(0, min(int(n_a or 0), int(n_b or 0)))
    if n_min < int(max(1, min_games_gate)):
        return {"applied": False, "reason": "low_sample", "p_out": pm, "w_used": wf, "w_extra": 0.0}

    div_pp = abs(float(pm) - float(pf)) * 100.0
    gate = max(0.0, float(divergence_gate_pp))
    if div_pp <= gate:
        return {"applied": False, "reason": "low_divergence", "p_out": pm, "w_used": wf, "w_extra": 0.0}

    # More games + larger divergence => stronger reinforcement.
    sample_sig = float(n_min) / float(n_min + 10.0)
    div_sig = max(0.0, min(1.0, (div_pp - gate) / 22.0))
    target = 0.40 + 0.48 * (0.55 * sample_sig + 0.45 * div_sig)
    w_cap = max(0.0, min(0.95, float(max_weight)))
    w_used = max(wf, min(w_cap, target))
    p_out = float(clip_prob((1.0 - w_used) * float(pm) + w_used * float(pf)))

    return {
        "applied": bool(abs(w_used - wf) > 1e-12),
        "reason": "ok",
        "p_out": float(p_out),
        "w_used": float(w_used),
        "w_extra": float(max(0.0, w_used - wf)),
        "n_min": int(n_min),
        "divergence_pp": float(div_pp),
    }


def _smooth_binary_rate(df: pd.DataFrame, col: str, beta: float = 2.0) -> tuple[float, int]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return float("nan"), 0
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    n = int(s.shape[0])
    if n <= 0:
        return float("nan"), 0
    k = int((s > 0.5).sum())
    bp = max(0.5, float(beta))
    p = (float(k) + bp) / (float(n) + 2.0 * bp)
    return float(clip_prob(p)), n


def _smooth_win_rate(df: pd.DataFrame, beta: float = 2.0) -> tuple[float, int]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return float("nan"), 0
    s = pd.Series(dtype=float)
    if "win" in df.columns:
        s = pd.to_numeric(df["win"], errors="coerce").dropna()
    elif "result" in df.columns:
        s = pd.to_numeric(df["result"], errors="coerce").dropna()
    else:
        return float("nan"), 0
    n = int(s.shape[0])
    if n <= 0:
        return float("nan"), 0
    k = int((s > 0.5).sum())
    bp = max(0.5, float(beta))
    p = (float(k) + bp) / (float(n) + 2.0 * bp)
    return float(clip_prob(p)), n


def _share_stat(df: pd.DataFrame, num_col: str, den_col: str, beta: float = 2.0) -> tuple[float, int]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return float("nan"), 0
    if num_col not in df.columns or den_col not in df.columns:
        return float("nan"), 0
    num = pd.to_numeric(df[num_col], errors="coerce")
    den = pd.to_numeric(df[den_col], errors="coerce")
    good = num.notna() & den.notna()
    if int(good.sum()) <= 0:
        return float("nan"), 0
    num_sum = float(num[good].sum())
    den_sum = float(den[good].sum())
    bp = max(0.5, float(beta))
    p = (num_sum + bp) / (num_sum + den_sum + 2.0 * bp)
    return float(clip_prob(p)), int(good.sum())


def _pair_prob(p_a: float, p_b: float) -> float:
    if not (math.isfinite(float(p_a)) and math.isfinite(float(p_b))):
        return float("nan")
    a = max(1e-6, float(p_a))
    b = max(1e-6, float(p_b))
    return float(clip_prob(a / (a + b)))


def hybrid_momentum_blend(
    *,
    p_model: float,
    df_team_a: pd.DataFrame,
    df_team_b: pd.DataFrame,
    enabled: bool = True,
    max_weight: float = 0.70,
    knee_games: float = 12.0,
    momentum_games: int = 10,
    beta_prior: float = 2.0,
    agree_damp: float = 1.0,
) -> Dict[str, Any]:
    """Mistura o preco do modelo com os "pilares" de forma recente.

    `agree_damp` controla QUANTO a concordancia com o modelo aumenta o peso
    dado a forma. 1.0 e o comportamento de sempre; 0.0 desliga esse efeito.
    O padrao e 1.0 de proposito -- este botao existe para ser MEDIDO antes de
    ser mexido, nao para mudar o preco por conta propria.

    POR QUE ELE EXISTE (medido em 21/09/2026). Comparando as previsoes fora da
    amostra com o resultado real, a forma recente AINDA acrescenta depois deste
    estagio. E o ganho nao esta espalhado -- esta concentrado exatamente onde o
    `agree` manda calar:

        mapa 1, forma CONCORDA com o modelo ... +0,004 [-0,004, +0,012]  nada
        mapa 1, forma DISCORDA do modelo ...... +0,037 [+0,018, +0,058]  real

    Faz sentido: quando a forma concorda, ela nao tem nada a acrescentar; a
    informacao nova esta na discordancia. A formula de hoje pesa ao contrario.
    """
    out: Dict[str, Any] = {
        "applied": False,
        "enabled": bool(enabled),
        "p_model": float(p_model) if math.isfinite(float(_safe_float(p_model, float("nan")))) else None,
        "p_hybrid": None,
        "p_blended": float(p_model) if math.isfinite(float(_safe_float(p_model, float("nan")))) else None,
        "w_hybrid": 0.0,
        "pillars": {},
        "reason": "disabled",
    }
    pm = _safe_float(p_model, float("nan"))
    if (not bool(enabled)) or (not math.isfinite(pm)) or pm <= 0.0 or pm >= 1.0:
        return out

    a = df_team_a if isinstance(df_team_a, pd.DataFrame) else pd.DataFrame()
    b = df_team_b if isinstance(df_team_b, pd.DataFrame) else pd.DataFrame()
    a_last = a.tail(int(max(1, momentum_games))).copy()
    b_last = b.tail(int(max(1, momentum_games))).copy()

    p_mom_a, n_mom_a = _smooth_win_rate(a_last, beta=float(beta_prior))
    p_mom_b, n_mom_b = _smooth_win_rate(b_last, beta=float(beta_prior))
    p_mom = _pair_prob(p_mom_a, p_mom_b)
    n_mom = min(int(n_mom_a), int(n_mom_b))

    p_wr_a, n_wr_a = _smooth_win_rate(a, beta=float(beta_prior))
    p_wr_b, n_wr_b = _smooth_win_rate(b, beta=float(beta_prior))
    p_wr = _pair_prob(p_wr_a, p_wr_b)
    n_wr = min(int(n_wr_a), int(n_wr_b))

    early_cols = ["firstblood", "firsttower", "firstdragon"]
    p_early_parts: list[float] = []
    n_early_parts: list[int] = []
    for c in early_cols:
        pa, na = _smooth_binary_rate(a, c, beta=float(beta_prior))
        pb, nb = _smooth_binary_rate(b, c, beta=float(beta_prior))
        p_col = _pair_prob(pa, pb)
        if math.isfinite(float(p_col)):
            p_early_parts.append(float(p_col))
            n_early_parts.append(min(int(na), int(nb)))
    p_early = float(np.mean(p_early_parts)) if p_early_parts else float("nan")
    n_early = int(np.mean(n_early_parts)) if n_early_parts else 0

    p_drg_a, n_drg_a = _share_stat(a, "dragons_for", "dragons_against", beta=float(beta_prior))
    p_drg_b, n_drg_b = _share_stat(b, "dragons_for", "dragons_against", beta=float(beta_prior))
    p_drg = _pair_prob(p_drg_a, p_drg_b)
    n_drg = min(int(n_drg_a), int(n_drg_b))
    p_bar_a, n_bar_a = _share_stat(a, "barons_for", "barons_against", beta=float(beta_prior))
    p_bar_b, n_bar_b = _share_stat(b, "barons_for", "barons_against", beta=float(beta_prior))
    p_bar = _pair_prob(p_bar_a, p_bar_b)
    n_bar = min(int(n_bar_a), int(n_bar_b))
    macro_vals = [x for x in [p_drg, p_bar] if math.isfinite(float(x))]
    macro_ns = [n for x, n in [(p_drg, n_drg), (p_bar, n_bar)] if math.isfinite(float(x))]
    p_macro = float(np.mean(macro_vals)) if macro_vals else float("nan")
    n_macro = int(np.mean(macro_ns)) if macro_ns else 0

    pillars = {
        "momentum": {"p": p_mom, "n": int(n_mom)},
        "winrate": {"p": p_wr, "n": int(n_wr)},
        "early": {"p": p_early, "n": int(n_early)},
        "macro": {"p": p_macro, "n": int(n_macro)},
    }

    items: list[tuple[str, float, float, int]] = []
    for k, v in pillars.items():
        p = float(v.get("p", float("nan")))
        n = int(v.get("n", 0) or 0)
        if not (math.isfinite(p) and 0.0 < p < 1.0 and n > 0):
            continue
        sig = max(0.0, min(1.0, abs(p - 0.5) / 0.5))
        w = (math.sqrt(float(n)) * (0.25 + 0.75 * sig))
        items.append((k, p, w, n))

    if not items:
        out["reason"] = "no_pillars"
        out["pillars"] = pillars
        return out

    wsum = float(sum(it[2] for it in items))
    if wsum <= 0.0:
        out["reason"] = "zero_weight"
        out["pillars"] = pillars
        return out

    p_h = float(sum(it[1] * it[2] for it in items) / wsum)
    n_eff = float(np.mean([it[3] for it in items]))
    kng = max(1.0, float(knee_games))
    sample_factor = n_eff / (n_eff + kng)
    agree = float(np.mean([max(0.0, 1.0 - abs(it[1] - pm) / 0.5) for it in items]))
    w_cap = max(0.0, min(0.95, float(max_weight)))
    # `fator_concordancia` = 1.0 quando agree_damp=0 (a concordancia nao pesa),
    # e (0.35 + 0.65*agree) quando agree_damp=1 (o comportamento de sempre).
    d = max(0.0, min(1.0, _safe_float(agree_damp, 1.0)))
    fator_concordancia = (1.0 - d) + d * (0.35 + 0.65 * agree)
    w_h = float(w_cap * sample_factor * fator_concordancia)
    w_h = max(0.0, min(0.95, w_h))
    p_blended = float(clip_prob((1.0 - w_h) * pm + w_h * p_h))

    out.update(
        {
            "applied": True,
            "p_hybrid": float(p_h),
            "p_blended": float(p_blended),
            "w_hybrid": float(w_h),
            "n_eff": float(n_eff),
            "agree_to_model": float(agree),
            "agree_damp": float(d),
            "fator_concordancia": float(fator_concordancia),
            "pillars": pillars,
            "reason": "ok",
        }
    )
    return out
