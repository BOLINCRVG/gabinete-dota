"""Score e análise temporal para os candidatos do Resumo.

O score premia tendência recente, não só média histórica.
Exemplo: 10/40 total MAS 30/30 recente → "recovering" → bonus alto.
Exemplo: 38/40 total MAS 0/5 recente → "declining" → penalty.

Componentes do score (0-100):
  conf_pts         0-30  Confiança (amostra+acordo)
  edge_pts         0-30  Edge vs odd da casa (ou neutro 15 se sem odd)
  roi_pts          0-25  ROI histórico do mercado nas suas planilhas
  trend_bonus    -10+10  Trend recente: recovering/improving/flat/declining
  anti_loss_pen    0-15  Penalidade se mercado tem ROI < -10% no seu hist
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from app_core.picks_anti_loss_core import (
    BAD_MARKET_ROI,
    market_historical_roi,
)
from app_core.resumo_core import (
    edge_pct,
    market_confidence_parts,
    odd_to_prob,
)
from app_core.resumo_paste_core import (
    discrete_stake_units,
    kelly_stake_fraction,
)

# ---------------------------------------------------------------------------
# Metric column derivation (for temporal analysis from team_games).
# Each tuple: (col_for, col_against). For time: (col, None).
# ---------------------------------------------------------------------------
_METRIC_PAIR: dict[str, tuple[str, Optional[str]]] = {
    "kills":      ("kills_for", "kills_against"),
    "towers":     ("towers_for", "towers_against"),
    "dragons":    ("dragons_for", "dragons_against"),
    "barons":     ("barons_for", "barons_against"),
    "inhibitors": ("inhibitors_for", "inhibitors_against"),
    "time":       ("game_time_min", None),
}

_METRIC_LABEL_PT: dict[str, str] = {
    "kills":      "Kills",
    "towers":     "Torres",
    "dragons":    "Dragões",
    "barons":     "Barões",
    "inhibitors": "Inibidores",
    "time":       "Tempo",
    "ml":         "ML (Vencedor)",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WindowHitRate:
    label: str           # "Ano todo" / "L15" / "L10" / "L5"
    n: int
    hits: int
    pct: Optional[float]  # 0-100, None when n=0


@dataclass
class TemporalProfile:
    """Hit-rate breakdown across four time windows for one team's perspective."""
    team: str
    windows: list[WindowHitRate]      # [Ano, L15, L10, L5]
    trend: str                        # "recovering"/"improving"/"flat"/"declining"/"volatile"
    trend_label: str                  # human-readable Portuguese
    trend_score: float                # -10 to +10
    summary_line: str                 # e.g. "Ano 58% → L15 62% → L10 70% → L5 80% ↑"


@dataclass
class CandidateScore:
    row: dict
    # Temporal profiles for both teams
    temporal_a: TemporalProfile
    temporal_b: TemporalProfile
    # Score breakdown
    total: int              # 0-100, rounded
    conf_pts: float
    edge_pts: float
    roi_pts: float
    trend_bonus: float
    anti_loss_penalty: float
    # Calculated from provided odd casa
    edge_hist: Optional[float]
    odd_hist: Optional[float]
    odd_casa: Optional[float]
    stake_units: Optional[float]
    # Natural language explanation bullets
    pros: list[str]
    cons: list[str]


# ---------------------------------------------------------------------------
# Temporal profile computation
# ---------------------------------------------------------------------------

def _derive_combined_series(df: pd.DataFrame, metric_key: str) -> Optional[pd.Series]:
    """Derive the combined (total) series for a metric from team_games rows.

    For "kills": kills_for + kills_against (= total kills in the game)
    For "time":  game_time_min directly
    """
    pair = _METRIC_PAIR.get(str(metric_key))
    if pair is None:
        return None
    col_for, col_against = pair
    if col_for not in df.columns:
        return None
    s = pd.to_numeric(df[col_for], errors="coerce")
    if col_against and col_against in df.columns:
        s2 = pd.to_numeric(df[col_against], errors="coerce")
        s = s + s2.fillna(0)
    return s


def _hit_rate(series: pd.Series, line: float, side_key: str) -> tuple[int, int]:
    """Return (hits, n) for a given line and side."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return 0, 0
    if str(side_key).strip().lower() == "over":
        hits = int((s > float(line)).sum())
    else:
        hits = int((s < float(line)).sum())
    return hits, n


def _trend_from_windows(windows: list[WindowHitRate]) -> tuple[str, float, str, str]:
    """Derive trend label, score, and summary from the four windows.

    Returns (trend_key, score, trend_label_pt, summary_line).

    Key insight: a recovering bet (bad early, good recent) gets a POSITIVE
    score — don't punish teams that improved through the split.
    """
    valid = [w for w in windows if w.n >= 3 and w.pct is not None]
    if len(valid) < 2:
        return "insufficient", 0.0, "Amostra insuficiente", _summary_line(windows)

    # Use year vs l5 as primary signal (biggest time spread).
    year_w = next((w for w in windows if w.label == "Ano todo"), None)
    l5_w   = next((w for w in windows if w.label == "L5"),       None)
    l10_w  = next((w for w in windows if w.label == "L10"),      None)

    if year_w is None or l5_w is None or year_w.n < 5 or l5_w.n < 3:
        return "flat", 0.0, "Dados insuficientes para tendência", _summary_line(windows)

    year_pct = float(year_w.pct)
    l5_pct   = float(l5_w.pct)
    l10_pct  = float(l10_w.pct) if (l10_w and l10_w.pct is not None) else year_pct

    delta_recent = l5_pct - year_pct   # positive = improving vs season baseline

    # "Recovering": was bad overall, recent is strong.
    # Threshold: L5 > 65% AND L5 > Ano + 15pp
    if l5_pct >= 65.0 and delta_recent >= 15.0:
        return ("recovering", +10.0,
                f"Início fraco, forma recente boa (L5 {l5_pct:.0f}% vs Ano {year_pct:.0f}%)",
                _summary_line(windows))

    # "Improving": consistently getting better window by window.
    if l5_pct > l10_pct > year_pct and delta_recent >= 8.0:
        return ("improving", +5.0,
                f"Tendência crescente (Ano {year_pct:.0f}% → L5 {l5_pct:.0f}%)",
                _summary_line(windows))

    # "Flat": all within 5pp of each other.
    if abs(delta_recent) < 5.0:
        return ("flat", 0.0,
                f"Estável ({year_pct:.0f}% consistente)",
                _summary_line(windows))

    # "Declining": was good, recently worse.
    if l5_pct < year_pct - 8.0:
        label = (f"Forma recente em queda (Ano {year_pct:.0f}% → L5 {l5_pct:.0f}%)")
        score = -5.0 if delta_recent >= -15.0 else -10.0
        return ("declining", score, label, _summary_line(windows))

    # Default: mild variation.
    sign = "↑" if delta_recent > 0 else "↓"
    return ("variable", 0.0,
            f"Variação moderada {sign} ({year_pct:.0f}% → L5 {l5_pct:.0f}%)",
            _summary_line(windows))


def _summary_line(windows: list[WindowHitRate]) -> str:
    parts: list[str] = []
    for w in windows:
        if w.pct is not None:
            parts.append(f"{w.label} {w.pct:.0f}%")
        else:
            parts.append(f"{w.label} —")
    return " → ".join(parts)


def compute_temporal_profile(
    df_team: pd.DataFrame,
    *,
    metric_key: str,
    line: float,
    side_key: str,
    team_name: str,
    full_year_df: Optional[pd.DataFrame] = None,
) -> TemporalProfile:
    """Compute the temporal hit-rate profile for a single team.

    df_team: recorte das janelas RECENTES (L15/L10/L5) — pela regra do usuário,
    em cross-league é o campeonato MAIS RECENTE do time.
    full_year_df: quando fornecido, alimenta a janela "Ano todo" (regra: ano
    inteiro SEM filtro de liga — todas as ligas/torneios). Sem ele, "Ano todo"
    cai no df_team (comportamento antigo).
    """
    if df_team is None or df_team.empty:
        empty_windows = [WindowHitRate(lbl, 0, 0, None) for lbl in ("Ano todo", "L15", "L10", "L5")]
        return TemporalProfile(
            team=team_name, windows=empty_windows,
            trend="insufficient", trend_label="Sem dados",
            trend_score=0.0, summary_line="— sem dados —",
        )

    is_ml = str(metric_key).lower() in ("ml", "ml_a", "ml_b")

    def _series_of(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        d = df.copy()
        if "date" in d.columns:
            d["_d"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["_d"]).sort_values("_d")
        if is_ml:
            s = pd.to_numeric(d.get("win", pd.Series(dtype=float)), errors="coerce")
            if str(side_key).strip().upper() == "B":
                s = s.apply(lambda x: (1.0 - x) if pd.notna(x) else float("nan"))
            return s
        return _derive_combined_series(d, metric_key)

    if is_ml:
        def _ml_hit(s: pd.Series, _line: float, _side: str) -> tuple[int, int]:
            dropped = s.dropna()
            return int((dropped > 0.5).sum()), len(dropped)
        hit_fn = lambda s, _l, _sd: _ml_hit(s, _l, _sd)
    else:
        hit_fn = _hit_rate

    combined = _series_of(df_team)
    combined_year = _series_of(full_year_df)
    if combined_year is None or combined_year.dropna().empty:
        combined_year = combined

    if combined is None or combined.dropna().empty:
        empty_windows = [WindowHitRate(lbl, 0, 0, None) for lbl in ("Ano todo", "L15", "L10", "L5")]
        return TemporalProfile(
            team=team_name, windows=empty_windows,
            trend="insufficient", trend_label="Métrica indisponível",
            trend_score=0.0, summary_line="— métrica indisponível —",
        )

    windows: list[WindowHitRate] = []
    for label, n_last in [("Ano todo", None), ("L15", 15), ("L10", 10), ("L5", 5)]:
        sub = combined_year if n_last is None else combined.iloc[-int(n_last):]
        h, n = hit_fn(sub, float(line), str(side_key))
        pct = (100.0 * h / n) if n > 0 else None
        windows.append(WindowHitRate(label=label, n=n, hits=h, pct=pct))

    trend_key, trend_score, trend_label, summary = _trend_from_windows(windows)
    return TemporalProfile(
        team=team_name, windows=windows,
        trend=trend_key, trend_label=trend_label,
        trend_score=float(trend_score), summary_line=summary,
    )


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _conf_pts(row: dict) -> float:
    """0-30 from market confidence parts."""
    n_liga = int(row.get("Amostra Liga") or 0)
    n_h2h  = int(row.get("Amostra H2H") or 0)
    parts = market_confidence_parts(
        n_liga=n_liga, n_h2h=n_h2h,
        odd_model=row.get("Odd (Modelo)"), odd_hist=row.get("Odd (Hist)"),
    )
    raw = float(parts.get("total", 0.0))
    # Map [0,55] → [0,30].
    return min(30.0, max(0.0, raw / 55.0 * 30.0))


def _edge_pts(edge_hist: Optional[float]) -> float:
    """0-30 from edge vs odd casa. 15 neutral when no odd casa."""
    if edge_hist is None:
        return 15.0   # neutral: no odd casa yet
    if edge_hist > 10.0:
        return 30.0
    if edge_hist > 5.0:
        return 25.0
    if edge_hist > 0.0:
        return 18.0
    if edge_hist > -5.0:
        return 10.0
    return 5.0


def _roi_pts(metric_key: str, side_key: str) -> float:
    """0-25 from user's historical ROI in this market (planilhas)."""
    roi = market_historical_roi(str(metric_key), str(side_key))
    if roi is None:
        return 12.0   # neutral: no data
    if roi >= 15.0:
        return 25.0
    if roi >= 8.0:
        return 20.0
    if roi >= 0.0:
        return 15.0
    if roi >= -10.0:
        return 8.0
    return 3.0


def _anti_loss_penalty(metric_key: str, side_key: str) -> float:
    """0 to -15 from bad-market list."""
    key = (str(metric_key).lower().strip(), str(side_key).strip())
    roi = BAD_MARKET_ROI.get(key)
    if roi is None:
        return 0.0
    if roi <= -20.0:
        return -15.0
    if roi <= -10.0:
        return -8.0
    return -3.0


def _best_temporal_trend(ta: TemporalProfile, tb: TemporalProfile) -> float:
    """Use the better (higher) trend score of the two teams as the bonus."""
    return max(ta.trend_score, tb.trend_score)


def score_candidate(
    row: dict,
    *,
    oc_map: dict[str, float],
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    team_a: str,
    team_b: str,
    df_a_full: Optional[pd.DataFrame] = None,
    df_b_full: Optional[pd.DataFrame] = None,
) -> CandidateScore:
    """Compute a full CandidateScore for one row from the Resumo rows list."""
    row_key = f"{row.get('Mercado','')}|{row.get('Lado','')}|{row.get('Linha','')}"
    metric_key = str(row.get("_metric") or "")
    # Normalize composite keys.
    if metric_key.endswith("_ml_A") or metric_key == "ml_A":
        metric_key = "ml"
    elif metric_key.endswith("_ml_B") or metric_key == "ml_B":
        metric_key = "ml"
    elif metric_key.startswith("zz_mltot_"):
        metric_key = "combo"

    lado = str(row.get("Lado") or "")
    side_key = "Over" if lado.upper() in ("ACIMA", "OVER") else (
               "Under" if lado.upper() in ("ABAIXO", "UNDER") else lado)

    line_raw = row.get("_line")
    try:
        line = float(line_raw) if line_raw is not None else 0.0
    except (TypeError, ValueError):
        line = 0.0

    odd_hist = row.get("Odd (Hist)")
    try:
        odd_hist = float(odd_hist) if odd_hist is not None else None
    except (TypeError, ValueError):
        odd_hist = None

    # 29/10/2026: com o metodo dele, a chance e a do metodo (resumo_metodo_core).
    p_met = row.get("_p_metodo")
    if p_met is not None:
        odd_hist = 1.0 / float(p_met)
    oc = oc_map.get(row_key)
    p_hist = float(p_met) if p_met is not None else odd_to_prob(odd_hist)
    e_hist = edge_pct(p_hist, oc) if (p_hist and oc) else None

    k_frac = kelly_stake_fraction(p_hist, oc, fraction=0.25) if (p_hist and oc and e_hist and e_hist > 0) else None
    stake_u = discrete_stake_units(k_frac)

    # Temporal profiles.
    if metric_key not in ("combo",):
        ta = compute_temporal_profile(df_a, metric_key=metric_key, line=line,
                                       side_key=side_key, team_name=team_a,
                                       full_year_df=df_a_full)
        tb = compute_temporal_profile(df_b, metric_key=metric_key, line=line,
                                       side_key=side_key, team_name=team_b,
                                       full_year_df=df_b_full)
    else:
        empty_w = [WindowHitRate(lbl, 0, 0, None) for lbl in ("Ano todo", "L15", "L10", "L5")]
        ta = TemporalProfile(team=team_a, windows=empty_w, trend="insufficient",
                              trend_label="Combo — sem temporal", trend_score=0.0,
                              summary_line="—")
        tb = TemporalProfile(team=team_b, windows=empty_w, trend="insufficient",
                              trend_label="Combo — sem temporal", trend_score=0.0,
                              summary_line="—")

    if p_met is not None:
        from app_core.resumo_metodo_core import pontos, pros_e_contras
        pt = pontos(row, e_hist)
        pros_m, cons_m = pros_e_contras(row, e_hist, oc)
        return CandidateScore(
            row=row, temporal_a=ta, temporal_b=tb, total=int(round(pt["total"])),
            conf_pts=pt["chance"], edge_pts=pt["valor"], roi_pts=pt["passos"],
            trend_bonus=0.0, anti_loss_penalty=0.0,
            edge_hist=e_hist, odd_hist=odd_hist, odd_casa=oc,
            stake_units=float(stake_u) if stake_u is not None else None,
            pros=pros_m, cons=cons_m,
        )

    # Score components.
    c_pts  = _conf_pts(row)
    e_pts  = _edge_pts(e_hist)
    r_pts  = _roi_pts(metric_key, side_key)
    t_bon  = _best_temporal_trend(ta, tb)
    al_pen = _anti_loss_penalty(metric_key, side_key)

    raw_total = c_pts + e_pts + r_pts + t_bon + al_pen
    total = int(max(0, min(100, round(raw_total))))

    # Build pros / cons bullets for the veredito.
    pros: list[str] = []
    cons: list[str] = []

    roi_hist = market_historical_roi(metric_key, side_key)
    if roi_hist is not None and roi_hist >= 8.0:
        pros.append(f"ROI histórico positivo: {roi_hist:+.1f}% em {side_key} {_METRIC_LABEL_PT.get(metric_key, metric_key)}")
    elif roi_hist is not None and roi_hist < 0.0:
        cons.append(f"ROI histórico negativo: {roi_hist:+.1f}% neste mercado")

    n_liga = int(row.get("Amostra Liga") or 0)
    n_h2h  = int(row.get("Amostra H2H") or 0)
    if n_liga >= 12:
        pros.append(f"Amostra Liga robusta: {n_liga} jogos")
    elif n_liga < 8:
        cons.append(f"Amostra Liga pequena: {n_liga} jogos")
    if n_h2h >= 4:
        pros.append(f"H2H sólido: {n_h2h} confrontos diretos")
    elif n_h2h <= 1:
        cons.append(f"Quase sem H2H: {n_h2h} confronto(s)")

    for tp in (ta, tb):
        if tp.trend == "recovering":
            pros.append(f"{tp.team}: forma em recuperação — {tp.trend_label}")
        elif tp.trend == "improving":
            pros.append(f"{tp.team}: tendência crescente — {tp.trend_label}")
        elif tp.trend == "declining":
            cons.append(f"{tp.team}: forma em queda — {tp.trend_label}")

    gap_pp = row.get("_gap_pp")
    try:
        gap_pp = float(gap_pp) if gap_pp is not None else None
    except (TypeError, ValueError):
        gap_pp = None
    if gap_pp is not None and gap_pp <= 8.0:
        pros.append(f"Modelo e Histórico concordam (Δ {gap_pp:.0f}pp — sinal limpo)")
    elif gap_pp is not None and gap_pp >= 15.0:
        cons.append(f"Modelo e Histórico discordam (Δ {gap_pp:.0f}pp — fontes conflitam)")

    if e_hist is not None and e_hist >= 5.0:
        pros.append(f"Edge vs casa: {e_hist:+.1f}%")
    elif e_hist is not None and e_hist < 0.0:
        cons.append(f"Edge negativo vs casa: {e_hist:+.1f}%")

    return CandidateScore(
        row=row,
        temporal_a=ta, temporal_b=tb,
        total=total,
        conf_pts=c_pts, edge_pts=e_pts, roi_pts=r_pts,
        trend_bonus=t_bon, anti_loss_penalty=al_pen,
        edge_hist=e_hist, odd_hist=odd_hist, odd_casa=oc,
        stake_units=float(stake_u) if stake_u is not None else None,
        pros=pros, cons=cons,
    )


def rank_candidates(
    rows: list[dict],
    *,
    oc_map: dict[str, float],
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    team_a: str,
    team_b: str,
    top_n: int = 5,
    sort_by: str = "score",
    df_a_full: Optional[pd.DataFrame] = None,
    df_b_full: Optional[pd.DataFrame] = None,
) -> list[CandidateScore]:
    """Score all rows and return the top_n ordered by the chosen key.

    sort_by:
      "score"  → highest score first (default)
      "odd"    → smallest Hist odd first (safest/most likely); missing odds last
      "gestao" → biggest suggested stake first; missing stakes last
    """
    scored: list[CandidateScore] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            s = score_candidate(r, oc_map=oc_map, df_a=df_a, df_b=df_b,
                                  team_a=team_a, team_b=team_b,
                                  df_a_full=df_a_full, df_b_full=df_b_full)
            scored.append(s)
        except Exception:
            continue
    sb = str(sort_by or "score").strip().lower()
    if sb == "odd":
        scored.sort(key=lambda c: (c.odd_hist is None, c.odd_hist if c.odd_hist is not None else float("inf")))
    elif sb in ("gestao", "gestão", "stake"):
        scored.sort(key=lambda c: (c.stake_units is None, -(c.stake_units or 0.0)))
    else:
        # 29/10/2026, a regra dele: ENTRA (3+ passos e valor) primeiro, depois ANALISAR / falta odd,
        # depois sem valor; dentro de cada faixa, o score.
        from app_core.resumo_metodo_core import faixa
        scored.sort(key=lambda c: (faixa(c.row, c.edge_hist), -c.total))
    return scored[:int(top_n)]


def generate_veredito(
    ranked: list[CandidateScore],
    *,
    team_a: str,
    team_b: str,
) -> dict:
    """Build the veredito dict for the UI.

    Returns {
        winner_label: str,
        winner_score: int,
        stake_str: str,
        pros: list[str],
        cons: list[str],        # cons of winner (honest view)
        tradeoff: str | None,   # if runner-up is within 5 pts of winner
    }
    """
    if not ranked:
        return {"winner_label": "—", "winner_score": 0, "stake_str": "—",
                "pros": [], "cons": [], "tradeoff": None}

    winner = ranked[0]
    metric_key = str(winner.row.get("_metric") or "")
    if metric_key.endswith("_ml_A") or metric_key == "ml_A":
        metric_key = "ml"
    elif metric_key.endswith("_ml_B") or metric_key == "ml_B":
        metric_key = "ml"
    metric_label = _METRIC_LABEL_PT.get(metric_key, metric_key.title())
    lado = str(winner.row.get("Lado") or "")
    linha = str(winner.row.get("Linha") or "")
    winner_label = f"{metric_label} {lado}" + (f" {linha}" if linha and linha != "-" else "")

    stake_str = (f"{winner.stake_units:g}u" if winner.stake_units else
                 ("preencha Sua odd para calcular stake" if winner.odd_casa is None else "<0.5u"))

    tradeoff: Optional[str] = None
    if len(ranked) >= 2:
        runner = ranked[1]
        r_metric = str(runner.row.get("_metric") or "")
        if r_metric.endswith("_ml_A") or r_metric == "ml_A":
            r_metric = "ml"
        elif r_metric.endswith("_ml_B") or r_metric == "ml_B":
            r_metric = "ml"
        r_label = _METRIC_LABEL_PT.get(r_metric, r_metric.title())
        r_lado = str(runner.row.get("Lado") or "")
        r_linha = str(runner.row.get("Linha") or "")
        r_name = f"{r_label} {r_lado}" + (f" {r_linha}" if r_linha and r_linha != "-" else "")

        if winner.total - runner.total <= 5:
            # Close race — surface the tradeoff.
            runner_roi = market_historical_roi(r_metric, r_lado)
            winner_roi = market_historical_roi(metric_key, lado)
            if runner_roi is not None and winner_roi is not None and runner_roi > winner_roi:
                tradeoff = (
                    f"{r_name} tem ROI histórico maior ({runner_roi:+.1f}% vs "
                    f"{winner_roi:+.1f}%), mas placar está apertado ({winner.total} vs "
                    f"{runner.total} pts). Considere os dois."
                )
            else:
                tradeoff = (
                    f"{r_name} chegou perto ({runner.total} pts vs {winner.total} pts). "
                    f"Ambas as picks são válidas."
                )

    return {
        "winner_label": winner_label,
        "winner_score": winner.total,
        "stake_str": stake_str,
        "pros": winner.pros[:4],
        "cons": winner.cons[:3],
        "tradeoff": tradeoff,
    }


__all__ = [
    "_METRIC_LABEL_PT",
    "CandidateScore",
    "TemporalProfile",
    "WindowHitRate",
    "compute_temporal_profile",
    "generate_veredito",
    "rank_candidates",
    "score_candidate",
]
