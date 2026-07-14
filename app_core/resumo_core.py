"""Core helpers for Resumo tab computations."""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from app_core.calibration_core import calibrate_market_odds
from app_core.shared_formatters import fmt_pct
from mlcore.utils import safe_float_or_none as _safe_float
from app_core.market import combine_ml_totals_probs as core_combine_ml_totals_probs
from app_core.odds_core import (
    format_scope_odds_clean as core_format_scope_odds_clean,
    laplace_odds as core_laplace_odds,
    laplace_pick_odd as core_laplace_pick_odd,
    odd_mix as core_odd_mix,
    odd_to_prob as core_odd_to_prob,
    prob_to_odd as core_prob_to_odd,
)


def counts_ou(df: pd.DataFrame, value_col: str, line: float) -> tuple[int, int, int]:
    """Conta jogos acima, abaixo e total para uma coluna numérica em relação a uma linha."""
    d = df if isinstance(df, pd.DataFrame) else pd.DataFrame([])
    x = pd.to_numeric(d.get(value_col), errors="coerce").dropna()
    n = int(x.shape[0])
    if n <= 0:
        return 0, 0, 0
    over = int((x > float(line)).sum())
    under = int((x < float(line)).sum())
    return over, under, n


def counts_win(df: pd.DataFrame, win_col: str = "win") -> tuple[int, int]:
    """Retorna o total de vitórias e o número de jogos no DataFrame."""
    d = df if isinstance(df, pd.DataFrame) else pd.DataFrame([])
    x = pd.to_numeric(d.get(win_col), errors="coerce").dropna()
    n = int(x.shape[0])
    if n <= 0:
        return 0, 0
    wins = int((x > 0).sum())
    return wins, n


def wr_laplace(df: pd.DataFrame, win_col: str = "win") -> tuple[float, int]:
    """Calcula a taxa de vitória suavizada por Laplace e o número de jogos."""
    wins, n = counts_win(df, win_col=win_col)
    if n <= 0:
        return 0.5, 0
    return float((float(wins) + 1.0) / (float(n) + 2.0)), int(n)


def counts_ml_totals(df: pd.DataFrame, side: str, line: float, value_col: str = "total_kills", win_col: str = "win") -> tuple[int, int, int]:
    """Conta hits de over/under condicionais à vitória do lado especificado para mercados combo ML+Total."""
    d = df if isinstance(df, pd.DataFrame) else pd.DataFrame([])
    wins = pd.to_numeric(d.get(win_col), errors="coerce")
    vals = pd.to_numeric(d.get(value_col), errors="coerce")
    ok = wins.notna() & vals.notna()
    if not bool(ok.any()):
        return 0, 0, 0
    wins = wins[ok]
    vals = vals[ok]
    n = int(vals.shape[0])
    if str(side).strip().upper() == "A":
        ml_hit = wins > 0
    else:
        ml_hit = wins <= 0
    over = int((ml_hit & (vals > float(line))).sum())
    under = int((ml_hit & (vals < float(line))).sum())
    return over, under, n


def odd_to_prob(odd: Any) -> Optional[float]:
    """Converte uma odd decimal para probabilidade implícita."""
    return core_odd_to_prob(odd)


def prob_to_odd(p: Any) -> Optional[float]:
    """Converte uma probabilidade para odd decimal."""
    return core_prob_to_odd(p)


def _guardrail_model_odd(odd_model: Any, *, max_odd: float, req_model: bool) -> bool:
    om = _safe_float(odd_model)
    mx = _safe_float(max_odd)
    mx = float(mx) if mx is not None else 999.0
    if bool(req_model):
        return om is not None and om <= mx
    return True


def market_confidence_parts(*, n_liga: int, n_h2h: int, odd_model: Any, odd_hist: Any) -> dict[str, float]:
    """Return the named contributions that make up the confidence score.

    Each component is bounded to its own cap (sum ≤ 55 of the 100-point
    base). The UI can render these as a tooltip so the user understands why
    a pick scores 47 vs 62.
    """
    parts = {
        "amostra_liga": min(18.0, max(0.0, float(n_liga)) * 2.0),
        "amostra_h2h": min(16.0, max(0.0, float(n_h2h)) * 4.0),
        "modelo": 8.0 if _safe_float(odd_model) is not None else 0.0,
        "historico": 5.0 if _safe_float(odd_hist) is not None else 0.0,
        "acordo": 0.0,
    }
    pm = odd_to_prob(odd_model)
    ph = odd_to_prob(odd_hist)
    if pm is not None and ph is not None:
        parts["acordo"] = max(0.0, 8.0 - (abs(pm - ph) * 100.0))
    parts["total"] = float(sum(v for k, v in parts.items() if k != "total"))
    return parts


def _market_confidence_score(*, n_liga: int, n_h2h: int, odd_model: Any, odd_hist: Any) -> int:
    total = market_confidence_parts(
        n_liga=n_liga, n_h2h=n_h2h, odd_model=odd_model, odd_hist=odd_hist,
    )["total"]
    return int(max(0.0, min(100.0, round(total))))


def model_hist_gap_pp(odd_model: Any, odd_hist: Any) -> Optional[float]:
    """Absolute difference between model and historical probability, in pp.

    Returns None when either side is missing. Use it to flag picks where the
    two sources disagree strongly (>= 15pp is typically "alerta")."""
    pm = odd_to_prob(odd_model)
    ph = odd_to_prob(odd_hist)
    if pm is None or ph is None:
        return None
    return float(abs(pm - ph) * 100.0)


def edge_pct(p_true: Any, odd_casa: Any) -> Optional[float]:
    """Edge in % for a given true probability vs the bookmaker's offered odd.

    Edge = p_true * odd_casa - 1. Positive means the bet has +EV.
    Returns None if either input is missing/invalid.
    """
    p = _safe_float(p_true)
    o = _safe_float(odd_casa)
    if p is None or o is None:
        return None
    if not (0.0 < p < 1.0) or o <= 1.0:
        return None
    return float((p * o - 1.0) * 100.0)


def _mix_obs_from_cal(cal: dict[str, Any]) -> str:
    reason = str((cal or {}).get("reason", "") or "")
    source = str((cal or {}).get("source", "") or "")
    if source == "mix":
        return ""
    if source == "model":
        return "Odd Mix indisponivel; usando Modelo"
    if source == "hist":
        return "Odd Mix indisponivel; usando Historico"
    return reason


def passed_recortes(odds_dict: dict[str, Any], prefix: str, side_key: str, max_odd: float) -> list[str]:
    """Retorna quais recortes temporais possuem odd dentro do limite máximo para um lado e prefixo dados."""
    out: list[str] = []
    mx = _safe_float(max_odd)
    mx = float(mx) if mx is not None else 999.0
    for rec in ("Ano", "15", "10", "5"):
        fv = _safe_float(odds_dict.get(f"{prefix}_{rec}_{side_key}"))
        if fv is None:
            continue
        if fv <= mx:
            out.append(rec)
    return out


def sample_gate_ok(*, n_liga: int, n_h2h: int, req_liga: bool, req_h2h: bool, min_sample_liga: int, min_sample_h2h: int) -> bool:
    """Verifica se os tamanhos de amostra da liga e H2H satisfazem os mínimos configurados."""
    if bool(req_liga) and int(n_liga) < int(min_sample_liga):
        return False
    if bool(req_h2h) and int(n_h2h) < int(min_sample_h2h):
        return False
    return True


def fmt_recortes_disp(recortes: list[str]) -> str:
    """Formata a lista de recortes aprovados como string separada por vírgulas ou '-' se vazia."""
    return ",".join(str(x) for x in recortes if str(x).strip()) if recortes else "-"


def make_pricing_adapters(*, laplace_pick: str, w_model: float, w_hist: float, w_stats: float, floor_odd: float = 1.10) -> dict[str, Callable[..., Any]]:
    """Cria closures de precificação (Laplace, mix de odds) parametrizadas com os pesos e pick configurados."""
    def _laplace_odds(k: int, n: int) -> Optional[float]:
        return core_laplace_odds(k, n, floor_odd=float(floor_odd))

    def _laplace_pick_odd(odds_dict: dict, side_key: str) -> Optional[float]:
        return core_laplace_pick_odd(odds_dict, side_key, str(laplace_pick or "Liga 15"))

    def _odd_mix(odd_model: Any, odd_hist: Any, stats_pct: Any, model_boost: float = 1.0) -> Optional[float]:
        return core_odd_mix(
            odd_model,
            odd_hist,
            stats_pct,
            w_model=float(w_model),
            w_hist=float(w_hist),
            w_stats=float(w_stats),
            model_boost=float(model_boost),
        )

    def _fmt_laplace_filtered(odds_dict: dict, prefix: str, side_key: str, odd_max: float) -> str:
        # User-facing, jargon-free, scope-correct (Liga shows Liga, H2H shows H2H).
        # No "pick" concept — odds shown whenever within the limit.
        return core_format_scope_odds_clean(odds_dict, prefix, side_key, odd_max)

    return {
        "laplace_odds_fn": _laplace_odds,
        "laplace_pick_odd_fn": _laplace_pick_odd,
        "odd_mix_fn": _odd_mix,
        "fmt_laplace_filtered_fn": _fmt_laplace_filtered,
    }


_CONF_ALIASES = ("Confiança", "Confianca", "ConfianÃ§a", "ConfianÃƒÂ§a")





def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _get_operational_confidence_score(guardrail_cfg: Optional[dict]) -> Optional[float]:
    if not isinstance(guardrail_cfg, dict):
        return None

    vals: list[float] = []
    raw_vals = [
        guardrail_cfg.get("operational_confidence_score"),
        guardrail_cfg.get("operational_confidence_marginal"),
        guardrail_cfg.get("operational_confidence_layer"),
    ]
    for raw in raw_vals:
        fv = _safe_float(raw)
        if fv is None:
            continue
        # Some traces emit confidence in 0-1 scale.
        if 0.0 <= fv <= 1.0:
            fv = fv * 100.0
        vals.append(float(max(0.0, min(100.0, fv))))

    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def _blend_confidence(base_score: int, guardrail_cfg: Optional[dict]) -> int:
    op = _get_operational_confidence_score(guardrail_cfg)
    if op is None:
        return int(base_score)

    w = 0.25
    if isinstance(guardrail_cfg, dict):
        wf = _safe_float(guardrail_cfg.get("operational_confidence_weight"))
        if wf is not None:
            w = float(max(0.0, min(1.0, wf)))

    blended = (1.0 - w) * float(base_score) + w * float(op)
    return int(max(0, min(100, round(blended))))


def _call_odd_mix_3(
    odd_mix_fn: Callable[..., Optional[float]],
    odd_model: Any,
    odd_hist: Any,
    stats_pct: Any,
) -> Optional[float]:
    try:
        return odd_mix_fn(odd_model, odd_hist, stats_pct)
    except TypeError:
        # Combo lambda may require model_boost as 4th positional argument.
        try:
            return odd_mix_fn(odd_model, odd_hist, stats_pct, 1.0)
        except Exception:
            return None
    except Exception:
        return None


def _normalize_confidence_key(row: dict[str, Any]) -> None:
    for k in _CONF_ALIASES:
        if k in row:
            row["Confiança"] = row.get(k)
            break


def _refresh_row_confidence(row: dict[str, Any], guardrail_cfg: Optional[dict]) -> None:
    n_liga = _safe_int(row.get("Amostra Liga"), 0)
    n_h2h = _safe_int(row.get("Amostra H2H"), 0)
    parts = market_confidence_parts(
        n_liga=n_liga,
        n_h2h=n_h2h,
        odd_model=row.get("Odd (Modelo)"),
        odd_hist=row.get("Odd (Hist)"),
    )
    row["Confiança"] = _blend_confidence(int(parts["total"]), guardrail_cfg)
    row["_conf_parts"] = parts
    row["_gap_pp"] = model_hist_gap_pp(row.get("Odd (Modelo)"), row.get("Odd (Hist)"))


def _recalibrate_rows(
    rows: list[dict],
    *,
    max_odd: float,
    req_model: bool,
    odd_mix_fn: Callable[..., Optional[float]],
    guardrail_cfg: Optional[dict],
    recalibrate_mix: bool,
) -> list[dict]:
    out: list[dict] = []
    mx = _safe_float(max_odd)
    mx = float(mx) if mx is not None else 999.0

    for row in rows or []:
        if not isinstance(row, dict):
            continue
        r = dict(row)
        _normalize_confidence_key(r)

        if recalibrate_mix:
            cal = calibrate_market_odds(
                odd_model=r.get("Odd (Modelo)"),
                odd_hist=r.get("Odd (Hist)"),
                stats_pct=r.get("StatsCamps %"),
                max_odd=mx,
                require_model=bool(req_model),
                odd_mix_fn=lambda om, oh, sc: _call_odd_mix_3(odd_mix_fn, om, oh, sc),
            )
            if not bool(cal.get("ok")):
                continue
            r["Odd (Mix)"] = _safe_float(cal.get("odd_mix"))
            r["_calib_reason"] = str(cal.get("reason", ""))
            r["_calib_source"] = str(cal.get("source", ""))
            r["Obs Mix"] = _mix_obs_from_cal(cal)

        r["Odd (Modelo)"] = _safe_float(r.get("Odd (Modelo)"))
        r["Odd (Hist)"] = _safe_float(r.get("Odd (Hist)"))
        _refresh_row_confidence(r, guardrail_cfg)
        out.append(r)

    return out


def _format_odd_value(v: Any, format_num_fn: Callable[[float, int], str]) -> str:
    fv = _safe_float(v)
    if fv is None:
        return "-"
    return format_num_fn(float(fv), 2)



def _format_conf_value(v: Any) -> str:
    iv = _safe_int(v, -1)
    if iv < 0:
        return "-"
    return str(max(0, min(100, iv)))


def finalize_resumo_dataframe(df: pd.DataFrame, *, format_num_fn: Callable[[float, int], str]) -> pd.DataFrame:
    """Ordena, formata e filtra colunas do DataFrame de resumo para exibição na UI."""
    if df is None or df.empty:
        return pd.DataFrame([])

    out = df.sort_values(["_metric", "_side_ord", "_line"]).reset_index(drop=True).copy()

    # Normalize confidence aliases to one stable column.
    conf_src = None
    for c in _CONF_ALIASES:
        if c in out.columns:
            conf_src = c
            break
    if conf_src is None:
        out["Confiança"] = "-"
    else:
        out["Confiança"] = out[conf_src].apply(_format_conf_value)

    for col in ("Odd (Modelo)", "Odd (Hist)", "Odd (Mix)"):
        if col in out.columns:
            out[col] = out[col].apply(lambda x: _format_odd_value(x, format_num_fn))
        else:
            out[col] = "-"

    # Keep the UI-friendly Stats column.
    if "StatsCamps%" in out.columns and "StatsCamps %" not in out.columns:
        out = out.rename(columns={"StatsCamps%": "StatsCamps %"})
    if "Stats Detail" in out.columns and bool((out["Stats Detail"].fillna("").astype(str).str.strip() != "").any()):
        out["Stats"] = out["Stats Detail"].apply(lambda x: str(x).strip() if x is not None else "-")
    elif "StatsCamps %" in out.columns:
        out["StatsCamps %"] = out["StatsCamps %"].apply(lambda v: fmt_pct(v, 0))
        out["Stats"] = out["StatsCamps %"]
    elif "StatsCamps" in out.columns:
        out["Stats"] = out["StatsCamps"].apply(lambda x: str(x).strip() if x is not None else "-")
    else:
        out["Stats"] = "-"

    if "Obs Mix" in out.columns:
        out["Obs Mix"] = out["Obs Mix"].apply(lambda x: str(x).strip() if x is not None else "")

    out = out.drop(columns=["_metric", "_side_ord", "_line", "_edge_pp", "_conf_parts", "_gap_pp"], errors="ignore")

    preferred_cols = [
        "Mercado",
        "Lado",
        "Linha",
        "Confiança",
        "Laplace (Liga)",
        "Laplace (H2H)",
        "Odd (Modelo)",
        "Stats",
        "Odd (Hist)",
        "Odd (Mix)",
    ]

    for c in preferred_cols:
        if c not in out.columns:
            out[c] = "-"

    if "Obs Mix" in out.columns:
        has_obs = bool((out["Obs Mix"].astype(str).str.strip() != "").any())
        if has_obs:
            preferred_cols.append("Obs Mix")

    final_cols = [c for c in preferred_cols if c in out.columns]
    return out[final_cols]


def _build_win_odds_map(
    *,
    df_year: pd.DataFrame,
    df_15: pd.DataFrame,
    df_10: pd.DataFrame,
    df_5: pd.DataFrame,
    side_key: str,
    laplace_odds_fn: Callable[[int, int], Optional[float]],
) -> dict[str, Optional[float]]:
    def _odd(df: pd.DataFrame) -> Optional[float]:
        wins, n = counts_win(df)
        if str(side_key).strip().upper() == "B":
            wins = max(0, int(n) - int(wins))
        return laplace_odds_fn(wins, n)

    return {
        f"Liga_Ano_{side_key}": _odd(df_year),
        f"Liga_15_{side_key}": _odd(df_15),
        f"Liga_10_{side_key}": _odd(df_10),
        f"Liga_5_{side_key}": _odd(df_5),
    }


def _build_total_odds_map(
    *,
    df_year: pd.DataFrame,
    df_15: pd.DataFrame,
    df_10: pd.DataFrame,
    df_5: pd.DataFrame,
    value_col: str,
    line: float,
    laplace_odds_fn: Callable[[int, int], Optional[float]],
) -> dict[str, Optional[float]]:
    def _odd(df: pd.DataFrame, side_key: str) -> Optional[float]:
        over, under, n = counts_ou(df, value_col, line)
        k = over if side_key == "Over" else under
        return laplace_odds_fn(k, n)

    return {
        "Liga_Ano_Over": _odd(df_year, "Over"),
        "Liga_15_Over": _odd(df_15, "Over"),
        "Liga_10_Over": _odd(df_10, "Over"),
        "Liga_5_Over": _odd(df_5, "Over"),
        "Liga_Ano_Under": _odd(df_year, "Under"),
        "Liga_15_Under": _odd(df_15, "Under"),
        "Liga_10_Under": _odd(df_10, "Under"),
        "Liga_5_Under": _odd(df_5, "Under"),
    }


def _merge_h2h_odds(
    base: dict[str, Optional[float]],
    *,
    df_year: pd.DataFrame,
    df_15: pd.DataFrame,
    df_10: pd.DataFrame,
    df_5: pd.DataFrame,
    side_builder: Callable[[pd.DataFrame], tuple[Optional[float], Optional[float]]],
) -> dict[str, Optional[float]]:
    out = dict(base)
    for rec, df in (("Ano", df_year), ("15", df_15), ("10", df_10), ("5", df_5)):
        over_odd, under_odd = side_builder(df)
        out[f"H2H_{rec}_Over"] = over_odd
        out[f"H2H_{rec}_Under"] = under_odd
    return out


def _row_common(
    *,
    mercado: str,
    lado: str,
    linha: str,
    amostra_liga: int,
    amostra_h2h: int,
    liga_txt: str,
    h2h_txt: str,
    laplace_liga_txt: str,
    laplace_h2h_txt: str,
    stats_badge: str,
    stats_pct: Optional[float],
    odd_model: Optional[float],
    odd_hist: Optional[float],
    odd_mix: Optional[float],
    mismatch: str = "N/A",
) -> dict[str, Any]:
    return {
        "Mercado": mercado,
        "Lado": lado,
        "Linha": linha,
        "Amostra Liga": int(amostra_liga),
        "Amostra H2H": int(amostra_h2h),
        "Liga": liga_txt,
        "H2H": h2h_txt,
        "Mismatch": mismatch,
        "Laplace (Liga)": laplace_liga_txt,
        "Laplace (H2H)": laplace_h2h_txt,
        "StatsCamps": stats_badge,
        "StatsCamps %": stats_pct,
        "Odd (Modelo)": odd_model,
        "Odd (Hist)": odd_hist,
        "Odd (Mix)": odd_mix,
        "Obs Mix": "",
    }


def build_ml_rows(
    *,
    p_ml_row: float,
    team_a: str,
    team_b: str,
    req_liga: bool,
    req_h2h: bool,
    req_model: bool,
    min_sample_liga: int,
    min_sample_h2h: int,
    max_odd: float,
    df_league_year: pd.DataFrame,
    df_league_15: pd.DataFrame,
    df_league_10: pd.DataFrame,
    df_league_5: pd.DataFrame,
    df_h2h_year: pd.DataFrame,
    df_h2h_15: pd.DataFrame,
    df_h2h_10: pd.DataFrame,
    df_h2h_5: pd.DataFrame,
    odd_from_prob_fn: Callable[[float], Optional[float]],
    laplace_odds_fn: Callable[[int, int], Optional[float]],
    laplace_pick_odd_fn: Callable[[dict, str], Optional[float]],
    odd_mix_fn: Callable[[Any, Any, Any], Optional[float]],
    fmt_laplace_filtered_fn: Callable[[dict, str, str, float], str],
    fmt_recortes_disp_fn: Callable[[list[str]], str],
    guardrail_cfg: Optional[dict] = None,
    matchup_league: str = "",
) -> list[dict]:
    """Constrói linhas de mercado ML (moneyline) para os dois times com odds e amostras calculados."""
    p_a = _safe_float(p_ml_row)
    if p_a is None:
        return []
    p_b = float(max(0.0, min(1.0, 1.0 - float(p_a))))

    liga_n = int(getattr(df_league_year, "shape", [0])[0] or 0)
    h2h_n = int(getattr(df_h2h_year, "shape", [0])[0] or 0)

    merged_a = dict(
        _build_win_odds_map(
            df_year=df_league_year,
            df_15=df_league_15,
            df_10=df_league_10,
            df_5=df_league_5,
            side_key="A",
            laplace_odds_fn=laplace_odds_fn,
        )
    )
    merged_b = dict(
        _build_win_odds_map(
            df_year=df_league_year,
            df_15=df_league_15,
            df_10=df_league_10,
            df_5=df_league_5,
            side_key="B",
            laplace_odds_fn=laplace_odds_fn,
        )
    )

    def _h2h_win_odds(df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        wins, n = counts_win(df)
        return laplace_odds_fn(wins, n), laplace_odds_fn(max(0, int(n) - int(wins)), n)

    for rec, df in (("Ano", df_h2h_year), ("15", df_h2h_15), ("10", df_h2h_10), ("5", df_h2h_5)):
        a_odd, b_odd = _h2h_win_odds(df)
        merged_a[f"H2H_{rec}_A"] = a_odd
        merged_b[f"H2H_{rec}_B"] = b_odd

    rows: list[dict] = []
    for side_key, team_name, p_side, metric_key, side_ord in (
        ("A", team_a, p_a, "00_ml_A", 0),
        ("B", team_b, p_b, "00_ml_B", 1),
    ):
        odds_map = merged_a if side_key == "A" else merged_b
        odd_model = odd_from_prob_fn(p_side)
        odd_hist = laplace_pick_odd_fn(odds_map, side_key)
        if not sample_gate_ok(
            n_liga=liga_n,
            n_h2h=h2h_n,
            req_liga=req_liga,
            req_h2h=req_h2h,
            min_sample_liga=min_sample_liga,
            min_sample_h2h=min_sample_h2h,
        ):
            continue
        if not _guardrail_model_odd(odd_model, max_odd=max_odd, req_model=req_model):
            continue
        row = _row_common(
            mercado="ML (Mapa)",
            lado=str(team_name),
            linha="-",
            amostra_liga=liga_n,
            amostra_h2h=h2h_n,
            liga_txt=fmt_recortes_disp_fn(passed_recortes(odds_map, "Liga", side_key, max_odd)),
            h2h_txt=fmt_recortes_disp_fn(passed_recortes(odds_map, "H2H", side_key, max_odd)),
            laplace_liga_txt=fmt_laplace_filtered_fn(odds_map, "Liga", side_key, max_odd),
            laplace_h2h_txt=fmt_laplace_filtered_fn(odds_map, "H2H", side_key, max_odd),
            stats_badge="-",
            stats_pct=None,
            odd_model=odd_model,
            odd_hist=odd_hist,
            odd_mix=odd_mix_fn(odd_model, odd_hist, None),
            mismatch="N/A",
        )
        row["_metric"] = metric_key
        row["_side_ord"] = int(side_ord)
        row["_line"] = 0.0
        rows.append(row)
    return _recalibrate_rows(
        rows,
        max_odd=max_odd,
        req_model=req_model,
        odd_mix_fn=odd_mix_fn,
        guardrail_cfg=guardrail_cfg,
        recalibrate_mix=True,
    )


def build_metric_rows(
    *,
    lines_by_metric: dict[str, list[float]],
    ensure_lines_fn: Callable[[str, Any], list[float]],
    metric_col_map: dict[str, str],
    totals_for_lines: dict[str, Any],
    total_over_prob_fn: Callable[[Any, float], float],
    odd_from_prob_fn: Callable[[float], Optional[float]],
    req_liga: bool,
    req_h2h: bool,
    req_model: bool,
    min_sample_liga: int,
    min_sample_h2h: int,
    max_odd: float,
    df_league_year: pd.DataFrame,
    df_league_15: pd.DataFrame,
    df_league_10: pd.DataFrame,
    df_league_5: pd.DataFrame,
    df_h2h_year: pd.DataFrame,
    df_h2h_15: pd.DataFrame,
    df_h2h_10: pd.DataFrame,
    df_h2h_5: pd.DataFrame,
    laplace_odds_fn: Callable[[int, int], Optional[float]],
    laplace_pick_odd_fn: Callable[[dict, str], Optional[float]],
    odd_mix_fn: Callable[[Any, Any, Any], Optional[float]],
    fmt_laplace_filtered_fn: Callable[[dict, str, str, float], str],
    fmt_recortes_disp_fn: Callable[[list[str]], str],
    fmt_line_disp_fn: Callable[[str, float], str],
    metric_label_fn: Callable[[str], str],
    side_label_fn: Callable[[str], str],
    sc_badge_fn: Callable[[str, str, float], str],
    sc_pct_fn: Callable[[str, str, float], Optional[float]],
    guardrail_cfg: Optional[dict] = None,
    matchup_league: str = "",
) -> list[dict]:
    """Constrói linhas de mercados de totais (over/under) para todas as métricas e linhas configuradas."""
    rows: list[dict] = []
    liga_n = int(getattr(df_league_year, "shape", [0])[0] or 0)
    h2h_n = int(getattr(df_h2h_year, "shape", [0])[0] or 0)
    if not sample_gate_ok(
        n_liga=liga_n,
        n_h2h=h2h_n,
        req_liga=req_liga,
        req_h2h=req_h2h,
        min_sample_liga=min_sample_liga,
        min_sample_h2h=min_sample_h2h,
    ):
        return []

    for metric_key, raw_lines in dict(lines_by_metric or {}).items():
        lines = ensure_lines_fn(metric_key, raw_lines)
        value_col = str(metric_col_map.get(metric_key) or "")
        if not value_col:
            continue
        for line in list(lines or []):
            ln = _safe_float(line)
            if ln is None:
                continue
            odds_map = _build_total_odds_map(
                df_year=df_league_year,
                df_15=df_league_15,
                df_10=df_league_10,
                df_5=df_league_5,
                value_col=value_col,
                line=float(ln),
                laplace_odds_fn=laplace_odds_fn,
            )

            def _h2h(df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
                over, under, n = counts_ou(df, value_col, float(ln))
                return laplace_odds_fn(over, n), laplace_odds_fn(under, n)

            odds_map = _merge_h2h_odds(
                odds_map,
                df_year=df_h2h_year,
                df_15=df_h2h_15,
                df_10=df_h2h_10,
                df_5=df_h2h_5,
                side_builder=_h2h,
            )

            p_over_model = None
            if metric_key in dict(totals_for_lines or {}):
                try:
                    p_over_model = total_over_prob_fn(totals_for_lines[metric_key], float(ln))
                except Exception:
                    p_over_model = None
            model_over_odd = odd_from_prob_fn(p_over_model) if _safe_float(p_over_model) is not None else None
            model_under_odd = odd_from_prob_fn(1.0 - float(p_over_model)) if _safe_float(p_over_model) is not None else None

            for side_key, side_ord in (("Under", 0), ("Over", 1)):
                odd_model = model_over_odd if side_key == "Over" else model_under_odd
                if not _guardrail_model_odd(odd_model, max_odd=max_odd, req_model=req_model):
                    continue
                odd_hist = laplace_pick_odd_fn(odds_map, side_key)
                # Teto de odd: aplicado CENTRALMENTE em resumo_tab_core (vale p/
                # TODOS os mercados — vencedor ML, totais e combos), não aqui.
                stats_pct = sc_pct_fn(metric_key, side_key, float(ln))
                row = _row_common(
                    mercado=f"{metric_label_fn(metric_key)} ({fmt_line_disp_fn(metric_key, float(ln))})",
                    lado=side_label_fn(side_key),
                    linha=fmt_line_disp_fn(metric_key, float(ln)),
                    amostra_liga=liga_n,
                    amostra_h2h=h2h_n,
                    liga_txt=fmt_recortes_disp_fn(passed_recortes(odds_map, "Liga", side_key, max_odd)),
                    h2h_txt=fmt_recortes_disp_fn(passed_recortes(odds_map, "H2H", side_key, max_odd)),
                    laplace_liga_txt=fmt_laplace_filtered_fn(odds_map, "Liga", side_key, max_odd),
                    laplace_h2h_txt=fmt_laplace_filtered_fn(odds_map, "H2H", side_key, max_odd),
                    stats_badge=sc_badge_fn(metric_key, side_key, float(ln)),
                    stats_pct=stats_pct,
                    odd_model=odd_model,
                    odd_hist=odd_hist,
                    odd_mix=odd_mix_fn(odd_model, odd_hist, stats_pct),
                    mismatch="N/A",
                )
                row["_metric"] = metric_key
                row["_side_ord"] = int(side_ord)
                row["_line"] = float(ln)
                rows.append(row)
    return _recalibrate_rows(
        rows,
        max_odd=max_odd,
        req_model=req_model,
        odd_mix_fn=odd_mix_fn,
        guardrail_cfg=guardrail_cfg,
        recalibrate_mix=True,
    )


def build_combo_rows(
    *,
    combo_lines_by_metric: dict[str, list[float]],
    metric_col_map: dict[str, str],
    p_ml_combo: float,
    team_a: str,
    team_b: str,
    combo_hist_w: float,
    combo_style_w: float,
    combo_form_w: float,
    combo_use_mismatch: bool,
    req_liga: bool,
    req_h2h: bool,
    req_model: bool,
    min_sample_liga: int,
    min_sample_h2h: int,
    max_odd: float,
    get_combo_bases_fn: Callable[[str], dict[str, pd.DataFrame]],
    wr_laplace_fn: Callable[[pd.DataFrame], tuple[float, int]],
    counts_ml_totals_fn: Callable[[pd.DataFrame, str, float], tuple[int, int, int]],
    totals_for_lines: dict[str, Any],
    total_over_prob_fn: Callable[[Any, float], float],
    odd_from_prob_fn: Callable[[float], Optional[float]],
    laplace_odds_fn: Callable[[int, int], Optional[float]],
    laplace_pick_odd_fn: Callable[[dict, str], Optional[float]],
    odd_mix_fn: Callable[[Any, Any, Any, float], Optional[float]],
    fmt_line_disp_fn: Callable[[str, float], str],
    metric_label_fn: Callable[[str], str],
    fmt_recortes_disp_fn: Callable[[list[str]], str],
    fmt_laplace_filtered_fn: Callable[[dict, str, str, float], str],
    sc_badge_fn: Callable[[str, str, float], str],
    sc_pct_fn: Callable[[str, str, float], Optional[float]],
    guardrail_cfg: Optional[dict] = None,
    matchup_league: str = "",
) -> list[dict]:
    """Constrói linhas de mercados combo ML+Total para ambos os times com probabilidade conjunta ajustada."""
    rows: list[dict] = []
    p_map = _safe_float(p_ml_combo)
    if p_map is None:
        return []

    for metric_key, raw_lines in dict(combo_lines_by_metric or {}).items():
        value_col = str(metric_col_map.get(metric_key) or "")
        if not value_col:
            continue
        if metric_key not in dict(totals_for_lines or {}):
            continue
        for line in list(raw_lines or []):
            ln = _safe_float(line)
            if ln is None:
                continue
            p_over_model = _safe_float(total_over_prob_fn(totals_for_lines[metric_key], float(ln)))
            if p_over_model is None:
                continue
            for team_side, team_name, p_side_ml, metric_prefix in (
                ("A", team_a, float(p_map), "zz_mltot_A"),
                ("B", team_b, float(max(0.0, min(1.0, 1.0 - p_map))), "zz_mltot_B"),
            ):
                bases = dict(get_combo_bases_fn(team_side) or {})
                l_year = bases.get("l_year", pd.DataFrame([]))
                l_5 = bases.get("l_5", pd.DataFrame([]))
                l_10 = bases.get("l_10", pd.DataFrame([]))
                l_15 = bases.get("l_15", pd.DataFrame([]))
                h_year = bases.get("h_year", pd.DataFrame([]))
                h_5 = bases.get("h_5", pd.DataFrame([]))
                h_10 = bases.get("h_10", pd.DataFrame([]))
                h_15 = bases.get("h_15", pd.DataFrame([]))
                liga_n = int(getattr(l_year, "shape", [0])[0] or 0)
                h2h_n = int(getattr(h_year, "shape", [0])[0] or 0)
                if not sample_gate_ok(
                    n_liga=liga_n,
                    n_h2h=h2h_n,
                    req_liga=req_liga,
                    req_h2h=req_h2h,
                    min_sample_liga=min_sample_liga,
                    min_sample_h2h=min_sample_h2h,
                ):
                    continue

                def _combo_map(side_key: str) -> dict[str, Optional[float]]:
                    def _odd(df: pd.DataFrame, prefix: str) -> Optional[float]:
                        over, under, n = counts_ml_totals_fn(df, team_side, float(ln))
                        k = over if side_key == "Over" else under
                        return laplace_odds_fn(k, n)
                    out = {
                        f"Liga_Ano_{side_key}": _odd(l_year, "Liga"),
                        f"Liga_15_{side_key}": _odd(l_15, "Liga"),
                        f"Liga_10_{side_key}": _odd(l_10, "Liga"),
                        f"Liga_5_{side_key}": _odd(l_5, "Liga"),
                        f"H2H_Ano_{side_key}": _odd(h_year, "H2H"),
                        f"H2H_15_{side_key}": _odd(h_15, "H2H"),
                        f"H2H_10_{side_key}": _odd(h_10, "H2H"),
                        f"H2H_5_{side_key}": _odd(h_5, "H2H"),
                    }
                    return out

                odds_over = _combo_map("Over")
                odds_under = _combo_map("Under")
                p_form_side, _ = wr_laplace_fn(l_year)
                p_ml_adj = float((1.0 - float(combo_form_w)) * float(p_side_ml) + float(combo_form_w) * float(p_form_side))
                dep = float(combo_style_w if combo_use_mismatch else 0.0)
                p_joint_over, p_joint_under = core_combine_ml_totals_probs(
                    p_ml_adj,
                    float(p_over_model),
                    dependence=dep,
                )
                hist_over_odd = laplace_pick_odd_fn(odds_over, "Over")
                hist_under_odd = laplace_pick_odd_fn(odds_under, "Under")
                hist_over_p = odd_to_prob(hist_over_odd)
                hist_under_p = odd_to_prob(hist_under_odd)
                model_boost_over = 1.0
                model_boost_under = 1.0
                if combo_use_mismatch:
                    if hist_over_p is not None:
                        model_boost_over = float(max(0.75, min(1.35, 1.0 + (float(combo_hist_w) * ((hist_over_p - p_joint_over) * 2.0)))))
                    if hist_under_p is not None:
                        model_boost_under = float(max(0.75, min(1.35, 1.0 + (float(combo_hist_w) * ((hist_under_p - p_joint_under) * 2.0)))))

                for side_key, side_ord, odd_hist, p_joint_model, model_boost, odds_map in (
                    ("Over", 1, hist_over_odd, p_joint_over, model_boost_over, odds_over),
                    ("Under", 0, hist_under_odd, p_joint_under, model_boost_under, odds_under),
                ):
                    odd_model = odd_from_prob_fn(p_joint_model)
                    if not _guardrail_model_odd(odd_model, max_odd=max_odd, req_model=req_model):
                        continue
                    stats_pct = sc_pct_fn(metric_key, side_key, float(ln))
                    row = _row_common(
                        mercado=f"{team_name} + {metric_label_fn(metric_key)} ({fmt_line_disp_fn(metric_key, float(ln))})",
                        lado=f"ML & {'Acima' if side_key == 'Over' else 'Abaixo'}",
                        linha=fmt_line_disp_fn(metric_key, float(ln)),
                        amostra_liga=liga_n,
                        amostra_h2h=h2h_n,
                        liga_txt=fmt_recortes_disp_fn(passed_recortes(odds_map, "Liga", side_key, max_odd)),
                        h2h_txt=fmt_recortes_disp_fn(passed_recortes(odds_map, "H2H", side_key, max_odd)),
                        laplace_liga_txt=fmt_laplace_filtered_fn(odds_map, "Liga", side_key, max_odd),
                        laplace_h2h_txt=fmt_laplace_filtered_fn(odds_map, "H2H", side_key, max_odd),
                        stats_badge=sc_badge_fn(metric_key, side_key, float(ln)),
                        stats_pct=stats_pct,
                        odd_model=odd_model,
                        odd_hist=odd_hist,
                        odd_mix=odd_mix_fn(odd_model, odd_hist, stats_pct, model_boost),
                        mismatch="ON" if combo_use_mismatch else "OFF",
                    )
                    row["_metric"] = f"{metric_prefix}_{metric_key}"
                    row["_side_ord"] = int(side_ord)
                    row["_line"] = float(ln)
                    rows.append(row)
    # Keep legacy combo mix behavior (model_boost-aware) and only refresh confidence harmonization.
    return _recalibrate_rows(
        rows,
        max_odd=max_odd,
        req_model=req_model,
        odd_mix_fn=odd_mix_fn,
        guardrail_cfg=guardrail_cfg,
        recalibrate_mix=False,
    )
