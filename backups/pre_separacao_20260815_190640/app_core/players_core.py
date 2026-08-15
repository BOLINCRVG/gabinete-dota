"""Core builders for Players tab (means/fair/markets)."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import pandas as pd
from app_core.calibration_core import calibrate_market_odds


def norm_pos_local(x: str) -> str:
    s = str(x or "").strip().lower()
    if s in ("top",):
        return "top"
    if s in ("jng", "jungle"):
        return "jng"
    if s in ("mid",):
        return "mid"
    if s in ("bot", "adc"):
        return "bot"
    if s in ("sup", "support"):
        return "sup"
    return ""


def safe_norm_cdf(z: float) -> float:
    try:
        return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))
    except Exception:
        return 0.5


def player_metric_series(
    dfb: pd.DataFrame,
    player_name: str,
    pos_name: str,
    metric_col: str,
    norm_pos_fn: Callable[[str], str] = norm_pos_local,
) -> pd.Series:
    d0 = dfb[dfb["player"].astype(str) == str(player_name)].copy()
    if pos_name:
        pos_ser = d0["pos"].astype(str) if "pos" in d0.columns else pd.Series([""] * len(d0), index=d0.index)
        d0_pos = d0[pos_ser.apply(norm_pos_fn) == str(pos_name)].copy()
        if not d0_pos.empty:
            d0 = d0_pos
    if d0.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(d0.get(metric_col), errors="coerce").dropna()


def player_metric_h2h_series(
    dfb: pd.DataFrame,
    player_name: str,
    opponent_player_name: str,
    pos_name: str,
    metric_col: str,
    norm_pos_fn: Callable[[str], str] = norm_pos_local,
) -> pd.Series:
    """Serie do jogador contra um adversario especifico no mesmo matchup de lane (mesmo gameid/pos)."""
    if (
        dfb is None
        or dfb.empty
        or "gameid" not in dfb.columns
        or "player" not in dfb.columns
        or "side" not in dfb.columns
    ):
        return pd.Series(dtype=float)
    p_self = str(player_name or "").strip()
    p_opp = str(opponent_player_name or "").strip()
    if not p_self or not p_opp:
        return pd.Series(dtype=float)

    d_self = dfb[dfb["player"].astype(str) == p_self].copy()
    if d_self.empty:
        return pd.Series(dtype=float)
    if pos_name and "pos" in d_self.columns:
        pos_ser = d_self["pos"].astype(str).apply(norm_pos_fn)
        d_self = d_self[pos_ser == str(pos_name)].copy()
    if d_self.empty:
        return pd.Series(dtype=float)

    d_opp = dfb[dfb["player"].astype(str) == p_opp].copy()
    if d_opp.empty:
        return pd.Series(dtype=float)
    if pos_name and "pos" in d_opp.columns:
        pos_ser_opp = d_opp["pos"].astype(str).apply(norm_pos_fn)
        d_opp = d_opp[pos_ser_opp == str(pos_name)].copy()
    if d_opp.empty:
        return pd.Series(dtype=float)

    # Matchup real: mesmo gameid e lados opostos.
    d_self = d_self[["gameid", "side", metric_col]].copy()
    d_opp = d_opp[["gameid", "side"]].copy().rename(columns={"side": "opp_side"})
    d_join = d_self.merge(d_opp, on="gameid", how="inner")
    d_join = d_join[d_join["side"].astype(str) != d_join["opp_side"].astype(str)].copy()
    if d_join.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(d_join.get(metric_col), errors="coerce").dropna()


def build_player_means_rows(
    *,
    dfb: pd.DataFrame,
    lane_rows: list[dict],
    team_a: str,
    team_b: str,
    format_num_fn: Callable[[float, int], str],
) -> list[dict]:
    rows: list[dict] = []
    for rr in lane_rows:
        for team_name, player_name in [(team_a, rr.get("A")), (team_b, rr.get("B"))]:
            if not str(player_name or "").strip():
                continue
            xk = player_metric_series(dfb, str(player_name), str(rr.get("Pos", "")), "kills")
            xd = player_metric_series(dfb, str(player_name), str(rr.get("Pos", "")), "deaths")
            xa = player_metric_series(dfb, str(player_name), str(rr.get("Pos", "")), "assists")
            n = int(max(xk.shape[0], xd.shape[0], xa.shape[0]))
            if n <= 0:
                continue

            def _mu_sd(x: pd.Series) -> tuple[str, str]:
                if x is None or x.empty:
                    return "-", "-"
                mu = float(x.mean())
                sd = float(x.std(ddof=1)) if int(x.shape[0]) >= 2 else 0.0
                return format_num_fn(mu, 2), format_num_fn(sd, 2)

            k_mu, k_sd = _mu_sd(xk)
            d_mu, d_sd = _mu_sd(xd)
            a_mu, a_sd = _mu_sd(xa)
            rows.append(
                {
                    "Lane": rr.get("Lane"),
                    "Time": team_name,
                    "Player": str(player_name),
                    "Amostra": int(n),
                    "Kills (media)": k_mu,
                    "Kills (sd)": k_sd,
                    "Deaths (media)": d_mu,
                    "Deaths (sd)": d_sd,
                    "Assists (media)": a_mu,
                    "Assists (sd)": a_sd,
                }
            )
    return rows


def build_recent_player_stats_rows(
    *,
    dfb: pd.DataFrame,
    lane_rows: list[dict],
    team_a: str,
    team_b: str,
    format_num_fn: Callable[[float, int], str],
    max_games_per_player: int = 8,
) -> list[dict]:
    """Build recent per-player match rows for the Players tab audit panel."""

    def _fmt_mmss(v: Any) -> str:
        try:
            sec = float(v)
        except Exception:
            return "-"
        if (not math.isfinite(sec)) or sec <= 0:
            return "-"
        m = int(sec // 60.0)
        s = round(sec - 60.0 * m)
        if s >= 60:
            m += 1
            s -= 60
        return f"{m:02d}:{s:02d}"

    rows: list[dict] = []
    if dfb is None or dfb.empty:
        return rows

    selected_players: list[tuple[str, str, str, str]] = []
    for rr in lane_rows or []:
        selected_players.append((str(team_a), str(rr.get("A") or ""), str(rr.get("Pos") or ""), str(rr.get("Lane") or "")))
        selected_players.append((str(team_b), str(rr.get("B") or ""), str(rr.get("Pos") or ""), str(rr.get("Lane") or "")))

    seen_players: set[tuple[str, str, str]] = set()
    selected_unique: list[tuple[str, str, str, str]] = []
    for tm, pln, posn, lanen in selected_players:
        key = (tm, pln, posn)
        if (not str(pln).strip()) or (key in seen_players):
            continue
        seen_players.add(key)
        selected_unique.append((tm, pln, posn, lanen))

    opp_map: dict[tuple[str, str], str] = {}
    try:
        if {"gameid", "side", "team"}.issubset(set(dfb.columns)):
            g = dfb[["gameid", "side", "team"]].copy()
            g["gameid"] = g["gameid"].astype(str)
            g["side"] = g["side"].astype(str)
            g["team"] = g["team"].astype(str)
            g = g.dropna(subset=["gameid", "side", "team"]).drop_duplicates()
            for gid, gg in g.groupby("gameid"):
                for _, r0 in gg.iterrows():
                    side0 = str(r0.get("side") or "")
                    opps = gg.loc[gg["side"].astype(str) != side0, "team"].astype(str)
                    if not opps.empty:
                        opp_map[(str(gid), side0)] = str(opps.iloc[0])
    except Exception:
        opp_map = {}

    max_games_per_player = max(1, int(max_games_per_player or 8))

    for team_name, player_name, pos_name, lane_name in selected_unique:
        d0 = dfb[dfb["player"].astype(str) == str(player_name)].copy()
        if d0.empty:
            continue
        if pos_name and "pos" in d0.columns:
            d0 = d0[d0["pos"].astype(str).apply(norm_pos_local) == str(pos_name)].copy()
        if d0.empty:
            continue
        if "date" in d0.columns:
            try:
                d0 = d0.sort_values("date", ascending=False)
            except Exception:
                pass
        if "gameid" in d0.columns:
            try:
                d0 = d0.drop_duplicates(subset=["gameid"], keep="first")
            except Exception:
                pass
        d0 = d0.head(max_games_per_player)

        for _, rw in d0.iterrows():
            k_val = pd.to_numeric(pd.Series([rw.get("kills")]), errors="coerce").iloc[0]
            d_val = pd.to_numeric(pd.Series([rw.get("deaths")]), errors="coerce").iloc[0]
            a_val = pd.to_numeric(pd.Series([rw.get("assists")]), errors="coerce").iloc[0]
            tk_val = pd.to_numeric(pd.Series([rw.get("teamkills")]), errors="coerce").iloc[0]
            dmg_val = pd.to_numeric(pd.Series([rw.get("damagetochampions")]), errors="coerce").iloc[0]
            kda = (
                (float(k_val) + float(a_val)) / max(1.0, float(d_val))
                if (pd.notna(k_val) and pd.notna(d_val) and pd.notna(a_val))
                else float("nan")
            )
            kp = (
                100.0 * (float(k_val) + float(a_val)) / float(tk_val)
                if (pd.notna(k_val) and pd.notna(a_val) and pd.notna(tk_val) and float(tk_val) > 0.0)
                else float("nan")
            )
            gid = str(rw.get("gameid") or "")
            side = str(rw.get("side") or "")
            opp = opp_map.get((gid, side), "-")
            dtv = rw.get("date")
            try:
                dt_txt = dtv.strftime("%Y-%m-%d") if hasattr(dtv, "strftime") else str(dtv)[:10]
            except Exception:
                dt_txt = str(dtv)[:10]
            res_num = pd.to_numeric(pd.Series([rw.get("result")]), errors="coerce").iloc[0]
            res_txt = "Vitoria" if (pd.notna(res_num) and float(res_num) >= 1.0) else "Derrota"

            rows.append(
                {
                    "Lane": lane_name,
                    "Time": team_name,
                    "Player": str(player_name),
                    "Campeão": str(rw.get("champ") or "-"),
                    "Resultado": res_txt,
                    "Tempo": _fmt_mmss(rw.get("gamelength")),
                    "K/D/A": (
                        f"{round(float(k_val))}/{round(float(d_val))}/{round(float(a_val))}"
                        if (pd.notna(k_val) and pd.notna(d_val) and pd.notna(a_val))
                        else "-"
                    ),
                    "KDA": format_num_fn(kda, 2) if math.isfinite(kda) else "-",
                    "KP%": format_num_fn(kp, 1) if math.isfinite(kp) else "-",
                    "Dano": format_num_fn(float(dmg_val), 0) if pd.notna(dmg_val) else "-",
                    "Oponente": opp,
                    "Data": dt_txt,
                }
            )
    return rows


def build_player_market_rows(
    *,
    dfb: pd.DataFrame,
    lane_rows: list[dict],
    per_player_lines_by_idx: dict[int, dict[str, dict[str, list[float]]]],
    team_a: str,
    team_b: str,
    wf_m: float,
    wf_h: float,
    wf_s: float,
    odd_from_prob_fn: Callable[[float], Optional[float]],
    format_num_fn: Callable[[float, int], str],
    format_pct_fn: Callable[[float], str],
    max_odd: float = 999.0,
    require_model: bool = False,
    min_h2h_games: int = 3,
) -> tuple[list[dict], list[dict]]:
    min_h2h_games = max(1, int(min_h2h_games or 3))

    def _obs_odds_text(
        *,
        odd_model: Optional[float],
        odd_hist: Optional[float],
        odd_h2h: Optional[float],
        n_h2h: int,
        min_h2h: int,
        use_h2h: bool,
    ) -> str:
        msgs: list[str] = []
        if odd_model is None or (not math.isfinite(float(odd_model))):
            msgs.append("Odd modelo indisponível.")
        if odd_hist is None or (not math.isfinite(float(odd_hist))):
            msgs.append("Odd histórico indisponível.")
        if int(n_h2h) <= 0:
            msgs.append("Sem H2H deste matchup de lane.")
        elif odd_h2h is None or (not math.isfinite(float(odd_h2h))):
            msgs.append("Odd H2H indisponível.")
        elif not bool(use_h2h):
            msgs.append(f"H2H abaixo do mínimo para Mix ({int(n_h2h)}/{int(min_h2h)}).")
        return " ".join(msgs).strip()

    def _mix_prob_to_odd(
        p_model: Optional[float],
        p_hist: Optional[float],
        p_emp: Optional[float],
        p_h2h: Optional[float],
        use_h2h: bool,
    ) -> Optional[float]:
        src = []
        if p_model is not None and math.isfinite(float(p_model)):
            src.append((float(p_model), wf_m))
        if p_hist is not None and math.isfinite(float(p_hist)):
            src.append((float(p_hist), wf_h))
        if p_emp is not None and math.isfinite(float(p_emp)):
            src.append((float(p_emp), wf_s))
        if bool(use_h2h) and p_h2h is not None and math.isfinite(float(p_h2h)):
            # H2H entra como reforco historico quando existe amostra minima.
            src.append((float(p_h2h), wf_h))
        if not src:
            return None
        tw = sum(max(0.0, float(w)) for _, w in src)
        if tw <= 0:
            tw = float(len(src))
            src = [(p, 1.0) for p, _ in src]
        p = sum(float(pv) * max(0.0, float(w)) for pv, w in src) / float(tw)
        return odd_from_prob_fn(p)

    metric_to_col = {"Kills": "kills", "Deaths": "deaths", "Assists": "assists"}
    out_rows: list[dict] = []
    fair_rows: list[dict] = []

    for i, rr in enumerate(lane_rows):
        per_player_lines = per_player_lines_by_idx.get(i, {"A": {}, "B": {}})
        for side_key, team_name, player_name in [("A", team_a, rr.get("A")), ("B", team_b, rr.get("B"))]:
            if not str(player_name or "").strip():
                continue
            opp_name = rr.get("B") if side_key == "A" else rr.get("A")
            for metric_name, lines_list in (per_player_lines.get(side_key) or {}).items():
                if not lines_list:
                    continue
                x = player_metric_series(dfb, str(player_name), str(rr.get("Pos", "")), metric_to_col[metric_name])
                n = int(x.shape[0])
                if n <= 0:
                    continue
                x_h2h = player_metric_h2h_series(
                    dfb,
                    str(player_name),
                    str(opp_name or ""),
                    str(rr.get("Pos", "")),
                    metric_to_col[metric_name],
                )
                n_h2h = int(x_h2h.shape[0])
                mu = float(x.mean())
                sd = float(x.std(ddof=1)) if n >= 2 else 0.0
                if not math.isfinite(sd) or sd < 0.35:
                    sd = 0.35
                for ln in lines_list:
                    try:
                        lf = float(ln)
                    except Exception:
                        continue
                    if not math.isfinite(lf):
                        continue
                    z = (lf - mu) / sd
                    p_over_model = float(max(0.0, min(1.0, 1.0 - safe_norm_cdf(z))))
                    k_over = int((x > lf).sum())
                    k_under = int((x < lf).sum())
                    p_over_hist = (k_over + 1.0) / (n + 2.0)
                    p_under_hist = (k_under + 1.0) / (n + 2.0)
                    p_over_emp = float(k_over) / float(n)
                    p_under_emp = float(k_under) / float(n)
                    k_over_h2h = int((x_h2h > lf).sum()) if n_h2h > 0 else 0
                    k_under_h2h = int((x_h2h < lf).sum()) if n_h2h > 0 else 0
                    p_over_h2h = ((k_over_h2h + 1.0) / (n_h2h + 2.0)) if n_h2h > 0 else None
                    p_under_h2h = ((k_under_h2h + 1.0) / (n_h2h + 2.0)) if n_h2h > 0 else None
                    use_h2h = bool(n_h2h >= min_h2h_games)
                    odd_over_model = odd_from_prob_fn(p_over_model)
                    odd_under_model = odd_from_prob_fn(max(0.0, 1.0 - p_over_model))
                    odd_over_hist = odd_from_prob_fn(p_over_hist)
                    odd_under_hist = odd_from_prob_fn(p_under_hist)
                    odd_over_h2h = odd_from_prob_fn(p_over_h2h) if p_over_h2h is not None else None
                    odd_under_h2h = odd_from_prob_fn(p_under_h2h) if p_under_h2h is not None else None
                    odd_over_mix = _mix_prob_to_odd(p_over_model, p_over_hist, p_over_emp, p_over_h2h, use_h2h)
                    odd_under_mix = _mix_prob_to_odd(max(0.0, 1.0 - p_over_model), p_under_hist, p_under_emp, p_under_h2h, use_h2h)
                    cal_over = calibrate_market_odds(
                        odd_model=odd_over_model,
                        odd_hist=odd_over_hist,
                        stats_pct=p_over_emp * 100.0,
                        max_odd=float(max_odd),
                        require_model=bool(require_model),
                        odd_mix_fn=lambda om, oh, _sc: odd_over_mix,
                    )
                    cal_under = calibrate_market_odds(
                        odd_model=odd_under_model,
                        odd_hist=odd_under_hist,
                        stats_pct=p_under_emp * 100.0,
                        max_odd=float(max_odd),
                        require_model=bool(require_model),
                        odd_mix_fn=lambda om, oh, _sc: odd_under_mix,
                    )

                    h2h_msg = (
                        f"H2H nao considerado no Mix: amostra insuficiente ({n_h2h}/{min_h2h_games})."
                        if not use_h2h
                        else f"H2H considerado no Mix ({n_h2h} jogos)."
                    )
                    obs_over = h2h_msg
                    obs_under = h2h_msg
                    if cal_over.get("odd_mix") is None:
                        obs_over = f"{obs_over} Odd Mix indisponivel ({cal_over.get('reason', 'sem odd valida')!s})."
                    elif str(cal_over.get("source", "")) in {"model", "hist"}:
                        obs_over = f"{obs_over} Mix em fallback ({cal_over.get('source', '')!s})."
                    if cal_under.get("odd_mix") is None:
                        obs_under = f"{obs_under} Odd Mix indisponivel ({cal_under.get('reason', 'sem odd valida')!s})."
                    elif str(cal_under.get("source", "")) in {"model", "hist"}:
                        obs_under = f"{obs_under} Mix em fallback ({cal_under.get('source', '')!s})."

                    obs_odds_over = _obs_odds_text(
                        odd_model=odd_over_model,
                        odd_hist=odd_over_hist,
                        odd_h2h=odd_over_h2h,
                        n_h2h=n_h2h,
                        min_h2h=min_h2h_games,
                        use_h2h=use_h2h,
                    )
                    obs_odds_under = _obs_odds_text(
                        odd_model=odd_under_model,
                        odd_hist=odd_under_hist,
                        odd_h2h=odd_under_h2h,
                        n_h2h=n_h2h,
                        min_h2h=min_h2h_games,
                        use_h2h=use_h2h,
                    )

                    out_rows.append(
                        {
                            "Lane": rr.get("Lane"),
                            "Team": team_name,
                            "Player": str(player_name),
                            "Mercado": metric_name,
                            "Linha": f"{lf:g}",
                            "Lado": "Acima",
                            "Stats": f"{k_over}/{n} ({(100.0 * k_over / max(n, 1)):.0f}%)",
                            "H2H": (f"{k_over_h2h}/{n_h2h} ({(100.0 * k_over_h2h / max(n_h2h, 1)):.0f}%)" if n_h2h > 0 else "-"),
                            "Odd (Modelo)": format_num_fn(odd_over_model, 2) if odd_over_model is not None and math.isfinite(float(odd_over_model)) else "-",
                            "Odd (Hist)": format_num_fn(odd_over_hist, 2) if odd_over_hist is not None and math.isfinite(float(odd_over_hist)) else "-",
                            "Odd (H2H)": format_num_fn(odd_over_h2h, 2) if odd_over_h2h is not None and math.isfinite(float(odd_over_h2h)) else "-",
                            "Odd (Mix)": format_num_fn(cal_over.get("odd_mix"), 2) if cal_over.get("odd_mix") is not None and math.isfinite(float(cal_over.get("odd_mix"))) else "-",
                            "Obs Odds": obs_odds_over,
                            "Obs Mix": obs_over,
                            "_odd_model_raw": (float(odd_over_model) if odd_over_model is not None and math.isfinite(float(odd_over_model)) else None),
                            "_odd_hist_raw": (float(odd_over_hist) if odd_over_hist is not None and math.isfinite(float(odd_over_hist)) else None),
                            "_odd_h2h_raw": (float(odd_over_h2h) if odd_over_h2h is not None and math.isfinite(float(odd_over_h2h)) else None),
                            "_odd_mix_raw": (float(cal_over.get("odd_mix")) if cal_over.get("odd_mix") is not None and math.isfinite(float(cal_over.get("odd_mix"))) else None),
                            "_calib_reason": str(cal_over.get("reason", "")),
                            "_calib_source": str(cal_over.get("source", "")),
                            "_h2h_n": int(n_h2h),
                            "_h2h_min": int(min_h2h_games),
                            "_h2h_used": bool(use_h2h),
                        }
                    )
                    out_rows.append(
                        {
                            "Lane": rr.get("Lane"),
                            "Team": team_name,
                            "Player": str(player_name),
                            "Mercado": metric_name,
                            "Linha": f"{lf:g}",
                            "Lado": "Abaixo",
                            "Stats": f"{k_under}/{n} ({(100.0 * k_under / max(n, 1)):.0f}%)",
                            "H2H": (f"{k_under_h2h}/{n_h2h} ({(100.0 * k_under_h2h / max(n_h2h, 1)):.0f}%)" if n_h2h > 0 else "-"),
                            "Odd (Modelo)": format_num_fn(odd_under_model, 2) if odd_under_model is not None and math.isfinite(float(odd_under_model)) else "-",
                            "Odd (Hist)": format_num_fn(odd_under_hist, 2) if odd_under_hist is not None and math.isfinite(float(odd_under_hist)) else "-",
                            "Odd (H2H)": format_num_fn(odd_under_h2h, 2) if odd_under_h2h is not None and math.isfinite(float(odd_under_h2h)) else "-",
                            "Odd (Mix)": format_num_fn(cal_under.get("odd_mix"), 2) if cal_under.get("odd_mix") is not None and math.isfinite(float(cal_under.get("odd_mix"))) else "-",
                            "Obs Odds": obs_odds_under,
                            "Obs Mix": obs_under,
                            "_odd_model_raw": (float(odd_under_model) if odd_under_model is not None and math.isfinite(float(odd_under_model)) else None),
                            "_odd_hist_raw": (float(odd_under_hist) if odd_under_hist is not None and math.isfinite(float(odd_under_hist)) else None),
                            "_odd_h2h_raw": (float(odd_under_h2h) if odd_under_h2h is not None and math.isfinite(float(odd_under_h2h)) else None),
                            "_odd_mix_raw": (float(cal_under.get("odd_mix")) if cal_under.get("odd_mix") is not None and math.isfinite(float(cal_under.get("odd_mix"))) else None),
                            "_calib_reason": str(cal_under.get("reason", "")),
                            "_calib_source": str(cal_under.get("source", "")),
                            "_h2h_n": int(n_h2h),
                            "_h2h_min": int(min_h2h_games),
                            "_h2h_used": bool(use_h2h),
                        }
                    )

                    fair_rows.append(
                        {
                            "Lane": rr.get("Lane"),
                            "Time": team_name,
                            "Player": str(player_name),
                            "Mercado": metric_name,
                            "Media": format_num_fn(mu, 2),
                            "SD": format_num_fn(sd, 2),
                            "Linha": f"{lf:g}",
                            "Acima (%)": format_pct_fn(p_over_model),
                            "Odd Acima": format_num_fn(odd_over_model, 2) if odd_over_model is not None and math.isfinite(float(odd_over_model)) else "-",
                            "Abaixo (%)": format_pct_fn(max(0.0, 1.0 - p_over_model)),
                            "Odd Abaixo": format_num_fn(odd_under_model, 2) if odd_under_model is not None and math.isfinite(float(odd_under_model)) else "-",
                            "Amostra": int(n),
                        }
                    )
    return out_rows, fair_rows
