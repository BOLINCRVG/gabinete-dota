"""Series tab renderer."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from app_core.tab_ui_core import render_tab_hero, render_tab_section


def score_probs(p: float, bo: int) -> dict[str, float]:
    q = 1.0 - float(p)
    if int(bo) == 3:
        return {
            "A_2-0": p * p,
            "A_2-1": 2 * p * p * q,
            "B_2-1": 2 * p * q * q,
            "B_2-0": q * q,
        }
    if int(bo) == 5:
        return {
            "A_3-0": p**3,
            "A_3-1": 3 * (p**3) * q,
            "A_3-2": 6 * (p**3) * (q**2),
            "B_3-2": 6 * (q**3) * (p**2),
            "B_3-1": 3 * (q**3) * p,
            "B_3-0": q**3,
        }
    return {}


def render_series_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    render_tab_hero(st, title="Series", subtitle="Linhas de série (totais), ML da série e handicaps de mapas.")

    ml_info = ctx["ml_info"]
    team_a = ctx["teamA"]
    team_b = ctx["teamB"]
    bo = int(ctx["bo"])
    p_map_cal = float(ctx["p_map_cal"])
    state_get_float_fn = ctx["state_get_float_fn"]
    state_get_int_fn = ctx["state_get_int_fn"]
    state_get_str_fn = ctx["state_get_str_fn"]
    winner_pricing_bundle_fn = ctx["winner_pricing_bundle_fn"]
    prob_win_series_fn = ctx["prob_win_series_fn"]
    odd_from_prob_fn = ctx["odd_from_prob_fn"]
    team_with_league_label_fn = ctx["team_with_league_label_fn"]
    format_num_fn = ctx["format_num_fn"]
    format_pct_fn = ctx["format_pct_fn"]
    df_contract_view_fn = ctx["df_contract_view_fn"]
    parse_lines_fn = ctx["parse_lines_fn"]
    series_sims_cached_fn = ctx["series_sims_cached_fn"]
    combine_mean_fn = ctx["combine_mean_fn"]
    combine_sd_fn = ctx["combine_sd_fn"]
    matchup_totals_cls = ctx["matchup_totals_cls"]
    n_sims = int(ctx["n_sims"])
    totals_avg = ctx["totals_avg"]
    profile_a = ctx["profile_a"]
    profile_b = ctx["profile_b"]

    league_a = league_b = ""
    if ml_info is not None:
        league_a = str(((ml_info.get("blue") or {}).get("league")) or "").strip()
        league_b = str(((ml_info.get("red") or {}).get("league")) or "").strip()
    team_a_disp = team_with_league_label_fn(team_a, league_a)
    team_b_disp = team_with_league_label_fn(team_b, league_b)

    winner_pricing = winner_pricing_bundle_fn(
        p_team_a_map=state_get_float_fn(st.session_state, "_p_map_used", float(p_map_cal)),
        p_team_a_series=state_get_float_fn(st.session_state, "_p_series_used", float("nan")),
        bo_for_series=int(bo),
        prob_win_series_fn=prob_win_series_fn,
        odd_from_prob_fn=odd_from_prob_fn,
    )
    p_a_map = float(winner_pricing.get("pA_map", float("nan")))
    odd_a_series = winner_pricing.get("oddA_series")
    odd_b_series = winner_pricing.get("oddB_series")

    state_get_str_fn(st.session_state, "pl_year", "")
    state_get_str_fn(st.session_state, "pl_league", "")
    state_get_str_fn(st.session_state, "pl_split", "")
    state_get_str_fn(st.session_state, "pl_playoffs", "")

    render_tab_section(
        st,
        title="Preco da serie",
        subtitle="O ML do mapa alimenta a serie e os handicaps logo abaixo.",
    )
    with st.container(border=True):
        st.markdown(f"#### {team_a_disp} vs {team_b_disp} (bo{bo})")
        odd_a_map = winner_pricing.get("oddA_map")
        odd_b_map = winner_pricing.get("oddB_map")
        st.markdown(
            f"**ML mapa:** {team_a_disp} = **{format_num_fn(odd_a_map, 2)}** ({format_pct_fn(p_a_map)})"
            f" | {team_b_disp} = **{format_num_fn(odd_b_map, 2)}** ({format_pct_fn(1.0 - p_a_map)})"
        )
        st.markdown(
            f"**ML série:** {team_a_disp} = **{format_num_fn(odd_a_series, 2)}** | {team_b_disp} = **{format_num_fn(odd_b_series, 2)}**"
        )

    if odd_a_series is None or odd_b_series is None:
        # BO2 (Dota) ou confronto sem preço de série: tudo abaixo deriva desse
        # preço — sem ele, avisa em vez de quebrar (TypeError visto 2026-07-10).
        # Preço real do empate BO2 = backlog (draw pricing calibrado).
        st.info(
            "Sem preço de série para este formato/confronto (ex.: BO2). "
            "Use o ML do mapa acima; handicaps/placares de série ficam "
            "indisponíveis até haver preço de série para o formato."
        )
        return

    probs = score_probs(p_a_map, int(bo))
    fav_is_a = bool(odd_a_series <= odd_b_series)
    fav_team = team_a_disp if fav_is_a else team_b_disp
    dog_team = team_b_disp if fav_is_a else team_a_disp

    render_tab_section(st, title="Handicaps (mapas)", subtitle="Mostra as coberturas da serie derivadas do preco atual do confronto.")
    if int(bo) == 3:
        p_fav_m15 = probs.get("A_2-0") if fav_is_a else probs.get("B_2-0")
        p_dog_p15 = None if p_fav_m15 is None else max(0.0, 1.0 - float(p_fav_m15))
        df_h = pd.DataFrame(
            [
                {
                    "Linha": f"{fav_team} -1.5",
                    "Prob": p_fav_m15,
                    "Odd justa": odd_from_prob_fn(p_fav_m15) if p_fav_m15 is not None else None,
                },
                {
                    "Linha": f"{dog_team} +1.5",
                    "Prob": p_dog_p15,
                    "Odd justa": odd_from_prob_fn(p_dog_p15) if p_dog_p15 is not None else None,
                },
            ]
        )
    elif int(bo) == 5:
        p_sweep = probs.get("A_3-0") if fav_is_a else probs.get("B_3-0")
        p_win31 = probs.get("A_3-1") if fav_is_a else probs.get("B_3-1")
        p_fav_m25 = p_sweep
        p_fav_m15 = None if (p_sweep is None or p_win31 is None) else float(p_sweep) + float(p_win31)
        p_dog_p25 = None if p_fav_m25 is None else max(0.0, 1.0 - float(p_fav_m25))
        p_dog_p15 = None if p_fav_m15 is None else max(0.0, 1.0 - float(p_fav_m15))
        df_h = pd.DataFrame(
            [
                {
                    "Linha": f"{fav_team} -1.5",
                    "Prob": p_fav_m15,
                    "Odd justa": odd_from_prob_fn(p_fav_m15) if p_fav_m15 is not None else None,
                },
                {
                    "Linha": f"{fav_team} -2.5",
                    "Prob": p_fav_m25,
                    "Odd justa": odd_from_prob_fn(p_fav_m25) if p_fav_m25 is not None else None,
                },
                {
                    "Linha": f"{dog_team} +1.5",
                    "Prob": p_dog_p15,
                    "Odd justa": odd_from_prob_fn(p_dog_p15) if p_dog_p15 is not None else None,
                },
                {
                    "Linha": f"{dog_team} +2.5",
                    "Prob": p_dog_p25,
                    "Odd justa": odd_from_prob_fn(p_dog_p25) if p_dog_p25 is not None else None,
                },
            ]
        )
    else:
        df_h = pd.DataFrame([])

    with st.container(border=True):
        if df_h.empty:
            st.info("Selecione BO3 ou BO5 para ver handicaps.")
        else:
            df_h["Prob"] = df_h["Prob"].apply(
                lambda x: format_pct_fn(x) if x is not None and math.isfinite(float(x)) else "-"
            )
            df_h["Odd justa"] = df_h["Odd justa"].apply(
                lambda x: format_num_fn(x, 2) if x is not None and math.isfinite(float(x)) else "-"
            )
            df_h = df_contract_view_fn(df_h, ["Linha", "Prob", "Odd justa"], context="Series > Handicaps", fill_value="-")
            st.dataframe(df_h, width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Linhas secundárias da série (totais)")

    c_l1, c_l2, c_l3 = st.columns(3)
    with st.container(border=True):
        with c_l1:
            kills_s_txt = st.text_input(
                "Kills (série) - ex: 112.5, 115.5",
                key="series_kills_txt",
            )
            towers_s_txt = st.text_input(
                "Torres (série) - ex: 22.5",
                key="series_towers_txt",
            )
        with c_l2:
            dragons_s_txt = st.text_input(
                "Dragões (série) | ex: 9.5",
                key="series_dragons_txt",
            )
            barons_s_txt = st.text_input(
                "Barões (série) | ex: 3.5",
                key="series_barons_txt",
            )
        with c_l3:
            team1_k_txt = st.text_input(
                f"Total Kills {team_a} (série) - ex: 20.5",
                key="series_team1_k_txt",
            )
            team2_k_txt = st.text_input(
                f"Total Kills {team_b} (série) - ex: 20.5",
                key="series_team2_k_txt",
            )

    kills_s = parse_lines_fn(kills_s_txt)
    towers_s = parse_lines_fn(towers_s_txt)
    dragons_s = parse_lines_fn(dragons_s_txt)
    barons_s = parse_lines_fn(barons_s_txt)
    team1_k = parse_lines_fn(team1_k_txt)
    team2_k = parse_lines_fn(team2_k_txt)

    if not (math.isfinite(float(p_a_map)) and int(bo) in [3, 5]):
        st.warning("Sem p_map válido ou BO inválido.")
        return

    def _ou_from_sims(sims: np.ndarray, line: float) -> dict[str, float]:
        if sims is None or len(sims) == 0:
            return {"Over": float("nan"), "Under": float("nan")}
        line_f = float(line)
        p_over = float(np.mean(sims > line_f))
        p_under = float(np.mean(sims < line_f))
        return {"Over": odd_from_prob_fn(p_over), "Under": odd_from_prob_fn(p_under)}

    def _render_series_lines(metric: str, lines_list: list[float], total_obj: Any) -> Optional[pd.DataFrame]:
        if not lines_list:
            return None
        sims = series_sims_cached_fn(
            int(bo),
            float(p_a_map),
            total_obj,
            int(n_sims),
            state_get_int_fn(st.session_state, "_sim_seed_series", 1337),
        )
        rows = []
        for line in lines_list:
            ou = _ou_from_sims(sims, float(line))
            rows.append({"Linha": float(line), "Over": format_num_fn(ou["Over"], 2), "Under": format_num_fn(ou["Under"], 2)})
        return df_contract_view_fn(pd.DataFrame(rows), ["Linha", "Over", "Under"], context=f"Series > {metric}", fill_value="-")

    if kills_s:
        dfk = _render_series_lines("kills", kills_s, totals_avg["kills"])
        if dfk is not None:
            st.markdown("**Kills (série)**")
            st.dataframe(dfk, width="stretch", hide_index=True)
    if towers_s:
        dft = _render_series_lines("towers", towers_s, totals_avg["towers"])
        if dft is not None:
            st.markdown("**Torres (série)**")
            st.dataframe(dft, width="stretch", hide_index=True)
    if dragons_s and "dragons" in totals_avg:
        dfd = _render_series_lines("dragons", dragons_s, totals_avg["dragons"])
        if dfd is not None:
            st.markdown("**Dragões (série)**")
            st.dataframe(dfd, width="stretch", hide_index=True)
    if barons_s and "barons" in totals_avg:
        dfb = _render_series_lines("barons", barons_s, totals_avg["barons"])
        if dfb is not None:
            st.markdown("**Barões (série)**")
            st.dataframe(dfb, width="stretch", hide_index=True)

    def _team_kill_total_obj(p_for: float, p_for_sd: float, opp_against: float, opp_against_sd: float) -> Any:
        mean = combine_mean_fn(p_for, opp_against)
        sd = combine_sd_fn(p_for_sd, opp_against_sd)
        return matchup_totals_cls(mean=float(mean), sd=float(max(sd, 1.0)), dist="normal")

    @st.cache_data(show_spinner=False)
    def _series_team_sims_cached(bo_: int, p_map_: float, mean: float, sd: float, n_sims_: int, seed_: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed_))
        wins_needed = 2 if int(bo_) == 3 else 3
        out = []
        for _ in range(int(n_sims_)):
            wins_a = 0
            wins_b = 0
            total = 0.0
            while wins_a < wins_needed and wins_b < wins_needed:
                x = rng.normal(float(mean), float(sd))
                if x < 0:
                    x = 0.0
                total += float(x)
                if rng.random() < float(p_map_):
                    wins_a += 1
                else:
                    wins_b += 1
            out.append(total)
        return np.asarray(out, dtype=float)

    try:
        obj_a = _team_kill_total_obj(profile_a.kills_for, profile_a.kills_for_sd, profile_b.kills_against, profile_b.kills_against_sd)
        obj_b = _team_kill_total_obj(profile_b.kills_for, profile_b.kills_for_sd, profile_a.kills_against, profile_a.kills_against_sd)
        seed0 = state_get_int_fn(st.session_state, "_sim_seed_series", 1337)
        sims_a = _series_team_sims_cached(int(bo), float(p_a_map), float(obj_a.mean), float(obj_a.sd), int(n_sims), seed0 + 11)
        sims_b = _series_team_sims_cached(int(bo), float(1.0 - p_a_map), float(obj_b.mean), float(obj_b.sd), int(n_sims), seed0 + 29)

        def _render_team_k(lines_list: list[float], sims: np.ndarray, label: str) -> None:
            if not lines_list:
                return
            rows = []
            for line in lines_list:
                ou = _ou_from_sims(sims, float(line))
                rows.append({"Linha": float(line), "Over": format_num_fn(ou["Over"], 2), "Under": format_num_fn(ou["Under"], 2)})
            st.markdown(f"**{label} (série)**")
            st.dataframe(
                df_contract_view_fn(pd.DataFrame(rows), ["Linha", "Over", "Under"], context=f"Series > {label}", fill_value="-"),
                width="stretch",
                hide_index=True,
            )

        _render_team_k(team1_k, sims_a, f"Total Kills {team_a}")
        _render_team_k(team2_k, sims_b, f"Total Kills {team_b}")
    except Exception:
        pass
