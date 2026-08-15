"""Champions tab renderer."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Optional

import pandas as pd

from app_core.tab_filters_core import collect_year_split_league_options, scope_has_playoffs
from app_core.tab_state_core import blank_space_label
from app_core.tab_ui_core import render_tab_hero, render_tab_section


def _parse_names(text: str) -> list[str]:
    return [t.strip() for t in re.split(r"[\n\t,;]+", str(text or "").strip()) if t.strip()]


def _render_matchup_pool_block(
    *,
    st: Any,
    team_a: str,
    team_b: str,
    csv_sig: Any,
    load_player_rows_fn: Any,
) -> None:
    """Pool do confronto (redesign 2026-07-16): o que CADA time joga, cruzado
    com o tier do patch atual da aba Meta. Leitura descritiva, n visível."""
    ta = str(team_a or "").strip()
    tb = str(team_b or "").strip()
    if not ta or not tb:
        return
    try:
        from app_core.matchup_pool_core import (
            build_team_pool,
            cross_pool_with_meta,
            pool_verdict_lines,
        )

        rows = load_player_rows_fn(csv_sig)
        if rows is None or rows.empty:
            return
        year = None
        try:
            yr = pd.to_numeric(rows.get("year"), errors="coerce")
            year = int(yr.max()) if yr.notna().any() else None
        except Exception:
            year = None

        # tiers do patch atual (mesmo cache de sessão do Radar)
        meta_map: dict = {}
        try:
            meta_lu = st.session_state.get("_radar_meta_lu")
            if not (isinstance(meta_lu, tuple) and len(meta_lu) == 2 and meta_lu[0] == str(csv_sig)):
                from app_core.champion_knowledge_core import load_champion_knowledge_registry
                from app_core.champions_shared_core import _get_draft_rows
                from app_core.meta_patch_core import meta_lookup_for_radar

                meta_lu = (str(csv_sig), meta_lookup_for_radar(
                    _get_draft_rows(st.session_state, csv_sig),
                    registry=load_champion_knowledge_registry(),
                ))
                st.session_state["_radar_meta_lu"] = meta_lu
            meta_map = dict(meta_lu[1] or {})
        except Exception:
            meta_map = {}

        pools = {}
        for tm in (ta, tb):
            p = cross_pool_with_meta(build_team_pool(rows, tm, year=year), meta_map)
            pools[tm] = p
        if all(p.empty for p in pools.values()):
            return

        with st.container(border=True):
            render_tab_section(
                st,
                title=f"Pool do confronto — {ta} × {tb}",
                subtitle=f"O que cada time joga ({'temporada ' + str(year) if year else 'todo o CSV'}) × tier do patch atual (aba Meta). Leitura descritiva — não mexe em odd.",
            )
            c1, c2 = st.columns(2)
            for col, tm in ((c1, ta), (c2, tb)):
                with col:
                    st.markdown(f"**{tm}**")
                    p = pools[tm]
                    if p.empty:
                        st.caption("Sem picks no recorte.")
                        continue
                    view = pd.DataFrame({
                        "Campeão": p["champion"],
                        "J": p["games"].astype(int),
                        "WR": (p["wr"] * 100).round(0).map(lambda v: f"{v:.0f}%" if pd.notna(v) else "-"),
                        "Tier": p["tier"].replace("", "-"),
                        "Piloto": p["player"].fillna(""),
                    })
                    st.dataframe(view, hide_index=True, use_container_width=True)
                    for ln in pool_verdict_lines(tm, p):
                        st.caption(ln)
    except Exception:
        # bloco novo NUNCA derruba a aba inteira
        return


def render_champions_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    render_tab_hero(
        st,
        title="Campeões",
        subtitle="Pool do confronto: o que cada time joga × meta do patch atual.",
    )

    team_games = ctx["team_games"]
    team_a = ctx["teamA"]
    team_b = ctx["teamB"]
    csv_sig = ctx["csv_sig"]
    Filters = ctx["Filters"]
    state_get_str_fn = ctx["state_get_str_fn"]
    load_player_rows_fn = ctx["load_player_rows_fn"]
    apply_filters_players_fn = ctx["apply_filters_players_fn"]
    champion_vocab_fn = ctx["champion_vocab_fn"]
    fuzzy_match_one_fn = ctx["fuzzy_match_one_fn"]
    df_contract_view_fn = ctx["df_contract_view_fn"]
    format_num_fn = ctx["format_num_fn"]
    ensure_champions_filter_state_fn = ctx["ensure_champions_filter_state_fn"]
    reset_champions_league_dependents_fn = ctx["reset_champions_league_dependents_fn"]
    resolve_blue_side_choice_fn = ctx["resolve_blue_side_choice_fn"]
    sync_blue_side_choice_fn = ctx["sync_blue_side_choice_fn"]

    _render_matchup_pool_block(
        st=st,
        team_a=team_a,
        team_b=team_b,
        csv_sig=csv_sig,
        load_player_rows_fn=load_player_rows_fn,
    )
    if not (str(team_a or "").strip() and str(team_b or "").strip()):
        st.caption("Selecione um confronto (Time 1 × Time 2) para ver o pool dos dois times.")

    # Redesign 2026-07-16 (decisão dele): a aba vira SÓ o Pool do confronto.
    # Força global de campeão é papel da aba Meta; o restante (filtros,
    # colar draft, matchups por posição) foi retirado da UI — o código
    # antigo abaixo fica inalcançável de propósito (barato de reviver).
    return

    years, splits, leagues = collect_year_split_league_options(team_games)
    teams = []
    try:
        if team_games is not None and not team_games.empty and "team" in team_games.columns:
            teams = sorted([str(x) for x in team_games["team"].fillna("").astype(str).unique().tolist() if str(x)])
    except Exception:
        teams = []

    ensure_champions_filter_state_fn(st.session_state)

    def _ensure_choice(key: str, options: list[str], fallback: str | None = None) -> str:
        default = str(fallback if fallback is not None else (options[0] if options else ""))
        current = str(st.session_state.get(key, default) or default)
        allowed = {str(x) for x in options}
        if current not in allowed:
            current = default
            st.session_state[key] = current
        return current

    with st.container(border=True):
        render_tab_section(st, title="Filtros", subtitle="Feche o recorte e, se quiser, restrinja por time antes de colar o draft.")
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            year_options = [""] + [str(y) for y in years]
            _ensure_choice("ch_year", year_options, "")
            ch_year = st.selectbox("Ano", options=year_options, key="ch_year", format_func=blank_space_label)
        with bc4:
            league_options = ["", *leagues]
            _ensure_choice("ch_league", league_options, "")
            ch_league = st.selectbox(
                "Liga",
                options=league_options,
                key="ch_league",
                format_func=blank_space_label,
                on_change=lambda: reset_champions_league_dependents_fn(st.session_state),
            )

    splits_filtered = splits
    teams_filtered = teams
    try:
        if ch_league and team_games is not None and not team_games.empty:
            if "league" in team_games.columns and "split" in team_games.columns:
                splits_filtered = sorted(
                    [
                        str(x)
                        for x in team_games.loc[team_games["league"].astype(str) == str(ch_league), "split"]
                        .fillna("")
                        .astype(str)
                        .unique()
                        .tolist()
                        if str(x)
                    ]
                )
            if "league" in team_games.columns and "team" in team_games.columns:
                teams_filtered = sorted(
                    [
                        str(x)
                        for x in team_games.loc[team_games["league"].astype(str) == str(ch_league), "team"]
                        .fillna("")
                        .astype(str)
                        .unique()
                        .tolist()
                        if str(x)
                    ]
                )
    except Exception:
        splits_filtered = splits
        teams_filtered = teams

    with st.container(border=True):
        bc2, bc3 = st.columns(2)
        with bc2:
            split_options = ["", *splits_filtered]
            _ensure_choice("ch_split", split_options, "")
            ch_split = st.selectbox("Split", options=split_options, key="ch_split", format_func=blank_space_label)
        # Playoffs box appears ONLY when the chosen (liga, ano, split) has
        # playoff games — same pattern as the other tabs (scope_has_playoffs).
        if scope_has_playoffs(team_games, league=ch_league, year=ch_year, split=ch_split):
            with bc3:
                playoffs_options = ["", "Regular", "Playoffs"]
                _ensure_choice("ch_playoffs", playoffs_options, "")
                ch_playoffs = st.selectbox(
                    "Playoffs",
                    options=playoffs_options,
                    key="ch_playoffs",
                    format_func=blank_space_label,
                )
            ch_playoffs = st.session_state.get("ch_playoffs", "") or ""
        else:
            ch_playoffs = ""

        tc1, tc2 = st.columns([2, 1])
        with tc1:
            team_options = ["", *teams_filtered]
            _ensure_choice("ch_team", team_options, "")
            ch_team = st.selectbox("Time (opcional)", options=team_options, key="ch_team", format_func=blank_space_label)
        with tc2:
            min_games_mu = st.number_input(
                "Min jogos p/ matchup",
                min_value=1,
                max_value=30,
                step=1,
                key="ch_min_games_mu",
            )

    with st.spinner("Carregando players do CSV (1x por sessão / cache)..."):
        df_pl = load_player_rows_fn(csv_sig)

    if df_pl is None or df_pl.empty:
        st.warning("Não achei rows de players nesse CSV.")
        df_pl = pd.DataFrame(columns=["gameid", "pos", "side", "champ", "result", "team", "player"])

    base_filters = Filters(
        year=None if not ch_year else int(ch_year),
        league=None if not ch_league else ch_league,
        split=None if not ch_split else ch_split,
        playoffs=None if not ch_playoffs else (True if ch_playoffs == "Playoffs" else False),
    )
    dfb = apply_filters_players_fn(df_pl, base_filters).copy()
    if ch_team and "team" in dfb.columns:
        dfb = dfb[dfb["team"].astype(str) == str(ch_team)].copy()

    vocab = champion_vocab_fn(dfb)
    if not vocab:
        vocab = champion_vocab_fn(df_pl)

    st.markdown("#### Colar campeões")
    st.caption(
        "Formato esperado: 10 campeões em ordem Top/JG/Mid/ADC/Sup do Time A e depois Top/JG/Mid/ADC/Sup do Time B. "
        "Pode ser com TAB, espaço ou quebra de linha."
    )

    blue_opts, blue_idx = resolve_blue_side_choice_fn(st.session_state, team_a=team_a, team_b=team_b)
    blue_ui_key = "ch_blue_team_ui"
    if str(st.session_state.get(blue_ui_key, "") or "") not in [str(x) for x in blue_opts]:
        st.session_state[blue_ui_key] = str(blue_opts[int(blue_idx)])
    ch_blue_team = st.radio(
        "Quem está no Blue Side neste draft",
        options=blue_opts,
        horizontal=True,
        key=blue_ui_key,
    )
    sync_blue_side_choice_fn(st.session_state, selected_value=str(ch_blue_team))

    champs_txt = st.text_area("Cole aqui", height=90, key="ch_paste")
    tokens = [t for t in re.split(r"[\t\n,; ]+", champs_txt.strip()) if t]
    if len(tokens) < 2:
        st.info("Cole pelo menos 2 campeões para começar.")
        tokens = []

    resolved = []
    unmatched = []
    for token in tokens:
        match, suggestions = fuzzy_match_one_fn(token, vocab, cutoff=0.6)
        if match is None:
            unmatched.append((token, suggestions))
            resolved.append(None)
        else:
            resolved.append(match)

    if unmatched:
        st.warning("Alguns campeões não foram reconhecidos. Ajuste a escrita ou use uma das sugestões:")
        for token, suggestions in unmatched[:8]:
            st.write({"entrada": token, "sugestoes": suggestions[:5]})

    champs = [champ for champ in resolved if champ]
    tg_cols = ["gameid", "team", "total_kills"]
    df_tg = (
        team_games[tg_cols].copy()
        if (team_games is not None and not team_games.empty and all(col in team_games.columns for col in tg_cols))
        else None
    )

    def _champ_stats(champ_name: str) -> dict[str, Any]:
        data = dfb[dfb["champ"].astype(str) == str(champ_name)].copy()
        if data.empty:
            return {"Champion": champ_name, "Games": 0, "Win Rate": "-", "Avg game kills": "-"}

        wins = float(pd.to_numeric(data.get("result"), errors="coerce").fillna(0).sum())
        n = int(data.shape[0])
        wr = wins / max(n, 1)
        avg_k = None
        if df_tg is not None and "team" in data.columns:
            merged = data[["gameid", "team"]].dropna().drop_duplicates().merge(df_tg, on=["gameid", "team"], how="left")
            if "total_kills" in merged.columns:
                avg_k = float(pd.to_numeric(merged["total_kills"], errors="coerce").dropna().mean()) if merged["total_kills"].notna().any() else None
        if avg_k is None and "ckpm" in data.columns and "gamelength" in data.columns:
            try:
                game_len_min = pd.to_numeric(data["gamelength"], errors="coerce") / 60.0
                ckpm = pd.to_numeric(data["ckpm"], errors="coerce")
                total = game_len_min * ckpm
                avg_k = float(total.dropna().mean()) if total.notna().any() else None
            except Exception:
                avg_k = None

        return {
            "Champion": champ_name,
            "Games": n,
            "Win Rate": format_num_fn(100.0 * wr, 1) if math.isfinite(wr) else "-",
            "Avg game kills": format_num_fn(avg_k, 1) if (avg_k is not None and math.isfinite(float(avg_k))) else "-",
        }

    st.markdown("#### Resumo por campeão (na base selecionada)")
    rows = [_champ_stats(champ) for champ in champs]
    df_sum = pd.DataFrame(rows)
    if df_sum.empty:
        st.info("Nada para mostrar (provavelmente todos ficaram sem match).")
    else:
        df_sum = df_contract_view_fn(
            df_sum,
            ["Champion", "Games", "Win Rate", "Avg game kills"],
            context="Campeões > Resumo",
            fill_value="-",
        )
        st.dataframe(df_sum.sort_values(["Games", "Champion"], ascending=[False, True]), width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Matchups por posição")
    st.caption("Calcula (por posição) contra quais campeões o pick costuma ir melhor/pior. Usa só jogos do recorte.")

    d2 = dfb[["gameid", "pos", "side", "champ", "result"]].copy()
    d2 = d2[(d2["pos"] != "") & (d2["champ"] != "")].copy()
    d2["side"] = d2["side"].astype(str).str.title()
    blue = d2[d2["side"] == "Blue"].copy()
    red = d2[d2["side"] == "Red"].copy()
    pairs = blue.merge(red, on=["gameid", "pos"], suffixes=("_blue", "_red"))
    pairs["blue_win"] = pd.to_numeric(pairs["result_blue"], errors="coerce").fillna(0).astype(int)
    pairs["red_win"] = pd.to_numeric(pairs["result_red"], errors="coerce").fillna(0).astype(int)

    def _mu_table(champ_name: str, pos: str, max_rows: int = 3):
        a = pairs[(pairs["pos"] == pos) & (pairs["champ_blue"] == champ_name)].copy()
        a["opp"] = a["champ_red"]
        a["win"] = a["blue_win"]
        b = pairs[(pairs["pos"] == pos) & (pairs["champ_red"] == champ_name)].copy()
        b["opp"] = b["champ_blue"]
        b["win"] = b["red_win"]
        u = pd.concat([a[["opp", "win"]], b[["opp", "win"]]], ignore_index=True)
        if u.empty:
            return None, None
        grouped = u.groupby("opp").agg(games=("win", "size"), wins=("win", "sum")).reset_index()
        grouped["wr"] = (grouped["wins"] / grouped["games"]).astype(float)
        grouped = grouped[grouped["games"] >= int(min_games_mu)].copy()
        if grouped.empty:
            return None, None
        best = grouped.sort_values(["wr", "games"], ascending=[False, False]).head(max_rows).copy()
        worst = grouped.sort_values(["wr", "games"], ascending=[True, False]).head(max_rows).copy()
        for frame in (best, worst):
            frame["Win Rate"] = frame["wr"].apply(lambda x: format_num_fn(100.0 * float(x), 1))
        best = best[["opp", "games", "Win Rate"]].rename(columns={"opp": "Vs", "games": "Games"})
        worst = worst[["opp", "games", "Win Rate"]].rename(columns={"opp": "Vs", "games": "Games"})
        return best, worst

    roles = ["top", "jng", "mid", "bot", "sup"]
    picks = [champ for champ in resolved if champ]
    if len(picks) >= 10:
        a5 = picks[:5]
        b5 = picks[5:10]
        if str(ch_blue_team) == str(team_b):
            a5, b5 = b5, a5

        st.markdown("##### Draft (head-to-head por lane)")
        drows = []
        for i, pos in enumerate(roles):
            ca = a5[i] if i < len(a5) else None
            cb = b5[i] if i < len(b5) else None
            if not ca or not cb:
                continue
            x1 = pairs[(pairs["pos"] == pos) & (pairs["champ_blue"] == ca) & (pairs["champ_red"] == cb)].copy()
            w1 = int(x1["blue_win"].sum())
            n1 = int(x1.shape[0])
            x2 = pairs[(pairs["pos"] == pos) & (pairs["champ_red"] == ca) & (pairs["champ_blue"] == cb)].copy()
            w2 = int(x2["red_win"].sum())
            n2 = int(x2.shape[0])
            n = n1 + n2
            w = w1 + w2
            wr = (w / n) if n > 0 else None
            drows.append(
                {
                    "Pos": pos,
                    "Time A": ca,
                    "Time B": cb,
                    "H2H Games": n,
                    "Win Rate (A)": format_num_fn(100.0 * wr, 1) if (wr is not None and math.isfinite(float(wr))) else "-",
                }
            )
        if drows:
            st.dataframe(
                df_contract_view_fn(
                    pd.DataFrame(drows),
                    ["Pos", "Time A", "Time B", "H2H Games", "Win Rate (A)"],
                    context="Campeões > Draft H2H",
                    fill_value="-",
                ),
                width="stretch",
                hide_index=True,
            )

        st.markdown("##### Vantagem do draft (Liga + Player)")
        st.caption("Score por lane combina winrate do campeão no recorte da liga com winrate do player naquele campeão.")

        if "ch_pl_txtA" not in st.session_state:
            st.session_state["ch_pl_txtA"] = state_get_str_fn(st.session_state, "pl_txtA", "")
        if "ch_pl_txtB" not in st.session_state:
            st.session_state["ch_pl_txtB"] = state_get_str_fn(st.session_state, "pl_txtB", "")
        ch_pc1, ch_pc2 = st.columns(2)
        with ch_pc1:
            ch_pl_txtA = st.text_area(
                f"Players {team_a} (Top/JG/Mid/ADC/Sup)",
                height=110,
                key="ch_pl_txtA",
            )
        with ch_pc2:
            ch_pl_txtB = st.text_area(
                f"Players {team_b} (Top/JG/Mid/ADC/Sup)",
                height=110,
                key="ch_pl_txtB",
            )

        def _wr_parts(champ_name: str, player_name: str | None) -> dict[str, Any]:
            d_ch = dfb[dfb["champ"].astype(str) == str(champ_name)].copy()
            n_ch = int(d_ch.shape[0])
            w_ch = float(pd.to_numeric(d_ch.get("result"), errors="coerce").fillna(0).sum()) if n_ch > 0 else 0.0
            p_ch = (w_ch + 2.0) / (n_ch + 4.0) if n_ch > 0 else 0.5
            n_pl = 0
            w_pl = 0.0
            p_pl = p_ch
            if player_name:
                d_pl = d_ch[d_ch["player"].astype(str) == str(player_name)].copy()
                n_pl = int(d_pl.shape[0])
                if n_pl > 0:
                    w_pl = float(pd.to_numeric(d_pl.get("result"), errors="coerce").fillna(0).sum())
                    p_pl = (w_pl + 1.0) / (n_pl + 2.0)
            w_player = float(n_pl) / float(n_pl + 8.0) if n_pl > 0 else 0.0
            p_blend = (1.0 - w_player) * float(p_ch) + w_player * float(p_pl)
            return {
                "p_league": float(p_ch),
                "p_player": float(p_pl),
                "p_blend": float(p_blend),
                "n_league": int(n_ch),
                "n_player": int(n_pl),
            }

        names_a_raw = _parse_names(ch_pl_txtA)
        names_b_raw = _parse_names(ch_pl_txtB)
        player_vocab = sorted([x for x in dfb.get("player", pd.Series(dtype=str)).fillna("").astype(str).unique().tolist() if x])
        names_a = []
        names_b = []
        for name in names_a_raw[:5]:
            match, _ = fuzzy_match_one_fn(name, player_vocab, cutoff=0.6)
            names_a.append(match if match else name)
        for name in names_b_raw[:5]:
            match, _ = fuzzy_match_one_fn(name, player_vocab, cutoff=0.6)
            names_b.append(match if match else name)

        def _h2h_lane_wr(ca: str, cb: str, pos: str) -> tuple[Optional[float], int]:
            x1 = pairs[(pairs["pos"] == pos) & (pairs["champ_blue"] == ca) & (pairs["champ_red"] == cb)].copy()
            w1 = int(x1["blue_win"].sum())
            n1 = int(x1.shape[0])
            x2 = pairs[(pairs["pos"] == pos) & (pairs["champ_red"] == ca) & (pairs["champ_blue"] == cb)].copy()
            w2 = int(x2["red_win"].sum())
            n2 = int(x2.shape[0])
            n = int(n1 + n2)
            if n <= 0:
                return None, 0
            return float((w1 + w2) / float(n)), n

        st.markdown("##### Liga x Player x H2H (por lane)")
        names_a_pad = list(names_a[:5]) + [""] * max(0, 5 - len(names_a))
        names_b_pad = list(names_b[:5]) + [""] * max(0, 5 - len(names_b))
        det_rows = []
        for i, pos in enumerate(roles):
            ca = a5[i] if i < len(a5) else ""
            cb = b5[i] if i < len(b5) else ""
            pa = names_a_pad[i] if i < len(names_a_pad) else ""
            pb = names_b_pad[i] if i < len(names_b_pad) else ""
            if not ca or not cb:
                continue
            wa = _wr_parts(ca, pa if pa else None)
            wb = _wr_parts(cb, pb if pb else None)
            h2h_a, h2h_n = _h2h_lane_wr(ca, cb, pos)
            h2h_b = (1.0 - float(h2h_a)) if h2h_a is not None else None
            det_rows.append(
                {
                    "Pos": pos,
                    "Team": team_a,
                    "Player": pa if pa else "-",
                    "Champ": ca,
                    "WR Liga Champ": format_num_fn(100.0 * wa["p_league"], 1),
                    "N Liga": int(wa["n_league"]),
                    "WR Player Champ": format_num_fn(100.0 * wa["p_player"], 1),
                    "N Player": int(wa["n_player"]),
                    "WR H2H Lane": (format_num_fn(100.0 * float(h2h_a), 1) if h2h_a is not None else "-"),
                    "N H2H": int(h2h_n),
                }
            )
            det_rows.append(
                {
                    "Pos": pos,
                    "Team": team_b,
                    "Player": pb if pb else "-",
                    "Champ": cb,
                    "WR Liga Champ": format_num_fn(100.0 * wb["p_league"], 1),
                    "N Liga": int(wb["n_league"]),
                    "WR Player Champ": format_num_fn(100.0 * wb["p_player"], 1),
                    "N Player": int(wb["n_player"]),
                    "WR H2H Lane": (format_num_fn(100.0 * float(h2h_b), 1) if h2h_b is not None else "-"),
                    "N H2H": int(h2h_n),
                }
            )
        if det_rows:
            st.dataframe(
                df_contract_view_fn(
                    pd.DataFrame(det_rows),
                    ["Pos", "Team", "Player", "Champ", "WR Liga Champ", "N Liga", "WR Player Champ", "N Player", "WR H2H Lane", "N H2H"],
                    context="Campeões > Liga x Player x H2H",
                    fill_value="-",
                ),
                width="stretch",
                hide_index=True,
            )

        if len(names_a) >= 5 and len(names_b) >= 5:
            adv_rows = []
            for i, pos in enumerate(roles):
                ca = a5[i] if i < len(a5) else None
                cb = b5[i] if i < len(b5) else None
                pa = names_a[i] if i < len(names_a) else ""
                pb = names_b[i] if i < len(names_b) else ""
                if not ca or not cb:
                    continue
                wa = _wr_parts(ca, pa)
                wb = _wr_parts(cb, pb)
                edge = float(wa["p_blend"]) - float(wb["p_blend"])
                adv_rows.append(
                    {
                        "Pos": pos,
                        f"{team_a} player": pa,
                        f"{team_a} champ": ca,
                        f"WR blend {team_a}": format_num_fn(100.0 * wa["p_blend"], 1),
                        f"{team_b} player": pb,
                        f"{team_b} champ": cb,
                        f"WR blend {team_b}": format_num_fn(100.0 * wb["p_blend"], 1),
                        "Edge A-B (p.p.)": format_num_fn(100.0 * edge, 1),
                    }
                )
            if adv_rows:
                df_adv = pd.DataFrame(adv_rows)
                df_adv = df_contract_view_fn(df_adv, list(df_adv.columns), context="Campeões > Vantagem do draft", fill_value="-")
                st.dataframe(df_adv, width="stretch", hide_index=True)
                try:
                    edge_total = float(pd.to_numeric(df_adv["Edge A-B (p.p.)"], errors="coerce").dropna().sum())
                except Exception:
                    edge_total = float("nan")
                if math.isfinite(edge_total):
                    fav = team_a if edge_total >= 0 else team_b
                    st.caption(f"Edge total do draft: {fav} ({format_num_fn(abs(edge_total), 1)} p.p.)")
        else:
            st.info("Para calcular a vantagem Liga+Player, informe 5 players de cada time (Top/JG/Mid/ADC/Sup).")

    st.markdown("##### Melhores / Piores matchups (por campeão e posição)")
    csel1, csel2 = st.columns([2, 1])
    champ_opts = sorted(set(champs))
    if not champ_opts:
        champ_opts = [""]
    with csel1:
        if str(st.session_state.get("ch_mu_champ", "") or "") not in champ_opts:
            st.session_state["ch_mu_champ"] = champ_opts[0]
        champ_sel = st.selectbox("Campeão", options=champ_opts, key="ch_mu_champ")
    with csel2:
        if str(st.session_state.get("ch_mu_pos", "") or "") not in {"top", "jng", "mid", "bot", "sup"}:
            st.session_state["ch_mu_pos"] = "top"
        pos_sel = st.selectbox("Posição", options=["top", "jng", "mid", "bot", "sup"], key="ch_mu_pos")

    best, worst = _mu_table(champ_sel, pos_sel) if str(champ_sel).strip() else (None, None)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Costuma ganhar contra**")
        if best is None:
            st.info("Poucos jogos para esse recorte (aumente a base ou reduza o min jogos).")
        else:
            st.dataframe(
                df_contract_view_fn(best, ["Vs", "Games", "Win Rate"], context="Campeões > Melhores matchups", fill_value="-"),
                width="stretch",
                hide_index=True,
            )
    with cc2:
        st.markdown("**Costuma perder contra**")
        if worst is None:
            st.info("Poucos jogos para esse recorte (aumente a base ou reduza o min jogos).")
        else:
            st.dataframe(
                df_contract_view_fn(worst, ["Vs", "Games", "Win Rate"], context="Campeões > Piores matchups", fill_value="-"),
                width="stretch",
                hide_index=True,
            )
