"""Players tab renderer."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app_core.tab_filters_core import collect_year_split_league_options, scope_has_playoffs
from app_core.tab_state_core import blank_space_label
from app_core.tab_ui_core import render_tab_hero, render_tab_section
from app_core.ui_design_core import TONE_HTML_COLOR, cyan_section_header, team_badge_html, tone_for_pct
from mlcore.team_artifact_normalize import _norm_team_key
from mlcore.utils import sigmoid as _sigmoid


def player_lines_opp_split(dfb: pd.DataFrame, players: list[str], team: str,
                           opp_today: str) -> list[dict]:
    """Por player do TIME: média de K/D/A no bucket de força do adversário
    (forte/médio/fraco por win-rate no recorte), + o bucket do adversário de HOJE.
    Resolve a dor: 'a linha bate mas só vs times fracos'. Puro/testável."""
    cols = {"player", "team", "gameid", "kills", "deaths", "assists", "result"}
    if dfb is None or dfb.empty or not cols.issubset(dfb.columns) or not players:
        return []
    # win-rate por time (1 linha por jogo)
    g = dfb.drop_duplicates(subset=["gameid", "team"])
    wr = g.groupby("team")["result"].apply(
        lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).mean())).to_dict()
    vals = sorted(wr.values())
    lo = vals[len(vals) // 3] if vals else 0.4
    hi = vals[2 * len(vals) // 3] if vals else 0.6

    def bucket(w):
        return "forte" if w >= hi else ("fraco" if w <= lo else "médio")

    # adversário de cada (gameid, team)
    opp = {}
    for gid, grp in dfb.groupby("gameid"):
        teams = list(dict.fromkeys(grp["team"].astype(str)))
        if len(teams) == 2:
            opp[(gid, teams[0])] = teams[1]
            opp[(gid, teams[1])] = teams[0]
    today_bucket = bucket(wr.get(str(opp_today), 0.5))

    out = []
    for p in players:
        rows = dfb[dfb["player"].astype(str) == str(p)]
        if rows.empty:
            continue
        acc = {"forte": [], "médio": [], "fraco": [], "geral": []}
        for r in rows.itertuples(index=False):
            o = opp.get((str(r.gameid), str(r.team)))
            if o is None:
                continue
            b = bucket(wr.get(o, 0.5))
            kda = (float(r.kills), float(r.deaths), float(r.assists))
            acc[b].append(kda)
            acc["geral"].append(kda)

        def avg(lst, i):
            return sum(x[i] for x in lst) / len(lst) if lst else float("nan")

        out.append({
            "player": str(p), "today_bucket": today_bucket,
            "geral": {"k": avg(acc["geral"], 0), "d": avg(acc["geral"], 1),
                      "a": avg(acc["geral"], 2), "n": len(acc["geral"])},
            "hoje": {"k": avg(acc[today_bucket], 0), "d": avg(acc[today_bucket], 1),
                     "a": avg(acc[today_bucket], 2), "n": len(acc[today_bucket])},
        })
    return out


def _team_rows(dfb: pd.DataFrame, team: str) -> pd.DataFrame:
    """Linhas do time com match exato e fallback normalizado (grafias divergem
    entre o CSV de team-games e o de players)."""
    if dfb is None or dfb.empty or "team" not in dfb.columns:
        return dfb.iloc[0:0] if dfb is not None else pd.DataFrame()
    sub = dfb[dfb["team"].astype(str) == str(team)]
    if sub.empty:
        key = _norm_team_key(team)
        if key:
            sub = dfb[dfb["team"].apply(lambda x: _norm_team_key(x) == key)]
    return sub


def _subhead(st: Any, title: str) -> None:
    """Lightweight cyan-accented sub-header (consistent look, lighter than the
    full gradient bar) for the data sections inside the Players tab."""
    st.markdown(
        f"<div style='margin:14px 0 4px;padding-left:9px;border-left:3px solid #2daac4;"
        f"font-weight:700;font-size:1.02em;color:#2daac4;'>{title}</div>",
        unsafe_allow_html=True,
    )


def _render_players_result_card(
    st: Any, *, team_a: str, team_b: str, odd_a: float, odd_b: float, p_team_a: float,
    p_players_raw: float, p_team_ref: float, w_players: float,
    n_team_eff: float, n_players_eff: float, format_num_fn: Any, format_pct_fn: Any,
) -> None:
    """Headline output: two odd cards colored by win probability + a blend breakdown.

    Replaces the old plain-markdown line + two raw dataframes so the main result
    reads at a glance instead of as crude text.
    """
    pa = float(max(0.0, min(1.0, p_team_a))) * 100.0
    pb = 100.0 - pa
    col_a = TONE_HTML_COLOR.get({"elite": "good", "green": "good", "yellow": "warn", "red": "bad", "neutral": "neutral"}[tone_for_pct(pa)], "#fff")
    col_b = TONE_HTML_COLOR.get({"elite": "good", "green": "good", "yellow": "warn", "red": "bad", "neutral": "neutral"}[tone_for_pct(pb)], "#fff")

    def _odd_card(team: str, odd: float, pct: float, color: str) -> str:
        return (
            "<div style='flex:1;border:1px solid rgba(45,170,196,0.35);border-radius:10px;"
            "padding:12px 14px;background:rgba(45,170,196,0.06);text-align:center;'>"
            f"<div style='display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:6px;'>"
            f"{team_badge_html(team, size=34)}"
            f"<span style='font-weight:700;font-size:1.02em;'>{team}</span></div>"
            f"<div style='font-size:2.0em;font-weight:800;color:{color};line-height:1;'>{format_num_fn(odd, 2)}</div>"
            f"<div style='opacity:0.7;font-size:0.85em;margin-top:4px;'>{pct:.0f}% de vitória</div>"
            "</div>"
        )

    cyan_section_header(st, "E se os players pesassem? (simulação — NÃO é a odd do modelo)")
    st.caption(
        "⚠️ Esta odd **não sai do modelo que você aposta**. Ela mistura a probabilidade "
        "do time com um rating de player que é `logit(vitórias/jogos)` — ou seja, a taxa "
        "de vitória do TIME colada em cada jogador (todo mundo do Gen.G dá o mesmo número). "
        "Medido na season inteira, o peso ótimo da camada de players é **0,00**: o modelo "
        "já absorveu esse sinal pelo time. Use os blocos abaixo como leitura descritiva "
        "(quem joga, em que nível) — a odd de apostar é a da Visão Geral."
    )
    st.markdown(
        "<div style='display:flex;gap:12px;margin:8px 0 6px;'>"
        + _odd_card(str(team_a), odd_a, pa, col_a)
        + _odd_card(str(team_b), odd_b, pb, col_b)
        + "</div>",
        unsafe_allow_html=True,
    )

    def _chip(label: str, value: str) -> str:
        return (
            "<span style='display:inline-block;background:rgba(45,170,196,0.10);border-radius:6px;"
            "padding:3px 9px;margin:2px 4px 2px 0;font-size:0.86em;'>"
            f"<span style='opacity:0.6;'>{label}</span> <b>{value}</b></span>"
        )

    st.markdown(
        "<div style='margin:2px 0 4px;'>"
        + _chip("Players", format_pct_fn(p_players_raw))
        + _chip("Time (ML)", format_pct_fn(p_team_ref))
        + _chip("Blend", format_pct_fn(p_team_a))
        + _chip("Peso players", f"{format_num_fn(100.0 * w_players, 0)}%")
        + _chip("n_time", format_num_fn(n_team_eff, 1))
        + _chip("n_players", format_num_fn(n_players_eff, 1))
        + "</div>",
        unsafe_allow_html=True,
    )


def _team_rating(
    *,
    by_player: pd.DataFrame,
    players: list[str],
    format_num_fn,
) -> tuple[float, pd.DataFrame]:
    if not players:
        return 0.0, pd.DataFrame([])
    sub = by_player[by_player["player"].isin(players)].copy()
    if sub.empty:
        return 0.0, pd.DataFrame([])
    sub["w"] = sub["games"].clip(upper=20).astype(float)
    rating = float((sub["rating"] * sub["w"]).sum() / max(sub["w"].sum(), 1.0))
    view = sub[["player", "games", "wins", "p_smooth", "rating"]].copy()
    view["Win Rate"] = (100.0 * (view["wins"] / view["games"])).apply(lambda x: format_num_fn(x, 1))
    view["p_smooth"] = (100.0 * view["p_smooth"]).apply(lambda x: format_num_fn(x, 1))
    view["rating"] = view["rating"].apply(lambda x: format_num_fn(x, 3))
    view = view.rename(
        columns={"player": "Player", "games": "Games", "wins": "Wins", "p_smooth": "Smooth Win%", "rating": "Rating"}
    )
    return rating, view


def _build_trace_rows(out_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_rows = []
    for row in out_rows:
        trace_rows.append(
            {
                "lane": str(row.get("Lane", "")),
                "team": str(row.get("Team", "")),
                "player": str(row.get("Player", "")),
                "mercado": str(row.get("Mercado", "")),
                "lado": str(row.get("Lado", "")),
                "linha": str(row.get("Linha", "")),
                "odd_model": row.get("_odd_model_raw"),
                "odd_hist": row.get("_odd_hist_raw"),
                "odd_h2h": row.get("_odd_h2h_raw"),
                "odd_mix": row.get("_odd_mix_raw"),
                "calib_reason": str(row.get("_calib_reason", "")),
                "calib_source": str(row.get("_calib_source", "")),
                "h2h_n": row.get("_h2h_n"),
                "h2h_used": row.get("_h2h_used"),
                "obs_odds": str(row.get("Obs Odds", "")),
                "obs_mix": str(row.get("Obs Mix", "")),
            }
        )
    return trace_rows


def _render_lineup_value_block(
    *,
    st: Any,
    team_a: str,
    team_b: str,
    csv_sig: Any,
    load_player_rows_fn: Any,
    latest_lineup_names_fn: Any,
    norm_pos_fn: Any,
    fuzzy_match_one_fn: Any,
) -> None:
    """Manchete "quem joga hoje vale quanto" (redesign 2026-07-16): lineup
    habitual dos dois times com percentil de desempenho POR FUNÇÃO —
    independente de vitória (lição PandaSkill). Descritivo, n visível."""
    ta = str(team_a or "").strip()
    tb = str(team_b or "").strip()
    if not ta or not tb:
        return
    try:
        from app_core.players_value_core import (
            build_player_role_scores,
            lineup_value_view,
        )

        rows = load_player_rows_fn(csv_sig)
        if rows is None or rows.empty:
            return
        # NAO e `max(year)`. Duas temporadas se sobrepoem no calendario e o
        # maior ano pode ser um split recem-comecado com poucos times — o que
        # derrubou a lista do Radar de 394 para 18 em 28/08/2026.
        # Ver app_core/temporada_corrente_core.py.
        from app_core.temporada_corrente_core import ano_da_temporada_corrente
        year = ano_da_temporada_corrente(rows)
        scores = build_player_role_scores(rows, year=year)
        if scores.empty:
            return
        rows_year = rows
        if year is not None and "year" in rows.columns:
            rows_year = rows[pd.to_numeric(rows["year"], errors="coerce") == year]

        with st.container(border=True):
            render_tab_section(
                st,
                title=f"Quem joga hoje — {ta} × {tb}",
                subtitle=(
                    f"Lineup habitual + percentil de desempenho dentro da FUNÇÃO "
                    f"({'temporada ' + str(year) if year else 'CSV inteiro'}; KLA, dano/min, CS/min, visão/min; "
                    "independe de vitória — jogador bom em time ruim aparece)."
                ),
            )
            c1, c2 = st.columns(2)
            for col, tm in ((c1, ta), (c2, tb)):
                with col:
                    st.markdown(f"**{tm}**")
                    try:
                        lineup = latest_lineup_names_fn(df_in=rows_year, team_name=tm, norm_pos_fn=norm_pos_fn) or []
                    except Exception:
                        lineup = []
                    lineup = [x for x in lineup if str(x).strip()]
                    if not lineup:
                        st.caption("Sem lineup recente no recorte.")
                        continue
                    v = lineup_value_view(lineup, scores, fuzzy_match_one_fn=fuzzy_match_one_fn)
                    view = pd.DataFrame({
                        "Função": v["pos"].replace("", "?"),
                        "Player": v["player"],
                        "Percentil": v["percentile"].map(
                            lambda x: f"P{x:.0f}" if pd.notna(x) else "sem amostra"
                        ),
                        "J": v["games"].map(lambda n: int(n) if n else "-"),
                    })
                    st.dataframe(view, hide_index=True, use_container_width=True)
                    med = v["percentile"].dropna()
                    if not med.empty:
                        st.caption(f"Mediana do lineup: **P{med.median():.0f}** ({len(med)}/5 com amostra)")
    except Exception:
        # bloco novo NUNCA derruba a aba
        return


_LAB_POS = ["Top", "Jungle", "Mid", "Bot", "Support"]


def _ratings_individuais(art_players_fn: Any, dfb: pd.DataFrame, nomes: list[str]) -> dict[str, float]:
    """{nome do jogador -> rating INDIVIDUAL do players_artifact}.

    Por que existe (achado 22/07): o rating desta aba é logit(vitórias/jogos), ou
    seja, a taxa de vitória do TIME colada em cada jogador — todos do Gen.G davam
    +1.070. Para simular troca de roster isso mente. O artefato de players tem o
    rating individual de verdade (é o que o pipeline do ML usa); ele é indexado
    por `playerid|||Posição`, então mapeamos nome -> playerid pelo CSV.

    Devolve {} se o artefato não estiver disponível — quem chama cai no proxy e
    avisa na tela (nunca fingimos precisão que não temos).
    """
    if not callable(art_players_fn) or dfb is None or getattr(dfb, "empty", True):
        return {}
    cols = getattr(dfb, "columns", [])
    if "player" not in cols or "playerid" not in cols:
        return {}
    try:
        art = art_players_fn()
        players = getattr(art, "players", None) or {}
        if not players:
            return {}
    except Exception:
        return {}
    alvo = {str(n).strip() for n in (nomes or []) if str(n).strip()}
    if not alvo:
        return {}
    try:
        sub = dfb[dfb["player"].astype(str).isin(alvo)]
        pid_de = sub.dropna(subset=["playerid"]).groupby("player")["playerid"].last().to_dict()
    except (KeyError, ValueError, TypeError):
        return {}
    # índice playerid -> r (a posição vem depois do '|||' na chave)
    r_por_pid: dict[str, float] = {}
    for chave, estado in players.items():
        pid = str(chave).split("|||")[0]
        r = getattr(estado, "r", None)
        if pid and isinstance(r, (int, float)) and pid not in r_por_pid:
            r_por_pid[pid] = float(r)
    out: dict[str, float] = {}
    for nome, pid in pid_de.items():
        r = r_por_pid.get(str(pid))
        if r is not None and math.isfinite(r):
            out[str(nome)] = float(r)
    return out


def _render_roster_lab(
    *,
    st: Any,
    team_a: str,
    team_b: str,
    dfb: pd.DataFrame,
    by_player: pd.DataFrame,
    norm_pos_fn: Any,
    p_team_ref: float,
    p_real: float,
    n_team_eff: float,
    n_players_eff: float,
    fuse_kwargs: dict[str, Any],
    fuse_probs_by_precision_fn: Any,
    format_num_fn: Any,
    odd_from_prob_fn: Any,
    art_players_fn: Any = None,
) -> None:
    """Laboratório de escalação — "e se trocar alguém?" (pedido dele 22/07).

    Desenho guiado por pesquisa (FanGraphs Lab / WhatIfSports / Versus Sports):
    (a) pré-preenche com a escalação real e deixa editar; (b) a saída é
    PROBABILIDADE DE VITÓRIA, não um delta solto; (c) mostra o fator
    ENTROSAMENTO — a literatura de esports é enfática que troca recente é
    amortecida no curto prazo (o time ainda não acertou a comunicação).
    """
    from app_core.lineup_runtime_core import build_lineup_name_choices

    with st.expander("🧪 Laboratório de escalação — e se trocar alguém?", expanded=False):
        st.caption(
            "Monte uma escalação hipotética e veja o ML recalculado. Serve para "
            "testar troca de roster de split ('e se o Zeus fosse para a HLE?') "
            "antes de o time jogar com ela."
        )
        libera = st.checkbox(
            "Permitir jogador de outro time (simular transferência)",
            key="lab_pool_all",
            help="Ligado, cada posição lista o pool inteiro da liga; os do próprio time vêm primeiro.",
        )
        cols = st.columns(2)
        escolhas: dict[str, list[str]] = {}
        trocas = 0
        for col, tm in zip(cols, (team_a, team_b)):
            with col:
                st.markdown(f"**{tm}**")
                ops, time_de = build_lineup_name_choices(
                    df_pr=dfb, team_ui=tm, pos_canon=_LAB_POS,
                    norm_pos_fn=norm_pos_fn, include_all_teams=bool(libera),
                )
                sel: list[str] = []
                for pos in _LAB_POS:
                    lista = list(ops.get(pos) or [])
                    if not lista:
                        continue
                    key = f"lab_{tm}_{pos}"
                    if st.session_state.get(key) not in lista:
                        st.session_state[key] = lista[0]  # pré-preenche com o titular
                    escolhido = st.selectbox(
                        pos, lista, key=key,
                        format_func=lambda n, _t=time_de: (
                            f"{n} · {_t[n]}" if n in _t else str(n)
                        ),
                    )
                    sel.append(escolhido)
                    if escolhido != lista[0]:
                        trocas += 1
                escolhas[tm] = sel

        sel_a, sel_b = escolhas.get(team_a) or [], escolhas.get(team_b) or []
        if not sel_a or not sel_b:
            st.info("Sem jogadores suficientes no recorte para simular.")
            return

        tr_a_sim, df_sa = _team_rating(by_player=by_player, players=sel_a, format_num_fn=format_num_fn)
        tr_b_sim, df_sb = _team_rating(by_player=by_player, players=sel_b, format_num_fn=format_num_fn)

        # FONTE DE VERDADE do rating (achado 22/07): o rating desta aba é
        # logit(vitórias/jogos) = a taxa de vitória do TIME colada em cada
        # jogador — todos do Gen.G davam +1.070, zero informação individual.
        # Para simular troca de roster isso MENTE. Quando o artefato de players
        # está disponível, usamos o rating INDIVIDUAL (o que o pipeline do ML
        # usa de fato); só caímos no proxy se ele faltar, e avisamos na tela.
        fonte_rating = "proxy (taxa de vitória do time)"
        r_ind = _ratings_individuais(art_players_fn, dfb, sel_a + sel_b)
        if r_ind:
            ra = [r_ind[n] for n in sel_a if n in r_ind]
            rb = [r_ind[n] for n in sel_b if n in r_ind]
            if ra and rb:
                tr_a_sim = float(sum(ra) / len(ra))
                tr_b_sim = float(sum(rb) / len(rb))
                fonte_rating = f"rating individual do modelo ({len(ra)+len(rb)}/10 jogadores)"

        p_players_sim = _sigmoid(tr_a_sim - tr_b_sim)

        # BUG CORRIGIDO (report dele 22/07: "troquei 3 jogadores e deu +0.0 pp").
        # Eu usava o n_players_eff da escalação REAL. Quando ela está vazia
        # ("nenhum player reconhecido"), esse peso é ZERO — e com peso zero a
        # fusão ignora a camada de players, então trocar jogador não movia nada.
        # A simulação agora calcula o PRÓPRIO peso, dos jogadores escolhidos.
        def _games(dfx: pd.DataFrame) -> int:
            if dfx is None or dfx.empty or "Games" not in dfx.columns:
                return 0
            return int(pd.to_numeric(dfx["Games"], errors="coerce").fillna(0).sum())

        n_players_sim = float(min(_games(df_sa), _games(df_sb)))
        if not (n_players_sim > 0.0):
            n_players_sim = float(n_players_eff)
        _kw_sim = dict(fuse_kwargs)
        _kw_sim["coverage"] = float(max(0.0, min(1.0, n_players_sim / max(1.0, float(n_team_eff) + n_players_sim))))
        fused_sim = fuse_probs_by_precision_fn(
            p_team=p_team_ref, p_players=p_players_sim,
            n_team=n_team_eff, n_players=n_players_sim, **_kw_sim,
        )
        p_sim = float(max(0.02, min(0.98, float(fused_sim.get("p_fused", p_team_ref)))))
        dpp = 100.0 * (p_sim - float(p_real))

        m1, m2, m3 = st.columns(3)
        m1.metric(f"ML real · {team_a}", f"{100.0*float(p_real):.1f}%")
        m2.metric(f"ML simulado · {team_a}", f"{100.0*p_sim:.1f}%", f"{dpp:+.1f} pp")
        m3.metric("Odd justa simulada", f"{odd_from_prob_fn(p_sim):.2f}")
        # Transparência da fonte: sem isso, o número parece igual nos dois casos
        # — e o proxy (taxa de vitória do time) NÃO distingue jogador.
        if fonte_rating.startswith("rating individual"):
            st.caption(f"Fonte do rating: **{fonte_rating}** — mede o jogador, não o time.")
        else:
            st.warning(
                f"Fonte do rating: **{fonte_rating}**. Esse proxy dá o MESMO valor "
                "para todos os jogadores de um time, então ele **não distingue** quem "
                "entrou ou saiu. Trate a simulação como indicativa até o artefato de "
                "players ficar disponível neste recorte."
            )

        if trocas <= 0:
            st.caption("Escalação igual à titular — nada trocado ainda.")
        else:
            # Entrosamento: a literatura (e o próprio modelo, via continuidade de
            # roster) diz que time recém-montado rende ABAIXO do papel no curto
            # prazo. Avisamos em vez de vender o número como certeza.
            st.warning(
                f"**{trocas} troca(s)** em relação à escalação titular. "
                "Na prática, elenco recém-montado costuma render **abaixo do papel** "
                "no começo (entrosamento/comunicação) — o modelo trata isso inflando "
                "a incerteza quando o roster muda de verdade. Leia este número como "
                "**teto otimista** da escalação nova, não como previsão pronta."
            )


_PROP_STATS = (("kills", "Kills"), ("deaths", "Deaths"), ("assists", "Assists"))


def _lineup_do_lab(st: Any, team: str, dfb: pd.DataFrame, norm_pos_fn: Any) -> list[tuple[str, str]]:
    """[(rota, jogador)] lida do laboratório — trocou lá, muda aqui (pedido dele).

    Cai na escalação habitual do recorte quando o lab ainda não foi tocado."""
    from app_core.lineup_runtime_core import build_lineup_name_choices
    ops, _ = build_lineup_name_choices(df_pr=dfb, team_ui=team, pos_canon=_LAB_POS,
                                       norm_pos_fn=norm_pos_fn, include_all_teams=True)
    out: list[tuple[str, str]] = []
    for pos in _LAB_POS:
        escolhido = str(st.session_state.get(f"lab_{team}_{pos}", "") or "")
        if not escolhido:
            lista = list(ops.get(pos) or [])
            escolhido = lista[0] if lista else ""
        if escolhido:
            out.append((pos, escolhido))
    return out


def _taxas_kda(dfb: pd.DataFrame, nomes: list[str]) -> dict[str, dict[str, float]]:
    """{jogador: {kills/deaths/assists: média por jogo}} no recorte."""
    cols = [c for c, _ in _PROP_STATS]
    if dfb is None or dfb.empty or "player" not in dfb.columns:
        return {}
    faltando = [c for c in cols if c not in dfb.columns]
    if faltando:
        return {}
    sub = dfb[dfb["player"].astype(str).isin([str(n) for n in nomes])]
    if sub.empty:
        return {}
    g = sub.groupby("player")[cols].mean()
    return {str(k): {c: float(v[c]) for c in cols} for k, v in g.iterrows()}


def _render_props_table(
    *,
    st: Any,
    team_a: str,
    team_b: str,
    dfb: pd.DataFrame,
    norm_pos_fn: Any,
    mu_gap: float,
    format_num_fn: Any,
) -> None:
    """Tabela de props dos 10 jogadores + resumo com filtro de atropelo.

    Desenho escolhido por ele (22/07): TABELA ÚNICA de 10 linhas (Time A do TOP
    ao SUP, depois Time B), colunas Kills/Deaths/Assists onde ele digita a linha
    da casa; e um RESUMO que ESCONDE por padrão as linhas movidas pelo atropelo,
    com botão para revelar.

    O filtro não é opinião: o motor projeta em componentes separados, então
    "quanto da projeção veio do massacre" (stomp_share) é um número.
    """
    from app_core.player_props_core import build_prop_row, separar_por_atropelo

    cyan_section_header(st, "Props dos jogadores — linhas da casa vs modelo")
    st.caption(
        "Espelha a escalação do laboratório acima: trocou lá, muda aqui. Digite a "
        "linha da casa em cada célula; a projeção, a probabilidade e a odd justa "
        "aparecem do lado. O resumo no fim esconde as linhas que só batem por atropelo."
    )

    lineup = [(team_a, pos, pl) for pos, pl in _lineup_do_lab(st, team_a, dfb, norm_pos_fn)]
    lineup += [(team_b, pos, pl) for pos, pl in _lineup_do_lab(st, team_b, dfb, norm_pos_fn)]
    if not lineup:
        st.info("Sem escalação identificada neste recorte.")
        return
    taxas = _taxas_kda(dfb, [pl for _, _, pl in lineup])
    if not taxas:
        st.info("Sem K/D/A por jogador neste recorte (o CSV precisa das colunas kills/deaths/assists).")
        return

    linhas_calc: list[dict] = []
    cab = st.columns([1.5, 0.7, 1.3, 1.3, 1.3])
    for c, t in zip(cab, ("Jogador", "Rota", "Kills", "Deaths", "Assists")):
        c.markdown(f"<div style='font-size:0.78em;opacity:0.65;text-transform:uppercase;"
                   f"letter-spacing:0.05em;'>{t}</div>", unsafe_allow_html=True)

    for tm, pos, pl in lineup:
        tx = taxas.get(str(pl)) or {}
        # o gap entra com sinal: positivo p/ o time A, negativo p/ o B
        gap_do_time = float(mu_gap) if tm == team_a else -float(mu_gap)
        cols = st.columns([1.5, 0.7, 1.3, 1.3, 1.3])
        cols[0].markdown(f"**{pl}**<br><span style='opacity:0.55;font-size:0.8em;'>{tm}</span>",
                         unsafe_allow_html=True)
        cols[1].markdown(f"<div style='padding-top:6px;'>{pos}</div>", unsafe_allow_html=True)
        for i, (stat, rot) in enumerate(_PROP_STATS):
            with cols[2 + i]:
                key = f"prop_{tm}_{pos}_{stat}"
                txt = st.text_input(rot, key=key, placeholder="linha", label_visibility="collapsed")
                base = tx.get(stat)
                if base is None:
                    continue
                try:
                    line = float(str(txt).replace(",", ".").strip()) if str(txt).strip() else None
                except ValueError:
                    line = None
                r = build_prop_row(player=pl, role=pos, team=tm, stat=stat,
                                   line=(line if line is not None else base),
                                   base_rate=base, mu_gap=gap_do_time)
                if not r.get("ok"):
                    continue
                if line is not None:
                    r["linha_informada"] = True
                    linhas_calc.append(r)
                    cor = "#22c55e" if r["p_over"] >= 0.55 else "#facc15" if r["p_over"] >= 0.45 else "#ef4444"
                    selo = " ⚠" if r["movido_por_atropelo"] else ""
                    st.markdown(
                        f"<div style='font-size:0.82em;margin-top:-6px;'>"
                        f"proj <b>{format_num_fn(r['proj'], 1)}</b> · "
                        f"<span style='color:{cor};font-weight:700;'>{100*r['p_over']:.0f}%</span> · "
                        f"odd {format_num_fn(r['odd_justa_over'], 2)}{selo}</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='font-size:0.78em;opacity:0.5;margin-top:-6px;'>"
                                f"média {format_num_fn(base, 1)}</div>", unsafe_allow_html=True)

    if not linhas_calc:
        st.caption("Digite ao menos uma linha para o resumo aparecer.")
        return

    limpas, atropelo = separar_por_atropelo(linhas_calc)
    limpas.sort(key=lambda r: -float(r.get("p_over") or 0.0))
    cyan_section_header(st, "Resumo — melhores oportunidades")

    def _tabela(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame([{
            "Jogador": r["player"], "Rota": r["role"], "Time": r["team"],
            "Mercado": r["stat"].capitalize(), "Linha": format_num_fn(r["line"], 1),
            "Projeção": format_num_fn(r["proj"], 1),
            "P(Over)": f"{100*r['p_over']:.0f}%",
            "Odd justa": format_num_fn(r["odd_justa_over"], 2),
        } for r in rows])

    if limpas:
        st.dataframe(_tabela(limpas), width="stretch", hide_index=True)
    else:
        st.info("Nenhuma linha limpa — todas foram marcadas como movidas pelo atropelo.")

    if atropelo:
        st.warning(
            f"**{len(atropelo)} linha(s) ocultada(s)** — nelas, a maior parte da projeção "
            "vem do desequilíbrio do confronto, não do jogador. É o caso Gen.G × Pain: "
            "batem (ou não) por causa do atropelo, e a casa já sabe disso."
        )
        if st.checkbox("Mostrar as linhas movidas pelo atropelo", key="props_show_stomp"):
            st.dataframe(_tabela(atropelo), width="stretch", hide_index=True)


def render_players_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    session_state = st.session_state
    render_tab_hero(
        st, title="Players",
        subtitle=(
            "Quem joga hoje e em que nível — leitura DESCRITIVA. A odd de apostar sai da "
            "Visão Geral: medido na season inteira, o peso ótimo dos players no modelo é 0,00."
        ),
    )
    _render_lineup_value_block(
        st=st,
        team_a=ctx["teamA"],
        team_b=ctx["teamB"],
        csv_sig=ctx["csv_sig"],
        load_player_rows_fn=ctx["load_player_rows_fn"],
        latest_lineup_names_fn=ctx["latest_lineup_names_fn"],
        norm_pos_fn=ctx["norm_pos_fn"],
        fuzzy_match_one_fn=ctx["fuzzy_match_one_fn"],
    )

    team_games = ctx["team_games"]
    Filters = ctx["Filters"]
    csv_sig = ctx["csv_sig"]
    team_a = ctx["teamA"]
    team_b = ctx["teamB"]
    load_player_rows_fn = ctx["load_player_rows_fn"]
    apply_filters_players_fn = ctx["apply_filters_players_fn"]
    latest_lineup_names_fn = ctx["latest_lineup_names_fn"]
    norm_pos_fn = ctx["norm_pos_fn"]
    parse_players_text_fn = ctx["parse_players_text_fn"]
    resolve_players_fn = ctx["resolve_players_fn"]
    fuzzy_match_one_fn = ctx["fuzzy_match_one_fn"]
    ensure_players_filter_state_fn = ctx["ensure_players_filter_state_fn"]
    ensure_players_text_state_fn = ctx["ensure_players_text_state_fn"]
    apply_players_autofill_state_fn = ctx["apply_players_autofill_state_fn"]
    format_num_fn = ctx["format_num_fn"]
    format_pct_fn = ctx["format_pct_fn"]
    odd_from_prob_fn = ctx["odd_from_prob_fn"]
    state_get_float_fn = ctx["state_get_float_fn"]
    fuse_probs_by_precision_fn = ctx["fuse_probs_by_precision_fn"]
    playoffs_opt = ctx["playoffs_opt"]

    years, splits, leagues = collect_year_split_league_options(team_games)
    ensure_players_filter_state_fn(session_state)

    def _ensure_choice(key: str, options: list[str], fallback: str | None = None) -> str:
        default = str(fallback if fallback is not None else (options[0] if options else ""))
        current = str(session_state.get(key, default) or default)
        allowed = {str(x) for x in options}
        if current not in allowed:
            current = default
            session_state[key] = current
        return current

    def _on_pl_league_change() -> None:
        session_state["pl_split"] = ""

    def _splits_for_league(league_sel: str) -> list[str]:
        """Splits available for the selected league (fallback: all splits)."""
        try:
            if (league_sel and team_games is not None and not team_games.empty
                    and "league" in team_games.columns and "split" in team_games.columns):
                return sorted(
                    str(x)
                    for x in team_games.loc[team_games["league"].astype(str) == str(league_sel), "split"]
                    .fillna("").astype(str).unique().tolist()
                    if str(x)
                )
        except Exception:
            pass
        return splits

    # Single clean filter row: Ano | Liga | Split | Playoffs.
    with st.container(border=True):
        render_tab_section(st, title="Filtros", subtitle="Feche o recorte antes de abrir lineup, props e medias por jogador.")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            year_options = [""] + [str(y) for y in years]
            _ensure_choice("pl_year", year_options, "")
            p_year = st.selectbox("Ano", options=year_options, key="pl_year", format_func=blank_space_label)
        with pc2:
            league_options = ["", *leagues]
            _ensure_choice("pl_league", league_options, "")
            p_league = st.selectbox(
                "Liga",
                options=league_options,
                key="pl_league",
                format_func=blank_space_label,
                on_change=_on_pl_league_change,
            )
        # Splits depend on the league just picked; compute before rendering Split.
        splits_pl = _splits_for_league(p_league)
        with pc3:
            split_options = ["", *splits_pl]
            _ensure_choice("pl_split", split_options, "")
            p_split = st.selectbox("Split", options=split_options, key="pl_split", format_func=blank_space_label)
        # Playoffs box appears ONLY when the chosen (liga, ano, split) actually
        # has playoff games — e.g. pick Split 2 and, once playoffs start, the
        # box shows up. Otherwise it stays hidden and no playoff filter applies.
        _has_po = scope_has_playoffs(team_games, league=p_league, year=p_year, split=p_split)
        if _has_po:
            with pc4:
                playoffs_options = ["", "Regular", "Playoffs"]
                _ensure_choice("pl_playoffs", playoffs_options, "")
                p_playoffs = st.selectbox(
                    "Playoffs",
                    options=playoffs_options,
                    key="pl_playoffs",
                    format_func=blank_space_label,
                )
            p_playoffs = st.session_state.get("pl_playoffs", "") or ""
        else:
            p_playoffs = ""

    base_filters = Filters(
        year=None if not p_year else int(p_year),
        league=None if not p_league else p_league,
        split=None if not p_split else p_split,
        playoffs=None if not p_playoffs else (True if p_playoffs == "Playoffs" else False),
    )

    with st.spinner("Carregando players do CSV (1x por sessão / cache)..."):
        df_pl = load_player_rows_fn(csv_sig)

    if df_pl is None or df_pl.empty:
        st.warning("Não achei rows de players nesse CSV.")
        dfb = pd.DataFrame()
    else:
        dfb = apply_filters_players_fn(df_pl, base_filters).copy()

    if dfb.empty:
        st.info("Sem jogos nesse recorte.")
        return

    group_df = dfb.copy()
    group_df["win"] = (pd.to_numeric(group_df.get("result"), errors="coerce").fillna(0).astype(int) > 0).astype(int)
    by_player = group_df.groupby("player").agg(games=("win", "size"), wins=("win", "sum")).reset_index()
    by_player["p_smooth"] = (by_player["wins"] + 2.0) / (by_player["games"] + 4.0)
    by_player["rating"] = np.log(by_player["p_smooth"] / (1.0 - by_player["p_smooth"]))

    vocab = sorted([x for x in by_player["player"].fillna("").astype(str).unique().tolist() if x])
    vocab_a = sorted([x for x in _team_rows(dfb, team_a)["player"].fillna("").astype(str).unique().tolist() if x])
    vocab_b = sorted([x for x in _team_rows(dfb, team_b)["player"].fillna("").astype(str).unique().tolist() if x])
    if not vocab_a:
        vocab_a = vocab
    if not vocab_b:
        vocab_b = vocab

    render_tab_section(st, title="Cole os players", subtitle="Cole os nomes por time ou use o autofill do lineup mais recente.")
    st.caption("Um por linha. O matching é fuzzy (se errar um pouco o nome, ele sugere).")
    a0, a1 = st.columns([1.4, 2.6])
    with a0:
        if st.button("Auto-preencher players dos times", key="pl_autofill_players"):
            auto_a = latest_lineup_names_fn(df_in=dfb, team_name=team_a, norm_pos_fn=norm_pos_fn)
            auto_b = latest_lineup_names_fn(df_in=dfb, team_name=team_b, norm_pos_fn=norm_pos_fn)
            apply_players_autofill_state_fn(session_state, team_a_players=auto_a, team_b_players=auto_b)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with a1:
        st.caption("Preenche Top/JG/Mid/ADC/Sup com o lineup mais recente encontrado no recorte atual.")

    ensure_players_text_state_fn(session_state)
    players_a = parse_players_text_fn(session_state.get("pl_txtA_applied", ""))
    players_b = parse_players_text_fn(session_state.get("pl_txtB_applied", ""))
    resolved_a, notes_a = resolve_players_fn(
        names=players_a,
        vocab=vocab_a,
        fuzzy_match_fn=lambda nm, vv, c: fuzzy_match_one_fn(nm, vv, cutoff=c),
        cutoff=0.6,
    )
    resolved_b, notes_b = resolve_players_fn(
        names=players_b,
        vocab=vocab_b,
        fuzzy_match_fn=lambda nm, vv, c: fuzzy_match_one_fn(nm, vv, cutoff=c),
        cutoff=0.6,
    )
    resolved_a_view = resolved_a if resolved_a else players_a
    resolved_b_view = resolved_b if resolved_b else players_b

    if notes_a or notes_b:
        with st.expander("Ver matching / sugestões", expanded=False):
            if notes_a:
                st.markdown("**Time A**")
                st.write(notes_a)
            if notes_b:
                st.markdown("**Time B**")
                st.write(notes_b)

    tr_a, df_a = _team_rating(by_player=by_player, players=resolved_a, format_num_fn=format_num_fn)
    tr_b, df_b = _team_rating(by_player=by_player, players=resolved_b, format_num_fn=format_num_fn)

    p_players_raw = _sigmoid(tr_a - tr_b)
    p_team_ref = state_get_float_fn(session_state, "_p_map_used", float("nan"))
    if not math.isfinite(p_team_ref):
        p_team_ref = 0.5
    games_a = int(pd.to_numeric(df_a.get("Games"), errors="coerce").fillna(0).sum()) if not df_a.empty else 0
    games_b = int(pd.to_numeric(df_b.get("Games"), errors="coerce").fillna(0).sum()) if not df_b.empty else 0
    n_players_eff = float(min(games_a, games_b))
    n_team_eff = state_get_float_fn(session_state, "_ml_team_games_min", float("nan"))
    # Mesmos parâmetros de fusão usados no card real E no laboratório abaixo —
    # se divergirem, a comparação "real vs simulado" vira mentira.
    _fuse_kwargs = dict(
        team_scale=state_get_float_fn(session_state, "_fit_team_scale_used", state_get_float_fn(session_state, "_fit_team_scale", 1.0)),
        players_scale=state_get_float_fn(
            st.session_state,
            "_fit_players_scale_used",
            state_get_float_fn(st.session_state, "_fit_players_scale", 1.0),
        ),
        coverage=float(max(0.0, min(1.0, n_players_eff / max(1.0, n_team_eff + n_players_eff)))),
        transfer_signal=0.0,
        season_phase=("playoffs" if str(playoffs_opt).lower() in {"playoffs", "true", "1"} else "auto"),
    )
    fused = fuse_probs_by_precision_fn(
        p_team=p_team_ref,
        p_players=p_players_raw,
        n_team=n_team_eff,
        n_players=n_players_eff,
        **_fuse_kwargs,
    )
    p_team_a = float(max(0.02, min(0.98, float(fused.get("p_fused", p_team_ref)))))
    w_players = float(fused.get("w_players", float("nan")))
    odd_a = odd_from_prob_fn(p_team_a)
    odd_b = odd_from_prob_fn(1.0 - p_team_a)

    _render_players_result_card(
        st, team_a=team_a, team_b=team_b, odd_a=odd_a, odd_b=odd_b, p_team_a=p_team_a,
        p_players_raw=p_players_raw, p_team_ref=p_team_ref, w_players=w_players,
        n_team_eff=n_team_eff, n_players_eff=n_players_eff,
        format_num_fn=format_num_fn, format_pct_fn=format_pct_fn,
    )

    try:
        _render_roster_lab(
            st=st, team_a=team_a, team_b=team_b,
            dfb=dfb, by_player=by_player, norm_pos_fn=norm_pos_fn,
            p_team_ref=p_team_ref, p_real=p_team_a,
            n_team_eff=n_team_eff, n_players_eff=n_players_eff,
            fuse_kwargs=_fuse_kwargs,
            fuse_probs_by_precision_fn=fuse_probs_by_precision_fn,
            format_num_fn=format_num_fn, odd_from_prob_fn=odd_from_prob_fn,
            art_players_fn=ctx.get("players_artifact_fn"),
        )
    except (KeyError, TypeError, ValueError, AttributeError, ImportError) as _lab_exc:
        # NUNCA derruba a aba — mas também NUNCA some calado (lição 22/07: erro
        # engolido vira mistério). Mostra o motivo para dar p/ consertar.
        st.warning(
            f"Laboratório de escalação indisponível neste recorte "
            f"({type(_lab_exc).__name__}: {_lab_exc}). O resto da aba segue normal."
        )

    try:
        # gap de força entre os times, na mesma fonte do laboratório (rating
        # individual quando disponível). É ele que dirige o filtro de atropelo.
        _r_ind = _ratings_individuais(
            ctx.get("players_artifact_fn"), dfb,
            [p for _, p in _lineup_do_lab(st, team_a, dfb, norm_pos_fn)]
            + [p for _, p in _lineup_do_lab(st, team_b, dfb, norm_pos_fn)],
        )
        _ra = [v for k, v in _r_ind.items()
               if k in {p for _, p in _lineup_do_lab(st, team_a, dfb, norm_pos_fn)}]
        _rb = [v for k, v in _r_ind.items()
               if k in {p for _, p in _lineup_do_lab(st, team_b, dfb, norm_pos_fn)}]
        _gap = (sum(_ra) / len(_ra) - sum(_rb) / len(_rb)) if (_ra and _rb) else 0.0
        _render_props_table(
            st=st, team_a=team_a, team_b=team_b, dfb=dfb,
            norm_pos_fn=norm_pos_fn, mu_gap=float(_gap), format_num_fn=format_num_fn,
        )
    except (KeyError, TypeError, ValueError, AttributeError, ImportError) as _props_exc:
        st.warning(
            f"Tabela de props indisponível neste recorte "
            f"({type(_props_exc).__name__}: {_props_exc}). O resto da aba segue normal."
        )

    with st.expander("Detalhe por jogador (ratings)", expanded=False):
        cta, ctb = st.columns(2)
        with cta:
            st.markdown(f"**{team_a}**")
            if df_a.empty:
                st.info("Nenhum player reconhecido na base.")
            else:
                st.dataframe(df_a, width="stretch", hide_index=True)
        with ctb:
            st.markdown(f"**{team_b}**")
            if df_b.empty:
                st.info("Nenhum player reconhecido na base.")
            else:
                st.dataframe(df_b, width="stretch", hide_index=True)

    # ---- Linhas de player · ajuste por força do adversário (pedido dele 24/07) ----
    # Dor: "a linha bate mas só vs times fracos (stomp)". Mostra a média K/D/A
    # contra times do NÍVEL do adversário de hoje vs a média geral (inflada).
    from html import escape as _escp
    st.divider()
    render_tab_section(
        st, title="Linhas de player · ajuste por adversário",
        subtitle="Média K/D/A contra times do NÍVEL de hoje vs a média geral — pega linha inflada por stomp.",
    )
    st.caption("Números: **nível de hoje / geral**. Vermelho = geral inflada (cai contra o nível de hoje); verde = sobe.")
    _pc = st.columns(2)
    for _c, _team, _opp, _players in ((_pc[0], team_a, team_b, resolved_a_view),
                                      (_pc[1], team_b, team_a, resolved_b_view)):
        with _c:
            _rows = player_lines_opp_split(dfb, list(_players), _team, _opp)
            _lvl = _rows[0]["today_bucket"] if _rows else "?"
            st.markdown(f"**{_escp(str(_team))}** · adv de hoje: nível **{_escp(_lvl)}**")
            if not _rows:
                st.caption("Sem players/jogos no recorte.")
            for _r in _rows:
                _h, _g = _r["hoje"], _r["geral"]

                def _fmt(k, h=_h, g=_g):
                    hv, gv = h[k], g[k]
                    hs = f"{hv:.1f}" if hv == hv else "-"
                    gs = f"{gv:.1f}" if gv == gv else "-"
                    cor = ""
                    if h["n"] and g["n"] and hv == hv and gv == gv:
                        if hv < gv * 0.85:
                            cor = "#ef4444"
                        elif hv > gv * 1.15:
                            cor = "#22c55e"
                    hb = f"<b style='color:{cor}'>{hs}</b>" if cor else f"<b>{hs}</b>"
                    return f"{hb}<span style='opacity:.5'>/{gs}</span>"

                st.markdown(
                    f"<div style='font-size:.88em;margin:3px 0'><b>{_escp(str(_r['player']))}</b> "
                    f"<span style='opacity:.5'>(n{_h['n']}/{_g['n']})</span><br>"
                    f"K {_fmt('k')} · D {_fmt('d')} · A {_fmt('a')}</div>",
                    unsafe_allow_html=True,
                )
