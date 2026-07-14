from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import streamlit as st

# Opções do seletor de BO. Constante de módulo para permitir override por
# outros hosts (o Gabinete Dota inclui BO2; o LoL segue 1/3/5).
BO_OPTIONS: List[int] = [1, 3, 5]

from app_core.flow_runtime_core import (
    apply_flow_line_values,
    consume_pending_auto_analyze,
    ensure_matchup_selection_state,
    request_analysis_refresh,
    request_pending_reset_matchup,
    set_flow_stage,
    set_lines_loaded_key,
    set_matchup_key,
    set_pending_auto_analyze,
    set_pending_load,
)
from app_core.schedule_state_core import (
    get_schedule_games,
    get_schedule_line_fields,
    get_schedule_line_inputs,
    init_schedule_line_inputs,
    resolve_schedule_line_hc_team,
    save_schedule_games,
)


def _safe_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def render_matchup_selection(
    session_state: dict[str, Any],
    *,
    flow_stage: str,
    teams: Sequence[str],
    season_label: str,
    matchup_key_fn: Callable[[Any, Any, Any], str],
    save_lines_for_matchup_fn: Callable[[str], Any],
    load_lines_for_matchup_fn: Callable[[str], Any],
    reset_flow_fn: Callable[..., Any],
) -> tuple[str, str, int]:
    st.subheader("Time 1 - Time 2")
    st.subheader("Confronto")

    if flow_stage == "select":
        games = get_schedule_games(session_state)
        if isinstance(games, list) and games:
            with st.expander("Selecionar jogo salvo (Agenda)", expanded=False):
                labels: list[str] = []
                for game in games:
                    try:
                        d = str(game.get("date", "") or "").strip()
                        t = str(game.get("time", "") or "").strip()
                        dt = (d + (" " + t if t else "")).strip()
                        league = str(game.get("league", "") or "").strip()
                        ta = str(game.get("teamA", "") or "").strip()
                        tb = str(game.get("teamB", "") or "").strip()
                        bo_g = game.get("bo", 3)
                    except Exception:
                        dt, league, ta, tb, bo_g = "", "", "", "", 3
                    parts: list[str] = []
                    if dt:
                        parts.append(dt)
                    if league:
                        parts.append(league)
                    parts.append(f"{ta} x {tb}")
                    parts.append(f"BO{bo_g}")
                    labels.append(" | ".join(parts))

                pick = st.selectbox(
                    "Jogo salvo",
                    options=[-1, *list(range(len(labels)))],
                    key="main_saved_game_idx",
                    format_func=lambda x: "Selecione ..." if int(x) == -1 else labels[int(x)],
                )
                bcols = st.columns([1, 2])
                with bcols[0]:
                    do_pick = st.button(
                        "Usar este jogo",
                        type="primary",
                        width="stretch",
                        disabled=(int(pick) == -1),
                        key="main_saved_game_use",
                    )
                with bcols[1]:
                    st.caption("Dica: se o nome do time não bater com o CSV, use a aba **Jogos** para resolver/editar o texto.")

                if int(pick) >= 0:
                    try:
                        selected_game = games[int(pick)]
                    except Exception:
                        selected_game = None

                    if isinstance(selected_game, dict):
                        team_a_sel = str(selected_game.get("teamA", "") or "").strip()
                        team_b_sel = str(selected_game.get("teamB", "") or "").strip()
                        bo_sel = int(selected_game.get("bo", 3) or 3) if str(selected_game.get("bo", "")).strip() else 3

                        lines_saved = selected_game.get("lines_saved")
                        if not isinstance(lines_saved, dict):
                            lines_saved = {}

                        with st.expander("Linhas deste jogo (opcional) — salvar e/ou analisar", expanded=False):
                            prefix = f"ag_lines_{int(pick)}_"
                            fields = list(get_schedule_line_fields())
                            init_schedule_line_inputs(session_state, prefix, lines_saved, fields=tuple(fields))

                            st.markdown("**Mercados principais**")
                            g1, g2, g3 = st.columns(3)
                            with g1:
                                st.text_input("Kills (ex: 26.5, 27.5)", key=prefix + "lines_kills")
                                st.text_input("Torres (ex: 10.5)", key=prefix + "lines_towers")
                            with g2:
                                st.text_input("Dragões (ex: 4.5)", key=prefix + "lines_dragons")
                                st.text_input("Barões (ex: 0.5)", key=prefix + "lines_barons")
                            with g3:
                                st.text_input("Inibidores (ex: 1.5)", key=prefix + "lines_inhib")
                                st.text_input("Tempo (ex: 32, 32.5)", key=prefix + "lines_time")

                            with st.expander("Extras (opcional)", expanded=False):
                                st.text_input(f"Total Kills {team_a_sel or 'Time 1'} (ex: 16.5)", key=prefix + "teamA_kills_line_text")
                                st.text_input(f"Total Kills {team_b_sel or 'Time 2'} (ex: 15.5)", key=prefix + "teamB_kills_line_text")

                                st.caption("Handicap: interpreta como (métrica_for + handicap) > métrica_against (sem push; use .5).")
                                if team_a_sel and team_b_sel:
                                    _cur, hc_opts = resolve_schedule_line_hc_team(
                                        session_state,
                                        prefix,
                                        team_a=team_a_sel,
                                        team_b=team_b_sel,
                                    )
                                    st.selectbox(
                                        "Handicap Kills — Time",
                                        options=hc_opts,
                                        key=prefix + "hc_kills_team",
                                    )
                                    hc_team, _ = resolve_schedule_line_hc_team(
                                        session_state,
                                        prefix,
                                        team_a=team_a_sel,
                                        team_b=team_b_sel,
                                        write_back=False,
                                    )
                                else:
                                    hc_team = team_a_sel or team_b_sel

                                st.text_input(f"HC Kills ({hc_team or 'Time'}) (ex: -2.5)", key=prefix + "hc_kills_text")
                                st.text_input(f"HC Torres ({team_a_sel or 'Time 1'}) (ex: -3.5)", key=prefix + "hc_towers_text")
                                st.text_input(f"HC Dragões ({team_a_sel or 'Time 1'}) (ex: -1.5)", key=prefix + "hc_dragons_text")
                                st.text_input("Linha Kills (para ML + Totais)", key=prefix + "combo_kills_text")
                                st.text_input("Linha Tempo (para ML + Totais)", key=prefix + "combo_time_text")

                            s1, s2 = st.columns(2)
                            with s1:
                                save_lines = st.button("Salvar linhas", width="stretch", key=f"ag_save_lines_{int(pick)}")
                            with s2:
                                analyze = st.button(
                                    "Analisar confronto",
                                    type="primary",
                                    width="stretch",
                                    key=f"ag_analyze_{int(pick)}",
                                )

                            if save_lines or analyze:
                                line_values = get_schedule_line_inputs(session_state, prefix, fields=tuple(fields))
                                try:
                                    games[int(pick)]["lines_saved"] = line_values
                                    save_schedule_games(session_state, games)
                                except Exception:
                                    pass

                                if save_lines:
                                    st.success("Linhas salvas para este jogo.")
                                    _safe_rerun()

                                if analyze:
                                    set_pending_load(
                                        session_state,
                                        {
                                            "_reset_flow": True,
                                            "teamA": team_a_sel if team_a_sel in teams else "",
                                            "teamB": team_b_sel if team_b_sel in teams else "",
                                            "bo": bo_sel,
                                        },
                                    )
                                    set_pending_auto_analyze(session_state, {"lines": line_values})
                                    _safe_rerun()

                if do_pick and int(pick) >= 0:
                    try:
                        game = games[int(pick)]
                        prefix = f"ag_lines_{int(pick)}_"
                        current_line_values = get_schedule_line_inputs(session_state, prefix, fields=tuple(get_schedule_line_fields()))
                        if isinstance(game, dict):
                            game["lines_saved"] = dict(current_line_values)
                            save_schedule_games(session_state, games)
                        payload = {"_reset_flow": True}
                        if str(game.get("teamA", "")) in teams:
                            payload["teamA"] = str(game.get("teamA"))
                        if str(game.get("teamB", "")) in teams:
                            payload["teamB"] = str(game.get("teamB"))
                        if str(game.get("bo")) in {str(x) for x in BO_OPTIONS}:
                            payload["bo"] = int(game.get("bo"))
                        if isinstance(game, dict) and isinstance(game.get("lines_saved"), dict):
                            payload["_flow_line_values"] = dict(game.get("lines_saved") or {})
                        set_pending_load(session_state, payload)
                        _safe_rerun()
                    except Exception:
                        pass

    locked = flow_stage in ("lines", "results")
    ensure_matchup_selection_state(session_state)
    team_opts = ["", *list(teams)]
    if str(session_state.get("teamA", "") or "") not in team_opts:
        session_state["teamA"] = ""
    if str(session_state.get("teamB", "") or "") not in team_opts:
        session_state["teamB"] = ""
    if str(session_state.get("bo", "1") or "1") not in {str(x) for x in BO_OPTIONS}:
        session_state["bo"] = int(BO_OPTIONS[0])

    def _fmt_team_placeholder(x: str) -> str:
        return "" if str(x) == "" else str(x)

    cA, cB, cC, cD = st.columns([3, 3, 1, 3])
    with cA:
        team_a = st.selectbox("Time 1", options=team_opts, key="teamA", disabled=locked, format_func=_fmt_team_placeholder)
    with cB:
        team_b = st.selectbox("Time 2", options=team_opts, key="teamB", disabled=locked, format_func=_fmt_team_placeholder)
    with cC:
        bo = st.selectbox("BO", list(BO_OPTIONS), key="bo", disabled=locked)
    with cD:
        if flow_stage == "select":
            st.write("Ação")
            can_analyze = bool(team_a) and bool(team_b) and (str(team_a) != str(team_b))
            pending_auto = consume_pending_auto_analyze(session_state) if can_analyze else {}
            if can_analyze and pending_auto:
                lines_payload = pending_auto.get("lines") if isinstance(pending_auto, dict) else None
                if isinstance(lines_payload, dict):
                    key = matchup_key_fn(season_label, team_a, team_b)
                    set_matchup_key(session_state, key)
                    apply_flow_line_values(session_state, lines_payload, fields=tuple(get_schedule_line_fields()))
                    save_lines_for_matchup_fn(key)
                    set_flow_stage(session_state, "results")
                    request_analysis_refresh(session_state)
                    _safe_rerun()

            if st.button("Analisar Confronto", type="primary", width="stretch", disabled=not can_analyze, key="btn_go_lines_quick"):
                key = matchup_key_fn(season_label, team_a, team_b)
                set_matchup_key(session_state, key)
                load_lines_for_matchup_fn(key)
                set_lines_loaded_key(session_state, key)
                # Skip the legacy "lines" stage — users now input lines inside
                # the per-market sub-tabs (Análise de Linha v3). Go straight
                # to results so the tabs render immediately.
                set_flow_stage(session_state, "results")
                request_analysis_refresh(session_state)
                _safe_rerun()
        elif flow_stage == "lines":
            if st.button("Trocar confronto", width="stretch"):
                request_pending_reset_matchup(session_state)
                _safe_rerun()
        else:
            if st.button("Trocar confronto", width="stretch", key="btn_swap_matchup_results"):
                request_pending_reset_matchup(session_state)
                _safe_rerun()

    return str(team_a or ""), str(team_b or ""), int(bo)


def team_with_league_label(name: str, league: str) -> str:
    """Format team name with optional league suffix: 'TeamName (LCK)'."""
    team = str(name or "").strip()
    lg = str(league or "").strip()
    return f"{team} ({lg})" if lg else team


def canon_team_key(name: str) -> str:
    """Casefolded, whitespace-normalized team key for deduplication and comparison."""
    s = ("" if name is None else str(name)).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def norm_team_token(s: Any) -> str:
    """Normalize a team name token to lowercase alphanumeric words for fuzzy matching."""
    t = str(s or "").strip().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def team_match_score(name: str, patterns: List[str]) -> int:
    """Score a team name against search patterns: 100=exact, 70=substring, 0=no match."""
    nm = norm_team_token(name)
    score = 0
    for p in patterns:
        pp = norm_team_token(p)
        if not pp:
            continue
        if nm == pp:
            score = max(score, 100)
        elif pp in nm:
            score = max(score, 70)
    return score


def pick_team_name(candidates: List[str], patterns: List[str]) -> Optional[str]:
    """Return the best-matching team name from candidates using pattern scoring, or None."""
    best_name = None
    best_score = -1
    for c in candidates:
        sc = team_match_score(c, patterns)
        if sc > best_score:
            best_score = sc
            best_name = c
    return best_name if best_score >= 60 else None


def team_canon_maps(csv_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    """Build team name canonicalization maps from CSV (deduplication by frequency).

    Returns:
      - variant_to_best_key: canon_key -> best_display_name
      - best_to_variants: best_display_name -> [variants...]
      - best_list: [best_display_name...]
    """
    from mlcore.stats_teams import load_team_rows

    csv_path, _size, _mtime = csv_sig
    rows = load_team_rows(csv_path)
    s = rows.get("teamname")
    if s is None:
        return {"variant_to_best_key": {}, "best_to_variants": {}, "best_list": []}

    names = s.dropna().astype(str).map(lambda x: re.sub(r"\s+", " ", x.strip()))
    if names.empty:
        return {"variant_to_best_key": {}, "best_to_variants": {}, "best_list": []}

    vc = names.value_counts()
    key_to_variants: Dict[str, List[str]] = {}
    for v in vc.index.tolist():
        k = canon_team_key(v)
        key_to_variants.setdefault(k, []).append(v)

    key_to_best: Dict[str, str] = {}
    best_to_variants: Dict[str, List[str]] = {}
    for k, variants in key_to_variants.items():
        best = max(variants, key=lambda v: int(vc.get(v, 0)))
        key_to_best[k] = best
        best_to_variants[best] = sorted(list(set(variants)))

    best_list = sorted(list(best_to_variants.keys()))
    return {"variant_to_best_key": key_to_best, "best_to_variants": best_to_variants, "best_list": best_list}
