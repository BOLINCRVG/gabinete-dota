"""Schedule/agenda tab helpers."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from app_core.schedule_state_core import (
    append_schedule_games,
    ensure_schedule_state,
    get_schedule_default_league,
    get_schedule_games,
    save_schedule_games,
    set_schedule_pending_load,
)
from app_core.tab_ui_core import render_tab_hero, render_tab_section


def dedup_repeat_token(text: str) -> str:
    text0 = (text or "").strip()
    if not text0:
        return ""
    compact = re.sub(r"\s+", "", text0)
    n = len(compact)
    if n % 2 == 0:
        half = compact[: n // 2]
        if half == compact[n // 2 :]:
            return half
    return text0


def parse_schedule_text(text: str, *, default_bo: int = 3) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in str(text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    games: list[dict[str, Any]] = []
    current_date = ""
    idx = 0
    date_re = re.compile(r"(\b\d{2}\/\d{2}\b)")
    time_re = re.compile(r"(\b\d{2}\:\d{2}\b)")
    while idx < len(lines):
        line = lines[idx]
        mdate = date_re.search(line)
        if mdate:
            current_date = mdate.group(1)
            idx += 1
            continue
        mtime = time_re.search(line)
        if mtime:
            current_time = mtime.group(1)
            j = idx + 1
            while j < len(lines) and (lines[j] in ["-", "–", "—"] or lines[j].strip("- ") == ""):
                j += 1
            if j >= len(lines):
                break
            team_a = dedup_repeat_token(lines[j])
            j += 1
            while j < len(lines) and (lines[j] in ["-", "–", "—"] or lines[j].strip("- ") == ""):
                j += 1
            if j >= len(lines):
                break
            team_b = dedup_repeat_token(lines[j])
            games.append(
                {
                    "date": current_date,
                    "time": current_time,
                    "league": "",
                    "teamA": team_a,
                    "teamB": team_b,
                    "bo": int(default_bo) if str(default_bo).isdigit() else 3,
                    "lines": "",
                }
            )
            idx = j + 1
            continue
        idx += 1
    return games


def render_schedule_tab(*, st: Any, bo: int, teams: list[str]) -> None:
    session_state = st.session_state
    games_list = ensure_schedule_state(session_state)
    if "schedule_default_league" not in session_state:
        session_state["schedule_default_league"] = get_schedule_default_league(session_state)
    render_tab_hero(
        st,
        title="Jogos",
        subtitle="Cole jogos em formato livre, revise a agenda e carregue o confronto no painel principal.",
        eyebrow="Agenda",
        chips=[f"BO padrao {int(bo)}", f"{len(games_list)} jogo(s) na fila"],
    )
    render_tab_section(
        st,
        title="Importar jogos",
        subtitle="Cole o bloco bruto, ajuste liga/BO se quiser, e jogue os confrontos para a agenda ativa.",
    )

    pasted_text = st.text_area("Cole aqui a lista de jogos", height=180, key="schedule_paste")
    c_a, c_b, c_c = st.columns([1, 1, 2])
    with c_a:
        default_league = st.text_input(
            "Liga (opcional)",
            key="schedule_default_league",
        )
    with c_b:
        if str(session_state.get("schedule_default_bo", "") or "") not in {"1", "3", "5"}:
            session_state["schedule_default_bo"] = 3
        default_bo = st.selectbox("Formato", options=[1, 3, 5], key="schedule_default_bo")
    with c_c:
        if st.button("Adicionar jogos da colagem", type="primary", width="stretch", key="schedule_add_btn"):
            new_games = parse_schedule_text(pasted_text, default_bo=int(bo) if str(bo).isdigit() else 3)
            for game in new_games:
                if default_league:
                    game["league"] = default_league
                game["bo"] = int(default_bo)
            append_schedule_games(session_state, new_games)
            st.success(f"Adicionados: {len(new_games)} jogo(s).")

    games_list = get_schedule_games(session_state)
    if not games_list:
        st.info("Sua lista está vazia. Cole alguns jogos e clique em Adicionar.")
        return

    df_games = pd.DataFrame(games_list)
    if "remove" not in df_games.columns:
        df_games["remove"] = False
    st.markdown("#### Lista (editável)")
    edited = st.data_editor(df_games, num_rows="dynamic", width="stretch", hide_index=True, key="schedule_editor")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Salvar alterações", width="stretch", key="schedule_save"):
            try:
                save_schedule_games(session_state, edited.to_dict(orient="records"))
                st.success("Lista salva.")
            except Exception as exc:
                st.error(f"Falha ao salvar: {exc}")
    with c2:
        if st.button("Aplicar remoções", width="stretch", key="schedule_remove"):
            try:
                out = edited.to_dict(orient="records")
                out2 = [row for row in out if not bool(row.get("remove"))]
                save_schedule_games(session_state, out2)
                st.success(f"Removidos: {len(out) - len(out2)}")
            except Exception as exc:
                st.error(f"Falha ao remover: {exc}")
    with c3:
        idx = 0
        if len(edited) > 1:
            if "schedule_load_idx" not in session_state:
                session_state["schedule_load_idx"] = 1
            idx = (
                st.number_input(
                    "Carregar jogo #",
                    min_value=1,
                    max_value=len(edited),
                    step=1,
                    key="schedule_load_idx",
                )
                - 1
            )
        if st.button("Carregar no painel principal (Time 1/2)", width="stretch", key="schedule_load"):
            try:
                row = edited.iloc[int(idx)].to_dict()
                payload = {"_reset_flow": True}
                if row.get("teamA") in teams:
                    payload["teamA"] = row.get("teamA")
                if row.get("teamB") in teams:
                    payload["teamB"] = row.get("teamB")
                if str(row.get("bo")) in ["1", "3", "5"]:
                    payload["bo"] = int(row.get("bo"))
                set_schedule_pending_load(session_state, payload)
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao carregar: {exc}")
