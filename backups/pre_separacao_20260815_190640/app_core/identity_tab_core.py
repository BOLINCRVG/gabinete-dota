"""Identity registry tab renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from app_core.identity_registry_core import (
    load_identity_registry,
    search_identity_records,
    upsert_player_identity,
    upsert_team_identity,
)
from app_core.tab_ui_core import render_tab_hero, render_tab_section


def _render_sources(st: Any, urls: list[str]) -> None:
    if not urls:
        st.caption("Sem fontes cadastradas.")
        return
    lines = [f"- [{url}]({url})" for url in urls if str(url or "").strip()]
    if lines:
        st.markdown("\n".join(lines))


def _render_team_result(st: Any, record: Mapping[str, Any]) -> None:
    canonical = str(record.get("canonical_name") or "")
    current_name = str(record.get("current_name") or canonical)
    relation_type = str(record.get("relation_type") or "alias")
    confidence = str(record.get("confidence") or "medium")
    chips = [f"Relacao {relation_type}", f"Confianca {confidence}"]
    current_leagues = list(record.get("current_leagues") or [])
    if current_leagues:
        chips.append(" / ".join(current_leagues[:2]))
    render_tab_section(
        st,
        title=current_name or canonical or "Time",
        subtitle=str(record.get("summary") or "Sem resumo cadastrado."),
        eyebrow="Identidade competitiva",
    )
    st.caption(" | ".join(chips))
    prev_names = list(record.get("previous_names") or [])
    aliases = [v for v in list(record.get("aliases") or []) if v and v not in {canonical, current_name, *prev_names}]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Nome atual:** {current_name or '-'}")
        st.markdown(f"**Nome canonico:** {canonical or '-'}")
        st.markdown(f"**Nomes anteriores:** {', '.join(prev_names) if prev_names else '-'}")
    with c2:
        st.markdown(f"**Aliases:** {', '.join(aliases) if aliases else '-'}")
        st.markdown(f"**Notas:** {record.get('notes') or '-'!s}")
        st.markdown(f"**Atualizado em:** {record.get('updated_at') or '-'!s}")

    rows = []
    for label, items in (
        ("Mantidos", list(record.get("kept_players") or [])),
        ("Entraram", list(record.get("added_players") or [])),
        ("Sairam", list(record.get("removed_players") or [])),
        ("Staff mantido", list(record.get("kept_staff") or [])),
    ):
        if items:
            rows.append({"Grupo": label, "Itens": ", ".join(items)})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with st.expander("Fontes", expanded=False):
        _render_sources(st, list(record.get("source_urls") or []))
    with st.expander("Registro bruto", expanded=False):
        st.code(json.dumps(dict(record), ensure_ascii=False, indent=2), language="json")


def _render_player_result(st: Any, record: Mapping[str, Any]) -> None:
    canonical = str(record.get("canonical_name") or "")
    current_name = str(record.get("current_name") or canonical)
    relation_type = str(record.get("relation_type") or "alias")
    confidence = str(record.get("confidence") or "medium")
    render_tab_section(
        st,
        title=current_name or canonical or "Player",
        subtitle=str(record.get("summary") or "Sem resumo cadastrado."),
        eyebrow="Identidade de player",
    )
    st.caption(f"Relacao {relation_type} | Confianca {confidence}")
    prev_names = list(record.get("previous_names") or [])
    aliases = [v for v in list(record.get("aliases") or []) if v and v not in {canonical, current_name, *prev_names}]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Nome atual:** {current_name or '-'}")
        st.markdown(f"**Nome canonico:** {canonical or '-'}")
        st.markdown(f"**Nomes anteriores:** {', '.join(prev_names) if prev_names else '-'}")
    with c2:
        st.markdown(f"**Aliases:** {', '.join(aliases) if aliases else '-'}")
        st.markdown(f"**Time atual:** {record.get('current_team') or '-'!s}")
        st.markdown(f"**Atualizado em:** {record.get('updated_at') or '-'!s}")
    if str(record.get("notes") or "").strip():
        st.markdown(f"**Notas:** {record.get('notes')!s}")
    with st.expander("Fontes", expanded=False):
        _render_sources(st, list(record.get("source_urls") or []))
    with st.expander("Registro bruto", expanded=False):
        st.code(json.dumps(dict(record), ensure_ascii=False, indent=2), language="json")


def render_identity_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    registry_path = Path(str(ctx.get("registry_path") or "identity_registry.json"))
    registry = load_identity_registry(registry_path)
    teams = list(registry.get("teams") or [])
    players = list(registry.get("players") or [])

    render_tab_hero(
        st,
        title="Identidade",
        subtitle="Registro local de alias, rebrand, sucessao esportiva e mudancas de roster.",
        eyebrow="Identity registry",
        chips=[f"Times {len(teams)}", f"Players {len(players)}", f"Arquivo {registry_path.name}"],
    )

    tab_lookup, tab_team, tab_player = st.tabs(["Consulta", "Times", "Players"])

    with tab_lookup:
        render_tab_section(
            st,
            title="Consulta",
            subtitle="Busque por time ou player para ver nome atual, nome antigo, relacao e mudancas de roster.",
        )
        c1, c2 = st.columns([1.3, 1.0])
        with c1:
            if "identity_query" not in st.session_state:
                st.session_state["identity_query"] = ""
            query = st.text_input("Buscar time/player", key="identity_query")
        with c2:
            if str(st.session_state.get("identity_query_type", "") or "") not in {"Tudo", "Times", "Players"}:
                st.session_state["identity_query_type"] = "Tudo"
            mode = st.selectbox("Tipo", options=["Tudo", "Times", "Players"], key="identity_query_type")
        entity_type = {"Tudo": "all", "Times": "teams", "Players": "players"}.get(str(mode), "all")
        matches = search_identity_records(registry, query=query, entity_type=entity_type)
        if not matches:
            st.caption("Nenhum registro encontrado para esse termo.")
        else:
            st.caption(f"{len(matches)} registro(s) encontrado(s).")
            for match in matches[:12]:
                record = dict(match.get("record") or {})
                if str(match.get("entity_type")) == "team":
                    _render_team_result(st, record)
                else:
                    _render_player_result(st, record)

    with tab_team:
        render_tab_section(
            st,
            title="Cadastrar ou atualizar time",
            subtitle="Use o nome canonico como chave. Se repetir o nome, o registro e atualizado.",
        )
        st.caption(f"Arquivo: `{registry_path}`")
        with st.form("identity_team_upsert_form", clear_on_submit=False):
            team_canonical = st.text_input("Nome canonico", value="")
            team_current = st.text_input("Nome atual", value="")
            team_previous = st.text_area("Nomes anteriores", value="", help="Separe por virgula ou nova linha.")
            team_aliases = st.text_area("Aliases adicionais", value="", help="Separe por virgula ou nova linha.")
            cc1, cc2 = st.columns(2)
            with cc1:
                team_relation = st.selectbox(
                    "Relacao",
                    options=["alias", "rebrand", "sporting_successor", "roster_successor", "manual_note"],
                    index=2,
                )
            with cc2:
                team_confidence = st.selectbox("Confianca", options=["low", "medium", "high"], index=2)
            team_leagues = st.text_input("Ligas atuais", value="", help="Ex.: NLC, EMEA Masters")
            team_summary = st.text_area("Resumo", value="")
            c31, c32 = st.columns(2)
            with c31:
                team_kept = st.text_area("Players mantidos", value="")
                team_added = st.text_area("Players que entraram", value="")
            with c32:
                team_removed = st.text_area("Players que sairam", value="")
                team_staff = st.text_area("Staff mantido", value="")
            team_notes = st.text_area("Notas", value="")
            team_sources = st.text_area("Fontes (URLs)", value="")
            save_team = st.form_submit_button("Salvar time", width="stretch")
        if save_team:
            if not str(team_canonical or team_current).strip():
                st.error("Informe ao menos o nome canonico ou o nome atual.")
            else:
                upsert_team_identity(
                    registry_path,
                    {
                        "canonical_name": team_canonical,
                        "current_name": team_current,
                        "previous_names": team_previous,
                        "aliases": team_aliases,
                        "relation_type": team_relation,
                        "confidence": team_confidence,
                        "current_leagues": team_leagues,
                        "summary": team_summary,
                        "kept_players": team_kept,
                        "added_players": team_added,
                        "removed_players": team_removed,
                        "kept_staff": team_staff,
                        "notes": team_notes,
                        "source_urls": team_sources,
                    },
                )
                st.success("Registro de time salvo.")

        if teams:
            st.markdown("#### Times cadastrados")
            rows = [
                {
                    "Time": str(row.get("current_name") or row.get("canonical_name") or ""),
                    "Canonico": str(row.get("canonical_name") or ""),
                    "Relacao": str(row.get("relation_type") or ""),
                    "Confianca": str(row.get("confidence") or ""),
                    "Anterior": ", ".join(list(row.get("previous_names") or [])),
                }
                for row in teams
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with tab_player:
        render_tab_section(
            st,
            title="Cadastrar ou atualizar player",
            subtitle="Use o nome canonico como chave. Isso serve para nick antigo/novo e observacoes de identidade.",
        )
        with st.form("identity_player_upsert_form", clear_on_submit=False):
            player_canonical = st.text_input("Nome canonico", value="")
            player_current = st.text_input("Nome atual", value="")
            player_previous = st.text_area("Nomes anteriores", value="", help="Separe por virgula ou nova linha.")
            player_aliases = st.text_area("Aliases adicionais", value="", help="Separe por virgula ou nova linha.")
            cp1, cp2 = st.columns(2)
            with cp1:
                player_relation = st.selectbox(
                    "Relacao",
                    options=["alias", "rename", "same_identity", "manual_note"],
                    index=0,
                )
            with cp2:
                player_confidence = st.selectbox("Confianca", options=["low", "medium", "high"], index=1)
            player_team = st.text_input("Time atual", value="")
            player_summary = st.text_area("Resumo", value="")
            player_notes = st.text_area("Notas", value="")
            player_sources = st.text_area("Fontes (URLs)", value="")
            save_player = st.form_submit_button("Salvar player", width="stretch")
        if save_player:
            if not str(player_canonical or player_current).strip():
                st.error("Informe ao menos o nome canonico ou o nome atual.")
            else:
                upsert_player_identity(
                    registry_path,
                    {
                        "canonical_name": player_canonical,
                        "current_name": player_current,
                        "previous_names": player_previous,
                        "aliases": player_aliases,
                        "relation_type": player_relation,
                        "confidence": player_confidence,
                        "current_team": player_team,
                        "summary": player_summary,
                        "notes": player_notes,
                        "source_urls": player_sources,
                    },
                )
                st.success("Registro de player salvo.")

        if players:
            st.markdown("#### Players cadastrados")
            rows = [
                {
                    "Player": str(row.get("current_name") or row.get("canonical_name") or ""),
                    "Canonico": str(row.get("canonical_name") or ""),
                    "Relacao": str(row.get("relation_type") or ""),
                    "Confianca": str(row.get("confidence") or ""),
                    "Anteriores": ", ".join(list(row.get("previous_names") or [])),
                }
                for row in players
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
