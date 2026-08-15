from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import streamlit as st


def resolve_bootstrap_auto_selection(
    *,
    session_state: Mapping[str, Any],
    root: Path,
    list_csv_candidates_fn: Callable[[Path], list[dict[str, Any]]],
) -> dict[str, Any]:
    if bool(session_state.get("boot_show_selector", False)):
        return {"auto_apply": False}

    daily_dir = root / "CSV_DIARIO"
    hist_dir = root / "CSV_HISTORICO"
    daily_auto = list_csv_candidates_fn(daily_dir)
    hist_auto = list_csv_candidates_fn(hist_dir)
    cfg_csv = str(session_state.get("cfg_csv_override", "") or "").strip()
    cfg_csv_ok = bool(cfg_csv) and Path(cfg_csv).exists()
    hist_paths = [str(item.get("path")) for item in hist_auto if str(item.get("path", "")).strip()]

    if daily_auto:
        return {
            "auto_apply": True,
            "daily_choice": str(daily_auto[0]["path"]),
            "hist_choice": hist_paths,
        }
    if cfg_csv_ok:
        return {
            "auto_apply": True,
            "daily_choice": str(cfg_csv),
            "hist_choice": hist_paths,
        }
    return {"auto_apply": False}


def render_bootstrap_selector(
    *,
    session_state: dict[str, Any],
    root: Path,
    list_csv_candidates_fn: Callable[[Path], list[dict[str, Any]]],
    apply_bootstrap_selection_fn: Callable[..., Any],
) -> None:
    if session_state.get("boot_done", False):
        return

    auto_selection = resolve_bootstrap_auto_selection(
        session_state=session_state,
        root=root,
        list_csv_candidates_fn=list_csv_candidates_fn,
    )
    if bool(auto_selection.get("auto_apply", False)):
        apply_bootstrap_selection_fn(
            session_state,
            str(auto_selection.get("daily_choice") or ""),
            list(auto_selection.get("hist_choice") or []),
        )
        return

    st.markdown("## Selecione os CSVs")
    st.caption("Antes de escolher times/linhas: selecione o CSV **diario** (para analisar) e, se quiser, os CSVs **históricos** (para treinar o ML).")

    daily_dir = root / "CSV_DIARIO"
    hist_dir = root / "CSV_HISTORICO"
    daily = list_csv_candidates_fn(daily_dir)
    hist = list_csv_candidates_fn(hist_dir)

    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        st.markdown("### CSV diario (obrigatorio)")
        if not daily:
            st.warning(f"Nao encontrei CSVs em {daily_dir}. Coloque seu CSV diario la ou informe um caminho abaixo.")
            manual_daily = st.text_input("Caminho do CSV diario", value=str(session_state.get("boot_daily_csv", "")))
            daily_choice = manual_daily.strip()
        else:
            options = [item["name"] for item in daily]
            daily_name = st.selectbox("Arquivos em CSV_DIARIO", options, index=0)
            daily_choice = next((item["path"] for item in daily if item["name"] == daily_name), daily[0]["path"])
            st.caption(f"Selecionado: `{Path(daily_choice).name}`")

    with c2:
        st.markdown("### CSVs históricos (opcional - para treino)")
        if not hist:
            st.info(f"Nao encontrei CSVs em {hist_dir}. Se voce tiver (ex.: 2014/2015), coloque la.")
            hist_choice: list[str] = []
        else:
            options = [item["name"] for item in hist]
            selected = st.multiselect("Arquivos em CSV_HISTORICO", options, default=options[:])
            hist_choice = [next(item["path"] for item in hist if item["name"] == name) for name in selected]

        if hist_choice:
            st.caption(f"Historico selecionado: {len(hist_choice)} arquivo(s)")

    st.divider()
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if st.button("Continuar", width="stretch"):
            if not str(daily_choice or "").strip():
                st.error("Selecione um CSV diario para continuar.")
            else:
                apply_bootstrap_selection_fn(
                    session_state,
                    str(daily_choice),
                    list(hist_choice),
                    clear_selector_flag=True,
                )
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
    with col_b:
        if st.button("Recarregar lista", width="stretch"):
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with col_c:
        st.caption("Dica: coloque o diario em `CSV_DIARIO/` e 2014/2015 em `CSV_HISTORICO/`.")
