# -*- coding: utf-8 -*-
"""Modo VISUALIZAÇÃO (projeto app-online, 2026-07-15).

É a cara do app na nuvem (Streamlit Community Cloud): boota direto com o
modelo/CSVs que vieram do repositório, sem tela de manutenção e sem nunca
disparar retreino/subprocesso — quem atualiza dados e modelo é o robô do
GitHub Actions, não a instância web.

Como ligar (qualquer um dos dois):
- variável de ambiente GABINETE_VIEWER=1
- secret do Streamlit Cloud: GABINETE_VIEWER = "1"  (dashboard > Secrets)

No PC de casa a flag não existe -> app segue 100% como sempre.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

_TRUTHY = {"1", "true", "yes", "on"}


def is_viewer_mode() -> bool:
    v = str(os.environ.get("GABINETE_VIEWER", "") or "").strip().lower()
    if v in _TRUTHY:
        return True
    # Streamlit Community Cloud não expõe env vars custom; lá a flag vem
    # pelos secrets. Import tardio e à prova de falha: fora do Streamlit
    # (testes/scripts) simplesmente não há secret.
    try:
        import streamlit as st

        sv = str(st.secrets.get("GABINETE_VIEWER", "") or "").strip().lower()
        return sv in _TRUTHY
    except Exception:
        return False


def viewer_data_freshness_caption(csv_path: str) -> Optional[str]:
    """Selo 'dados até X' — a data de jogo mais recente no CSV ativo.

    Leitura barata (só a coluna date) e memoizada por assinatura do arquivo,
    para não pagar o parse do CSV a cada rerun.
    """
    p = Path(str(csv_path or ""))
    if not p.exists():
        return None
    try:
        sig = (str(p), int(p.stat().st_size), int(p.stat().st_mtime))
        return _freshness_cached(sig)
    except Exception:
        return None


def _freshness_cached(sig: tuple) -> Optional[str]:
    try:
        import streamlit as st

        cached = st.cache_data(ttl=6 * 3600, show_spinner=False)(_freshness_raw)
        return cached(sig)
    except Exception:
        return _freshness_raw(sig)


def _freshness_raw(sig: tuple) -> Optional[str]:
    import pandas as pd

    path = sig[0]
    try:
        s = pd.read_csv(path, usecols=["date"], low_memory=False)["date"]
        dmax = pd.to_datetime(s, errors="coerce").max()
        if pd.isna(dmax):
            return None
        return f"📅 Dados até {dmax.strftime('%d/%m/%Y %H:%M')} (última partida no CSV)"
    except Exception:
        return None


def render_viewer_badge(st: Any, csv_path: str) -> None:
    """Selo de frescor no topo — só no modo visualização."""
    if not is_viewer_mode():
        return
    cap = viewer_data_freshness_caption(csv_path)
    if cap:
        st.caption(cap + " · atualizado pelo robô · modo visualização")
    else:
        st.caption("modo visualização · atualização automática pelo robô")


__all__ = [
    "is_viewer_mode",
    "render_viewer_badge",
    "viewer_data_freshness_caption",
]
