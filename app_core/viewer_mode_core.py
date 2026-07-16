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
    render_update_now_button(st)


# --- Botão "Atualizar agora" (acorda o robô do GitHub Actions) -------------

_DISPATCH_COOLDOWN_S = 180
_WORKFLOW_FILE = "dados_diarios.yml"


def _gh_dispatch_config() -> tuple[str, str]:
    """(token, repo) dos secrets/env; token vazio = botão não aparece."""
    tok = str(os.environ.get("GABINETE_GH_TOKEN", "") or "").strip()
    repo = str(os.environ.get("GABINETE_GH_REPO", "") or "").strip()
    if not tok:
        try:
            import streamlit as st

            tok = str(st.secrets.get("GABINETE_GH_TOKEN", "") or "").strip()
            repo = repo or str(st.secrets.get("GABINETE_GH_REPO", "") or "").strip()
        except Exception:
            tok = tok or ""
    return tok, (repo or "BOLINAKT/gabinete-lol")


def dispatch_data_workflow(*, token: str, repo: str, post_fn: Any = None) -> tuple[bool, str]:
    """Dispara o workflow de dados via API do GitHub. Retorna (ok, msg)."""
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{_WORKFLOW_FILE}/dispatches"
    try:
        if post_fn is None:
            import requests

            post_fn = requests.post
        resp = post_fn(
            url,
            json={"ref": "master"},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=20,
        )
        code = int(getattr(resp, "status_code", 0))
        if code == 204:
            return True, "Robô acordado! Ele baixa o CSV e commita em ~2-5 min; o site renasce sozinho com os jogos novos."
        return False, f"GitHub recusou (HTTP {code}) — token/permissão do secret GABINETE_GH_TOKEN."
    except Exception as exc:
        return False, f"Falha de rede ao acordar o robô: {type(exc).__name__}"


def render_update_now_button(st: Any) -> None:
    """🔄 Atualizar agora — visível quando há token configurado (site).

    Cooldown de sessão evita spam de cliques; quem trabalha é o robô do
    GitHub (o site nunca roda download/treino localmente)."""
    tok, repo = _gh_dispatch_config()
    if not tok:
        return
    import time as _time

    last = float(st.session_state.get("_upd_now_last_ts", 0.0) or 0.0)
    remaining = int(_DISPATCH_COOLDOWN_S - (_time.time() - last))
    if remaining > 0:
        st.caption(f"🔄 Robô já acordado — aguarda {remaining}s para pedir de novo.")
        return
    if st.button("🔄 Atualizar dados agora", key="upd_now_btn",
                 help="Acorda o robô no GitHub: baixa o CSV novo do Oracle's Elixir e publica."):
        ok, msg = dispatch_data_workflow(token=tok, repo=repo)
        if ok:
            st.session_state["_upd_now_last_ts"] = _time.time()
            st.success(msg)
        else:
            st.error(msg)


__all__ = [
    "dispatch_data_workflow",
    "is_viewer_mode",
    "render_update_now_button",
    "render_viewer_badge",
    "viewer_data_freshness_caption",
]
