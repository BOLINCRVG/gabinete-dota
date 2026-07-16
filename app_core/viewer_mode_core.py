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


# --- Upload de CSV no site (pedido dele 16/07: "anexar o csv do meu pc") ---

def validate_csv_upload(file_bytes: bytes, current_csv_path: str) -> tuple[bool, str, dict]:
    """Valida o CSV enviado com as MESMAS guardas do robô: parseável, com as
    colunas do OE, e NUNCA regride (>= jogos e >= data máxima do atual)."""
    import io

    import pandas as pd

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), usecols=["gameid", "date"], low_memory=False)
    except Exception as exc:
        return False, f"Arquivo não parece o CSV do Oracle's Elixir ({type(exc).__name__}).", {}
    new_n = int(df["gameid"].nunique())
    new_dmax = str(df["date"].max())
    info = {"games": new_n, "date_max": new_dmax}
    cur = Path(str(current_csv_path or ""))
    if cur.exists():
        try:
            import pandas as pd2  # noqa: F401

            cdf = pd.read_csv(cur, usecols=["gameid", "date"], low_memory=False)
            cur_n = int(cdf["gameid"].nunique())
            cur_dmax = str(cdf["date"].max())
            if new_n < cur_n or new_dmax < cur_dmax:
                return False, (
                    f"REGRESSÃO: o enviado tem {new_n} jogos (máx {new_dmax}) vs "
                    f"{cur_n} ({cur_dmax}) do atual — mantendo o atual."
                ), info
            if new_n == cur_n and new_dmax == cur_dmax:
                return False, f"Sem novidade: mesmo conteúdo do atual ({cur_n} jogos).", info
        except Exception:
            pass  # atual ilegível não pode bloquear um upload válido
    return True, f"OK: {new_n} jogos, última partida {new_dmax}.", info


def commit_csv_to_repo(
    file_bytes: bytes,
    *,
    token: str,
    repo: str,
    repo_path: str,
    message: str,
    get_fn: Any = None,
    put_fn: Any = None,
) -> tuple[bool, str]:
    """Commita o CSV no GitHub via contents API (persistência real do site)."""
    import base64

    try:
        if get_fn is None or put_fn is None:
            import requests

            get_fn = get_fn or requests.get
            put_fn = put_fn or requests.put
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        folder = repo_path.rsplit("/", 1)[0]
        fname = repo_path.rsplit("/", 1)[1]
        sha = ""
        listing = get_fn(f"https://api.github.com/repos/{repo}/contents/{folder}",
                         headers=headers, timeout=30)
        if int(getattr(listing, "status_code", 0)) == 200:
            for item in (listing.json() or []):
                if str(item.get("name")) == fname:
                    sha = str(item.get("sha") or "")
                    break
        body = {
            "message": message,
            "content": base64.b64encode(file_bytes).decode("ascii"),
            "branch": "master",
        }
        if sha:
            body["sha"] = sha
        resp = put_fn(f"https://api.github.com/repos/{repo}/contents/{repo_path}",
                      json=body, headers=headers, timeout=300)
        code = int(getattr(resp, "status_code", 0))
        if code in (200, 201):
            return True, "CSV publicado! O site renasce com ele em ~2-3 min (recarrega a página)."
        return False, f"GitHub recusou o envio (HTTP {code})."
    except Exception as exc:
        return False, f"Falha ao publicar: {type(exc).__name__}"


def render_csv_upload_panel(st: Any, csv_path: str) -> None:
    """📤 Enviar CSV novo — só no site (viewer + token). Valida e commita;
    o arquivo enviado vira O oficial até alguém (você ou o robô) trocar."""
    if not is_viewer_mode():
        return
    tok, repo = _gh_dispatch_config()
    if not tok:
        return
    cur = Path(str(csv_path or ""))
    if not cur.name:
        return
    with st.expander("📤 Enviar CSV novo (quando o robô estiver bloqueado pela cota)"):
        st.caption(
            "Baixa o CSV no site do Oracle's Elixir e solta aqui. O arquivo é "
            "validado (nunca aceita um pior que o atual) e publicado — o site "
            "passa a usar ELE até a próxima troca (sua ou do robô)."
        )
        up = st.file_uploader("CSV do Oracle's Elixir", type=["csv"], key="viewer_csv_upload")
        if up is not None and st.button("Validar e publicar", key="viewer_csv_upload_go"):
            data = up.getvalue()
            ok, msg, _info = validate_csv_upload(data, str(cur))
            if not ok:
                st.error(msg)
                return
            with st.spinner("Publicando no repositório (arquivo grande, ~1 min)..."):
                ok2, msg2 = commit_csv_to_repo(
                    data, token=tok, repo=repo,
                    repo_path=f"CSV_DIARIO/{cur.name}",
                    message=f"Upload manual pelo site: {cur.name}",
                )
            (st.success if ok2 else st.error)(msg2 if ok2 else f"{msg} → {msg2}")


__all__ = [
    "commit_csv_to_repo",
    "dispatch_data_workflow",
    "is_viewer_mode",
    "render_csv_upload_panel",
    "render_update_now_button",
    "render_viewer_badge",
    "validate_csv_upload",
    "viewer_data_freshness_caption",
]
