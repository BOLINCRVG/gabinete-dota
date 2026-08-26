from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path
from typing import Any


_STREAMLIT_LOGGERS = (
    "streamlit",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.state.session_state_proxy",
)


def quiet_streamlit_bare_mode_warnings(*, disable_global_logging: bool = False) -> None:
    """Silence non-fatal bare-mode warnings emitted by Streamlit/AppTest."""
    for name in _STREAMLIT_LOGGERS:
        try:
            logger = logging.getLogger(name)
            logger.setLevel(logging.ERROR)
            logger.propagate = False
            logger.disabled = False
        except Exception:
            pass
    warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
    warnings.filterwarnings("ignore", message=".*can be ignored when running in bare mode.*")

    try:
        import streamlit.runtime.scriptrunner_utils.script_run_context as ctx_mod

        if not getattr(ctx_mod, "_gabinete_quiet_patch_applied", False):
            original = ctx_mod.get_script_run_ctx

            def _quiet_get_script_run_ctx(*args: Any, **kwargs: Any):
                kwargs["suppress_warning"] = True
                return original(*args, **kwargs)

            ctx_mod.get_script_run_ctx = _quiet_get_script_run_ctx  # type: ignore[assignment]
            ctx_mod._gabinete_quiet_patch_applied = True  # type: ignore[attr-defined]
    except Exception:
        pass

    if disable_global_logging:
        logging.disable(logging.CRITICAL)


# A entrada do app tem NOME DIFERENTE em cada repo: plays_app.py no LoL,
# gabinete_dota_app.py no Dota (que nasceu clone). Os testes de UI sao os mesmos
# arquivos nos dois, e pediam "plays_app.py" literal — no Dota isso derrubava
# 8 testes com FileNotFoundError, ou seja, a UI de la nao tinha cobertura
# nenhuma e ninguem via. Resolver por nome conhecido mantem os testes identicos.
_ENTRADAS_DO_APP = ("plays_app.py", "gabinete_dota_app.py")


def resolve_app_file(app_path: str | Path) -> Path:
    p = Path(app_path)
    if p.exists():
        return p
    raiz = Path(__file__).resolve().parents[1]
    for nome in (p.name, *_ENTRADAS_DO_APP):
        alvo = raiz / nome
        if alvo.exists():
            return alvo
    return p


def new_apptest_for_file(app_path: str | Path) -> Any:
    quiet_streamlit_bare_mode_warnings()
    from streamlit.testing.v1 import AppTest  # type: ignore

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        at = AppTest.from_file(str(resolve_app_file(app_path)))
    try:
        at.session_state["_probe_skip_csv_bootstrap_gate"] = True
        at.session_state["auto_maint_lock_poll_max"] = 1
        at.session_state["auto_maint_lock_poll_sleep_s"] = 0.2
    except Exception:
        pass
    return at


def run_apptest_quiet(at: Any, timeout: int) -> None:
    quiet_streamlit_bare_mode_warnings()
    try:
        if "_probe_skip_csv_bootstrap_gate" not in at.session_state:
            at.session_state["_probe_skip_csv_bootstrap_gate"] = True
        if "auto_maint_lock_poll_max" not in at.session_state:
            at.session_state["auto_maint_lock_poll_max"] = 1
        if "auto_maint_lock_poll_sleep_s" not in at.session_state:
            at.session_state["auto_maint_lock_poll_sleep_s"] = 0.2
    except Exception:
        pass
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        at.run(timeout=max(1, int(timeout)))
