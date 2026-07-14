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


def new_apptest_for_file(app_path: str | Path) -> Any:
    quiet_streamlit_bare_mode_warnings()
    from streamlit.testing.v1 import AppTest  # type: ignore

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        at = AppTest.from_file(str(Path(app_path)))
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
