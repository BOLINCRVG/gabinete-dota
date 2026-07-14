"""Subprocess runner with heartbeat callback and timeout support."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

# Raiz do repo (dona de mlcore/app_core). Subprocessos tipo
# `python -m mlcore.train_offline` precisam dela no PYTHONPATH: o app do LoL
# roda com cwd aqui (redundante e inofensivo), mas o Gabinete Dota roda em
# OUTRA pasta — sem isso, todo retrain da manutenção morria com rc=1
# (ModuleNotFoundError silencioso; visto 2026-07-11).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _env_with_repo_on_pythonpath() -> dict:
    env = dict(os.environ)
    prev = env.get("PYTHONPATH") or ""
    if _REPO_ROOT not in prev.split(os.pathsep):
        env["PYTHONPATH"] = _REPO_ROOT + (os.pathsep + prev if prev else "")
    return env


def _no_window_kwargs() -> dict:
    """Return Popen kwargs that suppress the console window on Windows.

    Without CREATE_NO_WINDOW, launching python.exe (or any console program)
    from a GUI/Streamlit process makes Windows flash a console window — the
    "shell that pops up out of nowhere". No-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        return {}
    # CREATE_NO_WINDOW = 0x08000000. Use getattr in case the attr is absent.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return {"creationflags": flags}


def run_cmd(
    cmd: list,
    *,
    timeout_s: Optional[int] = None,
    heartbeat_cb: Optional[Callable[[float], None]] = None,
    heartbeat_interval_s: float = 2.0,
) -> subprocess.CompletedProcess:
    _timeout = int(timeout_s or 0)
    if _timeout <= 0:
        _timeout = 1200
    _hb_interval = float(max(0.5, heartbeat_interval_s or 2.0))
    _t0 = time.perf_counter()
    _next_hb = _t0 + _hb_interval
    timed_out = False
    proc: Optional[subprocess.Popen] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=True) as _tmp:
            proc = subprocess.Popen(
                cmd,
                stdout=_tmp,
                stderr=_tmp,
                text=True,
                env=_env_with_repo_on_pythonpath(),
                **_no_window_kwargs(),
            )
            while True:
                rc = proc.poll()
                _tn = time.perf_counter()
                if rc is not None:
                    break
                if callable(heartbeat_cb) and _tn >= _next_hb:
                    try:
                        heartbeat_cb(float(max(0.0, _tn - _t0)))
                    except Exception:
                        pass
                    _next_hb = _tn + _hb_interval
                if (_tn - _t0) > float(_timeout):
                    timed_out = True
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    break
                time.sleep(0.10)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            _tmp.flush()
            _tmp.seek(0)
            _txt = str(_tmp.read() or "")
        _rc = int(124 if timed_out else int(proc.returncode if proc is not None else 1))
        if timed_out:
            _txt = (_txt + ("\n" if _txt and (not _txt.endswith("\n")) else "")) + f"[timeout] comando excedeu {int(_timeout)}s"
        return subprocess.CompletedProcess(cmd, returncode=_rc, stdout=_txt, stderr="")
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=f"[run_cmd_error] {type(exc).__name__}: {exc}")


__all__ = ["run_cmd"]
