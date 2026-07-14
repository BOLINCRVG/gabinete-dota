# -*- coding: utf-8 -*-
"""Descoberta PORTATIL do repo do LoL (requisito 2026-07-10: o usuario manda
as duas pastas num .rar pro amigo — nada de caminho absoluto com username).

Ordem de resolucao:
  1. env var GABINETE_LOL_DIR (override manual, se um dia precisar);
  2. pasta irma da GabineteDOTA (mesmo diretorio-pai), nome gabinetelolweb*;
  3. caminho legado da maquina do usuario (fallback p/ compatibilidade).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_LEGACY = Path(r"C:\Users\steam\Desktop\gabinetelolweb3.1")


def lol_repo_path() -> Path:
    env = (os.getenv("GABINETE_LOL_DIR") or "").strip()
    if env and (Path(env) / "app_core").exists():
        return Path(env)
    here = Path(__file__).resolve().parent
    # 1º a pasta irmã (desenvolvimento: sempre o codigo mais fresco);
    for cand in sorted(here.parent.glob("gabinetelolweb*")):
        if (cand / "app_core").exists():
            return cand
    # 2º a copia EMBUTIDA (sync_vendor.py) — e o que faz a pasta do Dota
    # funcionar SOZINHA em qualquer PC (requisito 2026-07-12: ele envia so o
    # GabineteDOTA pro amigo).
    if (here / "app_core").exists():
        return here
    return _LEGACY


def ensure_lol_on_path() -> Path:
    repo = lol_repo_path()
    s = str(repo)
    if s not in sys.path:
        sys.path.insert(0, s)
    return repo
