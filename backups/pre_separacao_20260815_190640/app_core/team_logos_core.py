"""Resolver de logo de time para a UI.

Se existir um arquivo em `logos/<time-normalizado>.<ext>`, mostra a imagem;
senão, cai graciosamente no badge de iniciais (ui_design_core.team_badge_html).
Assim, logos aparecem quando você adiciona os arquivos — sem fabricar nada e sem
quebrar quando faltam. Streamlit não serve arquivos locais em <img src=...>, então
a imagem vira um data-URI base64 (cacheado).

Como adicionar: coloque PNG/WEBP/JPG em `logos/` com o nome do time normalizado
(minúsculo, só letras/números). Ex.: "Gen.G" -> logos/geng.png; "T1" -> logos/t1.png.
"""
from __future__ import annotations

import base64
import re
from functools import lru_cache
from pathlib import Path

from app_core.ui_design_core import team_badge_html

LOGO_DIR = Path(__file__).resolve().parents[1] / "logos"
_EXTS = (".png", ".webp", ".jpg", ".jpeg", ".svg")
_MIME = {".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg",
         ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}


def _norm(team: str) -> str:
    """Chave de arquivo: minúsculo, só [a-z0-9] (tolerante a mojibake/pontuação)."""
    s = str(team or "").strip().lower()
    try:
        s = s.encode("ascii", "ignore").decode("ascii")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return re.sub(r"[^a-z0-9]+", "", s)


@lru_cache(maxsize=512)
def _logo_data_uri(norm_key: str) -> str | None:
    if not norm_key or not LOGO_DIR.is_dir():
        return None
    for ext in _EXTS:
        p = LOGO_DIR / f"{norm_key}{ext}"
        if p.exists():
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                return f"data:{_MIME[ext]};base64,{b64}"
            except OSError:
                return None
    return None


def has_logo(team: str) -> bool:
    return _logo_data_uri(_norm(team)) is not None


def team_logo_html(team: str, size: int = 40) -> str:
    """<img> do logo se houver arquivo; senão o badge de iniciais (fallback)."""
    uri = _logo_data_uri(_norm(team))
    if uri:
        return (
            f"<img src='{uri}' alt='{_norm(team)}' "
            f"style='width:{size}px;height:{size}px;object-fit:contain;"
            f"vertical-align:middle;border-radius:6px;flex:0 0 auto;'/>"
        )
    return team_badge_html(team, size=size)
