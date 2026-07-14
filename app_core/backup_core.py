# -*- coding: utf-8 -*-
"""Timestamped file backup utility."""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Union


def backup_file_if_exists(
    path_like: Union[str, Path],
    *,
    tag: str = "manual",
    app_root: Union[str, Path],
) -> str:
    """Create a timestamped backup under <app_root>/backups/. Returns backup path or ''."""
    try:
        src = Path(path_like).expanduser()
        if not src.exists() or not src.is_file():
            return ""
        backup_dir = Path(app_root) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(tag or "manual")).strip("_") or "manual"
        dst = backup_dir / f"{src.stem}.{safe_tag}.{ts}{src.suffix}"
        i = 1
        while dst.exists():
            dst = backup_dir / f"{src.stem}.{safe_tag}.{ts}.{i}{src.suffix}"
            i += 1
        shutil.copy2(src, dst)
        return str(dst)
    except OSError:
        return ""
