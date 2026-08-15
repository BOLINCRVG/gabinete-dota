"""Cache wrappers for heavy-but-stable computations in the UI hot path.

Streamlit re-runs the entire script on every user interaction. Without
caching, each click recomputes everything — and that's what makes the
app feel slow.

This module provides Streamlit-cached wrappers around the most-called
heavy functions in app_core/. The wrappers key on stable signatures
(CSV mtime, file path, primitive args) so cache hits are cheap.

Why a separate module
---------------------
- Decorating `load_champion_knowledge_registry` directly with @st.cache_resource
  would force every test/non-Streamlit caller to import Streamlit. Wrapping
  here keeps app_core/ free of Streamlit imports.
- Streamlit caches are global per-process. Wrapping in one place gives us a
  single audit point for cache hits and TTLs.

Usage
-----
    from app_core.ui_cache_core import (
        cached_champion_knowledge_registry,
        cached_champion_vocab,
    )

    registry = cached_champion_knowledge_registry()  # ~0ms on cache hit
    vocab = cached_champion_vocab(csv_sig)           # ~0ms on cache hit
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    import streamlit as st
    _CACHE_RESOURCE = st.cache_resource
    _CACHE_DATA = st.cache_data
except ImportError:  # pragma: no cover — used only when running outside Streamlit
    def _CACHE_RESOURCE(**kwargs):
        def deco(fn): return fn
        return deco
    def _CACHE_DATA(**kwargs):
        def deco(fn): return fn
        return deco


def _registry_path_signature(path: str | Path | None = None) -> Tuple[str, int, float]:
    """Build a hashable signature for the champion registry file.

    Returns (path, size_bytes, mtime) so the cache invalidates if the file
    on disk changes.
    """
    from app_core.champion_knowledge_core import default_registry_path
    p = Path(str(path)) if path is not None else default_registry_path()
    try:
        stat = p.stat()
        return (str(p), int(stat.st_size), float(stat.st_mtime))
    except OSError:
        return (str(p), 0, 0.0)


@_CACHE_RESOURCE(show_spinner=False)
def _cached_load_champion_knowledge(path_sig: Tuple[str, int, float]) -> dict:
    from app_core.champion_knowledge_core import load_champion_knowledge_registry
    path = path_sig[0]
    return load_champion_knowledge_registry(path)


def cached_champion_knowledge_registry(path: str | Path | None = None) -> dict:
    """Get the champion knowledge registry (cached, invalidates on file mtime)."""
    sig = _registry_path_signature(path)
    return _cached_load_champion_knowledge(sig)


@_CACHE_DATA(show_spinner=False)
def _cached_champion_vocab_impl(champ_names: Tuple[str, ...]) -> List[str]:
    """Build a sorted vocab from a tuple of champion strings."""
    return sorted({str(c).strip() for c in champ_names if str(c).strip()})


def cached_champion_vocab(df_players: pd.DataFrame | None) -> List[str]:
    """Cached version of mlcore champion_vocab — keyed on the unique champion strings."""
    if df_players is None or df_players.empty or "champion" not in df_players.columns:
        return []
    # Use a hashable signature: the sorted unique champion names
    champ_names = tuple(sorted({str(c).strip()
                                 for c in df_players["champion"].fillna("").astype(str).unique()
                                 if str(c).strip()}))
    return _cached_champion_vocab_impl(champ_names)


def clear_ui_cache() -> None:
    """Clear all Streamlit caches managed by this module."""
    try:
        _cached_load_champion_knowledge.clear()
    except Exception:
        pass
    try:
        _cached_champion_vocab_impl.clear()
    except Exception:
        pass


__all__ = [
    "cached_champion_knowledge_registry",
    "cached_champion_vocab",
    "clear_ui_cache",
]
