"""Loud-fail validation for the ctx=globals() render contract.

Background: plays_app passes `ctx=globals()` to the big tab renderers
(visao_tab, consistencia_tab). Those renderers pull dozens of imported
callables via `ctx.get("core_x")`. If an import is ever removed (e.g. a linter
auto-fix), the key silently becomes missing and `ctx.get` returns None, which
then crashes far away as "'NoneType' object is not callable" — hard to trace.

This module makes that failure LOUD and EARLY: before a renderer runs, we
check that every callable-looking key it requires is present and not None.
If something is missing we raise a clear error naming the exact keys, instead
of letting a None leak into the render and blow up deep in the call stack.

The required-key list is derived automatically from the renderer's source
(every `ctx.get("...")` / `_ctx(ctx, "...")`), so it never drifts out of sync
with the code. Only keys that look like imported callables are enforced
(prefix `core_` or a leading underscore); plain runtime vars like `teamA`,
`df`, `line` are not required to be non-None.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Mapping


_CTX_GET_RE = re.compile(r'ctx\.get\(\s*"([a-zA-Z_][a-zA-Z0-9_]*)"')
_CTX_HELPER_RE = re.compile(r'_ctx\(\s*ctx\s*,\s*"([a-zA-Z_][a-zA-Z0-9_]*)"')


def extract_ctx_keys(source: str) -> set[str]:
    """All keys a renderer reads from ctx (via ctx.get or _ctx(ctx, "x"))."""
    keys: set[str] = set()
    keys.update(_CTX_GET_RE.findall(source))
    keys.update(_CTX_HELPER_RE.findall(source))
    return keys


def is_enforced_key(key: str) -> bool:
    """Whether a key must be present-and-non-None.

    Enforce only `core_*` keys: these are aliased imports of core
    functions/classes wired in plays_app, never produced at runtime and never
    legitimately None. They are exactly the group that breaks silently when an
    import disappears (the failure mode this module guards against).

    Private helpers (`_xxx`) and plain runtime values (teamA, df, line, ...)
    are NOT enforced: some are optional on certain paths or produced mid-run,
    so requiring them non-None would create false alarms.
    """
    return key.startswith("core_")


@lru_cache(maxsize=8)
def _required_keys_for_source(source: str) -> frozenset[str]:
    return frozenset(k for k in extract_ctx_keys(source) if is_enforced_key(k))


def validate_render_ctx(
    ctx: Mapping[str, object],
    *,
    renderer_source: str,
    renderer_name: str,
    extra_optional: Iterable[str] = (),
) -> None:
    """Raise a clear error if any enforced key is missing or None in ctx.

    Call this at the top of a renderer (or right before invoking it) with the
    renderer's own source. `extra_optional` lets a caller whitelist enforced
    keys that are genuinely optional on the current path.

    Raises RuntimeError listing the offending keys. No-op when all present.
    """
    required = _required_keys_for_source(renderer_source)
    optional = set(extra_optional)
    missing = [
        k for k in sorted(required)
        if k not in optional and ctx.get(k) is None
    ]
    if missing:
        raise RuntimeError(
            f"[{renderer_name}] contrato de ctx quebrado: "
            f"{len(missing)} chave(s) exigida(s) ausente(s)/None — "
            f"provável import removido em plays_app. Chaves: {missing}"
        )


def build_render_ctx(
    global_ns: Mapping[str, object],
    *,
    renderer_source: str,
    renderer_name: str,
    validate: bool = True,
) -> dict[str, object]:
    """Build an explicit, minimal ctx for a renderer from a namespace.

    Instead of passing the whole `globals()` (which leaks hundreds of unrelated
    names and hides the real contract), this returns ONLY the keys the renderer
    actually reads — derived from its source via extract_ctx_keys. The result
    is a plain dict: explicit, inspectable, and decoupled from the caller's
    module internals.

    Safe only for renderers that access ctx by string literals (no ctx.keys()/
    iteration/**ctx). visao_tab and consistencia_tab qualify.

    When validate=True, also runs validate_render_ctx so a missing core_* key
    fails loud here at build time.
    """
    keys = extract_ctx_keys(renderer_source)
    ctx = {k: global_ns.get(k) for k in keys}
    if validate:
        validate_render_ctx(ctx, renderer_source=renderer_source, renderer_name=renderer_name)
    return ctx


__all__ = [
    "build_render_ctx",
    "extract_ctx_keys",
    "is_enforced_key",
    "validate_render_ctx",
]
