"""Tests for app_core.render_ctx_core — loud-fail ctx validation."""
from __future__ import annotations

import unittest

from app_core.render_ctx_core import (
    extract_ctx_keys,
    is_enforced_key,
    validate_render_ctx,
)


class TestExtractCtxKeys(unittest.TestCase):

    def test_ctx_get(self):
        src = 'x = ctx.get("core_foo")\ny = ctx.get("teamA")'
        self.assertEqual(extract_ctx_keys(src), {"core_foo", "teamA"})

    def test_ctx_helper(self):
        src = 'x = _ctx(ctx, "core_bar")'
        self.assertIn("core_bar", extract_ctx_keys(src))

    def test_both_forms(self):
        src = 'a = ctx.get("core_a")\nb = _ctx(ctx, "core_b")'
        self.assertEqual(extract_ctx_keys(src), {"core_a", "core_b"})

    def test_empty(self):
        self.assertEqual(extract_ctx_keys("no ctx here"), set())


class TestIsEnforcedKey(unittest.TestCase):

    def test_core_enforced(self):
        self.assertTrue(is_enforced_key("core_run_ml_pre_pipeline"))

    def test_runtime_var_not_enforced(self):
        for k in ("teamA", "df", "line", "filters", "bo"):
            self.assertFalse(is_enforced_key(k))

    def test_private_helper_not_enforced(self):
        # Private helpers may be optional on some paths — not enforced.
        self.assertFalse(is_enforced_key("_ml_info"))

    def test_dunder_not_enforced(self):
        self.assertFalse(is_enforced_key("__builtins__"))


class TestValidateRenderCtx(unittest.TestCase):

    SRC = '''
    def r(ctx):
        a = ctx.get("core_alpha")
        b = ctx.get("core_beta")
        t = ctx.get("teamA")   # runtime var, not enforced
    '''

    def test_all_present_passes(self):
        ctx = {"core_alpha": lambda: 1, "core_beta": lambda: 2, "teamA": "T1"}
        validate_render_ctx(ctx, renderer_source=self.SRC, renderer_name="r")  # no raise

    def test_missing_core_key_raises(self):
        ctx = {"core_alpha": lambda: 1, "teamA": "T1"}  # core_beta missing
        with self.assertRaises(RuntimeError) as cm:
            validate_render_ctx(ctx, renderer_source=self.SRC, renderer_name="r")
        self.assertIn("core_beta", str(cm.exception))
        self.assertIn("r", str(cm.exception))

    def test_none_core_key_raises(self):
        ctx = {"core_alpha": lambda: 1, "core_beta": None, "teamA": "T1"}
        with self.assertRaises(RuntimeError):
            validate_render_ctx(ctx, renderer_source=self.SRC, renderer_name="r")

    def test_missing_runtime_var_ok(self):
        # teamA absent is fine — not an enforced key.
        ctx = {"core_alpha": lambda: 1, "core_beta": lambda: 2}
        validate_render_ctx(ctx, renderer_source=self.SRC, renderer_name="r")  # no raise

    def test_extra_optional_whitelists(self):
        ctx = {"core_alpha": lambda: 1}  # core_beta missing
        # but we declare core_beta optional on this path
        validate_render_ctx(ctx, renderer_source=self.SRC, renderer_name="r",
                            extra_optional=["core_beta"])  # no raise


class TestRealRenderersContract(unittest.TestCase):
    """The actual renderers must validate clean against a full plays_app ctx."""

    def _plays_app_globals(self):
        import ast, io
        with io.open("plays_app.py", encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        g = {}
        for n in tree.body:
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    g[a.asname or a.name] = lambda: None
            elif isinstance(n, ast.Import):
                for a in n.names:
                    g[(a.asname or a.name).split(".")[0]] = lambda: None
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                g[n.name] = lambda: None
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        g[t.id] = lambda: None
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        for e in t.elts:
                            if isinstance(e, ast.Name):
                                g[e.id] = lambda: None
        return g

    def _enforced_missing(self, path: str) -> list[str]:
        import io
        from app_core.render_ctx_core import extract_ctx_keys, is_enforced_key
        g = self._plays_app_globals()
        with io.open(path, encoding="utf-8") as fh:
            src = fh.read()
        enforced = {k for k in extract_ctx_keys(src) if is_enforced_key(k)}
        return sorted(k for k in enforced if k not in g)

    def test_visao_core_keys_all_in_globals(self):
        missing = self._enforced_missing("app_core/visao_tab_core.py")
        self.assertEqual(missing, [], f"plays_app não importa: {missing}")

    def test_consistencia_core_keys_all_in_globals(self):
        missing = self._enforced_missing("app_core/consistencia_tab_core.py")
        self.assertEqual(missing, [], f"plays_app não importa: {missing}")


if __name__ == "__main__":
    unittest.main()
