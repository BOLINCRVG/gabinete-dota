import contextlib
import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = ROOT / "evidence" / "_tmp_tests"


@contextlib.contextmanager
def _tempdir():
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    td = TEST_TMP_ROOT / f"find_code_errors_{uuid.uuid4().hex}"
    td.mkdir(parents=True, exist_ok=True)
    try:
        yield str(td)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _load_module():
    mod_path = ROOT / "tools" / "find_code_errors.py"
    spec = importlib.util.spec_from_file_location("find_code_errors", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load tools/find_code_errors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FindCodeErrorsToolTests(unittest.TestCase):
    def test_find_syntax_errors_reports_bad_file(self) -> None:
        mod = _load_module()
        with _tempdir() as td:
            good = Path(td) / "good.py"
            bad = Path(td) / "bad.py"
            good.write_text("x = 1\n", encoding="utf-8")
            bad.write_text("def broken(:\n    pass\n", encoding="utf-8")
            issues = mod.find_syntax_errors([good, bad])
        self.assertEqual(len(issues), 1)
        self.assertTrue(str(issues[0]["path"]).endswith("bad.py"))

    def test_find_unbound_local_suspects_catches_simple_case(self) -> None:
        mod = _load_module()
        with _tempdir() as td:
            path = Path(td) / "sample.py"
            path.write_text(
                "def bad():\n"
                "    print(x)\n"
                "    x = 1\n",
                encoding="utf-8",
            )
            issues = mod.find_unbound_local_suspects([path])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["function"], "bad")
        self.assertEqual(issues[0]["name"], "x")

    def test_find_unbound_local_suspects_ignores_safe_function(self) -> None:
        mod = _load_module()
        with _tempdir() as td:
            path = Path(td) / "sample.py"
            path.write_text(
                "def ok(value):\n"
                "    total = value + 1\n"
                "    return total\n",
                encoding="utf-8",
            )
            issues = mod.find_unbound_local_suspects([path])
        self.assertEqual(issues, [])

    def test_find_undefined_name_suspects_catches_missing_global(self) -> None:
        mod = _load_module()
        with _tempdir() as td:
            path = Path(td) / "sample.py"
            path.write_text(
                "VALUE = 1\n"
                "def broken():\n"
                "    return VALUE + missing_name\n",
                encoding="utf-8",
            )
            issues = mod.find_undefined_name_suspects([path])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["scope"], "broken")
        self.assertEqual(issues[0]["name"], "missing_name")

    def test_find_undefined_name_suspects_ignores_free_variable(self) -> None:
        mod = _load_module()
        with _tempdir() as td:
            path = Path(td) / "sample.py"
            path.write_text(
                "def outer():\n"
                "    shared = 2\n"
                "    def inner():\n"
                "        return shared\n"
                "    return inner()\n",
                encoding="utf-8",
            )
            issues = mod.find_undefined_name_suspects([path])
        self.assertEqual(issues, [])

    def test_find_constant_branch_suspects_catches_static_if_and_while(self) -> None:
        mod = _load_module()
        with _tempdir() as td:
            path = Path(td) / "sample.py"
            path.write_text(
                "def branchy():\n"
                "    if False:\n"
                "        return 1\n"
                "    while 0:\n"
                "        return 2\n"
                "    return 3\n",
                encoding="utf-8",
            )
            issues = mod.find_constant_branch_suspects([path])
        self.assertEqual(len(issues), 2)
        self.assertEqual([row["kind"] for row in issues], ["if", "while"])
        self.assertEqual([row["scope"] for row in issues], ["branchy", "branchy"])

    def test_find_constant_branch_suspects_ignores_if_true_without_else(self) -> None:
        mod = _load_module()
        with _tempdir() as td:
            path = Path(td) / "sample.py"
            path.write_text(
                "def wrapper():\n"
                "    if True:\n"
                "        return 1\n",
                encoding="utf-8",
            )
            issues = mod.find_constant_branch_suspects([path])
        self.assertEqual(issues, [])

    def test_find_mutable_default_suspects_catches_list_default(self) -> None:
        mod = _load_module()
        with _tempdir() as td:
            path = Path(td) / "sample.py"
            path.write_text(
                "def broken(items=[]):\n"
                "    return len(items)\n",
                encoding="utf-8",
            )
            issues = mod.find_mutable_default_suspects([path])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["function"], "broken")
        self.assertEqual(issues[0]["name"], "items")

    def test_find_duplicate_dict_key_suspects_catches_literal_repeat(self) -> None:
        mod = _load_module()
        with _tempdir() as td:
            path = Path(td) / "sample.py"
            path.write_text(
                "CFG = {\n"
                "    'a': 1,\n"
                "    'a': 2,\n"
                "}\n",
                encoding="utf-8",
            )
            issues = mod.find_duplicate_dict_key_suspects([path])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["key"], "'a'")


if __name__ == "__main__":
    unittest.main()
