from __future__ import annotations

import contextlib
import shutil
import uuid
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = ROOT / "evidence" / "_tmp_tests"


@contextlib.contextmanager
def workspace_tempdir(prefix: str) -> Iterator[Path]:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    td = TEST_TMP_ROOT / f"{prefix}_{uuid.uuid4().hex}"
    td.mkdir(parents=True, exist_ok=True)
    try:
        yield td
    finally:
        shutil.rmtree(td, ignore_errors=True)
