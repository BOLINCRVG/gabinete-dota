"""Recovered stub — loads bytecode from mlcore/_recovered/ via marshal."""
from __future__ import annotations

import marshal
from pathlib import Path

_here = Path(__file__).resolve().parent
if _here.name == "__pycache__":
    _here = _here.parent
_pyc_path = _here / "_recovered" / "fit_players_weights_core.pyc"

with open(_pyc_path, "rb") as _f:
    _f.read(16)
    _code = marshal.load(_f)

_mod_globals: dict = {
    "__name__": "mlcore.fit_players_weights_core",
    "__file__": str(_pyc_path),
    "__package__": "mlcore",
    "__loader__": None,
    "__spec__": None,
    "__builtins__": __builtins__,
}
exec(_code, _mod_globals)
for _n, _v in list(_mod_globals.items()):
    if _n.startswith("__"):
        continue
    globals()[_n] = _v
