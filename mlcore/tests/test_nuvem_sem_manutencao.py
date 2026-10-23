# -*- coding: utf-8 -*-
"""A nuvem abre direto no app, sem a manutencao pesada da abertura (23/10/2026).
Arquivo identico nos dois repositorios."""
from __future__ import annotations

import inspect
import unittest
from unittest import mock

from app_core import csv_bootstrap_flow_core as fluxo
from app_core import live_models_status_core as vivo


class TestNuvemNaoFazManutencaoNaAbertura(unittest.TestCase):

    def test_trava_esta_no_fluxo_de_abertura(self):
        fonte = inspect.getsource(fluxo)
        self.assertIn("is_streamlit_cloud()", fonte)
        self.assertIn("_must_run_auto = False", fonte)

    def test_deteccao_da_nuvem(self):
        with mock.patch.object(vivo.os, "name", "posix"), \
             mock.patch.dict(vivo.os.environ, {"HOME": "/home/appuser"}):
            self.assertTrue(vivo.is_streamlit_cloud())
        with mock.patch.object(vivo.os, "name", "nt"):
            self.assertFalse(vivo.is_streamlit_cloud())


if __name__ == "__main__":
    unittest.main()
