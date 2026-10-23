# -*- coding: utf-8 -*-
"""Nomes de time com letra especial casam, e o apelido salvo na nuvem vai para
o GitHub (23/10/2026). Arquivo identico nos dois repositorios."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app_core import viewer_mode_core as vm
from app_core.schedule_import_core import (
    normalize_schedule_team_key,
    resolve_schedule_team,
    save_schedule_aliases,
)


class TestNomeComLetraEspecial(unittest.TestCase):

    def test_los_da_agenda_casa_com_los_do_csv(self):
        self.assertEqual(normalize_schedule_team_key("LØS"), "los")
        time, ok = resolve_schedule_team("LOS", ["LØS", "T1"], {})
        self.assertTrue(ok)
        self.assertEqual(time, "LØS")

    def test_acentos_viram_letra_e_nao_somem(self):
        self.assertEqual(normalize_schedule_team_key("Barça eSports"), "barcaesports")
        self.assertEqual(normalize_schedule_team_key("Barcząca Esports"), "barczacaesports")

    def test_apelido_salvo_com_a_chave_antiga_ainda_acha(self):
        # antes: letra acentuada era apagada ("Barça" -> "bara")
        time, ok = resolve_schedule_team("Barça", ["Barça eSports"], {"bara": "Barça eSports"})
        self.assertTrue(ok)
        self.assertEqual(time, "Barça eSports")


class _Resp:
    status_code = 204


class TestApelidoVaiParaOGitHub(unittest.TestCase):

    def test_na_nuvem_com_token_dispara_o_robo_com_os_apelidos(self):
        chamadas = []
        env = {"GABINETE_VIEWER": "1", "GABINETE_GH_TOKEN": "x", "GABINETE_GH_REPO": "dono/repo"}
        with mock.patch.dict(os.environ, env):
            ok, _ = vm.publicar_apelidos({"los": "LØS"},
                                         post_fn=lambda url, **k: chamadas.append((url, k)) or _Resp())
        self.assertTrue(ok)
        url, k = chamadas[0]
        self.assertIn("salvar_apelidos.yml", url)
        self.assertEqual(json.loads(k["json"]["inputs"]["apelidos"]), {"los": "LØS"})

    def test_no_pc_nao_dispara_nada(self):
        with mock.patch.dict(os.environ, {"GABINETE_VIEWER": "0"}):
            ok, _ = vm.publicar_apelidos({"los": "LØS"}, post_fn=lambda *a, **k: self.fail("disparou"))
        self.assertFalse(ok)

    def test_salvar_no_pc_grava_o_arquivo_e_nao_dispara(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"GABINETE_VIEWER": "0"}):
            arq = Path(d) / "apelidos.json"
            save_schedule_aliases({}, arq, {"los": "LØS"})
            self.assertEqual(json.loads(arq.read_text(encoding="utf-8"))["los"], "LØS")

    def test_na_nuvem_o_resultado_do_envio_fica_para_a_tela(self):
        with tempfile.TemporaryDirectory() as d,                 mock.patch.dict(os.environ, {"GABINETE_VIEWER": "1"}),                 mock.patch.object(vm, "publicar_apelidos", return_value=(False, "HTTP 403")):
            sessao = {}
            save_schedule_aliases(sessao, Path(d) / "a.json", {"los": "LØS"})
            self.assertEqual(sessao["_apelido_envio"], {"ok": False, "msg": "HTTP 403"})

    def test_o_robo_existe(self):
        raiz = Path(__file__).resolve().parents[2]
        yml = (raiz / ".github" / "workflows" / "salvar_apelidos.yml").read_text(encoding="utf-8")
        self.assertIn("user_team_aliases.json", yml)
        self.assertIn("workflow_dispatch", yml)


if __name__ == "__main__":
    unittest.main()
