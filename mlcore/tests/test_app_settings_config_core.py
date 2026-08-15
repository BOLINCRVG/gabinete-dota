# -*- coding: utf-8 -*-
"""Guard do registro de chaves persistidas (app_core/app_settings_config_core.py).

Pilar 2 da força-tarefa (23/07): a lista _PERSIST_KEYS é mantida à mão com ~170
chaves; uma DUPLICATA seria bug silencioso (some do diff, persiste 2x). Guard
barato contra isso + sanidade do nome do arquivo."""
import unittest

from app_core.app_settings_config_core import _PERSIST_KEYS, _SETTINGS_FILE


class PersistKeysTests(unittest.TestCase):

    def test_sem_chaves_duplicadas(self):
        vistos = {}
        for k in _PERSIST_KEYS:
            vistos[k] = vistos.get(k, 0) + 1
        dups = sorted(k for k, n in vistos.items() if n > 1)
        self.assertEqual(dups, [], f"chaves duplicadas em _PERSIST_KEYS: {dups}")

    def test_todas_strings_nao_vazias(self):
        for k in _PERSIST_KEYS:
            self.assertIsInstance(k, str)
            self.assertTrue(k.strip(), "chave vazia em _PERSIST_KEYS")

    def test_arquivo_de_settings_e_json_oculto(self):
        self.assertTrue(_SETTINGS_FILE.endswith(".json"))
        self.assertTrue(_SETTINGS_FILE.startswith("."))


if __name__ == "__main__":
    unittest.main()
