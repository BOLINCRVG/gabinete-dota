# -*- coding: utf-8 -*-
"""Auto-retreino dos modelos ao vivo (22/07, "não quero mais saber do ao vivo").

Os modelos ao vivo (Fase 6/7) não estavam na cadência do robô e ficaram 2
semanas velhos calados. Agora o app dispara o retreino SOZINHO quando detecta
que estão desatualizados — com trava de 6h para não relançar a cada F5. Estes
testes travam a regra da decisão (a parte que decide disparar ou não).
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from app_core.live_models_status_core import (
    hours_since_auto_retrain,
    mark_auto_retrain_attempt,
    should_auto_retrain,
)


class DecisaoTests(unittest.TestCase):
    _KW = dict(running=False, hours_since_attempt=99.0, min_cooldown_hours=6.0)

    def test_dispara_quando_velho_e_ninguem_treinando(self):
        for estado in ("stale", "very_stale", "no_manifest", "no_models"):
            self.assertTrue(should_auto_retrain(state=estado, **self._KW), estado)

    def test_nao_dispara_se_ja_esta_treinando(self):
        kw = {**self._KW, "running": True}
        self.assertFalse(should_auto_retrain(state="no_manifest", **kw))

    def test_nao_dispara_se_esta_fresco(self):
        self.assertFalse(should_auto_retrain(state="fresh", **self._KW))

    def test_trava_de_cooldown_evita_relancar(self):
        # tentou há 1h, cooldown é 6h -> NÃO relança (senão spammava a cada F5)
        kw = {**self._KW, "hours_since_attempt": 1.0}
        self.assertFalse(should_auto_retrain(state="no_manifest", **kw))

    def test_relanca_depois_do_cooldown(self):
        kw = {**self._KW, "hours_since_attempt": 7.0}
        self.assertTrue(should_auto_retrain(state="no_manifest", **kw))

    def test_hora_invalida_tratada_como_infinito(self):
        kw = {**self._KW, "hours_since_attempt": "lixo"}
        self.assertTrue(should_auto_retrain(state="no_manifest", **kw))


class MarcadorTests(unittest.TestCase):
    def test_sem_marcador_e_infinito(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(hours_since_auto_retrain(d), float("inf"))

    def test_carimba_e_le_perto_de_zero(self):
        with tempfile.TemporaryDirectory() as d:
            mark_auto_retrain_attempt(d)
            self.assertLess(hours_since_auto_retrain(d), 0.1)

    def test_marcador_antigo_da_horas_grandes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "last_auto_retrain.txt"
            antigo = (datetime.now().astimezone() - timedelta(hours=10)).isoformat()
            p.write_text(antigo, encoding="utf-8")
            self.assertGreater(hours_since_auto_retrain(d), 9.0)

    def test_marcador_corrompido_nao_quebra(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "last_auto_retrain.txt").write_text("nao e data", encoding="utf-8")
            self.assertEqual(hours_since_auto_retrain(d), float("inf"))


if __name__ == "__main__":
    unittest.main()
