# -*- coding: utf-8 -*-
"""Mesa do SUPER-LIVE (24/11/2026): rolando agora, abrir/fechar jogos. Identico nos repos."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app_core import super_live_mesa_core as M

AGORA = datetime(2026, 11, 24, 14, 0, tzinfo=timezone.utc)
BASE = {"Team Synapse": "Team Synapse", "Synapse": "Team Synapse", "Angry Birds": "Angry Birds",
        "Team Spirit": "Team Spirit", "PVISION": "PVISION"}


class TestRolandoAgora(unittest.TestCase):

    def test_janela_repetido_e_motivo(self):
        agenda = {"jogos": [
            {"teamA_raw": "Angry Birds", "teamB_raw": "Team Synapse", "inicio_utc": "2026-11-24T13:00:00Z", "bo": 3,
             "league": "EPL"},
            {"teamA_raw": "Team Spirit", "teamB_raw": "PVISION", "inicio_utc": "2026-11-24T20:00:00Z"},   # depois
            {"teamA_raw": "Time Novo", "teamB_raw": "Team Spirit", "inicio_utc": "2026-11-24T14:10:00Z"},  # comeca ja
            {"teamA_raw": "Team Spirit", "teamB_raw": "PVISION", "inicio_utc": "2026-11-24T08:00:00Z"}]}  # antes
        vivo = [{"azul": "Synapse", "vermelho": "Angry Birds", "liga": "EPL"}]           # o mesmo jogo
        js = M.rolando_agora(agenda, vivo, lambda n: BASE.get(n, ""), AGORA)
        self.assertEqual(len(js), 2)
        self.assertEqual((js[0]["a"], js[0]["b"]), ("Angry Birds", "Team Synapse"))
        self.assertIn("ao vivo", js[0]["fonte"])
        novo = [j for j in js if j["a_bruto"] == "Time Novo"][0]
        self.assertIn("Time Novo", novo["motivo"])          # nada some calado


class TestMesa(unittest.TestCase):

    def test_abrir_sem_repetir_limite_e_fechar(self):
        ss = {}
        a = M.abrir(ss, "dota", "Team Spirit", "PVISION")
        self.assertEqual(M.abrir(ss, "dota", "PVISION", "Team Spirit"), a)   # o mesmo jogo nao abre 2x
        for _ in range(M.MAX_JOGOS - 1):
            self.assertIsNotNone(M.abrir(ss, "dota"))
        self.assertIsNone(M.abrir(ss, "dota"))                                # 10 no maximo
        ss[f"slm_dota_{a}_t1"] = "x"
        M.fechar(ss, "dota", a)
        self.assertEqual(len(ss["slm_dota"]), M.MAX_JOGOS - 1)
        self.assertNotIn(f"slm_dota_{a}_t1", ss)                              # fechar limpa o que foi digitado
        self.assertEqual(ss["slm_dota"][0]["rotulo"], "Jogo 2")


class TestLado(unittest.TestCase):
    """24/11/2026 (ele: "voce errou o lado"): quem esta a esquerda sai do historico do elenco; sem folga, ninguem chuta."""

    def _estado(self, folga=True):
        from app_core import camada_draft_core as CD
        e = CD.EstadoDraft()
        t = datetime(2026, 11, 1)
        for _ in range(20):
            e.registrar([("1", "10"), ("2", "11"), ("3", "12"), ("4", "13"), ("5", "14")], 1, t)
            e.registrar([("6", "20"), ("7", "21"), ("8", "22"), ("9", "23"), ("10", "24")], 0, t)
            if not folga:   # os dois elencos jogam igual os dois conjuntos de herois: nao da para saber
                e.registrar([("1", "20"), ("2", "21"), ("3", "22"), ("4", "23"), ("5", "24")], 1, t)
                e.registrar([("6", "10"), ("7", "11"), ("8", "12"), ("9", "13"), ("10", "14")], 0, t)
        return e

    def test_detecta_e_nao_chuta(self):
        el = {"A": {"jogadores": ["1", "2", "3", "4", "5"]}, "B": {"jogadores": ["6", "7", "8", "9", "10"]}}
        esq, dir_ = ["10", "11", "12", "13", "14"], ["20", "21", "22", "23", "24"]
        self.assertIs(M.lado_provavel(self._estado(), el, "A", "B", esq, dir_)[0], True)
        self.assertIs(M.lado_provavel(self._estado(), el, "B", "A", esq, dir_)[0], False)
        self.assertIsNone(M.lado_provavel(self._estado(False), el, "A", "B", esq, dir_)[0])   # pouca folga
        self.assertIsNone(M.lado_provavel(self._estado(), {"A": el["A"]}, "A", "B", esq, dir_)[0])  # sem elenco


if __name__ == "__main__":
    unittest.main()
