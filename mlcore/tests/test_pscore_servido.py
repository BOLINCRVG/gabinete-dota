# -*- coding: utf-8 -*-
"""Testes do estagio de jogadores por desempenho NO CAMINHO SERVIDO.

O modulo mlcore/players_pscore_core.py ja tem os seus testes. Estes aqui
cobrem a outra metade: o estagio que o app roda, e principalmente as maneiras
de ele NAO agir — que sao as que mantem o caminho de volta honesto.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from app_core.ml_pre_live_pipeline_core import _aplicar_pscore, _carregar_pscore

ESCALACAO_A = {"Top": "a", "Jungle": "b", "Mid": "c", "Bot": "d", "Support": "e"}
ESCALACAO_B = {"Top": "v", "Jungle": "w", "Mid": "x", "Bot": "y", "Support": "z"}


def _pacote(w: float = 0.07) -> dict:
    notas = {}
    for j, rota in zip("abcde", ["Top", "Jungle", "Mid", "Bot", "Support"]):
        notas["{}|||{}".format(j, rota)] = {"n": 1.0, "gp": 50}
    for j, rota in zip("vwxyz", ["Top", "Jungle", "Mid", "Bot", "Support"]):
        notas["{}|||{}".format(j, rota)] = {"n": -1.0, "gp": 50}
    return {"versao": 1, "w": float(w), "beta": 10.0, "notas": notas}


class _ComArquivo(unittest.TestCase):
    """Escreve players_pscore.json ao lado de um artefato de mentira."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name)
        self.artefato = self.raiz / "ml_artifact.json"
        self.artefato.write_text("{}", encoding="utf-8")
        self.pscore = self.raiz / "players_pscore.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _grava(self, pac: dict) -> None:
        self.pscore.write_text(json.dumps(pac), encoding="utf-8")


class TestAplica(_ComArquivo):

    def test_escalacao_melhor_SOBE_a_probabilidade(self):
        self._grava(_pacote())
        r = _aplicar_pscore(
            p_map=0.5, artifact_path=str(self.artefato),
            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B, session_state={},
        )
        self.assertTrue(r["applied"], r.get("reason"))
        self.assertGreater(r["p_out"], 0.5)
        self.assertAlmostEqual(r["delta"], 10.0)   # 5*(1) - 5*(-1)
        self.assertEqual(r["cobertura"], 5)

    def test_trocar_os_lados_espelha_o_resultado(self):
        self._grava(_pacote())
        a = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={})
        b = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_B, red_lineup=ESCALACAO_A,
                            session_state={})
        self.assertAlmostEqual(a["p_out"], 1.0 - b["p_out"], places=12)

    def test_o_deslocamento_e_exatamente_w_vezes_delta_no_logito(self):
        self._grava(_pacote(w=0.05))
        p0 = 0.6
        r = _aplicar_pscore(p_map=p0, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={})
        z = math.log(p0 / (1 - p0)) + 0.05 * 10.0
        self.assertAlmostEqual(r["p_out"], 1.0 / (1.0 + math.exp(-z)), places=12)


class TestNasDuasEscalas(_ComArquivo):
    """O estagio servido tem de valer nos DOIS modos de rating.

    O modo 'media' e o 'ffa' produzem notas em escalas muito diferentes, e o
    peso compensa isso: medido nos arquivos reais em 13/09/2026,
      media : dispersao 0,68  peso 0,0686  -> desloca 0,149 no logito
      ffa   : dispersao 11,79 peso 0,0051  -> desloca 0,191 no logito
    Quem olhar so o peso conclui que o 'ffa' esta morto. Nao esta: e a escala
    da nota que e vinte vezes maior. Estes testes travam os dois.
    """

    def _pacote_escala(self, nota: float, w: float) -> dict:
        pac = _pacote(w=w)
        for k in pac["notas"]:
            pac["notas"][k]["n"] = nota if k[0] in "abcde" else -nota
        return pac

    def _deslocamento(self, nota: float, w: float) -> float:
        self._grava(self._pacote_escala(nota, w))
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={})
        self.assertTrue(r["applied"], r.get("reason"))
        return math.log(r["p_out"] / (1.0 - r["p_out"]))

    def test_as_duas_escalas_deslocam_de_forma_parecida(self):
        media = self._deslocamento(nota=0.6847 / 2, w=0.0686)
        ffa = self._deslocamento(nota=11.7865 / 2, w=0.00513)
        self.assertAlmostEqual(media, ffa, delta=0.6,
                               msg="media={:.3f} ffa={:.3f} — se descolarem, "
                                   "um dos dois esta com o peso errado"
                                   .format(media, ffa))

    def test_a_escala_grande_nao_satura_a_probabilidade(self):
        """Nota do 'ffa' e vinte vezes maior; o resultado tem de continuar odd."""
        self._grava(self._pacote_escala(nota=11.7865, w=0.00513))
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={})
        self.assertTrue(0.5 < r["p_out"] < 0.75,
                        "p_out={} — escalacao boa nao pode virar certeza"
                        .format(r["p_out"]))

    def test_caso_extremo_nao_estoura_nem_vira_0_ou_1(self):
        """Nota absurda (10 desvios) ainda tem de sair um numero utilizavel."""
        self._grava(self._pacote_escala(nota=118.0, w=0.00513))
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={})
        self.assertTrue(math.isfinite(r["p_out"]))
        self.assertLess(r["p_out"], 1.0)
        self.assertGreater(r["p_out"], 0.0)


class TestNaoAplica(_ComArquivo):
    """Cada um destes e uma forma do caminho de volta."""

    def test_sem_o_arquivo_fica_INERTE(self):
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={})
        self.assertFalse(r["applied"])
        self.assertIn("sem players_pscore", r["reason"])
        self.assertNotIn("p_out", r)

    def test_desligado_por_configuracao(self):
        self._grava(_pacote())
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={"ml_pscore_enabled": False})
        self.assertFalse(r["applied"])
        self.assertEqual(r["reason"], "desligado")

    def test_peso_zero_nao_mexe_em_nada(self):
        self._grava(_pacote(w=0.0))
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={})
        self.assertFalse(r["applied"])
        self.assertEqual(r["reason"], "peso zero")

    def test_escalacao_INCOMPLETA_nao_age(self):
        """PRINCIPIO IV: faltando gente, o estagio nao inventa um delta menor.

        Um delta calculado com 3 dos 5 nao e "quase certo" — e outra coisa,
        com outra escala. Aplicar seria transformar ausencia em numero.
        """
        self._grava(_pacote())
        meia = {"Top": "a", "Mid": "c"}
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=meia, red_lineup=ESCALACAO_B,
                            session_state={})
        self.assertFalse(r["applied"])
        self.assertIn("cobertura", r["reason"])
        self.assertEqual(r["cobertura"], 2)

    def test_jogador_desconhecido_derruba_a_cobertura(self):
        self._grava(_pacote())
        trocado = dict(ESCALACAO_A)
        trocado["Mid"] = "NINGUEM_CONHECE"
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=trocado, red_lineup=ESCALACAO_B,
                            session_state={})
        self.assertFalse(r["applied"])
        self.assertEqual(r["cobertura"], 4)

    def test_escalacoes_equivalentes_dao_delta_zero_e_nao_agem(self):
        self._grava(_pacote())
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_A,
                            session_state={})
        self.assertFalse(r["applied"])
        self.assertEqual(r["reason"], "delta zero")

    def test_arquivo_corrompido_nao_derruba_o_app(self):
        self.pscore.write_text("{isto nao e json", encoding="utf-8")
        r = _aplicar_pscore(p_map=0.5, artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={})
        self.assertFalse(r["applied"])

    def test_arquivo_sem_notas_e_recusado(self):
        self.pscore.write_text(json.dumps({"w": 0.07}), encoding="utf-8")
        self.assertIsNone(_carregar_pscore(str(self.artefato)))

    def test_probabilidade_invalida_nao_e_tocada(self):
        self._grava(_pacote())
        r = _aplicar_pscore(p_map=float("nan"), artifact_path=str(self.artefato),
                            blue_lineup=ESCALACAO_A, red_lineup=ESCALACAO_B,
                            session_state={})
        self.assertFalse(r["applied"])


class TestCache(_ComArquivo):

    def test_reler_apos_retreino_pega_o_peso_novo(self):
        """O cache e por mtime+tamanho: retreinar nao pode servir o peso velho."""
        import os
        import time

        self._grava(_pacote(w=0.05))
        a = _carregar_pscore(str(self.artefato))
        self.assertAlmostEqual(a["w"], 0.05)

        time.sleep(0.01)
        self._grava(_pacote(w=0.09))
        # garante mtime diferente mesmo em disco de resolucao grossa
        agora = time.time() + 5
        os.utime(str(self.pscore), (agora, agora))

        b = _carregar_pscore(str(self.artefato))
        self.assertAlmostEqual(b["w"], 0.09, msg="o cache serviu o peso velho depois do retreino")


if __name__ == "__main__":
    unittest.main()
