# -*- coding: utf-8 -*-
"""Testes da assinatura do cache do replay — mlcore/replay_pre_jogo.py.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ Dívida D9 (06/09/2026): `carregar()` devolvia o cache se o ARQUIVO          ║
║ EXISTISSE, sem conferir nada. O cache era de 26/08 23:34 e o robô atualizou ║
║ `ml_artifact.json` todo dia até 06/09 — então o gate deu números IDÊNTICOS  ║
║ depois de trocar o modelo, até a quinta casa decimal. Modelo diferente,     ║
║ nota igual, por 11 dias. Tudo o que foi medido em 04 e 05/09 saiu de um     ║
║ modelo velho sem ninguém saber.                                             ║
║                                                                             ║
║ A CURA É CHAVE ENDEREÇADA POR ENTRADA (conferido abrindo a página em        ║
║ 06/09/2026): num sistema de build, "o `digestKey` é um hash das entradas e  ║
║ saídas da ação", e como a chave incorpora o hash das próprias entradas,     ║
║ "qualquer modificação nessas entradas produz uma chave diferente, tornando  ║
║ o resultado antigo inacessível sob a chave nova e portanto impossível de    ║
║ ser reusado por engano".                                                    ║
║ https://blog.engflow.com/2024/05/13/the-many-caches-of-bazel/               ║
║                                                                             ║
║ Ou seja: não existe "esquecer de invalidar". Mudou a entrada, o cache erra  ║
║ sozinho. É isso que estes testes travam.                                    ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlcore.replay_pre_jogo import CACHE, CACHE_VERSAO, _assinatura_das_entradas


def _assina(**kw):
    base = dict(artifact="ml_artifact.json", players_artifact="players_artifact.json",
                csv_train="CSV_HISTORICO", csv_current="CSV_DIARIO",
                params="params_fitted.json", val_cutoff="2026-03-31")
    base.update(kw)
    return _assinatura_das_entradas(**base)


class TestAssinaturaDoCache(unittest.TestCase):

    def test_a_assinatura_e_estavel_quando_nada_muda(self):
        """Sem isso o cache nunca acertaria e o replay recalcularia sempre."""
        self.assertEqual(_assina(), _assina())

    def test_corte_diferente_muda_a_assinatura(self):
        """val_cutoff decide treino/val/teste — e entrada, nao detalhe."""
        self.assertNotEqual(_assina(), _assina(val_cutoff="2026-01-31"))

    def test_artefato_diferente_muda_a_assinatura(self):
        """O CASO REAL DA D9: trocar o modelo tem que vencer o cache."""
        self.assertNotEqual(_assina(), _assina(artifact="players_artifact.json"))

    def test_artefato_AUSENTE_nao_estoura_e_muda_a_assinatura(self):
        a = _assina(artifact="arquivo_que_nao_existe.json")
        self.assertIsInstance(a, str)
        self.assertNotEqual(a, _assina())

    def test_pasta_de_csv_diferente_muda_a_assinatura(self):
        self.assertNotEqual(_assina(), _assina(csv_current="CSV_HISTORICO"))

    def test_pasta_inexistente_nao_estoura(self):
        self.assertIsInstance(_assina(csv_current="pasta_que_nao_existe"), str)

    def test_a_assinatura_e_um_digest_curto(self):
        a = _assina()
        self.assertEqual(len(a), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in a))

    def test_a_versao_do_formato_entra_na_assinatura(self):
        """Subir CACHE_VERSAO tem que invalidar todo cache antigo de uma vez."""
        import mlcore.replay_pre_jogo as m
        antes = _assina()
        original = m.CACHE_VERSAO
        try:
            m.CACHE_VERSAO = original + 1
            self.assertNotEqual(antes, _assina())
        finally:
            m.CACHE_VERSAO = original
        self.assertEqual(antes, _assina())

    def test_o_cache_em_disco_carrega_assinatura(self):
        """Cache gravado sem assinatura e cache que mente calado.

        NAO se exige aqui que a assinatura CASE com as entradas de agora: o
        robo atualiza o modelo varias vezes por dia, entao cache vencido e
        estado NORMAL entre um retreino e a proxima medicao. Exigir igualdade
        transformaria o normal em teste vermelho, e teste que chora sem motivo
        ensina a ignorar teste. Quem cuida do vencimento e `carregar()`, em
        tempo de execucao — e ele avisa em voz alta quando recalcula."""
        if not CACHE.exists():
            self.skipTest("cache ainda nao foi gerado nesta maquina")
        b = json.loads(CACHE.read_text(encoding="utf-8"))
        self.assertIn("assinatura", b,
                      "cache sem assinatura: e o estado que causou a D9")
        self.assertIsInstance(b["assinatura"], str)
        self.assertEqual(len(b["assinatura"]), 32)


class TestConteudoDaAssinatura(unittest.TestCase):
    """Detalhes que, se mudarem sem querer, reabrem a D9."""

    def test_tamanho_do_arquivo_conta(self):
        with tempfile.TemporaryDirectory() as d:
            import mlcore.replay_pre_jogo as m
            raiz = m.ROOT
            try:
                m.ROOT = Path(d)
                alvo = Path(d) / "art.json"
                alvo.write_text("a" * 10, encoding="utf-8")
                a1 = _assinatura_das_entradas(
                    artifact="art.json", players_artifact="x", csv_train="y",
                    csv_current="z", params="w", val_cutoff="c")
                alvo.write_text("b" * 20, encoding="utf-8")
                a2 = _assinatura_das_entradas(
                    artifact="art.json", players_artifact="x", csv_train="y",
                    csv_current="z", params="w", val_cutoff="c")
                self.assertNotEqual(a1, a2, "mudar o tamanho tem que vencer")
            finally:
                m.ROOT = raiz


if __name__ == "__main__":
    unittest.main()
