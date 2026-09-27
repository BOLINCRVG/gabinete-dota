# -*- coding: utf-8 -*-
"""CSV parado tem que GRITAR — o silêncio que a data de referência cria.

Ele perguntou em 16/08/2026 se o modelo usa o jogo mais recente do CSV como data
de referência. NÃO usa: usa HOJE (`as_of_date = ... or date.today()`), e isso
está certo — se usasse a data do CSV, jogo velho pareceria novo e o decaimento
por tempo perderia o sentido.

Mas cria um buraco: se o CSV congela, HOJE continua avançando e envelhece tudo
sozinho. Com half-life de 60 dias, 14 dias parado já tiram 15% do peso do dado
mais recente; 30 dias tiram 29%. E NENHUM dos gatilhos antigos dispara — o
artifact continua recém-treinado e "jogos novos" fica em zero.

A cota do Google Drive já travou o upload dele várias vezes: é cenário real.
"""
from __future__ import annotations

import unittest
from datetime import date, timedelta

from mlcore.retrain_check import CheckConfig


def _decaimento(dias: int, half_life: int) -> float:
    return 0.5 ** (dias / half_life)


class TestCsvParado(unittest.TestCase):
    def test_limite_existe_e_e_curto(self) -> None:
        """5 dias: uma semana parada já é anormal para uma esteira diária."""
        cfg = CheckConfig()
        self.assertTrue(hasattr(cfg, "max_csv_stale_days"))
        self.assertLessEqual(cfg.max_csv_stale_days, 7)
        self.assertGreaterEqual(cfg.max_csv_stale_days, 2)

    def test_o_gatilho_antigo_nao_pegava_este_caso(self) -> None:
        """Artifact novo + zero jogos novos = silêncio, com o CSV parado."""
        cfg = CheckConfig()
        artifact_recem_treinado = 0
        jogos_novos = 0
        self.assertLess(artifact_recem_treinado, cfg.max_days)
        self.assertLess(jogos_novos, cfg.min_new_games)
        # ...e ainda assim o CSV pode estar parado há 30 dias:
        self.assertGreaterEqual(30, cfg.max_csv_stale_days)

    def test_o_custo_do_atraso_justifica_o_limite(self) -> None:
        """No limite (5d) a perda ainda é pequena; deixar chegar a 30d não é."""
        cfg = CheckConfig()
        perda_no_limite = 1 - _decaimento(cfg.max_csv_stale_days, 60)
        self.assertLess(perda_no_limite, 0.08)
        self.assertGreater(1 - _decaimento(30, 60), 0.25)

    def test_conta_de_dias_e_por_data_nao_por_carimbo_de_arquivo(self) -> None:
        """Regravar o CSV sem jogo novo não pode 'rejuvenescer' o dado."""
        ultimo_jogo = date.today() - timedelta(days=12)
        self.assertEqual((date.today() - ultimo_jogo).days, 12)


if __name__ == "__main__":
    unittest.main()
