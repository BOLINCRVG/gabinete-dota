# -*- coding: utf-8 -*-
"""Testes de qual temporada esta em curso.

O DEFEITO QUE ELES TRAVAM (13/09/2026, relatado pelo dono): ele procurou um
time pelo nome exato no Radar e a tela disse que nao existia. O time existia;
a LISTA e que tinha 18 times em vez de 394.

Tres telas escolhiam a temporada com `max(ano)`. Em 28/08/2026 as ligas
regionais europeias (EBL, LFL, LIT, PRM) comecaram o split WINTER de 2027 —
jogos reais, marcados com year=2027 pela fonte, e esta certo. A partir dali
"o maior ano" virou uma temporada de 46 jogos, enquanto a 2026 seguia com
394 times. Ninguem mexeu em nada: o codigo quebrou pelo calendario.
"""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.temporada_corrente_core import (
    ano_da_temporada_corrente as corrente,
    anos_disponiveis,
)


def _jogos(linhas) -> pd.DataFrame:
    """linhas = [(ano, data, quantidade)]"""
    out = []
    for ano, data, n in linhas:
        out += [{"year": ano, "date": pd.Timestamp(data)} for _ in range(n)]
    return pd.DataFrame(out)


class TestOCasoReal(unittest.TestCase):
    """Os numeros aqui sao os do CSV do dono, medidos em 13/09/2026."""

    def _base(self):
        return _jogos([
            (2026, "2026-09-12", 17128),   # temporada em curso
            (2027, "2026-09-10", 92),      # split Winter recem-comecado
        ])

    def test_escolhe_a_temporada_COM_JOGOS_e_nao_a_de_ano_maior(self):
        self.assertEqual(corrente(self._base()), 2026)

    def test_o_jeito_antigo_daria_2027(self):
        """Deixa explicito o que mudou, para ninguem 'simplificar' de volta."""
        df = self._base()
        self.assertEqual(int(pd.to_numeric(df["year"]).max()), 2027)
        self.assertNotEqual(corrente(df), 2027)

    def test_a_outra_temporada_continua_alcancavel(self):
        """Consertar o padrao nao pode virar um novo jeito de sumir com times."""
        self.assertEqual(anos_disponiveis(self._base()), [2027, 2026])


class TestSeCorrigeSozinha(unittest.TestCase):
    """Quando a temporada nova assumir de verdade, a conta vira sem ninguem mexer."""

    def test_quando_a_nova_domina_ela_passa_a_ser_a_corrente(self):
        df = _jogos([
            (2026, "2026-12-20", 30),     # a velha acabando
            (2027, "2027-02-01", 900),    # a nova a todo vapor
        ])
        self.assertEqual(corrente(df), 2027)

    def test_temporada_encerrada_sai_da_janela(self):
        """Jogos de 6 meses atras nao contam para decidir o que e agora."""
        df = _jogos([
            (2026, "2026-03-01", 5000),   # fora da janela
            (2027, "2026-09-10", 92),     # dentro
        ])
        self.assertEqual(corrente(df), 2027)


class TestAJanelaUsaOsDadosNaoORelogio(unittest.TestCase):
    """Abrir um CSV antigo nao pode dar janela vazia."""

    def test_csv_inteiro_de_2019_ainda_responde(self):
        df = _jogos([(2019, "2019-05-01", 100), (2019, "2019-06-01", 200)])
        self.assertEqual(corrente(df), 2019)

    def test_a_janela_e_contada_a_partir_do_jogo_mais_novo(self):
        df = _jogos([(2018, "2018-01-01", 900), (2019, "2019-06-01", 10)])
        self.assertEqual(corrente(df), 2019)


class TestEmpate(unittest.TestCase):

    def test_empate_no_volume_fica_com_a_temporada_mais_nova(self):
        df = _jogos([(2026, "2026-09-10", 50), (2027, "2026-09-10", 50)])
        self.assertEqual(corrente(df), 2027)


class TestSemDadoNaoInventa(unittest.TestCase):
    """PRINCIPIO IV: ausencia nao vira ano plausivel."""

    def test_vazio_devolve_None(self):
        self.assertIsNone(corrente(pd.DataFrame()))
        self.assertIsNone(corrente(None))

    def test_sem_coluna_de_ano_devolve_None(self):
        self.assertIsNone(corrente(pd.DataFrame([{"date": pd.Timestamp("2026-01-01")}])))

    def test_ano_todo_invalido_devolve_None(self):
        self.assertIsNone(corrente(pd.DataFrame([{"year": "abc", "date": "xyz"}])))

    def test_sem_coluna_de_data_cai_no_maior_ano(self):
        """Sem data nao da para saber o que e recente; o palpite esta dito no codigo."""
        df = pd.DataFrame([{"year": 2026}] * 10 + [{"year": 2027}])
        self.assertEqual(corrente(df), 2027)

    def test_data_toda_invalida_cai_no_maior_ano(self):
        df = pd.DataFrame([{"year": 2026, "date": "nao e data"}] * 10
                          + [{"year": 2027, "date": "nao e data"}])
        self.assertEqual(corrente(df), 2027)

    def test_anos_disponiveis_de_vazio_e_lista_vazia(self):
        self.assertEqual(anos_disponiveis(pd.DataFrame()), [])
        self.assertEqual(anos_disponiveis(None), [])


class TestAsTresTelasUsamARegra(unittest.TestCase):
    """Nao basta a regra existir: as tres telas que quebraram tem de chama-la."""

    def _fonte(self, modulo: str) -> str:
        import importlib
        import inspect
        return inspect.getsource(importlib.import_module(modulo))

    def test_radar_players_e_campeoes_usam_a_regra(self):
        for mod in ("app_core.radar_tab_core",
                    "app_core.players_tab_core",
                    "app_core.champions_tab_core"):
            src = self._fonte(mod)
            self.assertIn("ano_da_temporada_corrente", src,
                          "{} ainda escolhe a temporada sozinho".format(mod))

    def test_nenhuma_delas_voltou_a_usar_o_maior_ano(self):
        for mod in ("app_core.radar_tab_core",
                    "app_core.players_tab_core",
                    "app_core.champions_tab_core"):
            src = self._fonte(mod)
            self.assertNotIn('int(yr.max()) if yr.notna().any()', src,
                             "{} voltou ao max(ano)".format(mod))

    def test_o_radar_oferece_a_outra_temporada(self):
        src = self._fonte("app_core.radar_tab_core")
        self.assertIn("anos_disponiveis", src)
        self.assertIn('"Ano da temporada"', src)

    def test_o_rotulo_nao_colide_com_o_seletor_de_temporada_do_app(self):
        """O app ja tem um "Temporada" (S14/S15/S16), que e outra coisa.

        Dois seletores com o mesmo nome e significados diferentes e como se
        erra aposta. Visto ao dirigir o app em 15/09/2026.
        """
        src = self._fonte("app_core.radar_tab_core")
        self.assertNotIn('st.selectbox("Temporada"', src)


if __name__ == "__main__":
    unittest.main()
