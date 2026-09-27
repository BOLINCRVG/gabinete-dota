# -*- coding: utf-8 -*-
"""As tres tabelas no formato da bet365 (pedido dele, 25/09/2026).

"eu quero o design igual eu te mandei... e eu clique dentro das linhas para
que eu altere eles... a fair e padrao, nao altera, e se eu mudar a do
modelo, ele vai mudar a da pinnacle tambem".
"""
from __future__ import annotations

import json
import re
import unittest

from app_core.tabela_casa_core import altura, html_das_tabelas, linhas_da_tela


def _justa(**k):
    base = {
        "jogo": 1, "casa": "FlyQuest", "fora": "Shopify Rebellion", "periodo": 1,
        "mercado": "Total - Kills - mapa 1", "nome_pt": "Total - Kills - mapa 1",
        "forma": "total", "tipo_linha": "total", "conta": "modelo",
        "linha_modelo": 24.5, "odd_modelo_a": 2.05, "odd_modelo_b": 1.95,
        "grade": [
            {"linha": 24.5, "odd_modelo_a": 2.05, "odd_modelo_b": 1.95, "casa_a": 1.68,
             "casa_b": 2.08, "vantagem_a": -18.0, "vantagem_b": 6.8, "principal": False},
            {"linha": 25.5, "odd_modelo_a": 2.25, "odd_modelo_b": 1.80, "casa_a": 1.83,
             "casa_b": 1.83, "vantagem_a": -14.0, "vantagem_b": 1.7, "principal": True},
            {"linha": 23.5, "odd_modelo_a": 1.9, "odd_modelo_b": 2.1, "casa_a": None,
             "casa_b": None, "vantagem_a": None, "vantagem_b": None, "principal": False},
        ],
    }
    base.update(k)
    return base


class TestOQueATelaRecebe(unittest.TestCase):

    def test_totais_viram_a_secao_Totais_com_o_nome_da_casa(self):
        l = linhas_da_tela([_justa(),
                            _justa(mercado="(Map 1) Total Turrets Destroyed", nome_pt="Torres")])
        self.assertEqual({x["secao"] for x in l}, {"Totais"})
        self.assertEqual([x["rotulo"] for x in sorted(l, key=lambda x: x["ordem"])],
                         ["Total de Kills", "Total de Torres"])

    def test_comeca_na_linha_PRINCIPAL_da_pinnacle(self):
        self.assertEqual(linhas_da_tela([_justa()])[0]["inicial"], 25.5)

    def test_a_fair_e_a_linha_justa_do_modelo(self):
        f = linhas_da_tela([_justa()])[0]["fair"]
        self.assertEqual((f["linha"], f["oa"], f["ob"]), (24.5, 2.05, 1.95))

    def test_handicap_e_kills_por_time(self):
        l = linhas_da_tela([
            _justa(forma="handicap", tipo_linha="handicap", mercado="Handicap - Kills - mapa 1"),
            _justa(forma="total_time", time="away", mercado="Total do time - Kills - mapa 1")])
        por = {x["secao"]: x for x in l}
        self.assertEqual(por["Handicap de Kills"]["tipo"], "handicap")
        self.assertEqual(por["Kills por time"]["rotulo"], "Shopify Rebellion")

    def test_sem_grade_nao_entra(self):
        self.assertEqual(linhas_da_tela([_justa(grade=[])]), [])


class TestAPagina(unittest.TestCase):

    def _html(self):
        return html_das_tabelas(linhas_da_tela([_justa()]), "LCS — FlyQuest x Shopify")

    def test_os_numeros_vao_dentro_da_pagina(self):
        h = self._html()
        dados = re.search(r"const DADOS = (\[.*?\]);\n", h, re.S).group(1)
        self.assertEqual(json.loads(dados)[0]["inicial"], 25.5)

    def test_as_tres_tabelas_e_o_clique_na_linha(self):
        h = self._html()
        for t in ("Fair (modelo)", '"Modelo"', '"Pinnacle"', ".ln.clica", "sel[m.id] = Number"):
            self.assertIn(t, h)

    def test_nao_vai_a_internet(self):
        """O app nunca vai a rede: nada de script, fonte ou folha de fora."""
        h = self._html()
        self.assertNotRegex(h, r"(?i)(src|href)\s*=\s*[\"']?https?:")
        self.assertNotIn("@import", h)

    def test_texto_do_time_nao_quebra_a_pagina(self):
        h = html_das_tabelas(linhas_da_tela([_justa(casa="</script><b>x")]), "t")
        self.assertNotIn("</script><b>", h.split("const DADOS")[1].split(";\n")[0])

    def test_nomes_de_time_sao_protegidos_na_tela(self):
        h = self._html()
        self.assertIn("esc(m.casa)", h)
        self.assertIn("esc(m.rotulo)", h)

    def test_altura_cresce_com_as_linhas(self):
        um = altura(linhas_da_tela([_justa()]))
        dois = altura(linhas_da_tela([_justa(), _justa(mercado="(Map 1) Total Barons Killed")]))
        self.assertGreater(dois, um)


class TestOVencedorEmCima(unittest.TestCase):
    """"que em cima disso tudo tenha a parte do vencedor, baseado no BO que o
    calendario [da] por padrao" (25/09/2026)."""

    def _linhas(self):
        return [{"tipo": "moneyline", "unidade": "", "lado": "home", "periodo": 0, "odd": 1.47},
                {"tipo": "moneyline", "unidade": "", "lado": "away", "periodo": 0, "odd": 2.75},
                {"tipo": "moneyline", "unidade": "", "lado": "home", "periodo": 1, "odd": 1.60},
                {"tipo": "moneyline", "unidade": "", "lado": "away", "periodo": 1, "odd": 2.35},
                {"tipo": "moneyline", "unidade": "Kills", "lado": "home", "periodo": 1, "odd": 1.3},
                {"tipo": "moneyline", "unidade": "Maps", "lado": "A 3, B 0", "periodo": 0, "odd": 3}]

    def test_serie_e_mapa_com_a_odd_da_pinnacle(self):
        from app_core.tabela_casa_core import vencedores_da_tela
        v = vencedores_da_tela(self._linhas(), [])
        self.assertEqual([x["periodo"] for x in v], [0, 1])
        self.assertEqual((v[0]["pin_a"], v[0]["pin_b"]), (1.47, 2.75))
        self.assertNotIn("modelo_a", v[0])

    def test_o_modelo_e_o_numero_do_CARTAO(self):
        from app_core.tabela_casa_core import vencedores_da_tela
        achados = [{"metrica": "vencedor", "mercado": "Vencedor - serie", "lado": "home",
                    "odd_justa": 1.60, "vantagem_pct": -8.1},
                   {"metrica": "vencedor", "mercado": "Vencedor - mapa 1", "lado": "away",
                    "odd_justa": 2.10, "vantagem_pct": 11.9}]
        v = {x["periodo"]: x for x in vencedores_da_tela(self._linhas(), achados)}
        self.assertEqual(v[0]["modelo_a"], 1.60)
        self.assertEqual(v[1]["vant_b"], 11.9)

    def test_a_pagina_leva_o_BO_e_os_times(self):
        from app_core.tabela_casa_core import vencedores_da_tela
        h = html_das_tabelas([], "t", vencedores=vencedores_da_tela(self._linhas(), []),
                             bo=5, casa="FlyQuest", fora="Shopify")
        venc = json.loads(h.split("const VENC = ", 1)[1].split(";" + chr(10), 1)[0])
        self.assertEqual((venc["bo"], venc["casa"]), (5, "FlyQuest"))
        self.assertIn("Série - Vencedor", h)


class TestGrandeDemais(unittest.TestCase):

    def test_a_marca_usa_a_MESMA_regra_da_aba(self):
        from app_core.oportunidades_core import VANTAGEM_SUSPEITA_PCT
        h = html_das_tabelas(linhas_da_tela([_justa()]), "t")
        self.assertIn("const SUSPEITA = %s;" % float(VANTAGEM_SUSPEITA_PCT), h)
        self.assertIn("grande demais", h)


class TestAsCoresDoApp(unittest.TestCase):

    def test_ciano_do_modelo_ambar_da_pinnacle_e_tema_claro(self):
        h = html_das_tabelas(linhas_da_tela([_justa()]), "t")
        self.assertIn("#2daac4", h)
        self.assertIn("#d9a441", h)
        self.assertIn("prefers-color-scheme: light", h)
        self.assertNotIn("#f2d91a", h, "o amarelo da bet365 saiu a pedido dele")


if __name__ == "__main__":
    unittest.main()
