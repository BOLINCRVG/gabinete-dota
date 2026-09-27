# -*- coding: utf-8 -*-
"""O que cada mercado da Pinnacle E, alem do nome que ela deu a ele.

ELE PERGUNTOU (21/09/2026): "eu so preciso ter certeza que voce consegue pegar
as outras odds".

A auditoria respondeu que sim: 106 tipos, 14.937 leituras, com torres,
dragoes, baroes, inibidores, primeiro sangue, primeiro a N abates e par/impar.
A coleta nunca foi o problema.

O PROBLEMA ERA OUTRO. Quando a casa da nome ao mercado, o coletor guarda o
nome LETRA POR LETRA -- de proposito, porque e o que ele confere na tela da
Pinnacle. Mas esse nome vem assim:

    "(Map 3) Total Turrets Destroyed"

e o resto do programa fala "towers", "mapa 3". Resultado: 3.430 leituras
invisiveis para a aba de Oportunidades, entre elas TORRES -- um dos dois
mercados em que ele ganha dinheiro.

A traducao entra AO LADO: o nome da casa fica, e nascem campos que o programa
entende. Estes testes protegem as duas metades disso, e sobretudo a regra que
impede o estrago: nome desconhecido NAO VIRA PALPITE.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from app_core.mercado_pinnacle_core import (
    entender_mercado,
    nao_entendidos,
    resumir,
)

RAIZ = Path(__file__).resolve().parents[2]


class TestOsMercadosQueEstavamInvisiveis(unittest.TestCase):
    """Os nomes crus que o historico mostrou, um por um."""

    def _e(self, nome):
        return entender_mercado(nome)

    def test_TORRES_o_que_mais_importa(self):
        d = self._e("(Map 3) Total Turrets Destroyed")
        self.assertEqual(d["metrica"], "towers")
        self.assertEqual(d["forma"], "total")
        self.assertEqual(d["mapa"], 3)

    def test_baroes_dragoes_inibidores(self):
        self.assertEqual(self._e("(Map 1) Total Barons Killed")["metrica"], "barons")
        self.assertEqual(self._e("(Map 2) Total Elemental Dragons Killed")["metrica"],
                         "dragons")
        self.assertEqual(self._e("(Map 5) 1st Inhibitor")["metrica"], "inhibitors")

    def test_primeiro_sangue_e_primeiro_barao(self):
        d = self._e("(Map 4) 1st Blood")
        self.assertEqual((d["metrica"], d["forma"], d["mapa"]),
                         ("firstblood", "primeiro", 4))
        self.assertEqual(self._e("(Map 2) 1st Baron")["forma"], "primeiro")

    def test_corrida_ate_N_abates_guarda_o_N(self):
        for n in (5, 10, 15, 20):
            d = self._e("(Map 1) 1st To %d Kills" % n)
            self.assertEqual(d["metrica"], "kills")
            self.assertEqual(d["forma"], "corrida")
            self.assertEqual(d["alvo"], n, "perdeu o alvo da corrida")

    def test_os_dois_times_e_algum_time_sao_formas_DIFERENTES(self):
        """"os dois matam dragao" e "algum mata o Elder" sao apostas opostas
        em espirito. Confundir as duas seria apostar ao contrario."""
        ambos = self._e("(Map 3) Both Teams To Kill An Elemental Dragon?")
        algum = self._e("(Map 3) Will Either Team Slay The Elder Dragon?")
        self.assertEqual(ambos["forma"], "ambos")
        self.assertEqual(algum["forma"], "algum")
        self.assertNotEqual(ambos["metrica"], algum["metrica"])

    def test_par_ou_impar(self):
        self.assertEqual(self._e("(Map 2) Total Kills Odd/Even")["forma"], "par_impar")

    def test_placar_exato_e_da_SERIE_nao_de_um_mapa(self):
        d = self._e("Correct Score")
        self.assertEqual(d["mapa"], 0)
        self.assertEqual(d["metrica"], "placar")


class TestONomeDaCasaNAOSePerde(unittest.TestCase):
    """E o unico jeito de ele conferir a linha na tela da Pinnacle."""

    def test_o_nome_original_volta_intacto(self):
        for n in ("(Map 3) Total Turrets Destroyed", "Correct Score",
                  "Vencedor - serie", "coisa que eu nunca vi"):
            self.assertEqual(entender_mercado(n)["nome_casa"], n)

    def test_o_nome_em_portugues_e_um_campo_A_MAIS(self):
        d = entender_mercado("(Map 3) Total Turrets Destroyed")
        self.assertIn("Torres", d["nome_pt"])
        self.assertIn("mapa 3", d["nome_pt"])
        self.assertNotEqual(d["nome_pt"], d["nome_casa"])


class TestNomeDesconhecidoNAOViraPalpite(unittest.TestCase):
    """PRINCIPIO IV, na forma mais cara possivel.

    Chutar que um mercado desconhecido "deve ser kills" poria uma aposta de
    dragao na conta de abates -- e o numero sairia com toda a confianca.
    """

    def test_nome_que_ninguem_conhece_sai_SEM_metrica(self):
        d = entender_mercado("(Map 1) Mercado Que A Casa Inventou Ontem")
        self.assertIsNone(d["metrica"])
        self.assertFalse(d["entendido"])

    def test_mas_o_MAPA_dele_ainda_e_lido(self):
        """Saber menos nao e saber nada: o mapa vem do formato, nao do nome."""
        self.assertEqual(
            entender_mercado("(Map 4) Coisa Nova")["mapa"], 4)

    def test_vazio_e_None_nao_estouram(self):
        for v in ("", None, "   ", 7):
            d = entender_mercado(v)
            self.assertFalse(d["entendido"])

    def test_a_lista_de_NAO_ENTENDIDOS_conta_cada_um(self):
        c = nao_entendidos(["(Map 1) Coisa Nova", "(Map 1) Coisa Nova",
                            "(Map 3) Total Turrets Destroyed"])
        self.assertEqual(c, {"(Map 1) Coisa Nova": 2})

    def test_o_resumo_bate_as_contas(self):
        r = resumir(["(Map 3) Total Turrets Destroyed", "nada disso"])
        self.assertEqual((r["total"], r["entendidos"], r["sem_entender"]), (2, 1, 1))


class TestOsNossosProprosRotulos(unittest.TestCase):
    """O coletor monta alguns nomes sozinho quando a casa nao da descricao."""

    def test_total_de_kills_por_mapa(self):
        d = entender_mercado("Total - Kills - mapa 3")
        self.assertEqual((d["metrica"], d["forma"], d["mapa"]), ("kills", "total", 3))

    def test_total_do_TIME_e_diferente_de_total(self):
        """Um e o total da partida, o outro e o de um time so."""
        a = entender_mercado("Total do time - Kills - mapa 1")
        b = entender_mercado("Total - Kills - mapa 1")
        self.assertEqual(a["forma"], "total_time")
        self.assertEqual(b["forma"], "total")

    def test_vencedor_da_serie_e_mapa_zero(self):
        d = entender_mercado("Vencedor - serie")
        self.assertEqual((d["metrica"], d["mapa"]), ("ml", 0))

    def test_handicap_de_kills_nao_vira_handicap_de_vencedor(self):
        self.assertEqual(entender_mercado("Handicap - Kills - mapa 2")["metrica"],
                         "kills")
        self.assertEqual(entender_mercado("Handicap - serie")["metrica"], "ml")


class TestContraOHistoricoDeVerdade(unittest.TestCase):
    """A tabela foi escrita a partir do que REALMENTE apareceu. Se ela deixar
    de cobrir o que ja foi coletado, alguma coisa regrediu."""

    def _mercados(self):
        alvo = RAIZ / "odds_pinnacle_historico.jsonl"
        if not alvo.exists():
            self.skipTest("o historico nao esta nesta maquina")
        fora = []
        for linha in alvo.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if not linha:
                continue
            try:
                fora.append(json.loads(linha).get("mercado"))
            except ValueError:
                continue
        return fora

    def test_entende_pelo_menos_97_por_cento_do_historico(self):
        r = resumir(self._mercados())
        pct = 100.0 * r["entendidos"] / max(r["total"], 1)
        self.assertGreater(pct, 97.0, "cai para %.1f%%" % pct)

    def test_o_que_sobra_e_SO_o_primeiro_dia_sem_nome(self):
        """As 224 linhas sem nome e as 54 'Regular' sao de 15/09/2026, antes
        de o coletor passar a nomear mercado. Qualquer outro nome sobrando e
        mercado novo da casa, e ai ha trabalho a fazer."""
        sobrando = nao_entendidos(self._mercados())
        inesperados = {k: v for k, v in sobrando.items()
                       if str(k or "").strip() not in ("", "None", "Regular")}
        self.assertEqual(inesperados, {},
                         "mercado novo apareceu: acrescente na tabela")


class TestOColetorUsaEAvisa(unittest.TestCase):

    def setUp(self):
        self.fonte = (RAIZ / "tools" / "coletar_odds.py").read_text(encoding="utf-8")

    def test_cada_linha_gravada_leva_a_traducao(self):
        for campo in ('"metrica"', '"forma"', '"mapa"', '"nome_pt"'):
            self.assertIn(campo, self.fonte)

    def test_a_traducao_vem_ANTES_da_chave(self):
        """A chave identifica a aposta; se ela for montada antes, os campos
        novos nao entram nela."""
        i_trad = self.fonte.index('linhas[-1].update(traducao)')
        i_chave = self.fonte.index('linhas[-1]["chave"] = chave_da_linha')
        self.assertLess(i_trad, i_chave)

    def test_falhar_a_traducao_NAO_derruba_a_coleta_do_preco(self):
        """O preco e o que nao da para recuperar depois."""
        i = self.fonte.index("from app_core.mercado_pinnacle_core import entender_mercado")
        self.assertIn("except Exception", self.fonte[i:i + 900])

    def test_o_diario_conta_os_mercados_que_ninguem_soube_ler(self):
        """Mercado novo nao pode sumir calado."""
        self.assertIn("mercados_sem_ler=", self.fonte)
        self.assertIn("nomes_sem_ler=", self.fonte)


if __name__ == "__main__":
    unittest.main()
