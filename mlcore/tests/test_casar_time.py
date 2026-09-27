# -*- coding: utf-8 -*-
"""Casar sozinho o nome de um time -- so quando nao ha como errar.

ELE DISSE (23/09/2026): "ele ainda nao pega os times de forma correta, alguns
times eu ainda tenho que indicar os nomes".

A PROTECAO QUE NAO PODE CAIR: a regra antiga nao adivinhava por causa do caso
UP2U -- "UP2U Meavedron" casado com "UP2UMEDIA Cebulaki", de outra liga, e a
analise inteira saiu sobre o time errado. A regra nova so decide quando sobra
UM candidato depois de tres filtros: nome, liga do adversario, time ativo.

MEDIDO:
  - nos 52 apelidos que ele ensinou: 36 certos, 16 deixados para ele, 0 errados;
  - nas partidas REAIS da Pinnacle, fingindo que os 16 apelidos conferidos na
    Leaguepedia em 21/09 nao existiam: 13 nomes casados, todos certos --
    inclusive G2 (nao G2 NORD), Invictus (nao Macko Invictus) e paiN (nao a
    Academy). E NAO decidiu os dificeis: T1 Academy, Bilibili Junior,
    Gamespace MCE.

Os casos abaixo sao os reais dessas medicoes, nao inventados.
"""
from __future__ import annotations

import unittest
from datetime import date

from app_core import casar_time_core as ct
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol

HOJE = date(2026, 9, 21)
TIMES = ["Team Liquid", "G2 Esports", "G2 NORD", "Invictus Gaming",
         "Macko Invictus", "paiN Gaming", "paiN Gaming Academy", "T1",
         "T1 Esports Academy", "T1 Esports Academy Rookies", "Los Heretics",
         "Team Heretics Academy", "Otter Side", "Nongshim RedForce",
         "KT Rolster", "KT Rolster Challengers", "Karmine Corp", "FlyQuest"]
LIGAS = {
    "Team Liquid": {"LCS", "MSI"}, "FlyQuest": {"LCS"},
    "G2 Esports": {"LEC", "MSI", "EWC"}, "G2 NORD": {"EM", "PRM"},
    "Karmine Corp": {"LEC"},
    "Invictus Gaming": {"LPL"}, "Macko Invictus": {"LIT"},
    "paiN Gaming": {"CBLOL"}, "paiN Gaming Academy": {"CD"},
    "T1": {"LCK"}, "T1 Esports Academy": {"LCKC"},
    "T1 Esports Academy Rookies": {"LAS"},
    "Los Heretics": {"EM", "LVP SL"}, "Team Heretics Academy": {"EM", "LES"},
    "Otter Side": {"EM", "LPLOL"}, "Nongshim RedForce": {"LCK"},
    "KT Rolster": {"LCK"}, "KT Rolster Challengers": {"LCKC"},
}
ATIVO = date(2026, 9, 15)
ULTIMO = {t: ATIVO for t in TIMES}
ULTIMO["Los Heretics"] = date(2025, 11, 21)      # o nome velho, parado


def casar(cru, adversario):
    return ct.casar_sozinho(cru, TIMES, LIGAS, LIGAS.get(adversario), ULTIMO, HOJE)


class TestOsCasosReaisQueElaAcerta(unittest.TestCase):

    def test_sufixo_simples(self):
        self.assertEqual(casar("Liquid", "FlyQuest"), "Team Liquid")

    def test_G2_e_G2_Esports_e_NAO_G2_NORD(self):
        self.assertEqual(casar("G2", "Karmine Corp"), "G2 Esports")

    def test_Invictus_e_Invictus_Gaming_e_NAO_Macko(self):
        self.assertEqual(casar("Invictus", "T1"), None)       # T1 nao joga LPL
        self.assertEqual(ct.casar_sozinho("Invictus", TIMES, LIGAS, {"LPL"},
                                          ULTIMO, HOJE), "Invictus Gaming")

    def test_paiN_e_paiN_Gaming_e_NAO_a_Academy(self):
        self.assertEqual(ct.casar_sozinho("paiN", TIMES, LIGAS, {"CBLOL"},
                                          ULTIMO, HOJE), "paiN Gaming")

    def test_artigo_the_no_comeco(self):
        self.assertEqual(ct.casar_sozinho("The Otter Side", TIMES, LIGAS, {"EM"},
                                          ULTIMO, HOJE), "Otter Side")

    def test_prefixo_do_nome(self):
        self.assertEqual(ct.casar_sozinho("nong", TIMES, LIGAS, {"LCK"},
                                          ULTIMO, HOJE), "Nongshim RedForce")

    def test_KT_a_liga_desempata_o_principal_da_academia(self):
        self.assertEqual(ct.casar_sozinho("KT", TIMES, LIGAS, {"LCK"}, ULTIMO, HOJE),
                         "KT Rolster")
        self.assertEqual(ct.casar_sozinho("KT", TIMES, LIGAS, {"LCKC"}, ULTIMO, HOJE),
                         "KT Rolster Challengers")


class TestOsCasosEmQueElaTemDeSeCALAR(unittest.TestCase):
    """Abster e aceitavel. Errar nao e."""

    def test_T1_Academy_NUNCA_vira_o_T1_principal(self):
        """O palpite tentador e errado. A chave do time dentro da crua NAO
        conta -- so a crua no comeco de uma palavra do time."""
        self.assertNotEqual(ct.casar_sozinho("T1 Academy", TIMES, LIGAS, {"LCK"},
                                             ULTIMO, HOJE), "T1")

    def test_nome_renomeado_e_parado_nao_decide(self):
        """O ERRO REAL que a medicao pegou: "losheretics" achava "Los
        Heretics", que parou em 21/11/2025 -- renomeado para "Team Heretics
        Academy". O filtro de 60 dias ativo tira o nome velho."""
        self.assertIsNone(ct.casar_sozinho("losheretics", TIMES, LIGAS, {"EM"},
                                           ULTIMO, HOJE))

    def test_sem_liga_do_adversario_NAO_decide(self):
        """Sem liga, a medicao errou um caso em 52. O criterio e zero."""
        self.assertIsNone(ct.casar_sozinho("Liquid", TIMES, LIGAS, None, ULTIMO, HOJE))
        self.assertIsNone(ct.casar_sozinho("Liquid", TIMES, LIGAS, set(), ULTIMO, HOJE))

    def test_dois_candidatos_na_mesma_liga_NAO_decide(self):
        ligas = dict(LIGAS)
        ligas["paiN Gaming Academy"] = {"CBLOL"}
        self.assertIsNone(ct.casar_sozinho("paiN", TIMES, ligas, {"CBLOL"},
                                           ULTIMO, HOJE))

    def test_chave_curta_demais_nao_tenta(self):
        self.assertIsNone(ct.casar_sozinho("T", TIMES, LIGAS, {"LCK"}, ULTIMO, HOJE))

    def test_candidato_sem_data_de_jogo_nao_decide(self):
        ult = dict(ULTIMO)
        del ult["Team Liquid"]
        self.assertIsNone(ct.casar_sozinho("Liquid", TIMES, LIGAS, {"LCS"}, ult, HOJE))


class TestCompletarOPar(unittest.TestCase):

    def test_um_lado_casou_o_outro_e_completado(self):
        a, b, autos = ct.completar_par("FlyQuest", "FlyQuest", "Liquid", "",
                                       TIMES, LIGAS, ULTIMO, HOJE)
        self.assertEqual((a, b), ("FlyQuest", "Team Liquid"))
        self.assertEqual(autos, [("Time 2", "Liquid", "Team Liquid", "FlyQuest")])

    def test_os_dois_vazios_NAO_faz_nada(self):
        """Sem um lado casado nao ha liga para desempatar."""
        a, b, autos = ct.completar_par("Liquid", "", "G2", "", TIMES, LIGAS, ULTIMO, HOJE)
        self.assertEqual((a, b, autos), ("", "", []))

    def test_os_dois_casados_NAO_faz_nada(self):
        a, b, autos = ct.completar_par("x", "T1", "y", "KT Rolster",
                                       TIMES, LIGAS, ULTIMO, HOJE)
        self.assertEqual(autos, [])

    def test_nunca_casa_um_time_contra_ele_mesmo(self):
        a, b, autos = ct.completar_par("Team Liquid", "Team Liquid", "Liquid", "",
                                       TIMES, LIGAS, ULTIMO, HOJE)
        self.assertNotEqual(b, "Team Liquid")

    def test_hoje_padrao_e_a_data_mais_recente_do_CSV_e_nao_o_relogio(self):
        """Foi assim que a regra foi medida. O CSV dele anda um dia atras."""
        a, b, autos = ct.completar_par("FlyQuest", "FlyQuest", "Liquid", "",
                                       TIMES, LIGAS, ULTIMO)
        self.assertEqual(b, "Team Liquid")


class TestNadaEGravadoSemEle(unittest.TestCase):
    """Um palpite gravado sem ele ver viraria permanente e invisivel."""

    def test_o_modulo_nao_escreve_arquivo(self):
        from pathlib import Path
        fonte = Path(ct.__file__).read_text(encoding="utf-8")
        for proibido in ("write_text", "json.dump", "open(", "save_schedule_alias"):
            self.assertNotIn(proibido, fonte)

    def test_as_telas_MOSTRAM_o_que_foi_casado_sozinho(self):
        from pathlib import Path
        raiz = Path(ct.__file__).resolve().parents[1]
        agenda = (raiz / "app_core" / "matchup_selection_core.py").read_text(encoding="utf-8")
        aba = (raiz / "app_core" / "oportunidades_tab_core.py").read_text(encoding="utf-8")
        self.assertIn("_casados_sozinhos", agenda)
        self.assertIn("Esta certo, guardar", agenda)
        self.assertIn("_mostrar_casados_sozinhos", aba)

    @so_no_lol("so vale para o LoL")
    def test_a_medicao_usa_a_regra_de_PRODUCAO(self):
        """Uma copia na ferramenta mediria uma regra e a tela serviria outra."""
        from pathlib import Path
        raiz = Path(ct.__file__).resolve().parents[1]
        fonte = (raiz / "tools" / "eval_casar_times.py").read_text(encoding="utf-8")
        self.assertIn("from app_core.casar_time_core import", fonte)
        self.assertNotIn("def candidatos(", fonte)


if __name__ == "__main__":
    unittest.main()
