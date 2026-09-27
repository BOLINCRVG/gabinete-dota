# -*- coding: utf-8 -*-
"""Os jogos do calendario aparecem no seletor de confronto, sem importar nada.

O DEFEITO (relatado por ele em 16/09/2026, com print): abriu "Selecionar jogo
salvo (Agenda)" e so viu TRES jogos de 15/05 -- os antigos dele. Os 15 jogos
que o robo tinha coletado naquele dia nao estavam la.

A CAUSA foi decisao minha, nao bug: eu liguei a coleta a uma IMPORTACAO. Para
o jogo coletado chegar neste seletor era preciso ir na aba Jogos, clicar em
"Trazer os jogos coletados", conferir o mapeamento e clicar em "Adicionar a
agenda". Tres cliques, duas telas.

So que o pedido dele tinha sido, com estas palavras:

    "esse jogos (agenda) e onde eu quero que fiquem os jogos que voce pegar do
     calendario, PARA QUE EU NAO PRECISE FICAR INDO ATRAS DE JOGOS"

Coleta que exige ir atras dela nao resolve o problema que ele descreveu.
"""
from __future__ import annotations

import unittest

from app_core.agenda_coletada_core import (
    ORIGEM_COLETADO,
    ORIGEM_SALVO,
    juntar_para_escolher,
    rotulo_do_jogo,
)


def _salvo(**k):
    base = {"date": "15/05", "time": "13:00", "league": "EM",
            "teamA": "Barcząca Esports", "teamB": "French Flair", "bo": 3}
    base.update(k)
    return base


def _coletado(**k):
    base = {"date": "16/09", "time": "10:00", "league": "LPL",
            "teamA_raw": "Top Esports", "teamB_raw": "Invictus Gaming", "bo": 5,
            "estado": "unstarted"}
    base.update(k)
    return base


class TestOsColetadosAPARECEM(unittest.TestCase):

    def test_coletado_entra_na_lista_sem_importar(self):
        r = juntar_para_escolher([], [_coletado()])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["teamA"], "Top Esports")
        self.assertEqual(r[0]["teamB"], "Invictus Gaming")

    def test_os_salvos_DELE_continuam_na_lista(self):
        """Nao se troca a lista dele pela minha. Soma-se."""
        r = juntar_para_escolher([_salvo()], [_coletado()])
        self.assertEqual(len(r), 2)
        self.assertIn("Barcząca Esports", [j["teamA"] for j in r])

    def test_os_coletados_vem_PRIMEIRO(self):
        """Sao os jogos POR VIR -- e por eles que ele abre a lista.

        Os salvos dele no dia do relato eram de 15/05, quatro meses velhos.
        Deixar os de hoje embaixo deles seria manter o problema.
        """
        r = juntar_para_escolher([_salvo()], [_coletado()])
        self.assertEqual(r[0]["origem"], ORIGEM_COLETADO)
        self.assertEqual(r[-1]["origem"], ORIGEM_SALVO)

    def test_a_ordem_dos_coletados_entre_si_e_mantida(self):
        r = juntar_para_escolher([], [_coletado(teamA_raw="A"), _coletado(teamA_raw="B")])
        self.assertEqual([j["teamA"] for j in r], ["A", "B"])


class TestNadaDoQueEDELESeperde(unittest.TestCase):
    """Apagar trabalho dele seria pior que nao mostrar jogo nenhum."""

    def test_partida_nas_DUAS_listas_aparece_uma_vez_so(self):
        igual_salvo = _salvo(date="16/09", time="10:00",
                             teamA="Top Esports", teamB="Invictus Gaming")
        r = juntar_para_escolher([igual_salvo], [_coletado()])
        self.assertEqual(len(r), 1)

    def test_e_quem_fica_e_a_versao_DELE(self):
        """A dele pode ter LINHA DIGITADA junto; a coletada nunca tem."""
        igual_salvo = _salvo(date="16/09", time="10:00",
                             teamA="Top Esports", teamB="Invictus Gaming",
                             lines_saved={"kills_text": "27.5"})
        r = juntar_para_escolher([igual_salvo], [_coletado()])
        self.assertEqual(r[0]["origem"], ORIGEM_SALVO)
        self.assertEqual(r[0]["lines_saved"], {"kills_text": "27.5"})

    def test_a_comparacao_nao_e_sensivel_a_maiuscula(self):
        igual = _salvo(date="16/09", time="10:00",
                       teamA="TOP ESPORTS", teamB="invictus gaming")
        self.assertEqual(len(juntar_para_escolher([igual], [_coletado()])), 1)

    def test_jogo_parecido_em_OUTRO_horario_nao_e_o_mesmo(self):
        outro = _salvo(date="16/09", time="14:00",
                       teamA="Top Esports", teamB="Invictus Gaming")
        self.assertEqual(len(juntar_para_escolher([outro], [_coletado()])), 2)


class TestORotuloDizDeOndeVEIO(unittest.TestCase):
    """Sem a marca ele nao sabe se aquele jogo tem linha dele salva junto."""

    def test_o_coletado_vem_marcado(self):
        r = juntar_para_escolher([], [_coletado()])
        self.assertIn("calendario", rotulo_do_jogo(r[0]))

    def test_o_salvo_NAO_vem_marcado(self):
        r = juntar_para_escolher([_salvo()], [])
        self.assertNotIn("calendario", rotulo_do_jogo(r[0]))

    def test_o_rotulo_tem_data_liga_times_e_bo(self):
        texto = rotulo_do_jogo(juntar_para_escolher([], [_coletado()])[0])
        for pedaco in ("16/09", "10:00", "LPL", "Top Esports", "Invictus Gaming", "BO5"):
            self.assertIn(pedaco, texto)

    def test_bo_ilegivel_nao_estoura_e_cai_no_padrao(self):
        r = juntar_para_escolher([], [_coletado(bo="nao e numero")])
        self.assertIn("BO3", rotulo_do_jogo(r[0]))

    def test_jogo_sem_liga_nao_deixa_barra_solta(self):
        texto = rotulo_do_jogo(juntar_para_escolher([], [_coletado(league="")])[0])
        self.assertNotIn("|  |", texto)


class TestListaEstragadaNaoDerruba(unittest.TestCase):

    def test_lixo_nas_duas_listas_e_ignorado(self):
        r = juntar_para_escolher(["lixo", None, _salvo()], [7, _coletado(), None])
        self.assertEqual(len(r), 2)

    def test_listas_vazias_ou_ausentes(self):
        self.assertEqual(juntar_para_escolher([], []), [])
        self.assertEqual(juntar_para_escolher(None, None), [])


class TestATelaUSAOJuntador(unittest.TestCase):
    """O seletor nao pode voltar a ler so a lista salva."""

    def test_o_seletor_chama_o_juntador(self):
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("_jogos_do_seletor", src)
        self.assertNotIn("games = get_schedule_games(session_state)", src)

    def test_o_carregador_le_a_agenda_coletada(self):
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m._jogos_do_seletor)
        self.assertIn("carregar_agenda_coletada", src)
        self.assertIn("juntar_para_escolher", src)

    def test_se_a_leitura_FALHAR_a_lista_dele_continua(self):
        """O seletor nunca pode ficar pior do que ja era."""
        from app_core import matchup_selection_core as m

        class _EstadoQueSoTemOsSalvos(dict):
            pass

        est = _EstadoQueSoTemOsSalvos()
        est["games_list"] = [_salvo()]
        saida, coleta = m._jogos_do_seletor(est)
        self.assertTrue(any(j.get("teamA") == "Barcząca Esports" for j in saida))
        self.assertIsInstance(coleta, dict)


class TestSemRede(unittest.TestCase):
    def test_o_juntador_nao_vai_a_rede(self):
        import inspect

        from app_core import agenda_coletada_core as m
        src = inspect.getsource(m)
        self.assertNotIn("import requests", src)
        self.assertNotIn("urlopen", src)


if __name__ == "__main__":
    unittest.main()


class TestAOrdemDaLista(unittest.TestCase):
    """ELE RELATOU (16/09/2026): "a ordem so e meio cagada".

    A lista ia 16/09, 17/09, 18/09... 20/09 e entao caia para 15/05, sem nada
    explicando: eram os tres jogos que ele tinha salvo em maio. Data andando
    PARA TRAS no fim parece lista quebrada, nao "secao de jogos antigos".

    A causa era eu ter descartado `inicio_utc` na leitura. Sem o instante
    exato, ordenar dependia de adivinhar o ano de um "15/05" solto.
    """

    def _agora(self):
        from datetime import datetime, timezone
        return datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)

    def test_o_mais_PROXIMO_vem_primeiro(self):
        """PRAZOS RELATIVOS A AGORA, e nao datas fixas.

        A primeira versao usava "16/09" e "20/09" cravados. Passou no dia em
        que foi escrita e REPROVOU no dia seguinte, quando 16/09 virou
        passado e foi para o fim da lista -- que e o comportamento CERTO.

        Teste com data cravada mede o calendario, nao o codigo. O segundo
        dia seguido em que isso me pegou.
        """
        from datetime import datetime, timedelta, timezone
        agora = datetime.now(timezone.utc)
        def _em(horas):
            return (agora + timedelta(hours=horas)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = juntar_para_escolher([], [
            _coletado(inicio_utc=_em(96), teamA_raw="depois"),
            _coletado(inicio_utc=_em(3), teamA_raw="antes"),
        ])
        self.assertEqual([j["teamA"] for j in r], ["antes", "depois"])

    def test_o_que_JA_PASSOU_vai_para_o_fim(self):
        from datetime import datetime, timedelta, timezone
        agora = datetime.now(timezone.utc)
        futuro = (agora + timedelta(hours=96)).strftime("%Y-%m-%dT%H:%M:%SZ")
        passado = agora - timedelta(days=90)
        r = juntar_para_escolher(
            [_salvo(date=passado.strftime("%d/%m"), time="13:00", teamA="velho")],
            [_coletado(inicio_utc=futuro, teamA_raw="novo")])
        self.assertEqual(r[-1]["teamA"], "velho")

    def test_e_vem_MARCADO_em_vez_de_sumir(self):
        """Esconder o jogo dele seria pior: ele pode ter linha salva ali."""
        from datetime import datetime, timedelta, timezone
        passado = datetime.now(timezone.utc) - timedelta(days=90)
        r = juntar_para_escolher(
            [_salvo(date=passado.strftime("%d/%m"), time="13:00")], [])
        self.assertIn("JA PASSOU", rotulo_do_jogo(r[0]))

    def test_jogo_por_vir_NAO_e_marcado_como_passado(self):
        r = juntar_para_escolher([], [_coletado(inicio_utc="2099-01-01T00:00:00Z")])
        self.assertNotIn("JA PASSOU", rotulo_do_jogo(r[0]))


class TestDeQueANOEUmaDataSemAno(unittest.TestCase):
    """"15/05" visto em setembro e maio DESTE ano, que ja passou.

    Escolher o ano errado inverteria a lista inteira: o jogo de maio passaria
    a valer como maio do ANO QUE VEM e subiria para o topo, na frente dos
    jogos de amanha.
    """

    def _agora(self):
        from datetime import datetime, timezone
        return datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)

    def test_data_de_meses_atras_e_deste_ano(self):
        from app_core.agenda_coletada_core import quando_do_jogo
        q = quando_do_jogo({"date": "15/05", "time": "13:00"}, self._agora())
        self.assertEqual((q.year, q.month, q.day), (2026, 5, 15))

    def test_data_de_dezembro_vista_em_janeiro_e_do_ano_ANTERIOR(self):
        from datetime import datetime, timezone

        from app_core.agenda_coletada_core import quando_do_jogo
        jan = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
        q = quando_do_jogo({"date": "28/12", "time": "20:00"}, jan)
        self.assertEqual((q.year, q.month), (2025, 12))

    def test_o_instante_EXATO_tem_preferencia_sobre_o_palpite(self):
        from app_core.agenda_coletada_core import quando_do_jogo
        q = quando_do_jogo({"date": "01/01", "time": "00:00",
                            "inicio_utc": "2026-09-20T17:00:00Z"}, self._agora())
        self.assertEqual((q.year, q.month, q.day), (2026, 9, 20))

    def test_data_impossivel_devolve_NADA_em_vez_de_inventar(self):
        """PRINCIPIO IV: sem data legivel, None -- e None vai para o fim."""
        from app_core.agenda_coletada_core import quando_do_jogo
        self.assertIsNone(quando_do_jogo({"date": "", "time": ""}, self._agora()))
        self.assertIsNone(quando_do_jogo({"date": "amanha"}, self._agora()))
        self.assertIsNone(quando_do_jogo({"date": "99/99", "time": "13:00"}, self._agora()))

    def test_jogo_sem_data_vai_para_o_FIM_e_nao_derruba(self):
        r = juntar_para_escolher([_salvo(date="", time="")],
                                 [_coletado(inicio_utc="2099-01-01T00:00:00Z")])
        self.assertEqual(r[-1]["teamA"], "Barcząca Esports")


class TestOInstanteEXATOEGuardado(unittest.TestCase):
    """Eu vinha descartando `inicio_utc` ao ler a agenda. Era a causa."""

    def test_a_leitura_guarda_o_inicio_utc(self):
        import json
        import tempfile
        from pathlib import Path

        from app_core.agenda_coletada_core import carregar_agenda_coletada
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "agenda_coletada.json").write_text(json.dumps({
                "gerado_em": "2026-09-16T09:00:00-03:00",
                "jogos": [{"date": "16/09", "time": "11:00", "league": "LEC",
                           "teamA_raw": "A", "teamB_raw": "B", "bo": 5,
                           "inicio_utc": "2026-09-16T14:00:00Z"}],
            }), encoding="utf-8")
            r = carregar_agenda_coletada(Path(d))
        self.assertEqual(r["jogos"][0]["inicio_utc"], "2026-09-16T14:00:00Z")

    def test_agenda_sem_inicio_utc_nao_estoura(self):
        import json
        import tempfile
        from pathlib import Path

        from app_core.agenda_coletada_core import carregar_agenda_coletada
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "agenda_coletada.json").write_text(json.dumps({
                "gerado_em": "2026-09-16T09:00:00-03:00",
                "jogos": [{"date": "16/09", "time": "11:00", "league": "LEC",
                           "teamA_raw": "A", "teamB_raw": "B", "bo": 5}],
            }), encoding="utf-8")
            r = carregar_agenda_coletada(Path(d))
        self.assertEqual(r["jogos"][0]["inicio_utc"], "")


class TestOsAntigosSAEMDaLista(unittest.TestCase):
    """ELE RECLAMOU DUAS VEZES, e na segunda eu entendi.

    Primeiro: "a ordem so e meio cagada" -- eu ordenei e MARQUEI os antigos
    com "JA PASSOU". Ele olhou o resultado e disse: "aqui, ainda esta o
    mesmo".

    Estava certo. Marcar nao resolve: nao existe nada a fazer com um jogo de
    quatro meses atras numa tela de escolher confronto para apostar. Ele so
    ocupa espaco na frente dos jogos de amanha. Meu conserto tinha tratado o
    sintoma (a ordem) e deixado a causa (eles nao deveriam estar la).

    ESCONDIDO, NUNCA APAGADO: sao jogos DELE, podem ter linha digitada junto.
    """

    def test_a_tela_esconde_o_que_ja_passou(self):
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("_ja_passou", src)
        self.assertIn("main_saved_game_ver_passados", src)

    def test_a_tela_DIZ_quantos_escondeu(self):
        """Sumir em silencio trocaria um incomodo por uma duvida."""
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("ja aconteceram", src)

    def test_o_interruptor_traz_de_volta(self):
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("st.checkbox", src)

    def test_o_indice_guardado_e_limpo_quando_a_lista_encolhe(self):
        """Sem isto o Streamlit estoura.

        A lista encolhe quando os antigos somem; um indice salvo de antes
        apontaria para fora dela, e o Streamlit levanta erro quando o valor
        guardado nao esta mais entre as opcoes.
        """
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("main_saved_game_idx", src)
        self.assertIn(">= len(labels)", src)

    def test_ja_passou_separa_certo_os_dois_casos(self):
        from datetime import datetime, timezone

        from app_core.agenda_coletada_core import ja_passou
        agora = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        self.assertTrue(ja_passou({"date": "15/05", "time": "13:00"}, agora))
        self.assertFalse(ja_passou({"inicio_utc": "2026-09-20T17:00:00Z"}, agora))

    def test_jogo_que_comecou_ha_pouco_NAO_conta_como_passado(self):
        """Serie de BO5 dura horas; sumir no meio dela seria perder o jogo."""
        from datetime import datetime, timezone

        from app_core.agenda_coletada_core import ja_passou
        agora = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        self.assertFalse(ja_passou({"inicio_utc": "2026-09-16T10:00:00Z"}, agora))

    def test_jogo_sem_data_NAO_e_tratado_como_passado(self):
        """PRINCIPIO IV: "nao sei quando e" nao vira "ja foi", que sumiria
        com ele da lista sem ninguem poder conferir."""
        from datetime import datetime, timezone

        from app_core.agenda_coletada_core import ja_passou
        agora = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
        self.assertFalse(ja_passou({"date": "", "time": ""}, agora))


class TestOInstanteEXATOSobreviveAJuncao(unittest.TestCase):
    """DEFEITO MEU, pego pela suite um dia depois (16/09/2026).

    Eu guardei o `inicio_utc` na leitura da agenda -- justamente para a ordem
    parar de depender de adivinhar o ano de um "15/05" solto -- e o DESCARTEI
    tres linhas depois, ao juntar as listas.

    A ordem continuou parecendo certa porque sobrava o palpite por "dd/mm". So
    que o palpite trata a hora do rotulo, que e LOCAL, como se fosse UTC: erro
    de tres horas, que marca jogo como "ja passou" cedo demais. Um BO5 que
    comeca as 10:00 sumia da lista as 07:00.

    Foi preciso o relogio andar um dia para a suite reprovar. Teste que
    depende da hora e incomodo -- e foi ele que pegou.
    """

    def test_o_inicio_utc_do_coletado_atravessa(self):
        r = juntar_para_escolher([], [_coletado(inicio_utc="2026-09-20T17:00:00Z")])
        self.assertEqual(r[0]["inicio_utc"], "2026-09-20T17:00:00Z")

    def test_e_e_ELE_que_manda_na_ordem(self):
        """Com o instante presente, o rotulo "dd/mm" nao decide mais nada."""
        from app_core.agenda_coletada_core import quando_do_jogo
        j = juntar_para_escolher([], [_coletado(date="01/01", time="00:00",
                                                inicio_utc="2026-09-20T17:00:00Z")])[0]
        q = quando_do_jogo(j)
        self.assertEqual((q.year, q.month, q.day, q.hour), (2026, 9, 20, 17))

    def test_jogo_do_futuro_distante_nunca_e_dado_como_passado(self):
        """O caso que reprovou: sem o instante, o "dd/mm" de hoje mandava."""
        r = juntar_para_escolher([], [_coletado(date="16/09", time="10:00",
                                                inicio_utc="2099-01-01T00:00:00Z")])
        self.assertNotIn("JA PASSOU", rotulo_do_jogo(r[0]))

    def test_coletado_SEM_instante_nao_estoura(self):
        """Agenda antiga, gravada antes do campo existir."""
        cru = _coletado()
        cru.pop("inicio_utc", None)
        r = juntar_para_escolher([], [cru])
        self.assertEqual(r[0]["inicio_utc"], "")
