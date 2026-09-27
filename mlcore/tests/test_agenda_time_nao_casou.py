# -*- coding: utf-8 -*-
"""Testes do "Usar este jogo" quando o nome do time nao casa com o CSV.

O DEFEITO (relatado por ele em 15/09/2026, com print): ele escolheu um jogo
salvo da agenda, clicou em "Usar este jogo", e o **Time 1 ficou vazio**. O
Time 2 preencheu normalmente. Nenhuma mensagem, nenhum aviso — parecia que o
botao nao tinha funcionado.

A CAUSA: a checagem era `nome in teams`, exata letra por letra. A agenda dele
tinha "Barczaca Esports" e o CSV tem "Barczaca Esports by Yumisu". Sufixo
diferente, campo vazio, silencio.

E o mais frustrante: o resolvedor com apelidos JA EXISTIA no projeto — e o
mesmo que a aba Jogos usa ao importar — e nunca tinha sido chamado aqui. Alem
disso, o nome certo aparecia em TERCEIRO nas sugestoes automaticas.

CAMPO VAZIO SEM EXPLICACAO E O PIOR DESFECHO: nao da nem para saber se o erro
foi do app, do dado ou do clique.
"""
from __future__ import annotations

import unittest

from app_core.matchup_selection_core import (
    _CHAVE_NAO_RESOLVIDOS,
    _resolver_time_salvo,
    set_pending_unresolved,
)
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol

TIMES = ["Barcząca Esports by Yumisu", "French Flair", "Gen.G", "T1"]


class TestOCasoDele(unittest.TestCase):

    def test_nome_exato_passa_direto(self):
        self.assertEqual(_resolver_time_salvo({}, "French Flair", TIMES), "French Flair")

    def test_nome_com_sufixo_faltando_NAO_e_adivinhado(self):
        """PRINCIPIO IV: "Barczaca Esports" pode ser outro time.

        Adivinhar aqui seria pior que o campo vazio — analisaria o confronto
        errado com toda a confianca. O certo e DIZER que nao casou e deixar
        ele escolher; foi isso que passou a acontecer.
        """
        self.assertEqual(_resolver_time_salvo({}, "Barcząca Esports", TIMES), "")

    def test_nome_inexistente_devolve_vazio(self):
        self.assertEqual(_resolver_time_salvo({}, "Time Que Nao Existe", TIMES), "")

    def test_vazio_devolve_vazio(self):
        self.assertEqual(_resolver_time_salvo({}, "", TIMES), "")
        self.assertEqual(_resolver_time_salvo({}, "   ", TIMES), "")

    def test_o_nome_certo_esta_nas_SUGESTOES(self):
        """E o que torna o conserto de um clique: o palpite ja vem pronto."""
        from app_core.schedule_import_core import suggest_schedule_teams
        sug = suggest_schedule_teams("Barcząca Esports", TIMES)
        self.assertIn("Barcząca Esports by Yumisu", sug)


class TestApelidoAprendidoResolve(unittest.TestCase):
    """Depois de ensinar uma vez, o problema nao volta."""

    def test_com_apelido_salvo_o_nome_passa_a_casar(self):
        import tempfile
        from pathlib import Path

        from app_core.schedule_import_core import (
            normalize_schedule_team_key,
            resolve_schedule_team,
        )
        apelidos = {normalize_schedule_team_key("Barcząca Esports"):
                    "Barcząca Esports by Yumisu"}
        achado, ok = resolve_schedule_team("Barcząca Esports", TIMES, apelidos)
        self.assertTrue(ok)
        self.assertEqual(achado, "Barcząca Esports by Yumisu")


class TestOQueNaoCasouFicaDITO(unittest.TestCase):
    """Campo vazio sem explicacao parece botao quebrado."""

    def test_guarda_o_que_faltou_para_a_tela_avisar(self):
        est = {}
        set_pending_unresolved(est, [("Time 1", "Barcząca Esports")])
        self.assertEqual(est[_CHAVE_NAO_RESOLVIDOS],
                         [("Time 1", "Barcząca Esports")])

    def test_lista_vazia_nao_estoura(self):
        est = {}
        set_pending_unresolved(est, [])
        self.assertEqual(est[_CHAVE_NAO_RESOLVIDOS], [])

    def test_estado_que_nao_aceita_escrita_nao_derruba(self):
        class _Travado:
            def __setitem__(self, k, v):
                raise TypeError("nao aceita")
        set_pending_unresolved(_Travado(), [("Time 1", "X")])   # nao pode estourar


class TestATelaUsaOResolvedor(unittest.TestCase):
    """A comparacao exata nao pode voltar."""

    def test_nao_ha_mais_comparacao_exata_com_a_lista(self):
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertNotIn('if str(game.get("teamA", "")) in teams:', src)
        self.assertIn("_resolver_time_salvo", src)

    def test_a_tela_avisa_e_oferece_o_conserto(self):
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("ficou vazio", src)
        self.assertIn("suggest_schedule_teams", src)
        self.assertIn("save_schedule_aliases", src)


if __name__ == "__main__":
    unittest.main()


class TestOBOSobreviveAoEnsinarOApelido(unittest.TestCase):
    """O DEFEITO (relatado por ele em 16/09/2026, com print): escolheu
    "16/09 11:00 | Hitpoint Masters | eSuba x UP2U Meavedron | BO5", os times
    preencheram e o BO abriu em **1**.

    Dirigindo o app, o ponto exato apareceu:
        depois de "Usar este jogo"  -> bo = 5   (certo)
        depois de "Salvar e usar"   -> bo = 1   (perdido)

    A CAUSA nao e obvia: o botao "Salvar e usar" vive ACIMA da caixa de BO na
    tela, e o rerun para o script na hora -- antes de a caixa ser criada. O
    Streamlit descarta o estado de widget que nao chegou a ser desenhado,
    entao o BO voltava ao padrao. Escrever direto em session_state nao
    adiantava: o valor morria junto.

    O botao "Usar este jogo" nunca sofreu disso porque passa os valores pela
    FILA (pending_load), aplicada antes dos widgets. Agora os dois caminhos
    usam a mesma fila -- um so, e nao dois.

    UM BO5 LIDO COMO BO1 MUDA A ODD DA SERIE INTEIRA. Nao e detalhe de tela.
    """

    def test_o_apelido_viaja_pela_FILA_e_nao_por_escrita_direta(self):
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        corte = src.split("Salvar e usar", 1)[1]
        self.assertIn("set_pending_load", corte)
        self.assertNotIn('session_state["teamA" if _rot == "Time 1" else "teamB"]', corte)

    def test_o_BO_entra_na_fila(self):
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        corte = src.split("Salvar e usar", 1)[1]
        self.assertIn('_pendente["bo"]', corte)

    def test_o_OUTRO_time_tambem_entra_na_fila(self):
        """Senao ensinar o Time 2 apagaria o Time 1, pelo mesmo motivo."""
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        corte = src.split("Salvar e usar", 1)[1]
        self.assertIn("_outro", corte)

    def test_BO_fora_da_lista_NAO_entra_na_fila(self):
        """PRINCIPIO IV: formato desconhecido nao vira 3 chutado.

        A agenda grava bo=0 quando a fonte nao diz o formato. Deixar o 0
        passar faria a tela mostrar um BO que ninguem afirmou.
        """
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        corte = src.split("Salvar e usar", 1)[1]
        self.assertIn("BO_OPTIONS", corte)

    @so_no_lol("usa o plays_app.py, o app do LoL")
    def test_a_fila_e_aplicada_ANTES_dos_widgets(self):
        """Se fosse depois, o valor chegaria tarde e a tela mostraria o antigo.

        O plays_app chama apply_pending_load bem no alto, de proposito. Este
        teste existe para ninguem mover a chamada para baixo sem perceber.
        """
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        src = (raiz / "plays_app.py").read_text(encoding="utf-8")
        self.assertLess(src.index("core_apply_pending_load(st.session_state"),
                        src.index("core_render_matchup_selection("),
                        "a fila passou a ser aplicada depois dos widgets")


class TestALigaDesempataOPalpite(unittest.TestCase):
    """O CASO REAL (16/09/2026). Ele estranhou uma analise e o motivo era este.

    Jogo do Hitpoint Masters (Tchequia): eSuba x "UP2U Meavedron". A lista de
    palpites ofereceu em PRIMEIRO:

        UP2UMEDIA Cebulaki  -- time POLONES, Rift Legends,  4 jogos
        Meavedron           -- o certo,     Hitpoint,      63 jogos

    Ele escolheu o primeiro, o apelido ficou salvo, e a analise inteira saiu
    sobre o time errado -- com 4 jogos de historico, de outro pais.

    Nome parecido entre ligas NAO e coincidencia: e o mesmo patrocinador
    (UP2U) em paises diferentes. Sem a liga, o "parecido" e uma armadilha.
    """

    TIMES = ["Meavedron", "UP2UMEDIA Cebulaki", "Gity Meavedronu", "eSuba"]
    LIGAS = {"Meavedron": "HM", "eSuba": "HM",
             "UP2UMEDIA Cebulaki": "RL", "Gity Meavedronu": "HW"}

    def test_SEM_a_liga_o_errado_vinha_primeiro(self):
        """Guarda a armadilha: e assim que ela se parece."""
        from app_core.schedule_import_core import suggest_schedule_teams
        sug = suggest_schedule_teams("UP2U Meavedron", self.TIMES)
        self.assertEqual(sug[0], "UP2UMEDIA Cebulaki")

    def test_COM_a_liga_o_certo_vem_primeiro(self):
        from app_core.schedule_import_core import suggest_schedule_teams
        da_liga = [t for t, lg in self.LIGAS.items() if lg == "HM"]
        sug = suggest_schedule_teams("UP2U Meavedron", self.TIMES, prioridade=da_liga)
        self.assertEqual(sug[0], "Meavedron")

    def test_o_de_outra_liga_NAO_some_da_lista(self):
        """So muda a ordem. Quem decide e ele -- pode ser um time novo que
        de fato mudou de liga, e esconder tiraria a opcao."""
        from app_core.schedule_import_core import suggest_schedule_teams
        da_liga = [t for t, lg in self.LIGAS.items() if lg == "HM"]
        sug = suggest_schedule_teams("UP2U Meavedron", self.TIMES, prioridade=da_liga)
        self.assertIn("UP2UMEDIA Cebulaki", sug)

    def test_sem_saber_a_liga_a_ordem_fica_como_era(self):
        """Nao ter o mapa nao pode quebrar nada."""
        from app_core.schedule_import_core import suggest_schedule_teams
        self.assertEqual(suggest_schedule_teams("UP2U Meavedron", self.TIMES, prioridade=[]),
                         suggest_schedule_teams("UP2U Meavedron", self.TIMES))
        self.assertEqual(suggest_schedule_teams("UP2U Meavedron", self.TIMES, prioridade=None),
                         suggest_schedule_teams("UP2U Meavedron", self.TIMES))

    def test_a_tela_usa_a_liga_do_time_que_JA_casou(self):
        """Nao existe tabela de nome de liga no projeto, e uma tabela dessas
        envelheceria. O companheiro de confronto ja diz a liga."""
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("_liga_por_time", src)
        self.assertIn("prioridade=", src)

    def test_a_tela_AVISA_quando_o_time_e_de_outra_liga(self):
        """Ordenar nao basta: ele pode escolher o de baixo assim mesmo."""
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("joga na liga", src)

    def test_modulo_antigo_no_servidor_nao_derruba_a_tela(self):
        """A nuvem nao recarrega modulo importado so com o push.

        Se o schedule_import_core do servidor for o velho, passar `prioridade`
        levanta TypeError. A tela cai para a ordem antiga em vez de quebrar --
        foi assinatura incompativel que derrubou o site em 13/09/2026.
        """
        import inspect

        from app_core import matchup_selection_core as m
        src = inspect.getsource(m.render_matchup_selection)
        self.assertIn("except TypeError:", src)
