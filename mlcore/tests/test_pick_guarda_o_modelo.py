# -*- coding: utf-8 -*-
"""O pick passa a guardar o que o MODELO disse, e se foi pre ou live.

O QUE ELE DISSE (21/09/2026): "NORMALMENTE, os ml's pres sao vindos do modelo
de alguma forma, talvez 80% deles no minimo, os live, nos nao costumamos levar
o modelo em conta".

ISSO DERRUBOU UMA AFIRMACAO MINHA. Eu tinha usado "ML: -2,89% de ROI em 884
apostas" como prova independente de que o modelo de vencedor e ruim. Mas:

  - o tracker nao tem NENHUM campo separando pre de live, entao as 884
    misturam apostas que vieram do modelo com apostas que nunca passaram por
    ele;
  - e `p_hist` NAO e a probabilidade do modelo. E a da frequencia historica
    (`odd_to_prob(odd_hist)`). No codigo sao tres coisas distintas --
    `odd_model`, `odd_hist` e `p_stats` -- e so a do meio era gravada.

Resultado: o tracker nunca conseguiu responder "o modelo ganha dinheiro?".
Nem para as 4.788 apostas antigas, nem para as novas. Estes testes protegem o
conserto, e protegem sobretudo o que NAO pode ser feito com o passado:
inventar o dado que falta.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app_core.picks_tracker_core import Pick, load_picks, new_pick, save_picks


class TestOQueOModeloDisseFicaGravado(unittest.TestCase):

    def _pick(self, **kw):
        base = dict(team_a="Gen.G", team_b="T1", league="LCK", metric_key="ml",
                    side_key="A", line=None, odd_casa=2.10, stake_units=1.0)
        base.update(kw)
        return new_pick(**base)

    def test_guarda_a_probabilidade_do_modelo(self):
        self.assertAlmostEqual(self._pick(p_modelo=0.61).p_modelo, 0.61)

    def test_p_hist_e_p_modelo_sao_campos_DIFERENTES(self):
        """O erro que comecou tudo: tratar um pelo outro."""
        p = self._pick(p_hist=0.55, p_modelo=0.61)
        self.assertAlmostEqual(p.p_hist, 0.55)
        self.assertAlmostEqual(p.p_modelo, 0.61)

    def test_sem_o_modelo_fica_VAZIO_e_nao_vira_a_frequencia(self):
        """PRINCIPIO IV: ausencia nao vira numero. Copiar p_hist aqui faria
        toda medicao futura do modelo medir outra coisa."""
        p = self._pick(p_hist=0.55)
        self.assertIsNone(p.p_modelo)

    def test_valor_sem_sentido_nao_entra(self):
        for v in (float("nan"), float("inf")):
            self.assertIsNone(self._pick(p_modelo=v).p_modelo)

    def test_a_TELA_passa_a_odd_do_modelo_ao_registrar(self):
        import inspect

        from app_core import resumo_tab_core as m
        fonte = inspect.getsource(m)
        self.assertIn('p_modelo=odd_to_prob(row.get("Odd (Modelo)"))', fonte)
        self.assertIn('momento="pre"', fonte)


class TestPreOuLive(unittest.TestCase):

    def test_pick_feito_no_app_nasce_PRE(self):
        """O app so precifica pre-jogo."""
        p = new_pick(team_a="A", team_b="B", league="LCK", metric_key="ml",
                     side_key="A", line=None, odd_casa=2.0, stake_units=1.0)
        self.assertEqual(p.momento, "pre")

    def test_aposta_antiga_SEM_o_campo_fica_DESCONHECIDO(self):
        """As 4.788 ja gravadas nao dizem se foram pre ou live. Chuta-las como
        'pre' inventaria exatamente o dado que falta -- e foi a falta dele que
        me fez afirmar algo errado sobre o modelo."""
        velho = Pick(id="x", ts="2026-05-02T00:00:00", team_a="DK", team_b="GENG",
                     league="LCK", metric_key="ml", side_key="A", line=None,
                     odd_casa=3.42, stake_units=0.5)
        self.assertEqual(velho.momento, "desconhecido")
        self.assertIsNone(velho.p_modelo)

    def test_valor_estranho_vira_DESCONHECIDO_e_nao_pre(self):
        for v in ("qualquer", "", None, 7):
            p = new_pick(team_a="A", team_b="B", league="L", metric_key="ml",
                         side_key="A", line=None, odd_casa=2.0, stake_units=1.0,
                         momento=v)
            self.assertEqual(p.momento, "desconhecido")

    def test_live_e_aceito_quando_dito_explicitamente(self):
        p = new_pick(team_a="A", team_b="B", league="L", metric_key="ml",
                     side_key="A", line=None, odd_casa=2.0, stake_units=1.0,
                     momento="live")
        self.assertEqual(p.momento, "live")


class TestOArquivoAntigoContinuaAbrindo(unittest.TestCase):
    """4.788 apostas gravadas. Um campo novo nao pode apagar nenhuma."""

    def test_le_um_arquivo_sem_os_campos_novos(self):
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "picks.json"
            alvo.write_text(json.dumps({"picks": [{
                "id": "baebcbe97f", "ts": "2026-05-02T00:00:00",
                "team_a": "DK", "team_b": "GENG", "league": "LCK",
                "metric_key": "ml", "side_key": "A", "line": None,
                "odd_casa": 3.42, "stake_units": 0.5, "p_hist": None,
                "edge_hist_pct": None, "status": "lost", "profit_units": -0.5,
                "resolved_at": None, "resolved_gameid": None, "source": "xlsx"}]},
                ensure_ascii=False), encoding="utf-8")
            picks = load_picks(str(alvo))
            self.assertEqual(len(picks), 1)
            self.assertEqual(picks[0].momento, "desconhecido")
            self.assertEqual(picks[0].profit_units, -0.5)

    def test_ida_e_volta_preserva_os_campos_novos(self):
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "picks.json"
            p = new_pick(team_a="A", team_b="B", league="L", metric_key="ml",
                         side_key="A", line=None, odd_casa=2.0, stake_units=1.0,
                         p_modelo=0.61, momento="pre")
            save_picks(str(alvo), [p])
            lido = load_picks(str(alvo))[0]
            self.assertAlmostEqual(lido.p_modelo, 0.61)
            self.assertEqual(lido.momento, "pre")

    def test_o_tracker_de_verdade_continua_abrindo(self):
        raiz = Path(__file__).resolve().parents[2]
        alvo = raiz / "ops" / "picks_tracker.json"
        if not alvo.exists():
            self.skipTest("o tracker nao esta nesta maquina")
        picks = load_picks(str(alvo))
        self.assertGreater(len(picks), 1000)
        # Nenhuma das antigas pode ter ganhado um "pre" que ninguem afirmou.
        antigas = [p for p in picks if p.source == "xlsx"]
        self.assertTrue(antigas)
        self.assertTrue(all(p.momento == "desconhecido" for p in antigas))


class TestOQueONUMERODE884NAOPodeDizer(unittest.TestCase):
    """O registro do erro, para nao ser repetido.

    Enquanto houver apostas de ML com momento 'desconhecido', o ROI de ML NAO
    e leitura do modelo de vencedor. Este teste falha no dia em que alguem
    preencher esse campo sem ter de onde tirar a informacao.
    """

    def test_as_apostas_de_ml_importadas_seguem_sem_momento(self):
        raiz = Path(__file__).resolve().parents[2]
        alvo = raiz / "ops" / "picks_tracker.json"
        if not alvo.exists():
            self.skipTest("o tracker nao esta nesta maquina")
        mls = [p for p in load_picks(str(alvo))
               if p.metric_key == "ml" and p.source == "xlsx"]
        self.assertTrue(mls, "sumiram as apostas de ML importadas")
        self.assertTrue(all(p.p_modelo is None for p in mls),
                        "alguem preencheu p_modelo em aposta antiga -- de onde?")


if __name__ == "__main__":
    unittest.main()
