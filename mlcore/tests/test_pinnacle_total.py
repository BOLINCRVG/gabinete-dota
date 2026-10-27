# -*- coding: utf-8 -*-
"""A Pinnacle inteira, LoL e Dota -- com MENOS requisicoes, nao mais.

ELE PEDIU (23/09/2026): "eu quero utilizar a pinnacle de forma total, 100%...
pegar todos os mercados, tanto para o lol, quanto para o dota".

O QUE FOI MEDIDO ANTES DE MEXER:
  - LoL: 17 de 62 corridas bateram no teto de 40 chamadas; os jogos do fim da
    fila ficavam sem kills. Nas ligas grandes nada faltava; nas pequenas, sim.
  - Dota: a leitura geral trazia o Dota em TODA corrida, e ele era jogado fora.
  - A traducao dos mercados nunca rodou no robo: 591 de 591 leituras desde
    22/09 sem ela. E quando rodou pela primeira vez, traduzia so o ULTIMO lado
    de cada mercado (236 de 464 linhas sem).

O QUE MUDOU, e o que cada teste abaixo segura:
  1. duas chamadas por jogo (`related` + `markets/related/straight`) no lugar
     de uma por mercado: o mesmo teto cobre ~20 jogos em vez de 3 ou 4;
  2. o jogo que comeca antes e lido primeiro -- o teto corta o que tem outras
     corridas pela frente;
  3. 401 pula o jogo e, no segundo, para tudo (pode ser freio disfarcado);
  4. o mesmo arquivo coleta o Dota no repositorio do Dota;
  5. toda linha sai traduzida, inclusive no jeito que o robo roda.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tools import coletar_odds as co

RAIZ = Path(__file__).resolve().parents[2]


class _Resp:
    def __init__(self, code, data=None):
        self.status_code = code
        self._d = data if data is not None else []

    def json(self):
        return self._d


class _Sessao:
    """Responde por endereco e anota cada pedido."""

    def __init__(self, por_url=None, padrao=None):
        self.por_url = por_url or {}
        self.padrao = padrao
        self.urls = []

    def get(self, url, **k):
        self.urls.append(url)
        for trecho, resp in self.por_url.items():
            if url.endswith(trecho):
                return resp
        return self.padrao or _Resp(200, [])


AGORA = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def _jogo(i, hora="2026-09-23T15:00:00Z"):
    return {"id": i, "inicio_utc": hora, "tem_mercado_no_mapa": True}


class TestDuasChamadasPorJogo(unittest.TestCase):

    def test_um_jogo_custa_DUAS_chamadas_com_todos_os_mercados(self):
        se = _Sessao({
            "matchups/1/related": _Resp(200, [{"id": 1}, {"id": 11, "units": "Kills"},
                                              {"id": 12, "units": "1st Roshan"}]),
            "matchups/1/markets/related/straight": _Resp(200, [
                {"matchupId": 11, "type": "total"}, {"matchupId": 12, "type": "moneyline"}]),
        })
        r = co.buscar_derivados(se, {}, [_jogo(1)], agora=AGORA, pausa=0)
        self.assertEqual(r["chamadas"], 2)
        self.assertEqual(len(se.urls), 2)
        self.assertEqual({m["id"] for m in r["matchups"]}, {11, 12},
                         "o proprio jogo nao pode virar filho de si mesmo")
        self.assertTrue(all(m["_paiId"] == 1 for m in r["matchups"]))
        self.assertEqual(len(r["mercados"]), 2)

    def test_NAO_ha_mais_uma_chamada_por_mercado(self):
        fonte = Path(co.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"{}matchups/{}/markets/straight"', fonte)
        self.assertIn("markets/related/straight", fonte)

    def test_o_teto_cobre_VINTE_jogos(self):
        se = _Sessao(padrao=_Resp(200, []))
        jogos = [_jogo(i) for i in range(30)]
        r = co.buscar_derivados(se, {}, jogos, agora=AGORA, pausa=0, teto=40)
        self.assertEqual(r["jogos_lidos"], 20)
        self.assertLessEqual(r["chamadas"], 40)

    def test_nunca_le_MEIO_jogo(self):
        se = _Sessao(padrao=_Resp(200, []))
        r = co.buscar_derivados(se, {}, [_jogo(i) for i in range(5)],
                                agora=AGORA, pausa=0, teto=5)
        self.assertEqual(r["chamadas"], 4)


class TestOMaisPertoDoApitoPrimeiro(unittest.TestCase):

    def test_com_o_teto_curto_fica_o_jogo_que_comeca_antes(self):
        se = _Sessao(padrao=_Resp(200, []))
        jogos = [_jogo(1, "2026-09-23T19:00:00Z"), _jogo(2, "2026-09-23T13:00:00Z")]
        co.buscar_derivados(se, {}, jogos, agora=AGORA, pausa=0, teto=2)
        self.assertIn("matchups/2/", se.urls[0])


class TestO401(unittest.TestCase):

    def test_o_primeiro_pula_so_aquele_jogo(self):
        se = _Sessao({"matchups/1/related": _Resp(401)}, padrao=_Resp(200, []))
        r = co.buscar_derivados(se, {}, [_jogo(1), _jogo(2)], agora=AGORA, pausa=0)
        self.assertFalse(r["parado"])
        self.assertEqual(r["jogos_lidos"], 1)

    def test_o_segundo_PARA_a_corrida(self):
        se = _Sessao(padrao=_Resp(401))
        r = co.buscar_derivados(se, {}, [_jogo(i) for i in range(10)],
                                agora=AGORA, pausa=0)
        self.assertTrue(r["parado"])
        self.assertEqual(len(se.urls), 2, "insistiu depois do segundo 401")

    def test_403_continua_parando_na_hora(self):
        se = _Sessao(padrao=_Resp(403))
        r = co.buscar_derivados(se, {}, [_jogo(1), _jogo(2)], agora=AGORA, pausa=0)
        self.assertTrue(r["parado"])
        self.assertEqual(len(se.urls), 1)


def _matchup(i, liga, pai=None, **k):
    base = {"id": i, "parentId": pai, "isLive": False, "bestOfX": 3,
            "startTime": "2026-09-24T07:00:00Z", "league": {"name": liga},
            "participants": [{"alignment": "home", "name": "Xtreme", "id": 100},
                             {"alignment": "away", "name": "Nemesis", "id": 200}]}
    base.update(k)
    return base


def _ml(i, period=0):
    return {"matchupId": i, "type": "moneyline", "period": period,
            "prices": [{"designation": "home", "price": 120},
                       {"designation": "away", "price": -140}]}


class TestODota(unittest.TestCase):
    """Os confrontos reais de 23/09/2026, reduzidos ao que importa."""

    def _dados(self, jogo):
        ms = [_matchup(1, "Dota 2 - PGL Wallachia"),
              _matchup(2, "League of Legends - LCS"),
              _matchup(11, "Dota 2 - PGL Wallachia", pai=1, units="1st Roshan",
                       special={"description": "(Map 1) 1st Roshan"},
                       participants=[{"id": 301, "name": "Xtreme"},
                                     {"id": 302, "name": "Nemesis"}])]
        mk = [_ml(1), _ml(2),
              {"matchupId": 11, "type": "moneyline", "period": 1,
               "prices": [{"participantId": 301, "price": 110},
                          {"participantId": 302, "price": -130}]}]
        return co.normalizar(ms, mk, jogo=jogo)

    def test_no_dota_fica_so_o_dota(self):
        d = self._dados("dota")
        self.assertEqual([j["id"] for j in d["jogos"]], [1])
        self.assertEqual(d["jogos"][0]["liga"], "PGL Wallachia")

    def test_no_lol_nada_mudou(self):
        d = self._dados("lol")
        self.assertEqual([j["id"] for j in d["jogos"]], [2])
        self.assertEqual(d["jogos"][0]["liga"], "LCS")
        d_antigo = co.normalizar(*[x for x in (
            [_matchup(1, "Dota 2 - PGL Wallachia"), _matchup(2, "League of Legends - LCS")],
            [_ml(1), _ml(2)])])
        self.assertEqual([j["id"] for j in d_antigo["jogos"]], [2])

    def test_roshan_e_traduzido_nos_DOIS_lados(self):
        d = self._dados("dota")
        rosh = [l for l in d["linhas"] if "Roshan" in l["mercado"]]
        self.assertEqual(len(rosh), 2)
        for l in rosh:
            self.assertEqual(l["metrica"], "barons")
            self.assertEqual(l["forma"], "primeiro")
            self.assertEqual(l["nome_pt"], "Primeiro Roshan - mapa 1")

    def test_o_repositorio_do_dota_se_reconhece(self):
        self.assertEqual(co.JOGO_DO_REPOSITORIO,
                         "dota" if (co.RAIZ / "dota_paths.py").exists() else "lol")


class TestTodaLinhaSaiTraduzida(unittest.TestCase):

    def test_os_dois_lados_de_cada_mercado(self):
        d = co.normalizar([_matchup(2, "League of Legends - LCS")], [_ml(2), _ml(2, 1)])
        self.assertEqual(len(d["linhas"]), 4)
        for l in d["linhas"]:
            self.assertEqual(l.get("metrica"), "ml", l)
            self.assertTrue(l.get("chave"))

    def test_mercado_sem_preco_nao_traduz_a_linha_de_OUTRO(self):
        vazio = {"matchupId": 2, "type": "total", "period": 1, "prices": []}
        d = co.normalizar([_matchup(2, "League of Legends - LCS")], [_ml(2), vazio])
        for l in d["linhas"]:
            self.assertEqual(l["mercado"], "Vencedor - serie")
            self.assertEqual(l["metrica"], "ml")

    def test_do_jeito_que_o_ROBO_roda_o_programa_e_enxergado(self):
        """O robo roda `python tools/coletar_odds.py`: so a pasta tools/ fica
        visivel. `-I` reproduz isso (ignora pasta atual e PYTHONPATH)."""
        codigo = (
            "import importlib.util,sys\n"
            "s=importlib.util.spec_from_file_location('c', r'%s')\n"
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            "from app_core.mercado_pinnacle_core import entender_mercado\n"
            "print(entender_mercado('(Map 1) 1st Roshan')['metrica'])\n"
        ) % (RAIZ / "tools" / "coletar_odds.py")
        r = subprocess.run([sys.executable, "-I", "-c", codigo], cwd=str(Path(co.__file__).parent),
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertEqual(r.stdout.strip(), "barons")


class TestOsNomesDoDota(unittest.TestCase):

    def test_os_mercados_vistos_ao_vivo(self):
        from app_core.mercado_pinnacle_core import entender_mercado
        esperado = {
            "(Map 2) 1st Roshan": ("barons", "primeiro", 2),
            "(Map 1) 1st Barracks": ("inhibitors", "primeiro", 1),
            "(Map 1) 1st Blood": ("firstblood", "primeiro", 1),
            "(Map 2) Total Kills Odd/Even": ("kills", "par_impar", 2),
        }
        for nome, (met, forma, mapa) in esperado.items():
            e = entender_mercado(nome)
            self.assertEqual((e["metrica"], e["forma"], e["mapa"]), (met, forma, mapa), nome)


class TestOsDoisRepositoriosTemOMesmoColetor(unittest.TestCase):
    """Regra de arquivo compartilhado: identico no LoL e no Dota."""

    def test_copias_do_dota_sao_identicas(self):
        dota = RAIZ.parent / "GabineteDOTA"
        if not dota.exists():
            self.skipTest("o repositorio do Dota nao esta ao lado")
        for rel in ("tools/coletar_odds.py", "app_core/mercado_pinnacle_core.py",
                    "app_core/oportunidades_core.py", "app_core/oportunidades_tab_core.py",
                    "app_core/odds_coletadas_core.py", "app_core/preco_justo_casa_core.py",
                    "app_core/casar_time_core.py", "app_core/navigation.py",
                    "mlcore/abates_por_time.py", "tools/gerar_abates_por_time.py",
                    "mlcore/tests/test_abates_por_time.py", "app_core/tabela_casa_core.py",
                    "app_core/suas_casas_core.py", "mlcore/tests/test_suas_casas.py"):
            a = (RAIZ / rel).read_bytes().replace(b"\r\n", b"\n")
            b = (dota / rel).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(a, b, "%s divergiu entre LoL e Dota" % rel)

    def test_o_robo_do_dota_roda_o_coletor_no_modo_dota(self):
        yml = RAIZ.parent / "GabineteDOTA" / ".github" / "workflows" / "odds.yml"
        if not yml.exists():
            self.skipTest("o repositorio do Dota nao esta ao lado")
        texto = yml.read_text(encoding="utf-8")
        self.assertIn("python tools/coletar_odds.py --jogo dota", texto)
        self.assertIn("odds_coletas.jsonl", texto)


class TestAsFontesEstaoEscritas(unittest.TestCase):
    """Regra da casa: pesquisa que nao ficou escrita nao aconteceu."""

    def test_bloco_fontes(self):
        fonte = Path(co.__file__).read_text(encoding="utf-8")
        self.assertIn("FONTES", fonte)
        self.assertIn("pretrehr/Sports-betting", fonte)
        self.assertIn("pinnacleapi-documentation", fonte)

    def test_nao_mandamos_a_chave_do_site(self):
        """Mandar a X-API-Key copiada do site seria nos passar por ele."""
        import inspect
        self.assertNotIn("x-api-key", inspect.getsource(co.main).lower())
        self.assertNotIn("x-api-key", inspect.getsource(co.buscar_derivados).lower())


if __name__ == "__main__":
    unittest.main()
