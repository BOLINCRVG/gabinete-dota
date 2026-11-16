# -*- coding: utf-8 -*-
"""SUPER-LIVE: camada pos-draft, leitura dos feeds e p ao vivo (arquivo identico nos dois repos)."""
from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app_core import camada_draft_core as CD
from app_core import super_live_core as SL


def _estado():
    e = CD.EstadoDraft()
    t = datetime(2026, 1, 1)
    for _ in range(10):
        e.registrar([("a", "ahri")], 1, t)
        e.registrar([("b", "zed")], 0, t)
    return e


class _Modelo:
    def __init__(self, m, p):
        self.minute, self.p = m, p

    def predict_proba_one(self, s):
        return self.p


def _test_evento_polymarket_pelo_historico(tmp_path):
    agora = datetime(2026, 11, 16, 20, tzinfo=timezone.utc)
    (tmp_path / "polymarket_dota_historico.jsonl").write_text(
        '{"time_a": "Team Spirit", "time_b": "PARIVISION", "inicio_utc": "2026-11-16 19:00:00+00", "evento": "s1"}\n'
        '{"time_a": "X", "time_b": "Y", "inicio_utc": "2026-11-16 19:00:00+00", "evento": "s2"}\n', encoding="utf-8")
    ev = SL.evento_polymarket(tmp_path, "dota", "PARIVISION", "Team Spirit", agora=agora)
    assert ev and ev[0] == "s1"
    assert SL.evento_polymarket(tmp_path, "dota", "A", "B", agora=agora) is None



class TestSuperLive(unittest.TestCase):
    def test_chave_campeao_casa_feed_e_base(self):
        assert CD.chave_campeao("Kai'Sa") == CD.chave_campeao("Kaisa") == "kaisa"
        assert CD.chave_campeao("MonkeyKing") == CD.chave_campeao("Wukong") == "wukong"
        assert CD.chave_campeao("Renata") == CD.chave_campeao("Renata Glasc")
        assert CD.chave_campeao(49) == CD.chave_campeao("49") == CD.chave_campeao(49.0) == "49"

    def test_nome_sem_tag(self):
        assert CD.nome_sem_tag("NTH Excellents", ["NTH", "SGW"]) == "Excellents"
        assert CD.nome_sem_tag("Faker") == "Faker"
        assert CD.nome_sem_tag("T1 Faker", ["T1"]) == "Faker"

    def test_sinais_e_registro(self):
        e = _estado()
        s = e.sinais([("a", "ahri")], [("b", "zed")], datetime(2026, 1, 2))
        assert s["campeao"] > 0 and s["conforto"] > 0
        assert e.jogos_no_campeao("a", "ahri") == (10.0, 10.0)

    def test_jogador_fora_da_base_nao_vira_sinal(self):
        """REGRA ABSOLUTA: jogador que nao existe na base (time novo) nao pode virar 'falta de pratica'."""
        e = _estado()
        s, faltam = e.sinais_no_app([("a", "ahri")], [("novato", "zed")], datetime(2026, 1, 2))
        assert faltam == ["novato"]
        assert s["conforto"] == 0.0 and s["pratica"] == 0.0
        assert s["campeao"] != 0.0  # o sinal do campeao continua valendo
        s2, f2 = e.sinais_no_app([("a", "ahri")], [("b", "zed")], datetime(2026, 1, 2))
        assert f2 == [] and s2["pratica"] == 0.0 and s2["conforto"] > 0

    def test_p_pos_draft_sem_sinal_devolve_o_pre(self):
        coef = {"intercepto": 0.0, "p_pre": 1.0, "campeao": 9.0, "conforto": 3.5, "pratica": 0.28}
        assert abs(CD.p_pos_draft(0.62, {"campeao": 0, "conforto": 0, "pratica": 0}, coef) - 0.62) < 1e-9
        assert CD.p_pos_draft(0.5, {"campeao": 0.05, "conforto": 0, "pratica": 0}, coef) > 0.5

    def test_dota_ao_vivo_filtra_encerrado_e_le_estado(self):
        vivo = [
            {"league_id": 1, "team_name_radiant": "A", "team_name_dire": "B", "match_id": 9, "game_time": 600,
             "radiant_lead": 2500, "radiant_score": 7, "dire_score": 3, "deactivate_time": 0, "last_update_time": 100,
             "players": [{"team": 0, "hero_id": h, "account_id": h} for h in (1, 2, 3, 4, 5)]
             + [{"team": 1, "hero_id": h, "account_id": h} for h in (6, 7, 8, 9, 10)]},
            {"league_id": 1, "team_name_radiant": "C", "team_name_dire": "D", "deactivate_time": 50, "players": []},
            {"league_id": 0, "team_name_radiant": "pub", "team_name_dire": "pub"},
        ]
        js = SL.dota_ao_vivo(get=lambda url: vivo, agora=130)
        assert len(js) == 1
        e = js[0]["estado"]
        assert e["minuto"] == 10.0 and js[0]["atualizado_ha_s"] == 30
        assert e["azul"]["ouro"] - e["vermelho"]["ouro"] == 2500
        assert e["azul"]["mortes"] == 3 and len(e["azul"]["draft"]) == 5

    def test_lol_ao_vivo_le_lados_do_mapa_em_jogo(self):
        def get(url, h=None):
            if "getLive" in url:
                return {"data": {"schedule": {"events": [{"type": "match", "state": "inProgress", "match": {"id": "m1"}}]}}}
            return {"data": {"event": {"league": {"name": "LCK"}, "match": {
                "strategy": {"count": 5},
                "teams": [{"id": "t1", "name": "T1", "code": "T1", "result": {"gameWins": 1}},
                          {"id": "g", "name": "Gen.G", "code": "GEN", "result": {"gameWins": 0}}],
                "games": [{"id": "x1", "number": 1, "state": "completed", "teams": []},
                          {"id": "x2", "number": 2, "state": "inProgress",
                           "teams": [{"id": "g", "side": "blue"}, {"id": "t1", "side": "red"}]}]}}}}
        js = SL.lol_ao_vivo(get=get)
        assert js == [{"fonte": "lolesports", "id_jogo": "x2", "liga": "LCK", "azul": "Gen.G", "vermelho": "T1",
                       "siglas": ("GEN", "T1"), "mapa": 2, "bo": 5, "placar": (0, 1)}]

    def test_p_ao_vivo_interpola_no_logit(self):
        ms = {10: _Modelo(10, 0.6), 15: _Modelo(15, 0.8), 20: _Modelo(20, 0.8), 25: _Modelo(25, 0.9)}
        est = {"minuto": 12.5, "azul": {"ouro": 1}, "vermelho": {"ouro": 0}}
        r = SL.p_ao_vivo("lol", ms, 0.5, est)
        lg = lambda p: math.log(p / (1 - p))  # noqa: E731
        esperado = 1 / (1 + math.exp(-(0.5 * lg(0.6) + 0.5 * lg(0.8))))
        assert abs(r["p"] - esperado) < 1e-9
        r0 = SL.p_ao_vivo("lol", ms, 0.5, {**est, "minuto": 0.0})
        assert abs(r0["p"] - 0.5) < 1e-9  # no minuto 0 vale o pos-draft
        assert SL.p_ao_vivo("lol", ms, 0.5, {**est, "minuto": 40})["p"] == 0.9

    def test_vetor_zera_o_que_o_feed_nao_tem(self):
        est = {"azul": {"ouro": 10, "farm": 5, "assistencias": 3}, "vermelho": {"ouro": 0, "farm": 1, "assistencias": 1}}
        v = SL.vetor("dota", 0.5, est, 3, 1, False)
        assert v[1] == 10 and v[2] == 0 and v[3] == 0 and v[6] == 0  # dota: sem xp, farm, assistencias
        v = SL.vetor("lol", 0.5, est, 0, 2, True)
        assert v[2] == 0 and v[3] == 4 and v[6] == 2 and v[8] == 2 and v[9] == 1

    def test_evento_polymarket_pelo_historico(self):
        with tempfile.TemporaryDirectory() as d:
            _test_evento_polymarket_pelo_historico(Path(d))


if __name__ == "__main__":
    unittest.main()
