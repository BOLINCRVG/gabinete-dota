# -*- coding: utf-8 -*-
"""SONDA DA API OFICIAL DA VALVE (24/11/2026) -- so LE e registra; nao mexe em dado nenhum.

Por que: de 20 a 24/11 a OpenDota ficou instavel (429, 522, 500) e a base do Dota parou. A API da
Valve (Steam Web API, IDOTA2Match_570) e de onde a propria OpenDota tira os dados; ele criou a chave
(segredo STEAM_API_KEY no GitHub). Antes de escrever a reserva que grava jogos, esta sonda mostra o
que a API devolve de verdade: responde? GetMatchHistory por liga traz as partidas novas? GetMatchDetails
traz times, herois, jogadores e objetivos? Grava evidence/sonda_steam_dota.json (estrutura e contagens;
nunca a chave, nunca apelido de conta Steam).
Uso: STEAM_API_KEY=... python tools/sonda_steam_dota.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BASE = "https://api.steampowered.com/IDOTA2Match_570"
SAIDA = RAIZ / "evidence" / "sonda_steam_dota.json"


def _get(metodo: str, chave: str, **p) -> tuple:
    q = urllib.parse.urlencode({"key": chave, **p})
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{BASE}/{metodo}/v1/?{q}",
                                                           headers={"User-Agent": "GabineteDota/0.1"}), timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}", None


def _estrutura(o, prof=0):
    """So os nomes de campo e tipos (nunca valores) -- para eu escrever o leitor certo."""
    if prof > 3:
        return type(o).__name__
    if isinstance(o, dict):
        return {k: _estrutura(v, prof + 1) for k, v in list(o.items())[:60] if k not in ("personaname", "name")}
    if isinstance(o, list):
        return [_estrutura(o[0], prof + 1)] if o else []
    return type(o).__name__


def main() -> int:
    chave = os.environ.get("STEAM_API_KEY", "").strip()
    doc = {"quando": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tem_chave": bool(chave)}
    if not chave:
        doc["motivo"] = "sem STEAM_API_KEY no ambiente"
    else:
        d = pd.read_csv(RAIZ / "data.csv", usecols=["match_id", "league_id", "date"], low_memory=False)
        d = d.drop_duplicates("match_id").sort_values("date")
        ligas = [int(x) for x in d[d["date"] >= d["date"].max()[:10]]["league_id"].dropna().unique()[:3]]
        ultimo = int(d["match_id"].iloc[-1])
        doc["ultimo_jogo_na_base"] = {"match_id": ultimo, "data": str(d["date"].iloc[-1])}
        hist = {}
        for lg in ligas:
            st, js = _get("GetMatchHistory", chave, league_id=lg, matches_requested=25)
            ms = ((js or {}).get("result") or {}).get("matches") or []
            novos = [m for m in ms if int(m.get("match_id", 0)) > ultimo]
            hist[str(lg)] = {"status": st, "partidas": len(ms), "mais_novas_que_a_base": len(novos),
                             "resultado": ((js or {}).get("result") or {}).get("statusDetail") or
                             ((js or {}).get("result") or {}).get("status")}
            if ms and "estrutura_history" not in doc:
                doc["estrutura_history"] = _estrutura(ms[0])
        doc["GetMatchHistory_por_liga"] = hist
        st, js = _get("GetMatchDetails", chave, match_id=ultimo)
        res = (js or {}).get("result") or {}
        doc["GetMatchDetails"] = {"status": st, "erro": res.get("error"),
                                  "tem": {k: (k in res) for k in ("radiant_win", "duration", "radiant_team_id",
                                                                   "dire_team_id", "radiant_name", "dire_name",
                                                                   "leagueid", "picks_bans", "players", "tower_status_radiant",
                                                                   "barracks_status_radiant", "radiant_score",
                                                                   "first_blood_time", "start_time")},
                                  "estrutura": _estrutura(res)}
        st, js = _get("GetLiveLeagueGames", chave)
        jg = ((js or {}).get("result") or {}).get("games") or []
        com_sb = [g for g in jg if (g.get("scoreboard") or {}).get("radiant")]
        doc["GetLiveLeagueGames"] = {"status": st, "jogos": len(jg), "com_scoreboard": len(com_sb),
                                     "estrutura_jogo": _estrutura(com_sb[0] if com_sb else (jg[0] if jg else {}))}
        # o leitor do app (app_core/valve_ao_vivo_core.py) confere aqui com dado real
        try:
            sys.path.insert(0, str(RAIZ))
            from app_core import valve_ao_vivo_core as VA
            lidos = VA.ler_jogos(js or {}, {})
            doc["GetLiveLeagueGames"]["lidos_pelo_app"] = len(lidos)
            doc["GetLiveLeagueGames"]["exemplo_lido"] = ({k: lidos[0][k] for k in ("radiant", "dire", "picks", "bans",
                                                                                   "duracao")} if lidos else None)
        except Exception as e:  # noqa: BLE001
            doc["GetLiveLeagueGames"]["erro_leitor"] = f"{type(e).__name__}: {e}"
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if not k.startswith("estrutura")}, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
