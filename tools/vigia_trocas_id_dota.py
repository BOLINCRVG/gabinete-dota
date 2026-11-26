# -*- coding: utf-8 -*-
"""VIGIA DE TROCA DE ID/NOME DE TIME (Dota, 26/11/2026). Ele: "algo que pegue quando o id mudar?" + a regra fixa:
historico de um time partido, nunca.

Achado ao fazer: a uniao automatica por elenco (eval_whr_dota.roster_identity_map) deixava times PARTIDOS com os
MESMOS 5 jogadores -- Inner Circle x Insanity / "#10019843", Resp Gaming / "#10252235", RE ARISE -> 4ikibamboni,
CDUB, La Passion, SAND KING GOMEZ -> Stormrage... O robo dela so guarda os ultimos jogos de cada id.

Aqui, a partir do CSV + picks_dota.jsonl (elenco de cada lado em ~100% das partidas):
  - 5/5 jogadores iguais, os dois ids NUNCA se enfrentaram e NUNCA comecaram partidas a menos de 45 min
    -> entra SOZINHO em uniao_ids_dota.json (o adaptador aplica na mesma rodada);
  - 4/5 COM O MESMO NOME (a regra da lista auditada: "mesmo nome, elenco 4+ de 5") -> tambem une;
  - 4/5 com nomes diferentes -> evidence/ids_suspeitos_dota.json: o app avisa e ele decide (com 4/5 ja houve erro: HEROIC + LGD).
Uso: python tools/vigia_trocas_id_dota.py [--seco]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from app_core import identidade_time_core as it  # noqa: E402

UNIOES = RAIZ / "uniao_ids_dota.json"
SUSPEITOS = RAIZ / "evidence" / "ids_suspeitos_dota.json"
MESMA_HORA_MIN = 45


def _base():
    fs = sorted((RAIZ / "CSV_HISTORICO").glob("*.csv"))[-1:] + sorted((RAIZ / "CSV_DIARIO").glob("*.csv"))
    t = pd.concat([pd.read_csv(c, usecols=["gameid", "date", "teamname", "teamid", "position"], low_memory=False)
                   for c in fs])
    t = t[t.position == "team"].copy()
    t["dt"] = pd.to_datetime(t["date"], errors="coerce")
    return t.dropna(subset=["dt"])


def ids_do_nome(t: pd.DataFrame) -> dict:
    """{nome do CSV: [team_ids]} (o adaptador pode ter mais de um id sob o mesmo nome)."""
    out = defaultdict(Counter)
    for r in t.itertuples(index=False):
        try:
            out[str(r.teamname)][int(r.teamid)] += 1
        except (TypeError, ValueError):
            continue
    return {k: [i for i, _ in v.most_common()] for k, v in out.items()}


def _nome_base(nome: str) -> str:
    """Pura. 'Scarlet Lotus #10313314' -> 'scarlet lotus' (o adaptador poe #id no 2o id do mesmo nome)."""
    import re
    return re.sub(r"\s*#\d+$", "", str(nome)).strip().lower()


def conflito(t: pd.DataFrame, a: str, b: str) -> str:
    """'' se os dois nomes podem ser o mesmo time; senao o motivo (se enfrentaram / jogaram ao mesmo tempo)."""
    ga = t[t.teamname == a]
    gb = t[t.teamname == b]
    if set(ga.gameid) & set(gb.gameid):
        return "se enfrentaram"
    ia = sorted(ga.dt.drop_duplicates())
    for d in gb.dt.drop_duplicates():
        i = pd.Series(ia).searchsorted(d)
        for k in (i - 1, i):
            if 0 <= k < len(ia) and abs((ia[k] - d).total_seconds()) < MESMA_HORA_MIN * 60 and ia[k] != d:
                return "jogaram ao mesmo tempo"
    return ""


def compactar(unioes: list) -> list:
    """Pura. Linhas que dividem algum id viram UMA (o teste test_a_lista_real_nao_tem_id_repetido exige id unico):
    ids somados, nomes e provas juntados. Ex.: Scarlet Lotus com 3 ids."""
    out = []
    for u in unioes:
        ids = set(int(i) for i in u["ids"])
        juntos = [o for o in out if ids & set(o["ids"])]
        for o in juntos:
            out.remove(o)
            ids |= set(o["ids"])
        base = juntos[0] if juntos else None
        out.append({"nome": base["nome"] if base else u["nome"], "ids": sorted(ids),
                    "prova": " | ".join([o["prova"] for o in juntos] + [u["prova"]])})
    return out


def main() -> int:
    seco = "--seco" in sys.argv
    t = _base()
    pares = it.suspeitos_de_troca(it.historico(RAIZ, "dota"))
    ids = ids_do_nome(t)
    dados = json.loads(UNIOES.read_text(encoding="utf-8"))
    ja = [set(u["ids"]) for u in dados.get("unioes", [])]
    novas, suspeitos = [], []
    for p in pares:
        iv, inn = ids.get(p["velho"], []), ids.get(p["novo"], [])
        if not iv or not inn:
            continue
        par = {iv[0], inn[0]}
        if len(par) < 2 or any(par <= s for s in ja):
            continue
        motivo = conflito(t, p["velho"], p["novo"])
        item = {"velho": p["velho"], "novo": p["novo"], "ids": sorted(par), "jogadores_iguais": p["comum"],
                "velho_ate": f"{p['velho_ate']:%d/%m/%Y}", "novo_desde": f"{p['novo_desde']:%d/%m/%Y}"}
        mesmo_nome = _nome_base(p["velho"]) == _nome_base(p["novo"])
        if not motivo and (p["comum"] >= 5 or (p["comum"] >= 4 and mesmo_nome)):
            novas.append({"nome": p["novo"], "ids": sorted(par),
                          "prova": f"automatico {pd.Timestamp.now():%d/%m/%Y} (vigia_trocas_id_dota): elenco {p['comum']}/5"
                                   + (" e mesmo nome" if mesmo_nome else "") + "; "
                                   f"'{p['velho']}' ate {item['velho_ate']}, '{p['novo']}' desde {item['novo_desde']}; "
                                   "nunca se enfrentaram nem jogaram a menos de 45 min"})
            ja.append(par)
        else:
            suspeitos.append({**item, "por_que_nao_uni": motivo or "so 4 de 5 jogadores iguais e nomes diferentes: confira"})
    print(f"[vigia ids] pares: {len(pares)} | unir agora (5/5): {len(novas)} | para conferir: {len(suspeitos)}")
    for u in novas:
        print("  UNE:", u["ids"], u["prova"])
    for s in suspeitos:
        print("  CONFERIR:", s["velho"], "->", s["novo"], s["jogadores_iguais"], "/5 |", s["por_que_nao_uni"])
    if not seco:
        if novas:
            dados["unioes"] = compactar(dados.get("unioes", []) + novas)
            UNIOES.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
        SUSPEITOS.parent.mkdir(exist_ok=True)
        SUSPEITOS.write_text(json.dumps({"gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
                                         "suspeitos": suspeitos}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
