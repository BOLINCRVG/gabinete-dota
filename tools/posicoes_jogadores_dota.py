# -*- coding: utf-8 -*-
"""POSICAO DE COSTUME DE CADA JOGADOR (Dota, 26/11/2026). Ele: "esta trocado, o Ws esta de Dragon Knight e o
Mikoto de Slark". A captura da cyberscore (e o texto digitado) traz os herois NA ORDEM pos 1..5, mas o app casava
jogador com heroi so pelo historico de herois -- e trocou dois jogadores com historico parecido.

Aqui: em cada partida, os 5 de cada lado ordenados pelo ouro por minuto (o padrao do Dota profissional: pos 1 o
que mais farma, pos 5 o que menos) dao a posicao daquele jogo. Somado em todas as partidas da base, cada conta
fica com {posicao: jogos}. Fonte: OpenDota explorer (player_matches.gold_per_min), 400 partidas por chamada, so
as que ainda nao foram contadas (retomavel). Grava posicoes_jogadores_dota.json.
Uso: python tools/posicoes_jogadores_dota.py
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PICKS = RAIZ / "picks_dota.jsonl"
SAIDA = RAIZ / "posicoes_jogadores_dota.json"
LOTE = 400


def posicoes_da_partida(linhas: list) -> dict:
    """Pura. [{account_id, player_slot, gold_per_min}] de UMA partida -> {conta: posicao 1..5}. So lados completos."""
    lados = defaultdict(list)
    for x in linhas:
        if x.get("account_id") and x.get("gold_per_min") is not None and x.get("player_slot") is not None:
            lados[int(x["player_slot"]) < 128].append(x)
    fora = {}
    for js in lados.values():
        if len(js) != 5:
            continue
        for i, x in enumerate(sorted(js, key=lambda x: -float(x["gold_per_min"]))):
            fora[str(x["account_id"])] = i + 1
    return fora


def _explorer(ids: list):
    sql = ("select match_id, account_id, player_slot, gold_per_min from player_matches where match_id in ("
           + ",".join(map(str, ids)) + ")")
    url = "https://api.opendota.com/api/explorer?" + urllib.parse.urlencode({"sql": sql})
    for tent in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gabinete-dota/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["rows"]
        except Exception as e:  # noqa: BLE001
            print(f"[posicoes] explorer: {e} (tentativa {tent + 1})")
            time.sleep(20 * (tent + 1))
    return None


def main() -> int:
    d = json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() else {}
    contas = d.get("contas") or {}
    feitas = set(d.get("partidas_contadas") or [])
    ids = []
    for l in PICKS.read_text(encoding="utf-8").splitlines():
        if l.strip():
            m = int(json.loads(l)["m"])
            if m not in feitas:
                ids.append(m)
    print(f"[posicoes] {len(feitas)} partidas ja contadas, {len(ids)} novas")
    for k in range(0, len(ids), LOTE):
        lote = ids[k:k + LOTE]
        rows = _explorer(lote)
        if rows is None:
            print("[posicoes] explorer falhou; o resto fica para a proxima rodada")
            break
        por = defaultdict(list)
        for x in rows:
            por[int(x["match_id"])].append(x)
        for mid in lote:
            pos = posicoes_da_partida(por.get(mid, []))
            if len(pos) == 10:      # sem os 10 (OpenDota nao processou): tenta de novo na proxima
                for c, p in pos.items():
                    cc = contas.setdefault(c, {})
                    cc[str(p)] = cc.get(str(p), 0) + 1
                feitas.add(mid)
        time.sleep(2)
    SAIDA.write_text(json.dumps({"fonte": "OpenDota explorer, ordem de ouro por minuto no time em cada partida",
                                 "contas": contas, "partidas_contadas": sorted(feitas)},
                                separators=(",", ":")), encoding="utf-8")
    print(f"[posicoes] {len(contas)} contas, {len(feitas)} partidas -> {SAIDA.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
