# -*- coding: utf-8 -*-
"""SUPER-LIVE: gera camada_draft_<esporte>.json (tabelas do draft + coeficientes medidos).

  python tools/camada_draft.py                 # recalcula as tabelas, mantem os coeficientes
  python tools/camada_draft.py --ajustar P.csv # reajusta os coeficientes (P.csv: gameid,side,p_pre
                                               # = p de producao do gate ao vivo, ver dump no evidence)
  python tools/camada_draft.py --picks         # (Dota) baixa da OpenDota os herois que faltam

Tabelas: so jogos ja jogados, em ordem de data (app_core/camada_draft_core.EstadoDraft).
LoL: linhas de jogador do Oracle's Elixir (campeao, jogador). Dota: picks_dota.jsonl (heroi e
conta de cada jogador, OpenDota explorer) + linhas de time do CSV. Arquivo identico nos dois repos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from app_core.camada_draft_core import (  # noqa: E402
    EstadoDraft, arquivo, chave_campeao, chave_jogador,
)

E_DOTA = (RAIZ / "run_phase6_live_dota.py").exists()
ESPORTE = "dota" if E_DOTA else "lol"
PICKS = RAIZ / "picks_dota.jsonl"
AQUECIMENTO_DIAS = 180

# Coeficientes medidos em 16/11/2026 (treino ate 31/12/2025; teste 2026; ver o modulo core).
# Recalcular so com --ajustar e medicao nova.
COEF_PADRAO = {
    "lol": {"medido_em": "2026-11-16", "ganho_logloss": 0.0141, "ic95": [0.0112, 0.0173], "n_teste": 9864},
    "dota": {"medido_em": "2026-11-16", "ganho_logloss": 0.0118, "ic95": [0.0090, 0.0145], "n_teste": 8947},
}


def _csvs():
    if E_DOTA:
        return [p for p in (RAIZ / "CSV_HISTORICO/2025_Dota_match_data_oe_like.csv",
                            RAIZ / "CSV_DIARIO/2026_Dota_match_data_oe_like.csv") if p.exists()]
    return sorted((RAIZ / "CSV_HISTORICO").glob("*LoL*OraclesElixir*.csv")) + \
        sorted((RAIZ / "CSV_DIARIO").glob("*LoL*OraclesElixir*.csv"))


def _ler_picks():
    pk = {}
    if PICKS.exists():
        for linha in PICKS.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(linha)
            except ValueError:
                continue
            if "m" in j:  # formato compacto: [heroi, slot, conta]
                pk[int(j["m"])] = [{"hero_id": h, "player_slot": sl, "account_id": c} for h, sl, c in j["p"]]
            else:
                pk[int(j["match_id"])] = j.get("p") or []
    return pk


LIMITE_POR_JOGO = 150  # chamadas /matches/{id} por rodada (OpenDota gratis: 60/min)


def _gravar(f, mid, lista) -> None:
    f.write(json.dumps({"m": int(mid), "p": [[x["hero_id"], x["player_slot"], x["account_id"]] for x in lista]},
                       separators=(",", ":")) + "\n")


def _do_sidecar(ids, ja) -> int:
    """raw_matches.jsonl: o sync grava ali cada /matches/{id} completo da rodada (arquivo fora do
    git). 16/11/2026: os jogos das ligas promovidas entravam por ali e os herois se perdiam."""
    raw = RAIZ / "raw_matches.jsonl"
    if not raw.exists():
        return 0
    n = 0
    with PICKS.open("a", encoding="utf-8") as f:
        for linha in raw.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(linha)
            except ValueError:
                continue
            mid = int(j.get("match_id") or 0)
            ps = [p for p in (j.get("players") or []) if p.get("hero_id")]
            if mid in ids and mid not in ja and len(ps) == 10:
                _gravar(f, mid, [{"hero_id": p["hero_id"], "player_slot": 0 if p.get("isRadiant") else 128,
                                  "account_id": p.get("account_id")} for p in ps])
                ja.add(mid)
                n += 1
    return n


def _jogo_a_jogo(falta) -> int:
    """/api/matches/{id} para o que nem o sidecar nem o explorer trouxeram (ate LIMITE_POR_JOGO)."""
    n = 0
    with PICKS.open("a", encoding="utf-8") as f:
        for mid in falta[:LIMITE_POR_JOGO]:
            try:
                req = urllib.request.Request(f"https://api.opendota.com/api/matches/{mid}",
                                             headers={"User-Agent": "gabinete-dota/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    j = json.loads(r.read())
            except Exception as e:  # noqa: BLE001
                print(f"[camada_draft] /matches/{mid}: {e}")
                time.sleep(5)
                continue
            ps = [p for p in (j.get("players") or []) if p.get("hero_id")]
            if len(ps) == 10:
                _gravar(f, mid, [{"hero_id": p["hero_id"], "player_slot": p.get("player_slot", 0),
                                  "account_id": p.get("account_id")} for p in ps])
                n += 1
            time.sleep(1.1)
    return n


def baixar_picks(ids, ja) -> int:
    """Sidecar da rodada -> OpenDota explorer (400 por chamada) -> jogo a jogo. Retomavel."""
    ja = set(ja)
    n = _do_sidecar(set(ids), ja)
    falta = [i for i in ids if i not in ja]
    print(f"[camada_draft] picks: {len(ids)} jogos, {n} do sidecar, faltam {len(falta)}")
    with PICKS.open("a", encoding="utf-8") as f:
        for k in range(0, len(falta), 400):
            lote = falta[k:k + 400]
            sql = ("select match_id, hero_id, player_slot, account_id from player_matches where match_id in ("
                   + ",".join(map(str, lote)) + ")")
            url = "https://api.opendota.com/api/explorer?" + urllib.parse.urlencode({"sql": sql})
            rows = None
            for tent in range(4):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "gabinete-dota/1.0"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        rows = json.loads(r.read())["rows"]
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[camada_draft] explorer: {e} (tentativa {tent + 1})")
                    time.sleep(20 * (tent + 1))
            if rows is None:
                print("[camada_draft] explorer falhou; o resto fica para a proxima rodada")
                break
            por = {}
            for x in rows:
                por.setdefault(int(x["match_id"]), []).append(
                    {"hero_id": x.get("hero_id"), "player_slot": x.get("player_slot"), "account_id": x.get("account_id")})
            for mid in lote:
                # jogo sem linha (OpenDota ainda nao processou) NAO e gravado: tenta de novo amanha
                if len(por.get(mid, [])) == 10:
                    _gravar(f, mid, por[mid])
                    ja.add(mid)
                    n += 1
            f.flush()
            time.sleep(2)
    resto = [i for i in ids if i not in ja]
    if resto:
        m = _jogo_a_jogo(sorted(resto, reverse=True))  # os mais recentes primeiro
        print(f"[camada_draft] jogo a jogo: {m} de {min(len(resto), LIMITE_POR_JOGO)} (sobram {len(resto) - m})")
        n += m
    return n


def jogos():
    """Lista (data, gameid, azul[(jog,camp)], vermelho[(jog,camp)], azul_venceu) em ordem de data."""
    cols = ["gameid", "date", "side", "position", "playername", "champion", "result", "participantid"]
    d = pd.concat([pd.read_csv(c, usecols=lambda x: x in cols, low_memory=False) for c in _csvs()],
                  ignore_index=True)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    tm = d[d["position"] == "team"].drop_duplicates(["gameid", "side"])
    res = {(r.gameid, str(r.side).title()): int(r.result) for r in tm.itertuples(index=False)}
    data = tm.groupby("gameid")["date"].min()
    lados = {}
    if E_DOTA:
        pk = _ler_picks()
        for gid in data.index:
            mid = str(gid).split("_")[-1]
            ps = pk.get(int(mid)) if mid.isdigit() else None
            if not ps:
                continue
            az = [(chave_jogador(x.get("account_id")), chave_campeao(x["hero_id"])) for x in ps if int(x["player_slot"]) < 128]
            vm = [(chave_jogador(x.get("account_id")), chave_campeao(x["hero_id"])) for x in ps if int(x["player_slot"]) >= 128]
            lados[gid] = (az, vm)
    else:
        pl = d[(d["position"] != "team")].dropna(subset=["champion"])
        acc = {}
        for r in pl.itertuples(index=False):
            acc.setdefault(r.gameid, {"Blue": [], "Red": []}).setdefault(str(r.side).title(), []).append(
                (chave_jogador(r.playername), chave_campeao(r.champion)))
        lados = {g: (v.get("Blue", []), v.get("Red", [])) for g, v in acc.items()}
    out = []
    for gid, t in data.sort_values().items():
        lr = lados.get(gid)
        if not lr or len(lr[0]) != 5 or len(lr[1]) != 5 or (gid, "Blue") not in res:
            continue
        out.append((t.to_pydatetime(), str(gid), lr[0], lr[1], res[(gid, "Blue")]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ajustar", default=None)
    ap.add_argument("--picks", action="store_true")
    a = ap.parse_args()

    if E_DOTA and a.picks:
        ids = set()
        for c in _csvs():
            g = pd.read_csv(c, usecols=["gameid"], low_memory=False)["gameid"].astype(str)
            ids |= {int(x.split("_")[-1]) for x in g if x.split("_")[-1].isdigit()}
        print(f"[camada_draft] picks novos gravados: {baixar_picks(sorted(ids), set(_ler_picks()))}")

    lista = jogos()
    print(f"[camada_draft] {ESPORTE}: {len(lista)} jogos com draft completo")
    if not lista:
        print("[camada_draft] sem jogos; nada gravado")
        return 1
    est = EstadoDraft()
    linhas = []
    inicio = lista[0][0]
    for t, gid, az, vm, y in lista:
        if a.ajustar and (t - inicio).days >= AQUECIMENTO_DIAS:
            s = est.sinais(az, vm, t)
            linhas.append((gid, t, s["campeao"], s["conforto"], s["pratica"], y))
        est.registrar(az, y, t)
        est.registrar(vm, 1 - y, t)

    destino = arquivo(RAIZ, ESPORTE)
    velho = {}
    if destino.exists():
        try:
            velho = json.loads(destino.read_text(encoding="utf-8"))
        except ValueError:
            velho = {}
    coef = velho.get("coef")
    medicao = velho.get("medicao") or COEF_PADRAO[ESPORTE]
    if a.ajustar:
        # Ajuste com o p do modelo como OFFSET (peso fixo 1): a camada so SOMA o draft, nao
        # recalibra o modelo pre-jogo (congelado). Medido em 16/11/2026: LoL +0.0141, Dota +0.0118.
        import numpy as np
        pp = pd.read_csv(a.ajustar)
        pp = pp[pp["side"] == "Blue"].set_index("gameid")["p_pre"]
        X = pd.DataFrame(linhas, columns=["gid", "t", "campeao", "conforto", "pratica", "y"])
        X["p_pre"] = X["gid"].map(pp)
        X = X.dropna(subset=["p_pre"])
        q = X["p_pre"].clip(1e-6, 1 - 1e-6).to_numpy()
        off = np.log(q / (1 - q))
        cs = ["campeao", "conforto", "pratica"]
        A = np.column_stack([np.ones(len(X))] + [X[c].to_numpy() for c in cs])
        y = X["y"].to_numpy()
        w = np.zeros(A.shape[1])
        for _ in range(60):
            pr = 1 / (1 + np.exp(-(off + A @ w)))
            H = (A * (pr * (1 - pr))[:, None]).T @ A + 1e-6 * np.eye(len(w))
            w += np.linalg.solve(H, A.T @ (y - pr))
        coef = {"intercepto": float(w[0]), "p_pre": 1.0, **{c: float(v) for c, v in zip(cs, w[1:])},
                "ajustado_em": datetime.now(timezone.utc).date().isoformat(), "n_jogos": int(len(X))}
        print(f"[camada_draft] coeficientes: {coef}")
    if not coef:
        print("[camada_draft] sem coeficientes (rode com --ajustar); nada gravado")
        return 1
    doc = {"gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "ultimo_jogo": lista[-1][0].isoformat(timespec="seconds"), "n_jogos": len(lista),
           "coef": coef, "medicao": medicao, "estado": est.para_json()}
    destino.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[camada_draft] gravado {destino.name}: {len(est.campeao)} campeoes, {len(est.jogador)} pares "
          f"jogador-campeao, {destino.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
