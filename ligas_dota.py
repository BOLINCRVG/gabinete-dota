# -*- coding: utf-8 -*-
"""LIGAS DO DOTA: nenhuma partida de time profissional fica de fora calada (16/11/2026).

REGRA DELE (16/11/2026): "falta de jogos ou pegar informacoes erradas nao e aceitavel; os jogos tem
que ser explicado o porque nao estarem ali."

O QUE ACONTECEU. Ele comparou o app com um site de resultados: o Team Spirit Academy vinha de tres
2-0 na European Pro League e o app mostrava "Last 5: 40%" com jogos de 17/10; o Team Allied nem
existia. Duas causas, as duas na classificacao de ligas da OpenDota (campo `tier`):
  1. ela marca como "excluded" ligas REAIS de times profissionais (European Pro League 2026-2027,
     id 20276; Asgard Championship, 20026; EPL 2025-2026, 18866; Ultras Dota Pro League 2025-26;
     1win Essence I; CCT...). Partida de liga "excluded" nao aparece no /proMatches, entao o
     coletor diario parou de traze-las;
  2. o produto (dota_adapter / dota_engine) so usa tier professional + premium: mesmo as que estavam
     no data.csv eram descartadas sem aviso.

O QUE ESTE MODULO FAZ.
  - `classificar`: mede, liga por liga, quanto dela e jogado por times do universo profissional
    (time com 40+ partidas em liga professional/premium). Liga "excluded" com metade ou mais das
    participacoes desses times e PROMOVIDA: entra no produto como professional. Liga nova (poucas
    partidas) herda a decisao da edicao anterior de mesmo nome.
  - `python ligas_dota.py importar`: traz as partidas que faltam das ligas promovidas/acompanhadas
    pelo /leagues/{id}/matchIds (que lista ate liga "excluded") -- roda todo dia no robo.
  - `python ligas_dota.py varrer`: procura ligas "excluded" NOVAS onde jogam times conhecidos
    (amostra de 2 partidas por liga) e passa a acompanha-las.
  - `fora_do_produto`: para a tela -- as partidas de um time que estao no data.csv e NAO entram no
    produto, por liga, com o motivo. Nada some calado.

Arquivo versionado: ligas_promovidas_dota.json (gerado aqui; auditavel).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

RAIZ = Path(__file__).resolve().parent
ARQUIVO = RAIZ / "ligas_promovidas_dota.json"
PRO_TIERS = ("professional", "premium")
MIN_PARTIDAS_DO_TIME = 40          # "time do universo profissional"
MIN_PARTIDAS_DA_LIGA = 20          # abaixo disso a liga e nova demais para julgar sozinha
# 16/11/2026, MEDIDO (ele pediu: "faca"): incluir TODA liga com algum time profissional (Esportsprocircuit,
# NEXUS, Lunar Trophy, CIS Esports League... +11 ligas) PIORA a previsao nos jogos das ligas que ja
# entravam: -0,0090 IC95 [-0,0127; -0,0050] (cego, 935 jogos dos ultimos 30 dias), embora ajude nos
# jogos das proprias ligas novas (+0,052). Os jogos que ele aposta sao os das ligas principais: fica o corte.
PCT_PRO_MINIMO = 50.0              # metade das participacoes ou mais
DESDE = "2025-01-01"
_ANO = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,3}\b|season|series|temporada|[^a-z]+")


def raiz_do_nome(nome: Any) -> str:
    """"European Pro League  2026-2027 Season" e "European Pro League 41" -> "europeanproleague". Pura."""
    return _ANO.sub("", str(nome or "").lower())


def classificar(d: pd.DataFrame, tiers: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
    """{league_id: {nome, tier, partidas, pct_pro, ate, promovida, motivo}} das ligas fora do
    universo profissional com partida desde 2025. `d`: linhas do data.csv (match_id, league,
    league_id, team_id, date). Pura."""
    d = d.copy()
    d["league_id"] = pd.to_numeric(d["league_id"], errors="coerce").fillna(0).astype(int)
    d["team_id"] = pd.to_numeric(d["team_id"], errors="coerce").fillna(0).astype(int)
    d["tier"] = d["league_id"].map(tiers).fillna("sem_tier")
    d["dt"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
    pro = d[d["tier"].isin(PRO_TIERS)]
    raizes_pro = {raiz_do_nome(n) for n in pro["league"].dropna().unique()} - {""}
    fora = d[(~d["tier"].isin(PRO_TIERS)) & (d["dt"] >= pd.Timestamp(DESDE, tz="UTC")) & (d["league_id"] > 0)]
    res: Dict[int, Dict[str, Any]] = {}
    for lid, g in fora.groupby("league_id"):
        nome = str(g["league"].dropna().iloc[-1]) if g["league"].notna().any() else ""
        res[int(lid)] = {"nome": nome.strip(), "tier": str(g["tier"].iloc[0]), "partidas": int(g["match_id"].nunique()),
                         "pct_pro": 0.0, "ate": str(g["dt"].max())[:10], "promovida": False, "motivo": ""}
    # 1a passada, em RODADAS: a liga se sustenta sozinha. Time que so joga em liga "excluded" (Team
    # Allied) so vira "do universo profissional" depois que a liga dele e promovida -- por isso a
    # conta repete com as promovidas somadas ao universo, ate parar de mudar (no maximo 4 rodadas).
    for rodada in range(1, 5):
        ok = {lid for lid, r in res.items() if r["promovida"]}
        universo = d[d["tier"].isin(PRO_TIERS) | d["league_id"].isin(ok)]
        cont = universo.groupby("team_id")["match_id"].nunique()
        fortes: Set[int] = set(cont[cont >= MIN_PARTIDAS_DO_TIME].index) - {0}
        mudou = False
        for lid, g in fora.groupby("league_id"):
            r = res[int(lid)]
            if r["promovida"]:
                continue
            r["pct_pro"] = round(100.0 * float(g["team_id"].isin(fortes).mean()), 1)
            if r["partidas"] >= MIN_PARTIDAS_DA_LIGA and r["pct_pro"] >= PCT_PRO_MINIMO:
                r["promovida"], mudou = True, True
                r["motivo"] = (f"{r['pct_pro']:.0f}% das participacoes sao de times do universo profissional "
                               f"({r['partidas']} partidas; rodada {rodada})")
        if not mudou:
            break
    # 2a passada: liga nova herda da edicao anterior de mesmo nome (promovida ou profissional)
    raizes_ok = raizes_pro | {raiz_do_nome(r["nome"]) for r in res.values() if r["promovida"]}
    for r in res.values():
        if r["promovida"]:
            continue
        if r["partidas"] < MIN_PARTIDAS_DA_LIGA and raiz_do_nome(r["nome"]) in raizes_ok:
            r["promovida"] = True
            r["motivo"] = f"edicao nova ({r['partidas']} partidas) de uma liga do universo profissional"
        elif r["partidas"] < MIN_PARTIDAS_DA_LIGA and r["partidas"] >= 4 and r["pct_pro"] >= PCT_PRO_MINIMO:
            r["promovida"] = True
            r["motivo"] = f"liga nova: {r['pct_pro']:.0f}% das participacoes de times profissionais"
        else:
            r["motivo"] = f"liga de nível mais baixo: {r['pct_pro']:.0f}% de times profissionais (corte: {PCT_PRO_MINIMO:.0f}%)"
            r["nivel_baixo"] = True
    return res


def carregar() -> Dict[str, Any]:
    try:
        d = json.loads(ARQUIVO.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def promovidas() -> Set[int]:
    """Os ids das ligas promovidas (do arquivo versionado). DOTA_PROMOVIDAS=0 desliga (para medir)."""
    import os
    if os.environ.get("DOTA_PROMOVIDAS") == "0":
        return set()
    return {int(k) for k, v in (carregar().get("ligas") or {}).items() if (v or {}).get("promovida")}


def aplicar_promocao(tiers: Dict[int, str]) -> Dict[int, str]:
    """O mapa de tiers com as ligas promovidas como "professional". Pura (fora a leitura)."""
    pr = promovidas()
    if not pr:
        return tiers
    out = dict(tiers)
    for lid in pr:
        out[int(lid)] = "professional"
    return out


def _tiers_crus() -> Dict[int, str]:
    t = pd.read_csv(RAIZ / "league_tiers.csv")
    return {int(i): str(x) for i, x in zip(t["league_id"], t["tier"])}


def _linhas() -> pd.DataFrame:
    return pd.read_csv(RAIZ / "data.csv", low_memory=False,
                       usecols=["match_id", "date", "league", "league_id", "team_id", "teamname"])


def gravar(extra_acompanhar: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reclassifica com o data.csv atual e grava o arquivo. Mantem `acompanhar` e `julgadas`."""
    velho = carregar()
    res = classificar(_linhas(), _tiers_crus())
    acomp = dict(velho.get("acompanhar") or {})
    acomp.update(extra_acompanhar or {})
    doc = {"_como": "ligas que a OpenDota marca fora de professional/premium; promovida = entra no produto como "
                    "professional (ligas_dota.classificar). acompanhar = ligas cujas partidas o robo traz pelo "
                    "/leagues/{id}/matchIds. julgadas = ligas ja olhadas pela varredura.",
           "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "ligas": {str(k): v for k, v in sorted(res.items())},
           "acompanhar": acomp, "julgadas": velho.get("julgadas") or {}}
    ARQUIVO.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


# ------------------------------------------------------------------ para a tela: nada some calado
def ids_do_time(nome: str, d: pd.DataFrame) -> Set[int]:
    """Os team_id que o data.csv ja viu com este nome (sem diferenciar caixa). Pura."""
    alvo = str(nome or "").strip().casefold()
    if not alvo:
        return set()
    x = d[d["teamname"].astype(str).str.strip().str.casefold() == alvo]
    return set(pd.to_numeric(x["team_id"], errors="coerce").dropna().astype(int)) - {0}


def texto_fora_do_produto(nome: str, itens: List[Dict[str, Any]]) -> str:
    """A frase da tela. Pura."""
    if not itens:
        return ""
    partes = [f"{i['partidas']} partida(s) na \"{i['liga']}\" (até {i['ate'][8:10]}/{i['ate'][5:7]}): {i['motivo']}"
              for i in itens]
    txt = f"{nome} — jogos recentes FORA da análise: " + "; ".join(partes) + "."
    if any("nível mais baixo" in str(i.get("motivo")) for i in itens):
        txt += (" Esses jogos estão na base, mas não entram no modelo nem nas estatísticas: medido em 16/11/2026, "
                "pôr ligas de nível mais baixo na análise piora a previsão dos jogos das ligas principais.")
    return txt


def fora_do_produto(team_ids: Iterable[int], dias: int = 60, agora: Optional[pd.Timestamp] = None,
                    d: Optional[pd.DataFrame] = None, tiers: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
    """As partidas recentes destes team_id que estao no data.csv e NAO entram no produto, por liga:
    [{liga, partidas, ate, motivo}]. Pura quando `d` e `tiers` sao dados."""
    ids = {int(x) for x in team_ids if x}
    if not ids:
        return []
    d = _linhas() if d is None else d
    tiers = aplicar_promocao(_tiers_crus()) if tiers is None else tiers
    doc = carregar().get("ligas") or {}
    x = d[pd.to_numeric(d["team_id"], errors="coerce").fillna(0).astype(int).isin(ids)].copy()
    x["league_id"] = pd.to_numeric(x["league_id"], errors="coerce").fillna(0).astype(int)
    x["dt"] = pd.to_datetime(x["date"], errors="coerce", utc=True)
    agora = agora or pd.Timestamp.now(tz="UTC")
    x = x[(x["dt"] >= agora - pd.Timedelta(days=dias)) & ~x["league_id"].map(tiers).fillna("sem_tier").isin(PRO_TIERS)]
    fora = []
    for lid, g in x.groupby("league_id"):
        info = doc.get(str(int(lid))) or {}
        fora.append({"liga": str(g["league"].dropna().iloc[-1]).strip() if g["league"].notna().any() else f"liga {lid}",
                     "partidas": int(g["match_id"].nunique()), "ate": str(g["dt"].max())[:10],
                     "motivo": info.get("motivo") or f"a OpenDota classifica como {tiers.get(int(lid), 'sem tier')}"})
    return sorted(fora, key=lambda r: r["ate"], reverse=True)


# ------------------------------------------------------------------ robo: trazer o que falta
def _ids_da_liga(league_id: int) -> List[int]:
    from opendota_client import _request
    r = _request(f"leagues/{int(league_id)}/matchIds")
    return [int(x) for x in r] if isinstance(r, list) else []


def importar(max_partidas: int = 500, max_segundos: int = 1500, log=print, espera=time.sleep) -> Dict[str, Any]:
    """Traz as partidas que faltam das ligas promovidas + acompanhadas, as mais novas primeiro."""
    from opendota_sync import import_match_ids, load_existing_csv
    t0 = time.monotonic()
    doc = carregar()
    alvos = {int(k) for k, v in (doc.get("ligas") or {}).items() if (v or {}).get("promovida")}
    alvos |= {int(k) for k in (doc.get("acompanhar") or {})}
    existentes = load_existing_csv()
    vistos = set(pd.to_numeric(existentes.get("match_id"), errors="coerce").dropna().astype(int)) if not existentes.empty else set()
    faltam: List[int] = []
    falhou: List[int] = []
    for lid in sorted(alvos, reverse=True):
        try:
            faltam += [m for m in _ids_da_liga(lid) if m not in vistos]
        except Exception as e:  # rede/cota: segue para a proxima liga
            falhou.append(lid)
            log(f"[ligas] liga {lid}: nao deu para listar ({type(e).__name__})")
    faltam = sorted(set(faltam), reverse=True)
    total, trazidas = len(faltam), 0
    # lotes pequenos: cada lote grava o CSV, entao o que ja veio nao se perde. A OpenDota gratis
    # barra em ~60 pedidos por minuto (429): espera um minuto e repete o lote, ate 3 vezes.
    parar = False
    for i in range(0, min(total, max_partidas), 20):
        if parar or time.monotonic() - t0 > max_segundos:
            break
        lote = faltam[i:i + 20]
        for tentativa in range(3):
            try:
                rep = import_match_ids(lote)
                trazidas += int(rep.get("imported_matches") or 0)
                break
            except Exception as e:
                if tentativa == 2 or time.monotonic() - t0 > max_segundos:
                    log(f"[ligas] lote falhou ({type(e).__name__}); para aqui e continua na proxima corrida")
                    parar = True
                    break
                espera(65)
    rel = {"ligas": len(alvos), "faltavam": total, "trazidas": trazidas, "restam": max(0, total - trazidas),
           "ligas_sem_resposta": falhou}
    log(f"[ligas] {rel}")
    return rel


def varrer(max_ligas: int = 400, max_segundos: int = 900, log=print) -> Dict[str, Any]:
    """Procura ligas fora do universo profissional, NOVAS, onde jogam times conhecidos: lista as
    partidas da liga e abre duas; se ha time do universo profissional, passa a acompanhar."""
    from opendota_client import _request, get_match
    t0 = time.monotonic()
    doc = carregar()
    julgadas: Dict[str, Any] = dict(doc.get("julgadas") or {})
    acomp: Dict[str, Any] = dict(doc.get("acompanhar") or {})
    d = _linhas()
    tiers = _tiers_crus()
    tid = pd.to_numeric(d["team_id"], errors="coerce").fillna(0).astype(int)
    lig = pd.to_numeric(d["league_id"], errors="coerce").fillna(0).astype(int)
    pro = d[lig.map(tiers).fillna("sem_tier").isin(PRO_TIERS)]
    cont = pro.groupby(pd.to_numeric(pro["team_id"], errors="coerce").fillna(0).astype(int))["match_id"].nunique()
    fortes = set(cont[cont >= MIN_PARTIDAS_DO_TIME].index) - {0}
    raizes_ok = ({raiz_do_nome(n) for n in pro["league"].dropna().unique()}
                 | {raiz_do_nome((v or {}).get("nome")) for v in (doc.get("ligas") or {}).values() if (v or {}).get("promovida")}) - {""}
    ligas = [l for l in (_request("/leagues") or []) if str(l.get("tier")) not in PRO_TIERS]
    ligas = sorted(ligas, key=lambda l: -int(l.get("leagueid") or 0))[:max_ligas]
    novas = 0
    for l in ligas:
        if time.monotonic() - t0 > max_segundos:
            break
        lid = int(l.get("leagueid") or 0)
        ja = julgadas.get(str(lid))
        if lid <= 0 or str(lid) in acomp or (ja and ja.get("ids", 0) >= 6):
            continue
        if raiz_do_nome(l.get("name")) in raizes_ok:          # edicao nova de liga do universo profissional
            acomp[str(lid)] = {"nome": str(l.get("name") or "").strip(), "desde": time.strftime("%Y-%m-%d"),
                               "motivo": "edicao nova de uma liga do universo profissional (pelo nome)"}
            novas += 1
            log(f"[ligas] passa a acompanhar {lid} {l.get('name')!r} (pelo nome)")
            continue
        try:
            ids = _ids_da_liga(lid)
        except Exception:
            continue
        achou = False
        if len(ids) >= 2:
            for mid in (max(ids), min(ids)):
                try:
                    p = get_match(mid)
                except Exception:
                    continue
                times = {int((p.get("radiant_team") or {}).get("team_id") or p.get("radiant_team_id") or 0),
                         int((p.get("dire_team") or {}).get("team_id") or p.get("dire_team_id") or 0)}
                if times & fortes:
                    achou = True
                    break
        julgadas[str(lid)] = {"nome": str(l.get("name") or "").strip(), "ids": len(ids), "time_pro": achou}
        if achou:
            acomp[str(lid)] = {"nome": str(l.get("name") or "").strip(), "desde": time.strftime("%Y-%m-%d"),
                               "motivo": "time do universo profissional jogando nela (varredura)"}
            novas += 1
            log(f"[ligas] passa a acompanhar {lid} {l.get('name')!r} ({len(ids)} partidas)")
    doc["julgadas"], doc["acompanhar"] = julgadas, acomp
    ARQUIVO.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[ligas] varredura: {novas} liga(s) nova(s) acompanhada(s); {len(acomp)} no total")
    return {"novas": novas, "acompanhadas": len(acomp)}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass
    modo = sys.argv[1] if len(sys.argv) > 1 else "classificar"
    if modo == "varrer":
        varrer()
    if modo in ("importar", "varrer"):
        importar()
    doc = gravar()
    pr = [(k, v) for k, v in doc["ligas"].items() if v["promovida"]]
    print(f"[ligas] {len(doc['ligas'])} ligas fora do universo profissional com partida desde {DESDE}; "
          f"{len(pr)} promovidas; {len(doc['acompanhar'])} acompanhadas")
    for k, v in sorted(pr, key=lambda kv: -kv[1]["partidas"]):
        print(f"   PROMOVIDA {k} {v['nome']!r}: {v['motivo']}; ate {v['ate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
