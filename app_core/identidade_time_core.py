# -*- coding: utf-8 -*-
"""IDENTIDADE DO TIME PELOS JOGADORES (26/11/2026). Ele: "nao teria como garantir que o time em questao e aquele
time ali, que do nada so nao virou outro em outro lugar do mundo?"

Nome e ID de time mudam (DO KONCA -> BetBoom) e ha nomes parecidos pelo mundo; o que nao muda e o jogador. Aqui,
para cada time da base, o historico de elencos (Dota: contas de picks_dota.jsonl, pelo lado; LoL: playername do CSV) e:
  - mudanca(): quem do ultimo elenco NAO jogava pelo time entre 90 e 14 dias antes do ultimo jogo (= jogador novo
    ou stand-in), com a data em que apareceu;
  - confere(): quantos dos jogadores de um jogo (Valve ao vivo) sao do ultimo elenco: 4-5 confirmado, 3 mexido,
    0-2 provavel outro time.
Nunca muda preco: so diz quando desconfiar. Arquivo IDENTICO no LoL e no Dota.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import pandas as pd

JANELA = (90, 14)    # dias antes do ultimo jogo: o "elenco de costume" sai desta faixa
_CACHE: Dict[str, Any] = {}


def _csvs(raiz: Path, esporte: str) -> List[Path]:
    fs = sorted((raiz / "CSV_DIARIO").glob("*.csv"))
    hist = sorted((raiz / "CSV_HISTORICO").glob("*.csv"))
    return hist[-1:] + fs


def historico(raiz: Path, esporte: str) -> Dict[str, List[Tuple[pd.Timestamp, FrozenSet[str]]]]:
    """{time: [(data, elenco)]} em ordem de data, lido uma vez por versao dos arquivos."""
    raiz = Path(raiz)
    arqs = _csvs(raiz, esporte) + ([raiz / "picks_dota.jsonl"] if esporte == "dota" else [])
    versao = tuple((str(a), a.stat().st_mtime) for a in arqs if a.exists())
    if _CACHE.get((esporte, "v")) == versao:
        return _CACHE[(esporte, "h")]
    out: Dict[str, List[Tuple[pd.Timestamp, FrozenSet[str]]]] = defaultdict(list)
    if esporte == "dota":
        # Elenco pelo LADO, de picks_dota.jsonl ({"m", "p": [[heroi, slot, conta]]}, slot < 128 = Radiant = "Blue" no
        # CSV): cobre ~100% das partidas. Pelo team_id (escalacoes_dota.json) falhava quando o time ganhou ID novo
        # e a base usa o unificado (PVISION parava em 23/08).
        lados = {}
        try:
            for l in (raiz / "picks_dota.jsonl").read_text(encoding="utf-8").splitlines():
                if l.strip():
                    x = json.loads(l)
                    r = frozenset(str(c) for _h, sl, c in x["p"] if c and int(sl) < 128)
                    d = frozenset(str(c) for _h, sl, c in x["p"] if c and int(sl) >= 128)
                    lados[str(x["m"])] = {"Blue": r, "Red": d}
        except (OSError, ValueError, KeyError):
            pass
        for c in _csvs(raiz, esporte):
            t = pd.read_csv(c, usecols=["gameid", "date", "teamname", "side", "position"], low_memory=False)
            t = t[t.position == "team"]
            for r in t.itertuples(index=False):
                el = (lados.get(str(r.gameid).replace("DOTA_", "")) or {}).get(str(r.side))
                if el and len(el) >= 5:
                    out[str(r.teamname)].append((pd.Timestamp(r.date), el))
    else:
        for c in _csvs(raiz, esporte):
            t = pd.read_csv(c, usecols=["gameid", "date", "teamname", "playername", "position"], low_memory=False)
            t = t[(t.position != "team") & t.playername.notna()]
            for (g, tm), s in t.groupby(["gameid", "teamname"]):
                if len(s) >= 5:
                    out[str(tm)].append((pd.Timestamp(s.date.iloc[0]), frozenset(map(str, s.playername))))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    _CACHE[(esporte, "v")], _CACHE[(esporte, "h")] = versao, dict(out)
    return _CACHE[(esporte, "h")]


def mudanca(jogos: Sequence[Tuple[pd.Timestamp, FrozenSet[str]]]) -> Optional[Dict[str, Any]]:
    """Pura. {ultimo, elenco, novos: {jogador: data em que apareceu}, base_ok} ou None sem jogo."""
    if not jogos:
        return None
    dt_u, el = jogos[-1]
    ini, fim = dt_u - timedelta(days=JANELA[0]), dt_u - timedelta(days=JANELA[1])
    antes = set().union(*[e for d, e in jogos if ini <= d <= fim]) if any(ini <= d <= fim for d, _ in jogos) else None
    if antes is None:
        return {"ultimo": dt_u, "elenco": el, "novos": {}, "base_ok": False}
    novos = {}
    for j in el - antes:
        novos[j] = min(d for d, e in jogos if j in e and d > fim)
    return {"ultimo": dt_u, "elenco": el, "novos": novos, "base_ok": True}


def confere(contas: Sequence[str], elenco: FrozenSet[str]) -> Tuple[int, str]:
    """Pura. Quantos dos jogadores do jogo estao no ultimo elenco e o veredito: ok / mexido / outro."""
    n = len({str(c) for c in contas if c} & set(elenco))
    return n, ("ok" if n >= 4 else "mexido" if n == 3 else "outro")


def _nomes(raiz: Path, esporte: str) -> Dict[str, str]:
    if esporte != "dota":
        return {}
    try:
        return (json.loads((Path(raiz) / "nomes_jogadores_dota.json").read_text(encoding="utf-8")) or {}).get("nomes") or {}
    except (OSError, ValueError):
        return {}


def texto_elenco(raiz: Path, esporte: str, times: Sequence[str]) -> List[Tuple[str, str]]:
    """[(nivel, texto)] por time: nivel 'ok' (caption) ou 'atencao' (aviso amarelo)."""
    h = historico(raiz, esporte)
    nomes = _nomes(raiz, esporte)
    out = []
    for tm in times:
        if not tm:
            continue
        m = mudanca(h.get(tm) or [])
        if m is None:
            out.append(("atencao", f"**{tm}**: sem escalação na base — não dá para confirmar o time pelos jogadores."))
            continue
        if not m["base_ok"]:
            out.append(("ok", f"**{tm}**: elenco do último jogo ({m['ultimo']:%d/%m}); sem jogos antes para comparar."))
            continue
        if not m["novos"]:
            out.append(("ok", f"**{tm}**: ✅ mesmo elenco dos últimos meses (5 jogadores)."))
            continue
        qs = ", ".join(f"{nomes.get(j, j)} (desde {d:%d/%m})" for j, d in sorted(m["novos"].items(), key=lambda x: x[1]))
        n = len(m["novos"])
        extra = (" Pode ser outro time com o mesmo nome ou um time remontado — confira antes de usar o preço."
                 if n >= 3 else " A nota do time se apoia em jogos de outro elenco: vale menos.")
        out.append(("atencao", f"**{tm}**: ⚠ {n} jogador(es) novo(s) no elenco — {qs}.{extra}"))
    return out


def render(st: Any, raiz: Path, esporte: str, times: Sequence[str]) -> None:
    """Escreve os textos de elenco na tela (nunca derruba a tela)."""
    try:
        for nivel, txt in texto_elenco(raiz, esporte, times):
            (st.warning if nivel == "atencao" else st.caption)(txt)
    except Exception as e:  # noqa: BLE001
        st.caption(f"Conferência de elenco: não abriu ({type(e).__name__}).")


def suspeitos_de_troca(h: Dict[str, List[Tuple[pd.Timestamp, FrozenSet[str]]]], minimo: int = 4,
                       folga_dias: int = 21, recente_dias: int = 180) -> List[Dict[str, Any]]:
    """Pura (26/11/2026; ele: "algo que pegue quando o id mudar?"). Pares (velho, novo) com NOMES diferentes em
    que o novo comeca (ate `folga_dias` de sobreposicao) quando o velho para, e o primeiro elenco do novo tem
    `minimo`+ jogadores do ultimo elenco do velho = provavel o mesmo time com ID/nome novo, historico partido.
    5/5 a base ja une sozinha (eval_whr_dota.roster_identity_map); aqui sobram os 4/5, que pedem conferencia."""
    if not h:
        return []
    fim_base = max(v[-1][0] for v in h.values() if v)
    ult = {t: v[-1] for t, v in h.items() if v and v[-1][0] >= fim_base - timedelta(days=recente_dias)}
    pri = {t: v[0] for t, v in h.items() if v}
    out = []
    for velho, (dt_v, el_v) in ult.items():
        for novo, (dt_n, el_n) in pri.items():
            if novo == velho or dt_n < dt_v - timedelta(days=folga_dias) or dt_n > dt_v + timedelta(days=120):
                continue
            if h[novo][-1][0] <= dt_v:      # o "novo" tambem tem de seguir jogando depois do velho parar
                continue
            comum = len(el_v & el_n)
            if comum >= minimo:
                out.append({"velho": velho, "novo": novo, "comum": comum, "velho_ate": dt_v, "novo_desde": dt_n})
    return sorted(out, key=lambda x: -x["comum"])


def texto_troca(raiz: Path, esporte: str, times: Sequence[str]) -> List[str]:
    """Avisos para os times escolhidos que aparecem num par suspeito."""
    pares = suspeitos_de_troca(historico(raiz, esporte))
    out = []
    for p in pares:
        if p["novo"] in times or p["velho"] in times:
            out.append(f"**{p['novo']}** pode ser o mesmo time que **{p['velho']}** ({p['comum']} de 5 jogadores iguais; "
                       f"um parou em {p['velho_ate']:%d/%m/%y}, o outro começou em {p['novo_desde']:%d/%m/%y}) — o "
                       "histórico pode estar partido. Se for o mesmo, me avise que eu uno.")
    return out
