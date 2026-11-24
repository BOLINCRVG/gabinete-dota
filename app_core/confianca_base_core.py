# -*- coding: utf-8 -*-
"""FAIXA DE CONFIANCA DA BASE (24/11/2026): no topo do app, sempre: ate que jogo a base vai, quando
foi a ultima coleta boa e, se a ultima falhou, POR QUE.

Ele: "EU QUERO TER certeza de que o que eu estou vendo na minha tela e a informacao mais FIEL
POSSIVEL, [...] saber que 'esse jogo aqui, e o ultimo jogo do banco de dados que nos temos'. Se eu
tiver que ficar de tempos em tempos verificando isso, vai acabar com a confianca do modelo."

Fontes: o proprio CSV diario que o app usa (o ultimo jogo) + status_coleta_<esporte>.json, gravado
pelo robo em TODA rodada, inclusive quando falha (tools/status_coleta.py). Arquivo identico nos repos.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

CSV_DIA = {"dota": "CSV_DIARIO/2026_Dota_match_data_oe_like.csv",
           "lol": "CSV_DIARIO/2026_LoL_esports_match_data_from_OraclesElixir.csv",
           "cs": "mapas_cs.csv"}
NOME = {"dota": "Dota", "lol": "LoL", "cs": "CS2"}
HORAS_ATENCAO = 30.0          # o robo roda 2x/dia; mais de 30 h sem coleta boa = algo parou
BRT = timedelta(hours=-3)
_CACHE: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}


def ultimo_jogo(raiz: Path, esporte: str) -> Optional[Dict[str, Any]]:
    """O ultimo jogo do CSV diario que o app usa: {'quando': datetime UTC, 'a', 'b', 'liga'}."""
    arq = Path(raiz) / CSV_DIA.get(esporte, "")
    try:
        mt = arq.stat().st_mtime
    except OSError:
        return None
    c = _CACHE.get(str(arq))
    if c and c[0] == mt:
        return c[1]
    if esporte == "cs":   # uma linha por mapa: data (dia), time1, time2, pagina do torneio
        try:
            d = pd.read_csv(arq, usecols=["data", "time1", "time2", "pagina"], low_memory=False)
        except (OSError, ValueError):
            return None
        d["dt"] = pd.to_datetime(d["data"], errors="coerce")
        d = d.dropna(subset=["dt"])
        if d.empty:
            return None
        r = d.loc[d["dt"].idxmax()]
        out = {"quando": r["dt"].to_pydatetime().replace(tzinfo=timezone.utc), "a": str(r["time1"]),
               "b": str(r["time2"]), "liga": str(r["pagina"]).split("/")[0], "so_dia": True}
        _CACHE[str(arq)] = (mt, out)
        return out
    try:
        d = pd.read_csv(arq, usecols=["gameid", "date", "teamname", "league", "position"], low_memory=False)
    except (OSError, ValueError):
        return None
    t = d[d["position"] == "team"].copy()
    t["dt"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["dt"])
    if t.empty:
        return None
    g = t[t["gameid"] == t.loc[t["dt"].idxmax(), "gameid"]]
    nomes = [str(x) for x in g["teamname"].tolist()] + ["", ""]
    out = {"quando": g["dt"].max().to_pydatetime().replace(tzinfo=timezone.utc), "a": nomes[0], "b": nomes[1],
           "liga": str(g["league"].iloc[0])}
    _CACHE[str(arq)] = (mt, out)
    return out


def ler_completude(raiz: Path, esporte: str) -> Optional[Dict[str, Any]]:
    """evidence/completude_<esporte>.json: series que a fonte oficial diz que TERMINARAM e nao estao na
    base (tools/completude_<esporte>.py, no robo). Achado de 24/11/2026: a final do Mundial fora da base
    do LoL com a faixa dizendo "ok"."""
    try:
        return json.loads((Path(raiz) / "evidence" / f"completude_{esporte}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ler_status(raiz: Path, esporte: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads((Path(raiz) / f"status_coleta_{esporte}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dt(s: Any) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _brt(d: Optional[datetime]) -> str:
    return "—" if d is None else (d + BRT).strftime("%d/%m %H:%M")


def _ha(agora: datetime, d: Optional[datetime]) -> str:
    if d is None:
        return "—"
    h = (agora - d).total_seconds() / 3600.0
    return f"há {h:.0f} h" if h < 48 else f"há {h / 24:.1f} dias"


def veredito(esporte: str, ultimo: Optional[Dict[str, Any]], status: Optional[Dict[str, Any]],
             agora: Optional[datetime] = None, completude: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Pura: (nivel 'ok'|'atencao'|'erro', texto da faixa). Serie terminada fora da base sobe para atencao."""
    nivel, texto = _veredito_coleta(esporte, ultimo, status, agora)
    falt = list((completude or {}).get("faltando") or [])
    if falt:
        ex = falt[-1]
        texto += (f" · **faltam {len(falt)} série(s) que já terminaram** (fonte oficial, últimos "
                  f"{(completude or {}).get('janela_dias', 21)} dias), ex.: {ex.get('a')} x {ex.get('b')} "
                  f"({ex.get('liga')}, {str(ex.get('inicio'))[8:10]}/{str(ex.get('inicio'))[5:7]}) — veja os detalhes.")
        nivel = "erro" if nivel == "erro" else "atencao"
    return nivel, texto


def _veredito_coleta(esporte: str, ultimo: Optional[Dict[str, Any]], status: Optional[Dict[str, Any]],
                     agora: Optional[datetime] = None) -> Tuple[str, str]:
    agora = agora or datetime.now(timezone.utc)
    nome = NOME.get(esporte, esporte)
    if not ultimo:
        base = "Base sem jogo legível"
    else:
        quando = ultimo["quando"].strftime("%d/%m") if ultimo.get("so_dia") else f"{_brt(ultimo['quando'])} (Brasília)"
        base = f"Base {nome} até **{quando}**: {ultimo['a']} x {ultimo['b']} — {ultimo['liga']}"
    if not status:
        return "atencao", base + " · o robô ainda não registrou nenhuma coleta."
    ok_em = _dt(status.get("ultimo_sucesso"))
    rodou = _dt(status.get("quando"))
    if str(status.get("resultado")) != "success":
        mot = str(status.get("motivo") or "sem motivo registrado")
        return "erro", (base + f" · **a última coleta FALHOU** ({_brt(rodou)}): {mot}. "
                        f"Última coleta boa: {_brt(ok_em)} ({_ha(agora, ok_em)}).")
    if ok_em is None or (agora - ok_em).total_seconds() / 3600.0 > HORAS_ATENCAO:
        return "atencao", base + f" · última coleta boa {_ha(agora, ok_em)} (o robô roda 2x por dia)."
    return "ok", base + f" · coleta ok {_ha(agora, ok_em)}."


def render_faixa(st: Any, raiz: Path, esporte: str) -> None:
    try:
        ult, stt, cpl = ultimo_jogo(raiz, esporte), ler_status(raiz, esporte), ler_completude(raiz, esporte)
        nivel, texto = veredito(esporte, ult, stt, completude=cpl)
    except Exception as e:  # noqa: BLE001 -- a faixa nunca derruba o app, mas nunca some calada
        st.warning(f"A faixa de confiança da base não abriu ({type(e).__name__}).")
        return
    {"ok": st.success, "atencao": st.warning, "erro": st.error}[nivel](texto)
    falt = list((cpl or {}).get("faltando") or [])
    if falt:
        with st.expander(f"Séries que já terminaram e não estão na base ({len(falt)})"):
            st.dataframe([{"Data": str(f.get("inicio"))[:10], "Série": f"{f.get('a')} x {f.get('b')}",
                           "Placar": f.get("placar"), "Liga": f.get("liga"), "Motivo": f.get("motivo")}
                          for f in falt], hide_index=True, use_container_width=True)
    if nivel != "ok" and stt and (stt.get("diagnostico") or stt.get("rodada")):
        with st.expander("Detalhes da coleta"):
            if stt.get("rodada"):
                st.markdown(f"Rodada do robô: {stt['rodada']}")
            dg = stt.get("diagnostico") or {}
            if dg.get("perdas_por_liga"):
                st.write("Onde os jogos sumiram (liga):", dg["perdas_por_liga"])
            if dg.get("perdas_por_time"):
                st.write("Onde os jogos sumiram (time):", dg["perdas_por_time"])
