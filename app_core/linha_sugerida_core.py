# -*- coding: utf-8 -*-
"""A LINHA INICIAL da Analise de Linha: a do jogo, nao a padrao (24/10/2026).

Ele: "essas linhas padroes elas sao boas, mas as vezes elas enganam ... se dois
times com medias de 23,5 e 24,5 jogarem, e o modelo vier por padrao 27,5, logico
que vai ser excelente, mas nao vai ter na casa". Linha que a casa nao oferece
mostra um valor que ninguem consegue apostar.

ORDEM (a primeira que existir):
  1. a linha mais EQUILIBRADA da Pinnacle para este jogo e este mapa, entre
     a principal e as alternativas (over e under pagando parecido); se ate ela
     for torta, um passo para o lado do equilibrio. A Polymarket so tem vencedor;
  2. a MEDIA DAS MEDIAS dos dois times (no recorte que a tela ja usa),
     arredondada para o ,5 mais perto -- a linha onde o jogo "deveria" estar;
  3. a padrao do mercado (stats_config), como antes.
Ele continua podendo trocar a linha na mao; isto so muda de onde ela comeca.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]

# chave do mercado na Analise de Linha -> unidade da Pinnacle
UNIDADE_DO_MERCADO = {
    "kills": "kills", "towers": "turrets", "dragons": "dragons",
    "barons": "barons", "inhibitors": "inhibitors", "inibs": "inhibitors",
}


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def media_das_medias(serie_a, serie_b) -> Optional[float]:
    """Media de cada time, depois a media das duas, no ,5 mais perto."""
    import pandas as pd
    medias = []
    for s in (serie_a, serie_b):
        v = pd.to_numeric(s, errors="coerce").dropna() if s is not None else []
        if len(v):
            medias.append(float(v.mean()))
    if not medias:
        return None
    m = sum(medias) / len(medias)
    return math.floor(m) + 0.5


COLUNAS_DO_MERCADO = {
    "kills": ("kills_for", "kills_against"), "towers": ("towers_for", "towers_against"),
    "dragons": ("dragons_for", "dragons_against"), "barons": ("barons_for", "barons_against"),
    "inhibitors": ("inhibitors_for", "inhibitors_against"),
    "inibs": ("inhibitors_for", "inhibitors_against"),
}


def serie_recente(team_games, time: str, mercado: str):
    """O total do jogo nos ULTIMOS 10 jogos do time, qualquer liga -- a mesma
    janela do cruzamento estatistico. Nao usa o recorte da tela: num confronto
    entre ligas (Mundial) o recorte deixava um time com 4 jogos e o outro sem
    nenhum, e a "media" era de um time so (24/10/2026)."""
    import pandas as pd
    from app_core.cruzamento_estatistica_core import ultimos
    h = ultimos(team_games, time)
    if h is None or h.empty:
        return None
    if str(mercado) == "time":
        return pd.to_numeric(h.get("gamelength"), errors="coerce") / 60.0
    cols = COLUNAS_DO_MERCADO.get(str(mercado))
    if not cols or cols[0] not in h.columns or cols[1] not in h.columns:
        return None
    return pd.to_numeric(h[cols[0]], errors="coerce") + pd.to_numeric(h[cols[1]], errors="coerce")


def _media_bruta(serie_a, serie_b) -> Optional[float]:
    import pandas as pd
    medias = []
    for s in (serie_a, serie_b):
        v = pd.to_numeric(s, errors="coerce").dropna() if s is not None else []
        if len(v):
            medias.append(float(v.mean()))
    return sum(medias) / len(medias) if medias else None


def _chave(nome: str) -> str:
    from app_core.schedule_import_core import normalize_schedule_team_key
    return normalize_schedule_team_key(str(nome or ""))


def _apelidos() -> Dict[str, str]:
    try:
        return json.loads((RAIZ / "user_team_aliases.json").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def _e_o_time(nome_casa: str, time_csv: str, apelidos: Mapping[str, str]) -> bool:
    k = _chave(nome_casa)
    if k == _chave(time_csv):
        return True
    return _chave(apelidos.get(k, "")) == _chave(time_csv)


TORTA = 0.60   # chance justa de um lado acima disto = linha desequilibrada


def escada_da_pinnacle(odds: Mapping[str, Any], time_a: str, time_b: str, unidade: str,
                       mapa: int = 1, apelidos: Optional[Mapping[str, str]] = None) -> Dict[float, Dict[str, float]]:
    """{linha: {'over': odd, 'under': odd}} de TODAS as linhas (principal e
    alternativas) do total desse mercado nesse mapa, para este jogo. Pura."""
    apelidos = apelidos if apelidos is not None else {}
    jogo_id = None
    for j in odds.get("jogos") or []:
        casa, fora = str(j.get("casa") or ""), str(j.get("fora") or "")
        if ((_e_o_time(casa, time_a, apelidos) and _e_o_time(fora, time_b, apelidos))
                or (_e_o_time(casa, time_b, apelidos) and _e_o_time(fora, time_a, apelidos))):
            jogo_id = j.get("id")
            break
    escada: Dict[float, Dict[str, float]] = {}
    if jogo_id is None:
        return escada
    for l in odds.get("linhas") or []:
        if not isinstance(l, dict) or l.get("jogo") != jogo_id or str(l.get("tipo") or "") != "total":
            continue
        if str(l.get("unidade") or "").strip().lower() != unidade:
            continue
        try:
            if int(l.get("periodo") or 0) != int(mapa):
                continue
        except (TypeError, ValueError):
            continue
        p, odd = _num(l.get("pontos")), _num(l.get("odd"))
        lado = str(l.get("lado") or "").strip().lower()
        if p is not None and odd and odd > 1 and lado in ("over", "under"):
            escada.setdefault(p, {})[lado] = odd
    return {k: v for k, v in escada.items() if "over" in v and "under" in v}


def linha_equilibrada(escada: Mapping[float, Mapping[str, float]],
                      media: Optional[float] = None) -> Optional[Tuple[float, str]]:
    """(linha, explicacao). Pura.

    Pedido dele (24/10/2026): "vai colocar o 12,5 da Pinnacle, mas ele esta
    1,45 as vezes, logicamente e melhor mostrar o 11,5". A linha que interessa
    e a EQUILIBRADA -- over e under pagando parecido. Entre as que a Pinnacle
    oferece, fica a de odds mais proximas. Se ate essa for torta (um lado com
    chance justa acima de 60%), anda UMA linha para o lado do equilibrio: com
    uma linha so nao da para saber quanto, e um passo e o que a casa sugere."""
    if not escada:
        return None
    melhor = min(escada, key=lambda k: (abs(escada[k]["over"] - escada[k]["under"]), k))
    o, u = escada[melhor]["over"], escada[melhor]["under"]
    p_over = (1 / o) / (1 / o + 1 / u)
    odds_txt = "over {:.2f} x under {:.2f}".format(o, u).replace(".", ",")
    passo = -1.0 if p_over < 1 - TORTA else (1.0 if p_over > TORTA else 0.0)
    if passo:
        vizinha = melhor + passo
        # O PASSO PRECISA DE CONFIRMACAO (24/10/2026): em contagem pequena um
        # passo e grande demais -- baroes 1,5 torto nao vira 0,5 (quase todo
        # jogo tem barao). Com a media dos times: so anda se a vizinha estiver
        # mais perto dela. Sem media: so anda em linha de 3 para cima.
        if media is not None:
            anda = abs(vizinha - media) < abs(melhor - media)
        else:
            anda = melhor >= 3
        if anda and vizinha > 0:
            return vizinha, "a Pinnacle so tem {} e torta ({}); a linha equilibrada fica {}".format(
                f"{melhor:.1f}".replace(".", ","), odds_txt, "abaixo" if passo < 0 else "acima")
        return melhor, "a linha da Pinnacle ({}), torta, mas a media dos times fica perto dela".format(
            odds_txt) if media is not None else "a linha da Pinnacle ({}), torta".format(odds_txt)
    return melhor, "a linha mais equilibrada da Pinnacle ({})".format(odds_txt)


def linha_da_pinnacle(odds: Mapping[str, Any], time_a: str, time_b: str, unidade: str,
                      mapa: int = 1, apelidos: Optional[Mapping[str, str]] = None) -> Optional[float]:
    """A linha equilibrada da Pinnacle (ver `linha_equilibrada`), ou None."""
    r = linha_equilibrada(escada_da_pinnacle(odds, time_a, time_b, unidade, mapa, apelidos))
    return None if r is None else r[0]


def linha_inicial(mercado: str, time_a: str, time_b: str, serie_a, serie_b,
                  padrao: float, *, mapa: int = 1,
                  odds: Optional[Mapping[str, Any]] = None) -> Tuple[float, str]:
    """(linha, de onde veio). Ver a ordem no topo do arquivo."""
    unidade = UNIDADE_DO_MERCADO.get(str(mercado))
    if unidade:
        if odds is None:
            try:
                from app_core.odds_coletadas_core import carregar_odds_coletadas
                odds = carregar_odds_coletadas(RAIZ)
            except Exception:
                odds = {}
        r = linha_equilibrada(escada_da_pinnacle(odds or {}, time_a, time_b, unidade, mapa, _apelidos()),
                              media=_media_bruta(serie_a, serie_b))
        if r is not None:
            return float(r[0]), r[1]
    if str(mercado) in UNIDADE_DO_MERCADO or str(mercado) == "time":
        m = media_das_medias(serie_a, serie_b)
        if m is not None:
            return float(m), "a media das medias dos dois times (a Pinnacle nao tem este mercado aqui)"
    return float(padrao), "a linha padrao do mercado"
