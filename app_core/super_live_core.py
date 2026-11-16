# -*- coding: utf-8 -*-
"""SUPER-LIVE: quem esta na vantagem -- antes do jogo, depois do draft e com o mapa rolando.

Cadeia (P do time AZUL vencer o MAPA; no Dota azul = Radiant):
  1. pre-jogo  : o p do mapa do app (modelo congelado).
  2. pos-draft : app_core/camada_draft_core (medida as cegas; soma o draft ao p do modelo).
  3. ao vivo   : modelos ao vivo SO-FEED (live_models_feed/, tools/treinar_ao_vivo_feed.py), que
                 usam o p pos-draft como ponto de partida. Os modelos existem em 10/15/20/25 min;
                 entre eles a previsao e interpolada no logit; antes dos 10 min, entre o pos-draft
                 (minuto 0) e o de 10; depois dos 25, o de 25.

Fontes automaticas (dados publicos, sem chave nossa):
  LoL : API do site lolesports (a mesma chave publica que o proprio site usa) + livestats
        (feed.lolesports.com): draft, ouro, abates, farm a cada 10 s.
  Dota: OpenDota /api/live: herois, vantagem de patrimonio, placar de abates (com o atraso de
        transmissao que a Valve impoe).
Se a fonte falhar, a tela aceita o draft e o estado digitados.

Arquivo identico nos dois repositorios.
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

UA = {"User-Agent": "gabinete-super-live/1.0"}
LOL_API = "https://esports-api.lolesports.com/persisted/gw"
LOL_FEED = "https://feed.lolesports.com/livestats/v1"
# Chave publica embutida no proprio site lolesports.com (nao e credencial nossa).
LOL_CHAVE_PUBLICA = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
DOTA_LIVE = "https://api.opendota.com/api/live"
MINUTOS = (10, 15, 20, 25)
# Colunas que o feed NAO tem -- zeradas no treino (tools/treinar_ao_vivo_feed.py) e aqui.
FORA_DO_FEED = {"lol": ("xp_diff",), "dota": ("xp_diff", "cs_diff", "assists_diff")}


def _get(url: str, headers: Optional[Mapping[str, str]] = None, timeout: float = 12.0) -> Any:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ----------------------------------------------------------------------------------------------
# LoL (lolesports)
# ----------------------------------------------------------------------------------------------
def lol_ao_vivo(get=_get) -> List[Dict[str, Any]]:
    """Mapas de LoL rolando agora: [{fonte, id_jogo, liga, azul, vermelho, siglas, mapa, bo, placar}]."""
    h = {"x-api-key": LOL_CHAVE_PUBLICA}
    ev = (((get(f"{LOL_API}/getLive?hl=en-US", h) or {}).get("data") or {}).get("schedule") or {}).get("events") or []
    out = []
    for e in ev:
        if e.get("type") != "match" or e.get("state") != "inProgress":
            continue
        mid = (e.get("match") or {}).get("id") or e.get("id")
        try:
            det = ((get(f"{LOL_API}/getEventDetails?hl=en-US&id={mid}", h) or {}).get("data") or {}).get("event") or {}
        except Exception:  # noqa: BLE001
            continue
        m = det.get("match") or {}
        times = {str(t.get("id")): t for t in (m.get("teams") or [])}
        jogos = m.get("games") or []
        rolando = [g for g in jogos if g.get("state") == "inProgress"]
        if not rolando:
            continue
        g = rolando[-1]
        lados = {str(t.get("side")): str(t.get("id")) for t in (g.get("teams") or [])}
        az, vm = times.get(lados.get("blue", "")), times.get(lados.get("red", ""))
        if not az or not vm:
            continue
        placar = {str(t.get("id")): int(((t.get("result") or {}).get("gameWins")) or 0) for t in (m.get("teams") or [])}
        out.append({
            "fonte": "lolesports", "id_jogo": str(g.get("id")), "liga": str((det.get("league") or {}).get("name") or ""),
            "azul": str(az.get("name") or ""), "vermelho": str(vm.get("name") or ""),
            "siglas": (str(az.get("code") or ""), str(vm.get("code") or "")),
            "mapa": int(g.get("number") or 1), "bo": int(((m.get("strategy") or {}).get("count")) or 1),
            "placar": (placar.get(str(az.get("id")), 0), placar.get(str(vm.get("id")), 0)),
        })
    return out


def _iso10(t: datetime) -> str:
    t = t.replace(microsecond=0, second=(t.second // 10) * 10)
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def lol_estado(id_jogo: str, get=_get, agora: Optional[datetime] = None,
               inicio_cache: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Estado do mapa: draft + ultimo quadro. O minuto e contado do primeiro quadro do jogo."""
    agora = agora or datetime.now(timezone.utc)
    cache = inicio_cache if inicio_cache is not None else {}
    primeiro = get(f"{LOL_FEED}/window/{id_jogo}")
    if not primeiro or not primeiro.get("frames"):
        return None
    meta = primeiro.get("gameMetadata") or {}
    inicio = cache.get(id_jogo) or primeiro["frames"][0]["rfc460Timestamp"]
    cache[id_jogo] = inicio
    try:
        ult = get(f"{LOL_FEED}/window/{id_jogo}?startingTime={urllib.parse.quote(_iso10(agora - timedelta(seconds=60)))}")
        frames = (ult or {}).get("frames") or primeiro["frames"]
    except Exception:  # noqa: BLE001
        frames = primeiro["frames"]
    f = frames[-1]
    t0 = datetime.fromisoformat(inicio.replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(f["rfc460Timestamp"].replace("Z", "+00:00"))

    def lado(meta_lado, quadro):
        ps = quadro.get("participants") or []
        return {
            "draft": [(str(p.get("summonerName") or ""), str(p.get("championId") or ""), str(p.get("role") or ""))
                      for p in (meta_lado.get("participantMetadata") or [])],
            "ouro": float(quadro.get("totalGold") or 0), "abates": float(quadro.get("totalKills") or 0),
            "mortes": float(sum(float(p.get("deaths") or 0) for p in ps)),
            "assistencias": float(sum(float(p.get("assists") or 0) for p in ps)),
            "farm": float(sum(float(p.get("creepScore") or 0) for p in ps)),
            "torres": int(quadro.get("towers") or 0), "inibidores": int(quadro.get("inhibitors") or 0),
            "baroes": int(quadro.get("barons") or 0), "dragoes": list(quadro.get("dragons") or []),
        }

    return {"fonte": "lolesports", "minuto": max(0.0, (t1 - t0).total_seconds() / 60.0),
            "situacao": str(f.get("gameState") or ""), "lido_em": t1.isoformat(),
            "azul": lado(meta.get("blueTeamMetadata") or {}, f.get("blueTeam") or {}),
            "vermelho": lado(meta.get("redTeamMetadata") or {}, f.get("redTeam") or {})}


# ----------------------------------------------------------------------------------------------
# Dota (OpenDota /live)
# ----------------------------------------------------------------------------------------------
def dota_ao_vivo(get=_get, agora: Optional[float] = None) -> List[Dict[str, Any]]:
    """Mapas de Dota de liga rolando agora (com os dois times identificados)."""
    agora = agora if agora is not None else time.time()
    out = []
    for g in get(DOTA_LIVE) or []:
        if not g.get("league_id") or not g.get("team_name_radiant") or not g.get("team_name_dire"):
            continue
        if int(g.get("deactivate_time") or 0) > 0:  # mapa ja encerrado (a lista guarda por um tempo)
            continue
        ps = g.get("players") or []
        rad = [(str(p.get("account_id") or ""), str(p.get("hero_id") or ""), str(p.get("name") or ""))
               for p in ps if int(p.get("team", -1)) == 0 and int(p.get("hero_id") or 0) > 0]
        dire = [(str(p.get("account_id") or ""), str(p.get("hero_id") or ""), str(p.get("name") or ""))
                for p in ps if int(p.get("team", -1)) == 1 and int(p.get("hero_id") or 0) > 0]
        rs, ds = float(g.get("radiant_score") or 0), float(g.get("dire_score") or 0)
        lead = float(g.get("radiant_lead") or 0)
        out.append({
            "fonte": "opendota", "id_jogo": str(g.get("match_id") or ""), "liga": str(g.get("league_id")),
            "azul": str(g["team_name_radiant"]), "vermelho": str(g["team_name_dire"]),
            "siglas": ("", ""), "mapa": None, "bo": None, "placar": None,
            "atualizado_ha_s": max(0, int(agora - float(g.get("last_update_time") or agora))),
            "estado": {
                "fonte": "opendota", "minuto": max(0.0, float(g.get("game_time") or 0) / 60.0), "situacao": "",
                "azul": {"draft": rad, "ouro": max(lead, 0.0), "abates": rs, "mortes": ds},
                "vermelho": {"draft": dire, "ouro": max(-lead, 0.0), "abates": ds, "mortes": rs},
            },
        })
    return out


# ----------------------------------------------------------------------------------------------
# Probabilidade ao vivo
# ----------------------------------------------------------------------------------------------
def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def vetor(esporte: str, p_partida: float, estado: Mapping[str, Any], tier: int, n_mapa: int,
          playoffs: bool) -> List[float]:
    """Mesma ordem de mlcore.live_ml.features._FEATURE_COLS, do ponto de vista do AZUL."""
    a, b = estado["azul"], estado["vermelho"]

    def d(k):
        return float(a.get(k) or 0.0) - float(b.get(k) or 0.0)

    v = {"z_pre": _logit(p_partida), "gold_diff": d("ouro"), "xp_diff": 0.0, "cs_diff": d("farm"),
         "kills_diff": d("abates"), "deaths_diff": d("mortes"), "assists_diff": d("assistencias"),
         "league_tier": float(tier), "bo": float(n_mapa or 1), "playoffs": 1.0 if playoffs else 0.0}
    for c in FORA_DO_FEED[esporte]:
        v[c] = 0.0
    return [v[c] for c in ("z_pre", "gold_diff", "xp_diff", "cs_diff", "kills_diff", "deaths_diff",
                           "assists_diff", "league_tier", "bo", "playoffs")]


def carregar_modelos(raiz: Path) -> Dict[int, Any]:
    from mlcore.live_ml.state_models import StateToWinModel
    out = {}
    for m in MINUTOS:
        p = Path(raiz) / "live_models_feed" / f"minute_{m:02d}"
        if p.exists():
            try:
                out[m] = StateToWinModel.load(p)
            except Exception:  # noqa: BLE001
                pass
    return out


def _prever(modelo, x: Sequence[float]) -> float:
    import numpy as np
    from mlcore.live_ml.features import LiveSnapshot
    s = LiveSnapshot(gameid="ao_vivo", minute=int(modelo.minute), side="Blue", p_pre=_sig(x[0]),
                     features=np.array(x, dtype=float), y=0)
    return float(modelo.predict_proba_one(s))


def p_ao_vivo(esporte: str, modelos: Mapping[int, Any], p_partida: float, estado: Mapping[str, Any],
              tier: int = 5, n_mapa: int = 1, playoffs: bool = False) -> Optional[Dict[str, Any]]:
    """{'p': P(azul vence o mapa), 'como': texto} ou None sem modelo."""
    if not modelos:
        return None
    t = float(estado.get("minuto") or 0.0)
    x = vetor(esporte, p_partida, estado, tier, n_mapa, playoffs)
    ms = sorted(modelos)
    if t <= ms[0]:
        q = _prever(modelos[ms[0]], x)
        w = t / ms[0]
        return {"p": _sig((1 - w) * _logit(p_partida) + w * _logit(q)),
                "como": f"entre o pos-draft (min 0) e o modelo de {ms[0]} min"}
    if t >= ms[-1]:
        return {"p": _prever(modelos[ms[-1]], x), "como": f"modelo de {ms[-1]} min"}
    lo = max(m for m in ms if m <= t)
    hi = min(m for m in ms if m >= t)
    if lo == hi:
        return {"p": _prever(modelos[lo], x), "como": f"modelo de {lo} min"}
    w = (t - lo) / (hi - lo)
    z = (1 - w) * _logit(_prever(modelos[lo], x)) + w * _logit(_prever(modelos[hi], x))
    return {"p": _sig(z), "como": f"entre os modelos de {lo} e {hi} min"}


def odd_justa(p: Optional[float]) -> Optional[float]:
    if p is None or not (0.0 < p < 1.0):
        return None
    return round(1.0 / p, 2)


# ----------------------------------------------------------------------------------------------
# Polymarket com o jogo rolando
# ----------------------------------------------------------------------------------------------
def evento_polymarket(raiz: Path, esporte: str, azul: str, vermelho: str, resolver=None,
                      agora: Optional[datetime] = None) -> Optional[Tuple[str, str, str]]:
    """Slug do evento da Polymarket deste confronto: o robo grava cada leitura pre-jogo no historico;
    o evento mais recente com o mesmo par de times (inicio ate 18 h atras ou 6 h a frente)."""
    agora = agora or datetime.now(timezone.utc)
    hist = Path(raiz) / f"polymarket_{esporte}_historico.jsonl"
    if not hist.exists():
        return None
    r = resolver or (lambda x: str(x or "").strip().lower())
    alvo = {r(azul) or azul.lower(), r(vermelho) or vermelho.lower()}
    melhor = None
    for linha in hist.read_text(encoding="utf-8").splitlines()[-4000:]:
        try:
            j = json.loads(linha)
            ini = datetime.fromisoformat(str(j.get("inicio_utc")).replace("Z", "+00:00").replace(" ", "T"))
        except (ValueError, TypeError):
            continue
        if ini.tzinfo is None:
            ini = ini.replace(tzinfo=timezone.utc)
        if not (agora - timedelta(hours=18) <= ini <= agora + timedelta(hours=6)):
            continue
        par = {r(j.get("time_a")) or str(j.get("time_a")).lower(), r(j.get("time_b")) or str(j.get("time_b")).lower()}
        if par == alvo and j.get("evento"):
            melhor = (j["evento"], j.get("time_a"), j.get("time_b"))
    return melhor


def polymarket_ao_vivo(raiz: Path, slug: str) -> Dict[str, Any]:
    """Le o evento AGORA na Gamma: mercados de mapa abertos (o menor = o mapa em jogo) e o da serie.
    {'mapas': {n: (lados, p1, spread)}, 'serie': (lados, p1, spread) ou None, 'motivo': str}"""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("coletar_polymarket_sl", Path(raiz) / "tools" / "coletar_polymarket.py")
        cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cp)
        evs = cp._get("events", slug=slug)
    except Exception as e:  # noqa: BLE001
        return {"mapas": {}, "serie": None,
                "motivo": f"a Polymarket nao respondeu daqui ({type(e).__name__}); do Brasil ela e bloqueada, na nuvem abre"}
    ev = evs[0] if isinstance(evs, list) and evs else None
    if not ev:
        return {"mapas": {}, "serie": None, "motivo": "evento nao encontrado na Polymarket"}
    mapas = {}
    for m in cp.outros_mercados(ev):
        if m["tipo"] == "child_moneyline":
            mapas[int(m["mapa"])] = (m["lados"], m["p1"], m["spread"])
    serie = None
    ml = next((x for x in (ev.get("markets") or []) if x.get("sportsMarketType") == "moneyline" and not x.get("closed")), None)
    if ml:
        pr = cp._preco(ml)
        if pr:
            serie = ([str(t) for t in cp._lista(ml.get("outcomes"))], pr[0], pr[2])
    return {"mapas": mapas, "serie": serie, "motivo": "" if (mapas or serie) else "sem mercado aberto agora"}
