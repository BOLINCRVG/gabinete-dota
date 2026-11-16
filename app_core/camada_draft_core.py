# -*- coding: utf-8 -*-
"""SUPER-LIVE: camada POS-DRAFT -- o p do mapa depois que os campeoes/herois sao escolhidos.

Medido em 16/11/2026, as cegas (sinais so com jogos ANTERIORES; ajuste ate 31/12/2025, teste em
2026; base = p de producao do gate ao vivo, que entra com PESO FIXO 1: a camada so soma o draft,
nao recalibra o modelo pre-jogo):
  LoL : logloss 0.6003 -> 0.5862  ganho +0.0141 IC95 [+0.0112; +0.0173]  (9.864 mapas)
  Dota: logloss 0.6439 -> 0.6321  ganho +0.0118 IC95 [+0.0090; +0.0145]  (8.947 mapas)
(No Dota, deixar o peso do p livre rendia mais -- o pre-jogo de 2026 esta confiante demais -- mas
isso e recalibrar o modelo, assunto separado.)
Sinais (azul - vermelho; no Dota azul = Radiant), so do passado:
  campeao  = media do aproveitamento recente de cada campeao/heroi (decaimento, meia-vida 90 dias,
             encolhido para 50% com 30 jogos de peso)
  conforto = media do aproveitamento de cada jogador NAQUELE campeao (encolhido com 8 jogos)
  pratica  = media de log(1 + jogos do jogador naquele campeao)
p_pos = sigmoide(a + logit(p_pre) + c1*campeao + c2*conforto + c3*pratica).

O estado (tabelas + coeficientes) vem de camada_draft_<esporte>.json, gerado por
tools/camada_draft.py. Arquivo identico nos dois repositorios.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MEIA_VIDA_DIAS = 90.0
K_CAMPEAO = 30.0
K_JOGADOR = 8.0

# Nome interno do feed da lolesports -> nome do Oracle's Elixir (o resto casa pela normalizacao).
_APELIDO_CAMPEAO = {"monkeyking": "wukong", "nunu": "nunuwillump", "renata": "renataglasc"}


def chave_campeao(nome: Any) -> str:
    """'Kai'Sa' / 'Kaisa' / 'KaiSa' -> 'kaisa'; 'MonkeyKing' -> 'wukong'; heroi do Dota (id) -> '49'."""
    bruto = str(nome if nome is not None else "").strip()
    if re.fullmatch(r"\d+(\.0+)?", bruto):  # heroi do Dota: 49 / "49" / 49.0
        return str(int(float(bruto)))
    s = unicodedata.normalize("NFKD", bruto).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return _APELIDO_CAMPEAO.get(s, s)


def chave_jogador(nome: Any) -> str:
    s = unicodedata.normalize("NFKD", str(nome or "")).encode("ascii", "ignore").decode()
    s = s.strip().lower()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return re.sub(r"\s+", " ", s)


def nome_sem_tag(nome: str, tags: Iterable[str] = ()) -> str:
    """Feed da lolesports: 'NTH Excellents' -> 'Excellents' (tira a sigla do time na frente)."""
    s = str(nome or "").strip()
    partes = s.split(" ", 1)
    if len(partes) == 2:
        sig = {str(t or "").strip().lower() for t in tags if str(t or "").strip()}
        if partes[0].lower() in sig or (not sig and partes[0].isupper() and len(partes[0]) <= 5):
            return partes[1].strip()
    return s


class EstadoDraft:
    """Tabelas acumuladas em ordem de data: aproveitamento de campeao (com decaimento) e de
    jogador-campeao. Usado pelo gerador (sequencial) e pelo app (estado salvo)."""

    def __init__(self) -> None:
        self.campeao: Dict[str, List[Any]] = {}  # chave -> [vitorias, jogos, ultima_data_iso]
        self.jogador: Dict[str, List[float]] = {}  # "jogador|campeao" -> [vitorias, jogos]
        self._conhecidos: Optional[set] = None

    def conhece(self, jog: str) -> bool:
        """O jogador tem ALGUM jogo na base? (0 jogos no campeao so vale como 'pouca pratica' se ele
        existe na base; jogador fora da base -- time novo, nome diferente -- nao pode virar sinal.)"""
        if self._conhecidos is None:
            self._conhecidos = {k.rsplit("|", 1)[0] for k in self.jogador}
        return jog in self._conhecidos

    # -- leitura ---------------------------------------------------------
    def aproveitamento_campeao(self, camp: str, quando: datetime) -> float:
        v, n, u = self.campeao.get(camp, [0.0, 0.0, None])
        if u:
            dias = max(0.0, (quando - datetime.fromisoformat(str(u)[:19])).total_seconds() / 86400.0)
            f = 0.5 ** (dias / MEIA_VIDA_DIAS)
            v, n = v * f, n * f
        return (v + 0.5 * K_CAMPEAO) / (n + K_CAMPEAO)

    def jogos_no_campeao(self, jog: str, camp: str) -> Tuple[float, float]:
        v, n = self.jogador.get(f"{jog}|{camp}", [0.0, 0.0])
        return float(v), float(n)

    def sinais_do_lado(self, lado: Sequence[Tuple[str, str]], quando: datetime) -> Dict[str, float]:
        ch, cf, pr = [], [], []
        for jog, camp in lado:
            ch.append(self.aproveitamento_campeao(camp, quando))
            v, n = self.jogos_no_campeao(jog, camp)
            cf.append((v + 0.5 * K_JOGADOR) / (n + K_JOGADOR))
            pr.append(math.log1p(n))
        m = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
        return {"campeao": m(ch), "conforto": m(cf), "pratica": m(pr)}

    def sinais(self, azul, vermelho, quando: datetime) -> Dict[str, float]:
        a, b = self.sinais_do_lado(azul, quando), self.sinais_do_lado(vermelho, quando)
        return {k: a[k] - b[k] for k in a}

    def sinais_no_app(self, azul, vermelho, quando: datetime) -> Tuple[Dict[str, float], List[str]]:
        """Como `sinais`, mas se algum jogador nao esta na base, conforto e pratica ficam de fora
        (so o sinal do campeao vale) e a lista dos que faltam volta para a tela explicar."""
        s = self.sinais(azul, vermelho, quando)
        faltam = [j for j, _ in list(azul) + list(vermelho) if not self.conhece(j)]
        if faltam:
            s["conforto"] = 0.0
            s["pratica"] = 0.0
        return s, faltam

    # -- escrita ---------------------------------------------------------
    def registrar(self, lado, venceu: int, quando: datetime) -> None:
        for jog, camp in lado:
            v, n, u = self.campeao.get(camp, [0.0, 0.0, None])
            if u:
                dias = max(0.0, (quando - datetime.fromisoformat(str(u)[:19])).total_seconds() / 86400.0)
                f = 0.5 ** (dias / MEIA_VIDA_DIAS)
                v, n = v * f, n * f
            self.campeao[camp] = [v + venceu, n + 1.0, quando.isoformat(timespec="seconds")]
            k = f"{jog}|{camp}"
            w, g = self.jogador.get(k, [0.0, 0.0])
            self.jogador[k] = [w + venceu, g + 1.0]
        self._conhecidos = None

    def para_json(self) -> Dict[str, Any]:
        return {"campeao": self.campeao, "jogador": self.jogador}

    @classmethod
    def de_json(cls, d: Mapping[str, Any]) -> "EstadoDraft":
        e = cls()
        e.campeao = {str(k): list(v) for k, v in (d.get("campeao") or {}).items()}
        e.jogador = {str(k): list(v) for k, v in (d.get("jogador") or {}).items()}
        return e


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def p_pos_draft(p_pre_azul: float, sinais: Mapping[str, float], coef: Mapping[str, float]) -> float:
    """P(azul vence o mapa) depois do draft."""
    z = (float(coef.get("intercepto", 0.0)) + float(coef.get("p_pre", 1.0)) * _logit(p_pre_azul)
         + sum(float(coef.get(k, 0.0)) * float(sinais.get(k, 0.0)) for k in ("campeao", "conforto", "pratica")))
    return 1.0 / (1.0 + math.exp(-z))


def arquivo(raiz: Path, esporte: str) -> Path:
    return Path(raiz) / f"camada_draft_{esporte}.json"


def carregar(raiz: Path, esporte: str) -> Optional[Dict[str, Any]]:
    """{'coef':..., 'estado': EstadoDraft, 'gerado_em':..., 'medicao':...} ou None."""
    p = arquivo(raiz, esporte)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d.get("coef"), dict):
        return None
    return {"coef": d["coef"], "estado": EstadoDraft.de_json(d.get("estado") or {}),
            "gerado_em": d.get("gerado_em"), "medicao": d.get("medicao")}


def detalhe_do_lado(estado: EstadoDraft, lado: Sequence[Tuple[str, str]], quando: datetime,
                    nomes: Optional[Sequence[Tuple[str, str]]] = None) -> List[Dict[str, Any]]:
    """Linha por jogador para a tela: campeao, jogos e vitorias dele no campeao, aproveitamento do campeao."""
    out = []
    for i, (jog, camp) in enumerate(lado):
        v, n = estado.jogos_no_campeao(jog, camp)
        rot = nomes[i] if nomes and i < len(nomes) else (jog, camp)
        out.append({"jogador": rot[0], "campeao": rot[1], "na_base": estado.conhece(jog),
                    "jogos_no_campeao": int(round(n)),
                    "vitorias_no_campeao": int(round(v)),
                    "aproveitamento_campeao": estado.aproveitamento_campeao(camp, quando)})
    return out
