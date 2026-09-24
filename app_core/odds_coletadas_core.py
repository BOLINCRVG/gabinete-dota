# -*- coding: utf-8 -*-
"""Le as odds que o robo coletou da Pinnacle.

O APP NAO VAI A REDE. Quem coleta e o robo (tools/coletar_odds.py, no GitHub
Actions, de 3 em 3 horas); aqui so se le o arquivo que ele gravou. Cada visita
do app viraria uma ida a casa de apostas -- e o risco de bloqueio de IP cai
sobre ELE, que perderia o acesso ao site onde aposta.

A IDADE VEM SEMPRE JUNTO, e isto nao e enfeite. Odd e um preco vivo: a de tres
horas atras pode ja nao existir. Mostrar vantagem calculada sobre preco velho
seria mandar ele apostar num numero que nao esta mais na tela da casa -- e o
sintoma apareceria como "isso ai nao bate", que nao aponta para a causa.
Mesma licao da agenda coletada.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

NOME_DO_ARQUIVO = "odds_pinnacle.json"

# Acima disto o preco ja nao serve para decidir. O robo roda de 3 em 3 horas,
# entao passar de 4 significa que alguma corrida falhou.
HORAS_ATE_ENVELHECER = 4.0


def _idade_em_horas(carimbo: Any) -> float | None:
    """Quantas horas tem a leitura, ou None quando nao da para saber."""
    txt = str(carimbo or "").strip()
    if not txt:
        return None
    try:
        quando = datetime.fromisoformat(txt)
    except (TypeError, ValueError):
        return None
    agora = datetime.now(quando.tzinfo) if quando.tzinfo else datetime.now()
    return (agora - quando).total_seconds() / 3600.0


def carregar_odds_coletadas(raiz: Path, *, nome: str = NOME_DO_ARQUIVO) -> Dict[str, Any]:
    """As odds + a idade delas, ou o MOTIVO de nao haver.

    PRINCIPIO IV: ausencia nao vira lista vazia calada. "Nao ha aposta com
    vantagem" e "nao consegui ler as odds" sao coisas MUITO diferentes para
    quem vai apostar, e na tela vazia as duas ficam identicas.
    """
    alvo = Path(raiz) / str(nome)
    vazio = {"jogos": [], "linhas": [], "idade_horas": None, "lido_em": ""}
    if not alvo.exists():
        return {**vazio, "motivo": "o robo ainda nao coletou (arquivo ausente)"}
    try:
        dados = json.loads(alvo.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {**vazio, "motivo": "arquivo ilegivel ({})".format(type(e).__name__)}
    if not isinstance(dados, dict):
        return {**vazio, "motivo": "arquivo com formato inesperado"}

    jogos = dados.get("jogos")
    linhas = dados.get("linhas")
    if not isinstance(jogos, list) or not isinstance(linhas, list):
        return {**vazio, "motivo": "arquivo sem a lista de jogos ou de linhas"}

    jogos_ok: List[Dict[str, Any]] = [j for j in jogos if isinstance(j, dict)]
    linhas_ok: List[Dict[str, Any]] = [l for l in linhas if isinstance(l, dict)]

    idade = _idade_em_horas(dados.get("lido_em"))
    motivo = ""
    if not linhas_ok:
        motivo = "a leitura nao trouxe linha nenhuma"
    return {
        "jogos": jogos_ok,
        "linhas": linhas_ok,
        "idade_horas": idade,
        "lido_em": str(dados.get("lido_em") or ""),
        "velha": bool(idade is not None and idade > HORAS_ATE_ENVELHECER),
        "motivo": motivo,
    }


def times_da_coleta(odds: Dict[str, Any]) -> List[str]:
    """Os nomes de time como a CASA os escreve, sem repetir.

    Sao esses nomes que precisam ser casados com o CSV -- e e por aqui que a
    tela sabe quais ainda faltam ensinar.
    """
    vistos: List[str] = []
    for j in (odds.get("jogos") or []):
        if not isinstance(j, dict):
            continue
        for k in ("casa", "fora"):
            n = str(j.get(k) or "").strip()
            if n and n not in vistos:
                vistos.append(n)
    return vistos


# --- O DIARIO DAS CORRIDAS (16/09/2026) ---
#
# Ele perguntou se tudo tinha funcionado durante a noite e eu nao soube
# responder: o coletor parava em silencio quando a casa dizia 403/429, entao
# UMA JANELA BLOQUEADA FICAVA IDENTICA A UMA QUE NEM RODOU. Agora toda
# corrida deixa linha em `odds_coletas.jsonl`.
#
# Mostrar isso na TELA, e nao so no repositorio, e o que fecha o circuito:
# diario que exige abrir o GitHub para ler devolve o problema pela metade.

NOME_DO_DIARIO = "odds_coletas.jsonl"

_EM_PORTUGUES = {
    "ok": "leu",
    "bloqueado": "a casa barrou",
    "rede": "rede fora",
    "formato_mudou": "a API mudou de formato",
    "http_inesperado": "resposta estranha da casa",
}


def carregar_diario(raiz: Path, *, nome: str = NOME_DO_DIARIO,
                    ultimas: int = 8) -> List[Dict[str, Any]]:
    """As ultimas corridas do robo, da mais nova para a mais velha.

    Lista vazia quer dizer "ainda nao ha diario" -- o arquivo so comeca a
    existir da primeira corrida depois desta mudanca. Nao e erro.
    """
    alvo = Path(raiz) / str(nome)
    if not alvo.exists():
        return []
    linhas: List[Dict[str, Any]] = []
    try:
        for bruta in alvo.read_text(encoding="utf-8").splitlines():
            bruta = bruta.strip()
            if not bruta:
                continue
            try:
                d = json.loads(bruta)
            except ValueError:
                continue          # uma linha torta nao apaga o diario inteiro
            if isinstance(d, dict):
                linhas.append(d)
    except OSError:
        return []
    return list(reversed(linhas))[:max(int(ultimas), 0)]


def resumo_da_corrida(d: Dict[str, Any]) -> str:
    """Uma corrida em uma linha, em portugues."""
    quando = str(d.get("quando") or "")[:16].replace("T", " ")
    res = str(d.get("resultado") or "?")
    texto = _EM_PORTUGUES.get(res, res)
    if res == "ok":
        return "{} — {} {} jogo(s), {} preco(s) novo(s)".format(
            quando, texto, d.get("jogos", "?"), d.get("precos_novos", "?"))
    detalhe = str(d.get("detalhe") or "").strip()
    return "{} — {}{}".format(quando, texto, (" ({})".format(detalhe[:60]) if detalhe else ""))


def houve_bloqueio(corridas: List[Dict[str, Any]]) -> bool:
    """Alguma corrida recente foi barrada pela casa?

    E a pergunta que a ausencia de dados NAO respondia. Bloqueio e grave --
    muda o que fazer; janela atrasada pelo GitHub nao e nada.
    """
    return any(str(c.get("resultado") or "") == "bloqueado" for c in (corridas or []))


# --- O MOVIMENTO DO PRECO (16/09/2026) ---
#
# ELE OBSERVOU, e estava certo: "voce pode estar pegando algo que na realidade
# ja mudou de odd ja tem um tempo, ele nao necessariamente abriu nessa odd ai
# que voce esta pegando, ate porque voce nao pega a abertura".
#
# E verdade e nao tem como ser diferente: o robo le de 3 em 3 horas, entao o
# primeiro preco que vemos e o que estava la quando ele passou -- nao o que a
# casa abriu. Entre a abertura e a nossa primeira leitura pode ter entrado
# dinheiro e movido a linha.
#
# O que DA para fazer, e o que fazemos: mostrar desde quando aquele preco esta
# sendo visto e para onde andou. E chamar isso de "primeiro preco que vimos",
# nunca de "abertura" -- o nome errado transformaria uma limitacao conhecida
# numa afirmacao falsa.
#
# Por que o movimento importa: a linha andando CONTRA o seu lado quer dizer
# que o dinheiro esta indo para o outro lado. Nao prova nada sozinho, mas e
# informacao que nenhuma tela dava ate aqui.

NOME_DO_HISTORICO = "odds_pinnacle_historico.jsonl"


def carregar_movimento(raiz: Path, *, nome: str = NOME_DO_HISTORICO) -> Dict[str, Any]:
    """chave da aposta -> o que ja foi visto dela, em ordem de leitura.

    Devolve, por chave: o primeiro preco visto, quando, e quantas leituras.
    Dicionario vazio quer dizer "ainda nao ha historico" -- nao e erro.
    """
    alvo = Path(raiz) / str(nome)
    if not alvo.exists():
        return {}
    fora: Dict[str, Any] = {}
    try:
        for bruta in alvo.read_text(encoding="utf-8").splitlines():
            bruta = bruta.strip()
            if not bruta:
                continue
            try:
                d = json.loads(bruta)
            except ValueError:
                continue
            chave = str(d.get("chave") or "")
            odd = d.get("odd")
            quando = str(d.get("lido_em") or "")
            if not chave or odd is None:
                continue
            reg = fora.setdefault(chave, {"primeiro": None, "quando": "", "leituras": 0})
            if reg["primeiro"] is None:
                reg["primeiro"] = float(odd)
                reg["quando"] = quando
            reg["leituras"] += 1
    except OSError:
        return {}
    return fora


def movimento_da_linha(linha: Dict[str, Any], movimento: Dict[str, Any]) -> Dict[str, Any]:
    """Quanto esta aposta andou desde a primeira vez que a vimos.

    PRINCIPIO IV: sem historico, devolve "nao sei" -- e nao "nao andou".
    Zero de movimento e uma AFIRMACAO, e afirmar isso sobre uma linha que
    vimos uma vez so seria inventar estabilidade que ninguem observou.
    """
    vazio = {"sabido": False, "primeiro": None, "quando": "", "leituras": 0,
             "variacao_pct": None}
    chave = str((linha or {}).get("chave") or "")
    reg = (movimento or {}).get(chave)
    if not chave or not isinstance(reg, dict) or reg.get("primeiro") is None:
        return vazio
    try:
        agora = float(linha.get("odd"))
        antes = float(reg["primeiro"])
    except (TypeError, ValueError):
        return vazio
    if antes <= 0 or int(reg.get("leituras") or 0) < 2:
        # Uma leitura so nao e movimento: e uma foto. Dizer "0%" daria a
        # impressao de linha estavel sem ninguem ter visto duas vezes.
        return {**vazio, "primeiro": antes, "quando": reg.get("quando", ""),
                "leituras": int(reg.get("leituras") or 0)}
    return {"sabido": True, "primeiro": antes, "quando": str(reg.get("quando") or ""),
            "leituras": int(reg.get("leituras") or 0),
            "variacao_pct": 100.0 * (agora - antes) / antes}
