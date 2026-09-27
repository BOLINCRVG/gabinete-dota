# -*- coding: utf-8 -*-
"""Le a agenda que o robo coletou e entrega no formato que a previa ja usa.

O APP NAO VAI A REDE. Quem coleta e o robo (tools/coletar_agenda.py, no
GitHub Actions); aqui so se le o arquivo que ele gravou. Motivo: cada visita
ao site viraria uma ida ao alvo, e a hospedagem ja mostrou que nao e confiavel
para isso (o site caiu em 13/09/2026 por um problema dela, nao nosso).

O FORMATO DE SAIDA E O MESMO de `parse_schedule_text`, de proposito: assim a
coleta entra na MESMA previa do texto colado, e mapeamento de time, apelidos
aprendidos e "Adicionar a agenda" continuam funcionando sem mudar uma linha.

IDADE SEMPRE JUNTO. Uma agenda velha repete os jogos de ontem com toda a
confianca do mundo — e o sintoma aparece como "esse jogo nao existe", que nao
aponta para a causa. Mesma licao que a linha de nota por liga ja aplicava.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

NOME_DO_ARQUIVO = "agenda_coletada.json"

# Campos que a previa do app consome. Extras (bo, estado, inicio_utc) viajam
# junto e sao usados por quem souber o que fazer com eles.
CAMPOS = ("date", "time", "league", "teamA_raw", "teamB_raw")

# Distancia maxima, em dias, para decidir de que ANO e uma data "dd/mm" sem
# ano. Escolhe-se o ano que deixa a data mais perto de hoje.
_MEIO_ANO = 183


def _idade_em_horas(carimbo: Any) -> float | None:
    """Quantas horas tem a coleta, ou None quando nao da para saber."""
    txt = str(carimbo or "").strip()
    if not txt:
        return None
    try:
        quando = datetime.fromisoformat(txt)
    except (TypeError, ValueError):
        return None
    agora = datetime.now(quando.tzinfo) if quando.tzinfo else datetime.now()
    return (agora - quando).total_seconds() / 3600.0


def carregar_agenda_coletada(raiz: Path, *, nome: str = NOME_DO_ARQUIVO) -> Dict[str, Any]:
    """Jogos coletados + idade da coleta, ou o MOTIVO de nao haver.

    PRINCIPIO IV: ausencia nao vira lista vazia calada. Sem o motivo, "nao tem
    jogo hoje" e "nao consegui coletar" ficam identicos na tela — e sao coisas
    muito diferentes para quem vai apostar.
    """
    alvo = Path(raiz) / str(nome)
    if not alvo.exists():
        return {"jogos": [], "idade_horas": None,
                "motivo": "o robo ainda nao coletou (arquivo ausente)"}
    try:
        dados = json.loads(alvo.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"jogos": [], "idade_horas": None,
                "motivo": "arquivo ilegivel ({})".format(type(e).__name__)}
    if not isinstance(dados, dict):
        return {"jogos": [], "idade_horas": None, "motivo": "arquivo com formato inesperado"}

    brutos = dados.get("jogos")
    if not isinstance(brutos, list):
        return {"jogos": [], "idade_horas": None, "motivo": "arquivo sem a lista de jogos"}

    jogos: List[Dict[str, Any]] = []
    for j in brutos:
        if not isinstance(j, dict):
            continue
        linha = {c: str(j.get(c, "") or "").strip() for c in CAMPOS}
        if not (linha["teamA_raw"] and linha["teamB_raw"]):
            continue
        try:
            bo = int(j.get("bo") or 0)
        except (TypeError, ValueError):
            bo = 0
        linha["bo"] = bo if bo in (1, 3, 5) else 0
        linha["estado"] = str(j.get("estado", "") or "")
        # O INSTANTE EXATO, que eu vinha jogando fora. Sem ele, ordenar a
        # lista dependeria de adivinhar o ano a partir de "16/09" -- e a
        # ordem saia com os jogos de maio caindo depois dos de setembro.
        linha["inicio_utc"] = str(j.get("inicio_utc", "") or "")
        jogos.append(linha)

    return {
        "jogos": jogos,
        "idade_horas": _idade_em_horas(dados.get("gerado_em")),
        "gerado_em": str(dados.get("gerado_em") or ""),
        "motivo": "" if jogos else "a coleta nao trouxe jogo nenhum",
    }


def bo_do_jogo(jogo: Any, padrao: Any) -> int:
    """O BO REAL do jogo, caindo no padrao da tela so quando nao houver.

    Antes o app carimbava um unico BO em todos os jogos importados. Um BO5
    lido como BO3 muda a odd da serie inteira — nao e detalhe.
    """
    try:
        bo = int((jogo or {}).get("bo") or 0)
    except (TypeError, ValueError, AttributeError):
        bo = 0
    if bo in (1, 3, 5):
        return bo
    try:
        p = int(padrao)
    except (TypeError, ValueError):
        p = 3
    return p if p in (1, 3, 5) else 3


# --- OS JOGOS COLETADOS NO SELETOR DE CONFRONTO (16/09/2026) ---
#
# ELE RELATOU, com print: abriu "Selecionar jogo salvo (Agenda)" e so viu tres
# jogos de 15/05 -- os antigos dele. Os 15 jogos que o robo coletou nao
# estavam la.
#
# ERRO MEU DE DESENHO. Eu liguei a coleta a uma IMPORTACAO: ir na aba Jogos,
# clicar em "Trazer os jogos coletados", conferir o mapeamento, clicar em
# "Adicionar a agenda". Tres cliques e duas telas. Mas o que ele pediu foi:
#
#     "esse jogos (agenda) e onde eu quero que fiquem os jogos que voce pegar
#      do calendario, PARA QUE EU NAO PRECISE FICAR INDO ATRAS DE JOGOS"
#
# Coleta que exige ir atras dela nao resolve o problema que ele descreveu.
# Agora os coletados aparecem direto na lista, marcados, e o mapeamento de
# nome continua sendo resolvido na hora de usar -- com aviso e "Salvar e usar"
# quando nao casar, que e a parte que NAO pode ser automatica.

ORIGEM_SALVO = "salvo"
ORIGEM_COLETADO = "calendario"


def _chave_do_jogo(j: Dict[str, Any]) -> tuple:
    """Mesma partida, venha de onde vier."""
    def _t(k1: str, k2: str) -> str:
        v = str(j.get(k1) or j.get(k2) or "").strip().lower()
        return v
    return (str(j.get("date") or "").strip(),
            str(j.get("time") or "").strip(),
            _t("teamA", "teamA_raw"),
            _t("teamB", "teamB_raw"))


def quando_do_jogo(j: Dict[str, Any], agora: Any = None) -> Any:
    """O instante do jogo, ou None quando nao da para saber.

    Prefere `inicio_utc`, que vem exato do calendario. Sem ele, monta a partir
    de "dd/mm" + "hh:mm" escolhendo o ANO que deixa a data mais perto de hoje
    -- e por isso que "15/05", visto em setembro, e entendido como MAIO DESTE
    ANO (que ja passou) e nao como maio do ano que vem.

    PRINCIPIO IV: data que nao da para montar devolve None, e None vai para o
    fim da lista em vez de virar uma data inventada.
    """
    bruto = str(j.get("inicio_utc") or "").strip()
    if bruto:
        try:
            q = datetime.fromisoformat(bruto.replace("Z", "+00:00"))
            return q if q.tzinfo else q.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass

    data = str(j.get("date") or "").strip()
    hora = str(j.get("time") or "").strip() or "00:00"
    partes = data.replace("-", "/").split("/")
    if len(partes) < 2:
        return None
    try:
        dia, mes = int(partes[0]), int(partes[1])
        hh, mm = (int(x) for x in (hora.split(":") + ["0"])[:2])
    except (TypeError, ValueError):
        return None

    base = agora or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    melhor = None
    for ano in (base.year - 1, base.year, base.year + 1):
        try:
            cand = datetime(ano, mes, dia, hh, mm, tzinfo=timezone.utc)
        except ValueError:
            continue          # 29/02 em ano que nao tem
        dist = abs((cand - base).total_seconds())
        if melhor is None or dist < melhor[0]:
            melhor = (dist, cand)
    return melhor[1] if melhor else None


def _ordem_na_lista(j: Dict[str, Any], agora: Any) -> tuple:
    """Os que vem primeiro sao os PROXIMOS. Depois os que ja passaram.

    ELE RELATOU (16/09/2026): "a ordem so e meio cagada". A lista ia
    16/09, 17/09... 20/09 e entao caia para 15/05 sem nada explicando -- os
    tres jogos velhos que ele tinha salvos em maio. Datas andando para tras no
    fim parecem lista quebrada, e nao "secao de jogos antigos".

    Agora: por vir em ordem crescente (o mais proximo no topo), depois os que
    ja passaram do mais recente para o mais velho, e por ultimo o que nao tem
    data legivel.
    """
    q = quando_do_jogo(j, agora)
    if q is None:
        return (2, 0.0)
    delta = (q - agora).total_seconds()
    if delta >= -4 * 3600:        # ate 4h depois do inicio ainda conta como do dia
        return (0, delta)
    return (1, -delta)


def ja_passou(j: Dict[str, Any], agora: Any = None) -> bool:
    """Jogo que ja aconteceu. Aparece marcado, nunca escondido em silencio."""
    base = agora or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return _ordem_na_lista(j, base)[0] == 1


def juntar_para_escolher(salvos: Any, coletados: Any) -> List[Dict[str, Any]]:
    """Uma lista so para o seletor: primeiro os coletados, depois os dele.

    OS COLETADOS VEM PRIMEIRO de proposito: sao os jogos que estao POR VIR, e
    e por eles que ele abre a lista. Os salvos dele ficam logo abaixo, sem
    perder nada -- inclusive as linhas que ele ja tinha digitado neles.

    Partida que esta nas duas listas aparece UMA vez, e fica a versao SALVA:
    e a dele, pode ter linha digitada junto, e perder isso seria apagar
    trabalho dele.
    """
    saida: List[Dict[str, Any]] = []
    salvos_ok = [j for j in (salvos or []) if isinstance(j, dict)]
    coletados_ok = [j for j in (coletados or []) if isinstance(j, dict)]
    ja_salvas = {_chave_do_jogo(j) for j in salvos_ok}

    for j in coletados_ok:
        if _chave_do_jogo(j) in ja_salvas:
            continue
        saida.append({
            "date": str(j.get("date") or ""),
            "time": str(j.get("time") or ""),
            "league": str(j.get("league") or ""),
            "teamA": str(j.get("teamA") or j.get("teamA_raw") or ""),
            "teamB": str(j.get("teamB") or j.get("teamB_raw") or ""),
            "bo": j.get("bo", 3),
            # O INSTANTE EXATO TEM DE VIAJAR JUNTO.
            #
            # Eu o guardei na leitura (16/09/2026) e o descartei AQUI, tres
            # linhas depois. A ordem continuou parecendo certa porque sobrava
            # o palpite por "dd/mm" -- e o palpite trata a hora do rotulo, que
            # e LOCAL, como se fosse UTC: erro de tres horas, que marca jogo
            # como "ja passou" cedo demais.
            #
            # A suite so acusou no dia seguinte, quando o relogio andou o
            # bastante para o palpite errar. Teste que depende da hora e
            # incomodo, mas foi ele que pegou.
            "inicio_utc": str(j.get("inicio_utc") or ""),
            "origem": ORIGEM_COLETADO,
        })
    for j in salvos_ok:
        item = dict(j)
        item["origem"] = ORIGEM_SALVO
        saida.append(item)

    agora = datetime.now(timezone.utc)
    saida.sort(key=lambda j: _ordem_na_lista(j, agora))
    return saida


def rotulo_do_jogo(j: Dict[str, Any]) -> str:
    """O texto de uma linha do seletor.

    A origem aparece: sem isso ele nao saberia se aquele jogo foi ele que
    salvou (podendo ter linha digitada junto) ou se veio do calendario.
    """
    d = str(j.get("date") or "").strip()
    t = str(j.get("time") or "").strip()
    quando = (d + (" " + t if t else "")).strip()
    partes = [p for p in (quando, str(j.get("league") or "").strip()) if p]
    partes.append("{} x {}".format(str(j.get("teamA") or "").strip(),
                                   str(j.get("teamB") or "").strip()))
    try:
        bo = int(j.get("bo", 3) or 3)
    except (TypeError, ValueError):
        bo = 3
    partes.append("BO{}".format(bo))
    if str(j.get("origem") or "") == ORIGEM_COLETADO:
        partes.append("do calendario")
    if ja_passou(j):
        partes.append("JA PASSOU")
    return " | ".join(partes)
