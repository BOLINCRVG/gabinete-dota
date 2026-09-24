# -*- coding: utf-8 -*-
"""Coleta as odds de LoL da Pinnacle e guarda o MOVIMENTO delas.

POR QUE A PINNACLE, E NAO QUALQUER CASA
Ela trabalha com margem baixa e limite alto, e aceita dinheiro afiado em vez
de barrar a conta — por isso o preco dela e o mais proximo de probabilidade
verdadeira que o mercado oferece. Com esses precos guardados COM HORA, da para
medir se o modelo bate o fechamento (CLV), que e a unica medicao que responde
se existe lucro. Nenhuma regua deste projeto responde isso hoje: todas
comparam o modelo com ele mesmo.

DE ONDE VEM, SEM DISFARCE
`guest.api.arcadia.pinnacle.com` — a API interna que o site DELES usa, aberta
e sem credencial. NAO e a API oficial: essa fechou para o publico e, para
esports, o FAQ deles diz que precisa de autorizacao especial (b2b@pinnacle.com).
E a porta dos fundos, e ele decidiu usa-la sabendo disso (15/09/2026).

O risco concreto NAO e a conta de aposta dele — nada aqui usa login. E bloqueio
de IP, que tiraria o acesso ao site onde ele aposta. Por isso:
  - DUAS requisicoes por corrida para o retrato geral, de 3 em 3 horas, mais
    a varredura dos jogos que estao para comecar, com TETO de 40 chamadas.
    No pior caso 42 por corrida, 8 corridas por dia. O User-Agent diz isso.
  - User-Agent honesto, que diz o que e e para que serve.
  - 403/429 = para na hora e sai com rc=75. Nao insiste, nao tenta contornar,
    nao troca de identidade. Se eles nao quiserem, a resposta e parar.

O QUE EU ESTAVA JOGANDO FORA (descoberto em 15/09/2026)
A resposta ja trazia, sem custo nenhum, os confrontos FILHOS: o placar exato
e -- quando abre -- o mercado de KILLS, com total, handicap e total por time,
mapa a mapa. O filtro descartava todos eles como "filho". Duas correcoes
foram precisas: adotar o filho quando o pai esta na mesma resposta, e parar
de exigir "casa/fora" de um mercado cujos lados sao "Over 33.5" e "Under".

POR QUE SO O QUE MUDA
Guardar o retrato inteiro a cada corrida daria ~9 MB por mes no repositorio.
E o que a medicao de CLV precisa nao e o retrato: e o MOVIMENTO. Entao cada
linha so entra no historico quando o preco dela mudou desde a ultima leitura.
Encolhe enormemente e nao perde nada do que importa.

MEDIDO EM 15/09/2026, chamando de verdade:
  - 9 series de LoL por vir (LPL, LEC, LCS, CBLOL, Rift Legends, WSCI)
  - mercados: moneyline, handicap de mapas, total de mapas, vencedor de cada
    mapa e PLACAR EXATO -- este ultimo vinha junto na mesma resposta e eu o
    descartava de proposito; sao 54 linhas por corrida que custam zero
  - a margem denuncia onde ela esta confiante: LPL 4,8% · LCS 5,3% · WSCI 9,4%
  - o LIMITE tambem: R$ 2.000 no LPL contra R$ 100 no WSCI. E a propria casa
    dizendo em qual preco acredita — informacao que ninguem mais da de graca.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

def _achar_raiz() -> Path:
    """A raiz do repositorio pelo git, nunca pela pasta do arquivo.

    Este arquivo e IDENTICO no LoL e no Dota (regra de arquivo compartilhado),
    e a pasta do arquivo ja fez uma ferramenta do Dota medir a pasta errada
    por semanas. O `parents[1]` fica so de reserva, para quando nao ha git.
    """
    import subprocess
    aqui = Path(__file__).resolve().parent
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(aqui),
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return aqui.parent


RAIZ = _achar_raiz()
# DEFEITO ACHADO EM 23/09/2026: o robo roda `python tools/coletar_odds.py`, e
# o Python so enxerga a pasta tools/. O `from app_core...` falhava calado
# (dentro de um try), e desde 21/09 nenhuma das 591 leituras do robo levou a
# traducao do mercado -- nem o aviso de "mercado que eu nao sei ler", que
# existia justamente para mercado novo nao sumir calado. As telas nao
# perderam nada (traduzem ao ler), mas o vigia estava cego.
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# QUAL JOGO. A leitura geral da Pinnacle traz TODOS os esports juntos; ate
# 23/09/2026 o Dota vinha em toda corrida e era jogado fora. O repositorio do
# Dota tem o `dota_paths.py`; e assim que este arquivo, identico nos dois,
# sabe onde esta.
JOGO_DO_REPOSITORIO = "dota" if (RAIZ / "dota_paths.py").exists() else "lol"
PREFIXO_DA_LIGA = {"lol": "League of Legends - ", "dota": "Dota 2 - "}

AGORA_JSON = RAIZ / "odds_pinnacle.json"          # o retrato de agora
HISTORICO = RAIZ / "odds_pinnacle_historico.jsonl"  # so o que mudou
# QUEM jogava em cada partida. O historico de precos guarda so o ID do jogo, e
# `odds_pinnacle.json` (que tem os nomes) e sobrescrito a cada corrida.
#
# POR QUE ISTO EXISTE (21/09/2026): para comparar o modelo com o preco da casa
# eu precisei dos nomes das partidas antigas, e so consegui escavando o
# historico do git commit a commit. Funcionou por sorte -- o robo comita
# aquele arquivo. Medicao que depende de arqueologia nao e medicao.
PARTIDAS = RAIZ / "odds_pinnacle_partidas.jsonl"
DIARIO = RAIZ / "odds_coletas.jsonl"              # TODA corrida, deu certo ou nao

BASE = "https://guest.api.arcadia.pinnacle.com/0.1/"
ESPORTE_ESPORTS = 12
# O User-Agent e uma declaracao, e declaracao que mente nao serve para nada.
# Quando a varredura de derivados entrou, "8 requisicoes por dia" deixou de
# ser verdade -- o numero aqui acompanha o TETO de verdade do programa.
#
# O NUMERO, MEDIDO NO DIARIO (23/09/2026): o dia mais cheio (18 a 20/09, WSCI)
# teve ~270 requisicoes, com o teto batido em toda corrida. O teto por corrida
# nao mudou; o que mudou e que ele agora cobre ~20 jogos em vez de 3 ou 4.
AGENTE = ("gabinete-{}/1.2 (leitura de odds para analise pessoal, "
          "ate ~300 requisicoes por dia; sem login)").format(JOGO_DO_REPOSITORIO)
FUSO = "America/Sao_Paulo"

RC_OK = 0
RC_ERRO = 1
RC_TEMPORARIO = 75

# --- DERIVADOS (kills, torres, tempo) ---
#
# ERRO MEU, corrigido em 15/09/2026 depois de ele me corrigir: eu disse que a
# Pinnacle "so tem mercado de mapas". ELA TEM kills, torres e tempo — eu e que
# olhei so o nivel do esporte inteiro (`sports/12/markets/straight`), que traz
# os mercados PRINCIPAIS. Os derivados vivem por confronto:
#     matchups/{id}/related           -> quais derivados existem
#     matchups/{id}/markets/straight  -> os precos de cada um
#
# Isso custa requisicoes. Entao:
#   - a PROPRIA CASA avisa, em `periods`, se ja existe aposta dentro do mapa;
#     esse e o gatilho principal, e nao um palpite nosso;
#   - alem dele, os jogos que comecam nas proximas HORAS_A_FRENTE, porque ela
#     pode abrir a qualquer momento e queremos pegar o preco de abertura;
#   - PAUSA entre cada chamada;
#   - TETO por corrida, para uma agenda cheia nao virar enxurrada.
#
# ONDE KILLS REALMENTE ESTAVA (visto ao vivo em 15/09/2026, e NAO era aqui):
# kills nao entra em `periods` do jogo; ela abre um CONFRONTO FILHO proprio,
# com `units: "Kills"` e os times renomeados "Fuego (Kills)" x "9z (Kills)".
# Esse filho ja vem na leitura do esporte inteiro, de graca -- era ele que o
# filtro de "filho" jogava fora. Torre e tempo devem chegar do mesmo jeito,
# cada um com a sua unidade; quando chegarem, o coletor pega sozinho e o
# rotulo mostra a unidade, sem precisar mexer em nada.
#
# `tem_mercado_no_mapa` continua valendo para outra coisa: decidir se vale
# gastar requisicao varrendo os derivados daquele jogo.
#
# DUAS CHAMADAS POR JOGO, NAO UMA POR MERCADO (23/09/2026). Ele pediu a
# Pinnacle "de forma total, 100%", LoL e Dota. Do jeito antigo -- `related` e
# depois um `markets/straight` para CADA derivado -- uma partida do Dota
# custava ~11 chamadas, e o teto de 40 cobria 3 ou 4 jogos: 17 de 62 corridas
# bateram nele e cortaram o resto da fila. `matchups/{id}/markets/related/
# straight` traz os precos de TODOS os derivados de uma vez. Conferido ao vivo
# no mesmo dia: Xtreme x Nemesis (Dota) -> 57 mercados de 11 confrontos numa
# chamada, os mesmos 11 que o caminho antigo listava um a um.
HORAS_A_FRENTE = 8.0
PAUSA_ENTRE_CHAMADAS = 1.2
TETO_DE_CHAMADAS = 40
CHAMADAS_POR_JOGO = 2
RECUSAS_401_PARA_PARAR = 2
#
# FONTES (pesquisado em 23/09/2026)
#   - O endereco que traz todos os precos de uma vez, usado por um projeto
#     aberto que le a mesma porta:
#     github.com/pretrehr/Sports-betting/blob/master/sportsbetting/bookmakers/pinnacle.py
#     (`matchups/{}/markets/related/straight` e `matchups/{}/related`).
#     Esse projeto e o oddsR (rdrr.io/github/miken97/oddsR/src/R/pinnacle.R)
#     mandam a CHAVE copiada do site (X-API-Key). Nos NAO mandamos: seria nos
#     passar pelo site deles. Por isso um ou outro endereco pode responder 401.
#   - A API OFICIAL fechou ao publico em 23/07/2025 e limita a "1 request per
#     2 minutes, per endpoint, per sportId", com 429 para excesso:
#     github.com/pinnacleapi/pinnacleapi-documentation. Nao e a porta que
#     usamos, mas e o unico numero que a casa publicou. A leitura geral (2
#     chamadas) cabe nele com folga; a varredura por jogo usa enderecos
#     diferentes, espacados de PAUSA_ENTRE_CHAMADAS.
#   - 401 pode ser freio de excesso disfarcado, e nao so falta de credencial
#     (ha servidores que respondem assim: github.com/better-auth/better-auth/
#     issues/9504). Nao da para saber qual e -- entao o 401 pula o jogo, e o
#     segundo 401 na mesma corrida para tudo, como 403/429.


def anotar_corrida(caminho: Path, **campos: Any) -> None:
    """UMA LINHA POR CORRIDA, tenha ela dado certo ou nao.

    POR QUE ISTO EXISTE (16/09/2026). Ele perguntou se tudo tinha funcionado
    durante a noite. As odds deviam ser lidas as 21:25, 00:25, 03:25 e 06:25;
    o historico tinha UMA leitura. E eu nao consegui dizer por que.

    A culpa e de um desenho meu: quando a casa responde 403/429, o coletor
    para em silencio e sai com rc=75 -- de proposito, para nao insistir num
    bloqueio nem encher ele de e-mail por problema passageiro. So que o
    efeito colateral e que UMA JANELA BLOQUEADA FICA IDENTICA A UMA JANELA
    QUE NEM RODOU. E e justamente essa distincao que importa: se a casa
    comecou a barrar, isso e grave e muda o que fazer; se o GitHub so
    atrasou, nao e nada.

    Agora toda corrida deixa rastro. A gravacao NUNCA derruba a coleta: um
    diario que quebrasse o robo seria pior que nao ter diario.
    """
    try:
        linha = {"quando": _agora_local().isoformat(timespec="seconds")}
        linha.update(campos)
        with open(str(caminho), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(linha, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _agora_local() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(FUSO))
    except Exception:
        return datetime.now(timezone.utc)


def para_decimal(preco: Any) -> Optional[float]:
    """Odd americana -> decimal. None quando nao da para converter.

    PRINCIPIO IV: preco ilegivel nao vira 1,0 nem 2,0 — vira ausencia.
    """
    try:
        p = float(preco)
    except (TypeError, ValueError):
        return None
    if p == 0:
        return None
    return 1.0 + (p / 100.0 if p > 0 else 100.0 / abs(p))


def margem(odds: List[float]) -> Optional[float]:
    """Quanto a casa cobra, em %. Precisa de TODAS as pernas do mercado.

    EXIGE PELO MENOS DUAS PERNAS, e isso nao e detalhe: com uma so, a conta
    devolve um numero negativo perfeitamente plausivel (uma odd de 2,90
    sozinha da -65,5%) que nao quer dizer nada. Defeito meu, pego pelo teste
    em 15/09/2026 — o coletor so chamava com duas pernas, entao ele nunca
    apareceu na pratica, mas a funcao estava errada para quem a usasse depois.
    """
    if not odds or len(odds) < 2:
        return None
    if any(o is None or o <= 1.0 for o in odds):
        return None
    return (sum(1.0 / o for o in odds) - 1.0) * 100.0


def _exigir_lista(payload: Any, nome: str) -> List[Any]:
    """A resposta tem de ser lista. Se a API mudou, ESTOURA.

    Devolver [] calado seria o pior desfecho: a tela diria "sem odds hoje"
    com toda a confianca do mundo, num dia cheio de jogo.
    """
    if not isinstance(payload, list):
        raise ValueError("{} nao veio como lista (a API mudou?)".format(nome))
    return payload


def _mapa_de_participantes(m: Dict[str, Any]) -> Dict[Any, str]:
    """id do participante -> nome dele.

    Num mercado derivado a aposta NAO se chama "casa" ou "fora": chama-se
    "Yes", "No", "Odd", "Even". O preco aponta para o participante por
    `participantId`, e o nome mora aqui.
    """
    fora: Dict[Any, str] = {}
    for p in (m.get("participants") or []):
        if isinstance(p, dict) and p.get("id") is not None:
            fora[p.get("id")] = str(p.get("name") or "")
    return fora


def _lados_casa_fora(m: Dict[str, Any]) -> Dict[str, str]:
    return {str(p.get("alignment")): str(p.get("name") or "")
            for p in (m.get("participants") or []) if isinstance(p, dict)}


_TIPO_EM_PORTUGUES = {
    "moneyline": "Vencedor",
    "spread": "Handicap",
    "total": "Total",
    "team_total": "Total do time",
}


def periodo_legivel(n: Any) -> str:
    """0 e a serie inteira; 1, 2, 3... sao os mapas."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    return "serie" if n == 0 else "mapa {}".format(n)


def unidade_da_aposta(m: Dict[str, Any]) -> str:
    """Em que se conta esta aposta: "Kills", "Maps", "Rounds"...

    Campo proprio, e nao um pedaco do rotulo, porque e por ele que se decide
    se o modelo sabe precificar a linha. "Regular" nao e unidade -- e o nome
    que a casa da ao confronto principal --, entao vira vazio.
    """
    u = str(m.get("units") or "").strip()
    return "" if u.lower() == "regular" else u


def rotulo_da_linha(m: Dict[str, Any], mk: Dict[str, Any]) -> str:
    """Como chamar esta aposta na tela.

    Quando a casa da nome ao mercado -- "(Map 1) Total Kills", "Correct
    Score" -- esse nome vale letra por letra: e o que ele vai comparar com a
    tela da Pinnacle. Quando nao da (os mercados principais), montamos do que
    existe: tipo + unidade + periodo. NAO se inventa a unidade: se a casa nao
    disse se o total e de mapas ou de kills, o rotulo nao afirma.
    """
    esp = m.get("special") if isinstance(m.get("special"), dict) else {}
    d = str(esp.get("description") or "").strip()
    if d:
        return d
    bruto = str(mk.get("type") or "")
    partes = [_TIPO_EM_PORTUGUES.get(bruto, bruto or "?")]
    u = str(m.get("units") or "").strip()
    if u and u.lower() != "regular":
        partes.append(u)
    partes.append(periodo_legivel(mk.get("period", 0)))
    return " - ".join(partes)


def tem_mercado_no_mapa(m: Dict[str, Any]) -> bool:
    """A casa ja abriu aposta DENTRO de um mapa (kills, torre, tempo)?

    `periods` e uma lista em que cada item diz, para a serie (0) e para cada
    mapa (1, 2, 3...), quais mercados existem. Se nenhum mapa tem total nem
    handicap, kills e tempo ainda nao abriram -- e ir buscar seria bater na
    porta da casa para ouvir "nao tem".
    """
    for pr in (m.get("periods") or []):
        if not isinstance(pr, dict):
            continue
        try:
            n = int(pr.get("period", -1))
        except (TypeError, ValueError):
            continue
        if n >= 1 and (pr.get("hasTotal") or pr.get("hasTeamTotal")
                       or pr.get("hasSpread")):
            return True
    return False


def _sem_repetidos(matchups: List[Any]) -> List[Dict[str, Any]]:
    """`related` devolve o PROPRIO confronto junto com os derivados dele.

    DEFEITO DE 15/09/2026: sem isto, os jogos varridos entravam duas vezes --
    9 series viraram "11 series" no relatorio, e cada linha deles foi contada
    em dobro. Numero inflado e pior que numero faltando: parece certo.
    """
    onde: Dict[Any, int] = {}
    fora: List[Dict[str, Any]] = []
    for m in matchups:
        if not isinstance(m, dict):
            continue
        i = m.get("id")
        if i in onde:
            # SEGUNDA ARMADILHA, do mesmo dia: o derivado JA VEM na lista
            # geral, so que sem saber de quem e filho. Ficar com a primeira
            # copia jogava a marca `_paiId` fora e o mercado caia como
            # "filho" -- de novo em silencio. Guardamos a marca na copia que
            # fica.
            if m.get("_paiId") and not fora[onde[i]].get("_paiId"):
                fora[onde[i]] = {**fora[onde[i]], "_paiId": m["_paiId"]}
            continue
        onde[i] = len(fora)
        fora.append(m)
    return fora


def normalizar(matchups: Any, mercados: Any, *, so_lol: bool = True,
               matchups_extra: Any = None, mercados_extra: Any = None,
               jogo: Optional[str] = None) -> Dict[str, Any]:
    """As respostas -> uma lista plana de linhas de aposta.

    `matchups_extra`/`mercados_extra` trazem os DERIVADOS (kills, torres,
    tempo), que so existem por confronto. Eles entram na mesma lista, com o
    rotulo do mercado junto, pendurados no jogo PAI -- para quem le nao
    precisar saber de onde veio.
    """
    matchups = _exigir_lista(matchups, "matchups")
    mercados = _exigir_lista(mercados, "markets")
    if matchups_extra:
        matchups = list(matchups) + [m for m in matchups_extra if isinstance(m, dict)]
    if mercados_extra:
        mercados = list(mercados) + [m for m in mercados_extra if isinstance(m, dict)]
    matchups = _sem_repetidos(matchups)

    por_jogo: Dict[int, List[dict]] = {}
    ja_visto = set()
    for mk in mercados:
        if not isinstance(mk, dict):
            continue
        ident = (mk.get("matchupId"), str(mk.get("key") or ""),
                 mk.get("type"), mk.get("period"))
        if ident in ja_visto:
            continue          # mesmo mercado pelas duas portas: conta uma vez
        ja_visto.add(ident)
        por_jogo.setdefault(int(mk.get("matchupId", -1)), []).append(mk)

    por_id = {m.get("id"): m for m in matchups}
    linhas: List[Dict[str, Any]] = []
    jogos: List[Dict[str, Any]] = []
    descartes = {"ao_vivo": 0, "filho": 0, "sem_mercado": 0, "outro_jogo": 0,
                 "sem_lados": 0}

    def _de_quem_e_filho(m: Dict[str, Any]) -> Optional[Any]:
        """`_paiId` vem da varredura por confronto; `parentId` ja vinha na
        lista geral e eu o ignorava -- era o Placar Exato e, quando abre, o
        mercado de KILLS chegando DE GRACA e sendo jogados fora."""
        pai_id = m.get("_paiId") or m.get("parentId")
        if pai_id is not None and pai_id == m.get("id"):
            return None        # nao existe filho de si mesmo
        return pai_id

    def _linhas_de(m: Dict[str, Any], mks: List[dict], lados: Dict[str, str],
                   id_do_jogo: int) -> None:
        nomes = _mapa_de_participantes(m)
        for mk in mks:
            precos = mk.get("prices") if isinstance(mk.get("prices"), list) else []
            decs = [para_decimal(x.get("price")) for x in precos if isinstance(x, dict)]
            bons = [d for d in decs if d is not None]
            # Antes so calculava com 2 pernas; um "Placar Exato" tem 6, e a
            # margem dele e igualmente real.
            marg = margem(bons) if len(bons) >= 2 else None
            lim = (mk.get("limits") or [{}])
            limite = lim[0].get("amount") if isinstance(lim[0], dict) else None
            rotulo = rotulo_da_linha(m, mk)
            unidade = unidade_da_aposta(m)
            traducao: Dict[str, Any] = {}
            try:
                from app_core.mercado_pinnacle_core import entender_mercado
                _e = entender_mercado(rotulo)
                traducao = {"metrica": _e.get("metrica"), "forma": _e.get("forma"),
                            "mapa": _e.get("mapa"), "nome_pt": _e.get("nome_pt")}
            except Exception:
                # Traduzir e util; coletar o preco e essencial. Se a traducao
                # falhar, a linha vai sem ela e ainda serve.
                traducao = {}
            for x in precos:
                if not isinstance(x, dict):
                    continue
                d = para_decimal(x.get("price"))
                if d is None:
                    continue
                lado = str(x.get("designation") or "").strip()
                if not lado:
                    # DEFEITO DE 15/09/2026: num mercado derivado os lados
                    # nao sao casa/fora, sao "Over 33.5" e "Under 33.5", e o
                    # preco aponta para o participante por id.
                    lado = nomes.get(x.get("participantId"), "")
                linhas.append({
                    "jogo": id_do_jogo,
                    "tipo": str(mk.get("type") or ""),
                    "periodo": int(mk.get("period", -1) or 0),
                    "alternativa": bool(mk.get("isAlternate", False)),
                    "lado": lado,
                    "pontos": x.get("points"),
                    "odd": round(float(d), 4),
                    "americana": x.get("price"),
                    "limite": limite,
                    "margem_pct": (round(marg, 2) if marg is not None else None),
                    "fecha_em": str(mk.get("cutoffAt") or ""),
                    # "Total - Kills - mapa 1", "Correct Score", "Total -
                    # Maps - serie"... -- o que distingue uma linha de kills
                    # de uma de mapas.
                    "mercado": rotulo,
                    # A IDENTIDADE DA APOSTA, no proprio retrato.
                    #
                    # ELE OBSERVOU (16/09/2026): "voce pode estar pegando algo
                    # que na realidade ja mudou de odd ja tem um tempo, ele nao
                    # necessariamente abriu nessa odd ai... ate porque voce nao
                    # pega a abertura". Esta certo. Com a chave aqui, a tela
                    # consegue cruzar cada linha com o historico e mostrar
                    # DESDE QUANDO aquele preco esta ali e para onde ele andou.
                    # Nao e a abertura -- e o primeiro preco que NOS vimos, e a
                    # tela diz isso com essas palavras.
                    # A UNIDADE como campo proprio, nao so dentro do texto.
                    # Quem for comparar com a odd justa do modelo precisa
                    # saber se a linha e de kills, de torre ou de mapas; ler
                    # isso de dentro de um rotulo escrito para humano seria
                    # frageis por natureza -- bastaria a casa mudar uma
                    # palavra. "Regular" vira vazio: na serie, "Regular" nao
                    # nomeia unidade nenhuma.
                    "unidade": unidade,
                })
                # DE QUAL TIME, no total por time. A casa manda `side`
                # ("home"/"away") no mercado; sem isto, "over 30,5" nao diz
                # se e o total do mandante ou do visitante.
                if str(mk.get("type") or "") == "team_total" and mk.get("side"):
                    linhas[-1]["time"] = str(mk.get("side"))
                # A TRADUCAO AO LADO (21/09/2026). O nome da casa fica intacto
                # -- e o que ele confere na tela da Pinnacle. Junto dele vao os
                # campos que o programa entende (metrica, forma, mapa), senao
                # mercado com nome ingles fica invisivel para a aba de
                # Oportunidades. Foi o que aconteceu com TORRES, que e um dos
                # dois mercados em que ele ganha dinheiro: 3.430 leituras
                # coletadas e nenhuma usada.
                #
                # DEFEITO ACHADO EM 23/09/2026, na primeira corrida em que a
                # traducao de fato rodou: este bloco estava um nivel FORA do
                # laco dos precos, e so o ULTIMO lado de cada mercado era
                # traduzido (236 de 464 linhas sem). Num mercado sem preco
                # legivel, pior: traduzia a linha do mercado ANTERIOR.
                linhas[-1].update(traducao)
                linhas[-1]["chave"] = chave_da_linha(linhas[-1])

    # PRIMEIRA PASSADA: as partidas de verdade.
    #
    # Em duas passadas, e nao em uma, porque a resposta nao garante que o pai
    # venha antes do filho -- e sobretudo porque o filho so pode entrar se o
    # pai TIVER ENTRADO. Se o pai esta ao vivo, ou sem mercado, a linha de
    # kills dele apontaria para um jogo que nao esta na lista: a tela mostraria
    # uma aposta sem jogo. PRINCIPIO IV.
    aceitos: Dict[Any, Dict[str, str]] = {}
    filhos: List[Dict[str, Any]] = []
    # `jogo` escolhe o esporte; sem ele vale o `so_lol` de sempre.
    if jogo is None and so_lol:
        jogo = "lol"
    prefixo = PREFIXO_DA_LIGA.get(jogo or "", "")
    for m in matchups:
        liga = str((m.get("league") or {}).get("name") or "")
        if prefixo and prefixo.strip(" -").upper() not in liga.upper():
            descartes["outro_jogo"] += 1
            continue
        if _de_quem_e_filho(m) is not None:
            filhos.append(m)
            continue
        if m.get("isLive"):
            descartes["ao_vivo"] += 1   # escopo do projeto e PRE-JOGO
            continue
        mks = por_jogo.get(int(m.get("id", -1)), [])
        if not mks:
            descartes["sem_mercado"] += 1
            continue
        lados = _lados_casa_fora(m)
        if not (lados.get("home") and lados.get("away")):
            descartes["sem_lados"] += 1
            continue
        ident = int(m.get("id", -1))
        aceitos[m.get("id")] = lados
        jogos.append({
            "id": ident,
            "liga": liga.replace(prefixo, "") if prefixo else liga,
            "casa": lados["home"], "fora": lados["away"],
            "bo": m.get("bestOfX"),
            # A propria casa DIZ se ja existe aposta dentro do mapa. Medido em
            # 15/09/2026: ela vai abrindo esses mercados aos poucos.
            "tem_mercado_no_mapa": tem_mercado_no_mapa(m),
            "inicio_utc": str(m.get("startTime") or ""),
        })
        _linhas_de(m, mks, lados, ident)

    # SEGUNDA PASSADA: as outras apostas da MESMA partida -- placar exato,
    # kills, e o que mais ela abrir. Ficam penduradas no jogo pai, nunca
    # viram uma partida separada: "Fuego (Kills)" x "9z (Kills)" nao existe.
    for m in filhos:
        pai_id = _de_quem_e_filho(m)
        if pai_id not in aceitos:
            descartes["filho"] += 1
            continue
        if m.get("isLive"):
            descartes["ao_vivo"] += 1
            continue
        mks = por_jogo.get(int(m.get("id", -1)), [])
        if not mks:
            descartes["sem_mercado"] += 1
            continue
        _linhas_de(m, mks, aceitos[pai_id], int(pai_id))

    return {"jogos": jogos, "linhas": linhas, "descartes": descartes}


class Bloqueado(Exception):
    """A casa pediu para parar (403/429). Nao e erro nosso; e um NAO."""


class NaoAutorizado(Exception):
    """401: aquele endereco exige algo que nao mandamos. Ver FONTES."""


def _buscar(sessao: Any, url: str, cab: Dict[str, str]) -> Any:
    """Uma chamada, tratando o NAO da casa como NAO.

    403 e 429 nao viram retentativa nem espera-e-tenta-de-novo: viram parada.
    Insistir depois de um bloqueio e o caminho mais curto para ser bloqueado
    de vez — e quem perderia o acesso ao site e ele, nao eu.
    """
    r = sessao.get(url, headers=cab, timeout=30)
    if r.status_code in (403, 429):
        raise Bloqueado("a casa respondeu {} em {}".format(r.status_code, url[-60:]))
    if r.status_code == 401:
        raise NaoAutorizado("a casa respondeu 401 em {}".format(url[-60:]))
    if r.status_code != 200:
        raise ValueError("HTTP {} em {}".format(r.status_code, url[-60:]))
    return r.json()


def _comeca_em_breve(inicio_utc: str, agora: datetime, horas: float) -> bool:
    """A linha de um jogo de semana que vem nem abriu ainda."""
    try:
        q = datetime.fromisoformat(str(inicio_utc).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if q.tzinfo is None:
        q = q.replace(tzinfo=timezone.utc)
    delta = (q - agora).total_seconds() / 3600.0
    return -4.0 <= delta <= float(horas)


def buscar_derivados(sessao: Any, cab: Dict[str, str], jogos: List[Dict[str, Any]],
                     *, agora: Optional[datetime] = None,
                     pausa: float = PAUSA_ENTRE_CHAMADAS,
                     teto: int = TETO_DE_CHAMADAS) -> Dict[str, Any]:
    """Kills, torres, tempo — os mercados que so existem por confronto.

    Devolve {"matchups": [...], "mercados": [...], "chamadas": n, "parado": bool}.
    `parado` diz se veio um NAO da casa no meio; o que ja foi lido continua
    valendo, e a corrida termina quieta.
    """
    import time as _t

    agora = agora or datetime.now(timezone.utc)
    # Vale a pena bater na porta quando (a) a casa ja avisou que existe
    # aposta dentro do mapa, ou (b) o jogo esta perto e ela pode abrir a
    # qualquer momento. Longe e sem aviso, nao ha o que buscar.
    alvos = [j for j in jogos
             if j.get("tem_mercado_no_mapa")
             or _comeca_em_breve(j.get("inicio_utc", ""), agora, HORAS_A_FRENTE)]
    # O MAIS PERTO DO APITO PRIMEIRO. Se o teto cortar, corta o jogo que
    # ainda tem outras corridas pela frente, nunca o que comeca agora.
    alvos.sort(key=lambda j: str(j.get("inicio_utc") or "9999"))
    fora: Dict[str, Any] = {"matchups": [], "mercados": [], "chamadas": 0,
                            "parado": False, "jogos_olhados": len(alvos),
                            "jogos_lidos": 0, "recusas_401": 0}
    for j in alvos:
        # Cada jogo custa DUAS chamadas; meio jogo nao serve para nada.
        if fora["chamadas"] + CHAMADAS_POR_JOGO > teto:
            break
        try:
            # 1) quais mercados existem, com o NOME de cada um ("(Map 1) 1st
            #    Roshan"); os precos nao trazem nome.
            rel = _buscar(sessao, "{}matchups/{}/related".format(BASE, j["id"]), cab)
            fora["chamadas"] += 1
            _t.sleep(pausa)
            # 2) os precos de TODOS eles de uma vez (ver FONTES no topo).
            mks = _buscar(sessao, "{}matchups/{}/markets/related/straight".format(
                BASE, j["id"]), cab)
            fora["chamadas"] += 1
            _t.sleep(pausa)
        except Bloqueado:
            fora["parado"] = True
            return fora
        except NaoAutorizado:
            # 401 nao e 403/429, mas tambem nao e "vai la de novo": pode ser
            # mercado que exige a chave do site (que nao mandamos, de
            # proposito) ou um freio disfarcado (FONTES). Pula o jogo; no
            # segundo, para a corrida.
            fora["recusas_401"] += 1
            if fora["recusas_401"] >= RECUSAS_401_PARA_PARAR:
                fora["parado"] = True
                return fora
            continue
        except (ValueError, KeyError):
            continue
        fora["jogos_lidos"] += 1
        for r in (rel if isinstance(rel, list) else []):
            if not isinstance(r, dict):
                continue
            rid = r.get("id")
            if rid is None or rid == j["id"]:
                # O proprio confronto volta na lista de derivados. A marca de
                # pai apontando para si mesmo fazia o jogo sumir como se
                # fosse filho de si proprio.
                continue
            fora["matchups"].append({**r, "_paiId": j["id"]})
        # Os precos do proprio jogo tambem voltam aqui; o `normalizar` ja
        # descarta o repetido (mesmo mercado pelas duas portas conta uma vez).
        fora["mercados"].extend(x for x in (mks if isinstance(mks, list) else [])
                                if isinstance(x, dict))
    return fora


def rotulo_do_derivado(m: Dict[str, Any]) -> str:
    """Nome legivel do mercado derivado: "Total Kills", "Map 1 Duration"...

    Vem de `special.description` quando ha; senao da unidade (Maps, Regular).
    Sem inventar: o que nao tiver nome fica "?" e quem le decide.
    """
    esp = m.get("special") if isinstance(m.get("special"), dict) else {}
    d = str(esp.get("description") or "").strip()
    if d:
        return d
    u = str(m.get("units") or "").strip()
    return u or "?"


def chave_da_linha(l: Dict[str, Any]) -> str:
    """Identidade de uma aposta, para saber se o preco dela mudou.

    O TIME entra so quando existe (total por time). DEFEITO ACHADO EM
    23/09/2026: sem ele, "over 28,5" do mandante e "over 28,5" do visitante
    tinham a MESMA chave -- 49 vezes o historico misturou os dois numa mesma
    leitura. As outras chaves ficam identicas, para o historico antigo
    continuar casando.
    """
    base = "{}|{}|{}|{}|{}|{}".format(
        l.get("jogo"), l.get("mercado", ""), l.get("tipo"), l.get("periodo"),
        l.get("lado"), "" if l.get("pontos") is None else l.get("pontos"))
    time = str(l.get("time") or "")
    return base + "|" + time if time else base


def _ultimas_odds(caminho: Path) -> Dict[str, float]:
    """Ultimo preco visto de cada aposta, lido do historico."""
    vistos: Dict[str, float] = {}
    try:
        with open(str(caminho), "r", encoding="utf-8") as fh:
            for linha in fh:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    r = json.loads(linha)
                except ValueError:
                    continue   # linha truncada nao derruba o resto
                if isinstance(r, dict) and r.get("chave"):
                    vistos[str(r["chave"])] = r.get("odd")
    except OSError:
        pass
    return vistos


def _gravar_atomico(caminho: Path, conteudo: Dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(caminho.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(conteudo, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, str(caminho))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Coleta odds de LoL ou Dota da Pinnacle.")
    ap.add_argument("--jogo", choices=sorted(PREFIXO_DA_LIGA), default=JOGO_DO_REPOSITORIO)
    ap.add_argument("--agora", default=str(AGORA_JSON))
    ap.add_argument("--historico", default=str(HISTORICO))
    ap.add_argument("--diario", default=str(DIARIO))
    ap.add_argument("--partidas", default=str(PARTIDAS))
    args = ap.parse_args()
    diario = Path(str(args.diario)).expanduser()
    partidas = Path(str(args.partidas)).expanduser()

    import requests

    cab = {"User-Agent": AGENTE}
    try:
        r1 = requests.get("{}sports/{}/matchups".format(BASE, ESPORTE_ESPORTS),
                          headers=cab, timeout=30)
        if r1.status_code in (403, 429):
            print("[odds] a casa respondeu {} — parando na hora.".format(r1.status_code))
            print("[odds] nao insisto e nao tento contornar. Tenta na proxima janela.")
            anotar_corrida(diario, resultado="bloqueado", rc=RC_TEMPORARIO,
                           detalhe="HTTP {} na lista de confrontos".format(r1.status_code))
            return RC_TEMPORARIO
        r2 = requests.get("{}sports/{}/markets/straight".format(BASE, ESPORTE_ESPORTS),
                          headers=cab, timeout=30)
        if r2.status_code in (403, 429):
            print("[odds] a casa respondeu {} nos mercados — parando.".format(r2.status_code))
            anotar_corrida(diario, resultado="bloqueado", rc=RC_TEMPORARIO,
                           detalhe="HTTP {} nos mercados".format(r2.status_code))
            return RC_TEMPORARIO
    except requests.RequestException as e:
        print("[odds] rede fora do ar: {}".format(e))
        anotar_corrida(diario, resultado="rede", rc=RC_TEMPORARIO,
                       detalhe="{}: {}".format(type(e).__name__, str(e)[:120]))
        return RC_TEMPORARIO

    if r1.status_code != 200 or r2.status_code != 200:
        print("[odds] ERRO: HTTP {} e {}.".format(r1.status_code, r2.status_code))
        anotar_corrida(diario, resultado="http_inesperado", rc=RC_ERRO,
                       detalhe="HTTP {} e {}".format(r1.status_code, r2.status_code))
        return RC_ERRO

    # DERIVADOS: kills, torres, tempo. So dos jogos proximos, espacado, com
    # teto. Se a casa disser NAO em qualquer ponto, para e usa o que ja tem.
    derivados = {"matchups": [], "mercados": [], "chamadas": 0, "parado": False,
                 "jogos_olhados": 0}
    try:
        _pre = normalizar(r1.json(), r2.json(), jogo=args.jogo)
        sessao = requests.Session()
        derivados = buscar_derivados(sessao, cab, _pre["jogos"])
    except ValueError:
        pass   # formato tratado logo abaixo, no lugar certo
    except Exception:
        pass   # derivado e bonus: se falhar, o principal continua valendo

    try:
        dados = normalizar(r1.json(), r2.json(),
                           matchups_extra=derivados["matchups"],
                           mercados_extra=derivados["mercados"],
                           jogo=args.jogo)
    except ValueError as e:
        print("[odds] ERRO DE FORMATO: {}".format(e))
        print("[odds] Nada foi escrito. O retrato anterior continua valendo.")
        anotar_corrida(diario, resultado="formato_mudou", rc=RC_ERRO,
                       detalhe=str(e)[:200])
        return RC_ERRO

    lido_em = _agora_local().isoformat(timespec="seconds")
    alvo_agora = Path(str(args.agora)).expanduser()
    alvo_hist = Path(str(args.historico)).expanduser()

    _gravar_atomico(alvo_agora, {
        "lido_em": lido_em, "fonte": BASE, "fuso": FUSO,
        "jogos": dados["jogos"], "linhas": dados["linhas"],
        "descartes": dados["descartes"],
    })

    # HISTORICO: so o que MUDOU desde a ultima leitura.
    anteriores = _ultimas_odds(alhist := alvo_hist)
    novas = 0
    with open(str(alhist), "a", encoding="utf-8") as fh:
        for l in dados["linhas"]:
            k = chave_da_linha(l)
            if anteriores.get(k) == l["odd"]:
                continue
            fh.write(json.dumps({"lido_em": lido_em, "chave": k, **l},
                                ensure_ascii=False) + "\n")
            novas += 1

    print("[odds] {} serie(s) de {}, {} linha(s) de aposta lidas.".format(
        len(dados["jogos"]), args.jogo.upper(), len(dados["linhas"])))
    print("[odds] derivados: {} chamada(s), {} de {} jogo(s) proximo(s) lidos{}".format(
        derivados["chamadas"], derivados.get("jogos_lidos", 0), derivados["jogos_olhados"],
        " — PAREI no meio: a casa pediu para parar" if derivados["parado"] else ""))
    _por_mercado = {}
    for l in dados["linhas"]:
        _por_mercado[l.get("mercado", "?")] = _por_mercado.get(l.get("mercado", "?"), 0) + 1
    for _m, _n in sorted(_por_mercado.items(), key=lambda kv: -kv[1])[:10]:
        print("    {:>4}  {}".format(_n, _m))
    # PARTIDAS: quem jogava, uma vez por partida. O arquivo de precos so tem o
    # ID; sem isto, saber quem era exige escavar o historico do git.
    ja = set()
    if partidas.exists():
        try:
            with open(str(partidas), "r", encoding="utf-8") as fh:
                for linha in fh:
                    try:
                        ja.add(str(json.loads(linha).get("id")))
                    except (ValueError, AttributeError):
                        continue
        except OSError:
            pass
    partidas_novas = 0
    try:
        with open(str(partidas), "a", encoding="utf-8") as fh:
            for j in dados["jogos"]:
                if str(j.get("id")) in ja:
                    continue
                fh.write(json.dumps({"visto_em": lido_em, **j},
                                    ensure_ascii=False) + chr(10))
                partidas_novas += 1
    except OSError as e:
        # Perder o registro de quem jogava nao pode derrubar a coleta de preco.
        print("[odds] nao consegui gravar as partidas: {}".format(e))

    print("[odds] {} preco(s) mudaram e entraram no historico.".format(novas))
    print("[odds] {} partida(s) novas registradas com os nomes dos times.".format(
        partidas_novas))
    for j in dados["jogos"][:8]:
        print("    {:<14} {:<20} x {:<20} BO{}".format(
            j["liga"][:14], j["casa"][:20], j["fora"][:20], j["bo"]))

    # MERCADO NOVO NAO PODE SUMIR CALADO. Quando a Pinnacle inventa um
    # mercado, o nome dele nao bate com nada e a linha vira dado morto. O
    # diario passa a dizer quantos e quais, para virar uma linha de tabela na
    # proxima vez -- nunca um chute.
    _nao_lidos = {}
    try:
        from app_core.mercado_pinnacle_core import nao_entendidos
        _nao_lidos = nao_entendidos([l.get("mercado") for l in dados["linhas"]])
        if _nao_lidos:
            print("[odds] {} mercado(s) que eu nao sei ler:".format(len(_nao_lidos)))
            for _n, _q in sorted(_nao_lidos.items(), key=lambda kv: -kv[1])[:8]:
                print("    {:>4}  {}".format(_q, _n or "(sem nome)"))
    except Exception:
        _nao_lidos = {}

    anotar_corrida(
        diario, resultado="ok", rc=RC_OK,
        mercados_sem_ler=len(_nao_lidos),
        nomes_sem_ler=sorted(_nao_lidos)[:10],
        jogos=len(dados["jogos"]), linhas=len(dados["linhas"]),
        precos_novos=novas,
        kills=sum(1 for l in dados["linhas"]
                  if str(l.get("unidade") or "").lower() == "kills"),
        derivados_chamadas=derivados["chamadas"],
        derivados_parado=bool(derivados["parado"]),
        # Quantos jogos pediam varredura e quantos o teto deixou ler: e isto
        # que diz se o teto ainda corta a fila.
        derivados_jogos_olhados=derivados.get("jogos_olhados", 0),
        derivados_jogos_lidos=derivados.get("jogos_lidos", 0),
        recusas_401=derivados.get("recusas_401", 0),
        jogo=args.jogo,
    )
    return RC_OK


if __name__ == "__main__":
    raise SystemExit(main())
