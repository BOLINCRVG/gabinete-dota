# -*- coding: utf-8 -*-
"""Casar sozinho o nome de um time -- so quando nao ha como errar.

O QUE ELE DISSE (23/09/2026): "ele ainda nao pega os times de forma correta,
alguns times eu ainda tenho que indicar os nomes".

A REGRA ANTIGA nao adivinha, de proposito: casa so nome exato ou apelido
ensinado. Isso veio de um erro caro -- "UP2U Meavedron" foi casado com
"UP2UMEDIA Cebulaki", time de OUTRA liga, e a analise saiu inteira sobre o
time errado. Esta regra NAO desfaz essa protecao: ela so entra quando a
antiga desiste, e so decide quando sobra UM candidato depois de tres filtros.

OS TRES FILTROS, e o que cada um pegou na medicao:

  1. NOME: a chave tem de comecar, inteira, no inicio de alguma palavra do
     nome do CSV. "liquid" acha "Team Liquid"; "nong" acha "Nongshim
     RedForce". Nunca o contrario: "t1academy" NAO acha "T1" -- o time
     principal seria o palpite tentador e errado.

  2. LIGA: o candidato tem de ter jogado, no ultimo ano, alguma liga em comum
     com o ADVERSARIO ja casado. Foi a liga que desempatou todos os casos
     perigosos da conferencia de 21/09/2026: G2 Esports x G2 NORD, Invictus
     Gaming x Macko Invictus, paiN Gaming x paiN Academy.

  3. ATIVO: o candidato tem de ter jogado nos ultimos 60 dias. Sem isso a
     regra ERROU um caso real: "losheretics" achava "Los Heretics", que
     parou em 21/11/2025 -- o time foi RENOMEADO para "Team Heretics
     Academy", que estreou em 13/02/2026. Os 60 dias nao sao chute: na
     primeira partida do nome novo, o velho estava parado ha 84 dias. Um
     limite de 90 ou 180 teria errado ali.

MEDIDO contra os 52 apelidos que ele ensinou (tools/eval_casar_times.py),
fingindo que nao tinham sido ensinados, no cenario da tela (liga vinda do
adversario):

    acertaria sozinho ... 36
    deixaria para ele ... 16   (como hoje)
    erraria ............. 0

O QUE ESTE MODULO NAO FAZ: gravar. O time casado sozinho e usado na hora e
mostrado na tela como tal, com o motivo; vira apelido so se ele confirmar.
Um palpite gravado sem ele ver viraria permanente e invisivel.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# Chave com menos letras que isto casa com meio CSV.
MINIMO_DE_LETRAS = 2

# Candidato parado ha mais que isto nao decide sozinho. Ver o cabecalho: e o
# unico limite, entre os medidos, que protege a troca de nome real que houve.
DIAS_ATIVO = 60


def palavras(nome: Any) -> List[str]:
    """Minusculas, sem acento, so letras e numeros, separadas por palavra."""
    t = unicodedata.normalize("NFKD", str(nome or ""))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return [p for p in re.split(r"[^a-z0-9]+", t) if p]


def chave(nome: Any) -> str:
    """A forma comparavel do nome, sem o artigo "the" do comeco.

    "The Otter Side" / "theotterside" -> "otterside": sem tirar o artigo,
    "theotter" nunca acharia "Otter Side".
    """
    ps = palavras(nome)
    if len(ps) > 1 and ps[0] == "the":
        ps = ps[1:]
    elif len(ps) == 1 and ps[0].startswith("the") and len(ps[0]) > 5:
        ps = [ps[0][3:]]
    return "".join(ps)


def candidatos(cru: Any, times: Iterable[str]) -> List[str]:
    """Times cujo nome, a partir do INICIO de alguma palavra, comeca com a
    chave inteira. Nunca o contrario (chave do time dentro da crua)."""
    k = chave(cru)
    if len(k) < MINIMO_DE_LETRAS:
        return []
    fora = []
    for t in times:
        ps = palavras(t)
        for i in range(len(ps)):
            if "".join(ps[i:]).startswith(k):
                fora.append(str(t))
                break
    return fora


def _como_data(v: Any) -> Optional[date]:
    if v is None or v != v:
        return None
    if isinstance(v, datetime):
        d = v.date()
        return d if d == d else None
    if isinstance(v, date):
        return v
    try:
        return datetime.fromisoformat(str(v)[:19]).date()
    except ValueError:
        return None


def casar_sozinho(
    cru: Any,
    times: Sequence[str],
    ligas_do_time: Dict[str, Set[str]],
    ligas_do_adversario: Optional[Iterable[str]],
    ultimo_jogo: Dict[str, Any],
    hoje: Any,
    dias_ativo: int = DIAS_ATIVO,
) -> Optional[str]:
    """O time do CSV, ou None. None e abster -- a decisao fica com ele.

    Sem as ligas do adversario NAO decide: sem liga a medicao errou um caso
    em 52, e o criterio aqui e zero.
    """
    if not ligas_do_adversario:
        return None
    adv = {str(x) for x in ligas_do_adversario if str(x or "").strip()}
    if not adv:
        return None
    ref = _como_data(hoje)
    if ref is None:
        return None
    sobraram = []
    for c in candidatos(cru, times):
        if not adv & set(ligas_do_time.get(c) or ()):
            continue
        ultimo = _como_data(ultimo_jogo.get(c))
        if ultimo is None or (ref - ultimo).days > int(dias_ativo):
            continue
        sobraram.append(c)
    return sobraram[0] if len(sobraram) == 1 else None


def motivo(cru: Any, time: str, adversario: str) -> str:
    """O texto que a tela mostra ao lado de um casamento automatico."""
    return ("casei sozinho: \"%s\" = %s (unico nome parecido que joga a "
            "mesma liga de %s e esta ativo). Confira." % (cru, time, adversario))


def completar_par(cru_a: Any, res_a: str, cru_b: Any, res_b: str,
                  times: Sequence[str], ligas_do_time: Dict[str, Set[str]],
                  ultimo_jogo: Dict[str, Any], hoje: Any = None):
    """Um lado casou e o outro nao? Tenta o outro pela liga do que casou.

    Devolve (res_a, res_b, automaticos), onde `automaticos` lista
    (lado, nome_cru, time_escolhido, adversario) do que foi decidido aqui --
    para a tela MOSTRAR, nunca esconder.

    Com os dois vazios nao faz nada: sem um lado casado nao ha liga para
    desempatar, e sem liga a medicao errou.

    `hoje` padrao e a data do jogo mais recente do CSV, e nao o relogio: foi
    assim que a regra foi medida, e o CSV dele costuma andar um dia atras --
    usar o relogio deixaria todo time um dia "mais parado" do que na medicao.
    """
    res_a, res_b = str(res_a or ""), str(res_b or "")
    automaticos = []
    if bool(res_a) == bool(res_b):
        return res_a, res_b, automaticos
    if hoje is None:
        datas = [d for d in (_como_data(v) for v in (ultimo_jogo or {}).values()) if d]
        hoje = max(datas) if datas else None
    if hoje is None:
        return res_a, res_b, automaticos
    if res_a and not res_b and str(cru_b or "").strip():
        achado = casar_sozinho(cru_b, times, ligas_do_time,
                               ligas_do_time.get(res_a), ultimo_jogo, hoje)
        if achado and achado != res_a:
            res_b = achado
            automaticos.append(("Time 2", str(cru_b), achado, res_a))
    elif res_b and not res_a and str(cru_a or "").strip():
        achado = casar_sozinho(cru_a, times, ligas_do_time,
                               ligas_do_time.get(res_b), ultimo_jogo, hoje)
        if achado and achado != res_b:
            res_a = achado
            automaticos.append(("Time 1", str(cru_a), achado, res_b))
    return res_a, res_b, automaticos
