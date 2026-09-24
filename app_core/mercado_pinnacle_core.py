# -*- coding: utf-8 -*-
"""O que cada mercado da Pinnacle E, alem do nome que ela deu a ele.

O PROBLEMA (21/09/2026). Ele perguntou se conseguimos pegar as outras odds da
Pinnacle. A auditoria do historico respondeu: sim -- 106 tipos de mercado,
14.937 leituras, com torres, dragoes, baroes, inibidores, primeiro sangue,
primeiro a N abates e par/impar. Nada esta faltando na coleta.

O que falta e outra coisa. Quando a casa da nome ao mercado, o coletor guarda
esse nome LETRA POR LETRA, de proposito: e o que ele compara com a tela da
Pinnacle. So que o nome vem em ingles e com a forma dela --

    "(Map 3) Total Turrets Destroyed"
    "(Map 1) 1st To 15 Kills"
    "(Map 2) Both Teams To Kill An Elemental Dragon?"

-- e o resto do programa fala outra lingua ("towers", "kills", mapa 3). Com
isso, 3.430 leituras ficaram invisiveis para a aba de Oportunidades, entre
elas TORRES, que e um dos dois mercados em que ele ganha dinheiro.

A SOLUCAO NAO E RENOMEAR. Trocar o nome da casa quebraria a unica coisa que
permite a ele conferir a linha na tela dela. O que este modulo faz e traduzir
AO LADO: o nome original fica, e nascem campos que o programa entende.

POR QUE LER, E NAO SO GRAVAR DAQUI PARA A FRENTE: as 14.937 leituras ja
gravadas nao tem esses campos. Como isto e uma funcao pura do nome, quem le o
historico antigo chama aqui e obtem o mesmo resultado -- sem precisar
reescrever arquivo nenhum.

E O QUE ELE NAO FAZ: nao adivinha. Nome que nao bate com nenhum padrao sai com
`metrica=None`, e quem usa decide o que fazer. Chutar "deve ser kills" poria
uma aposta de dragao na conta de abates (PRINCIPIO IV).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

# A metrica no vocabulario que o resto do programa ja usa (o mesmo do
# picks_tracker e do modelo), para nao nascer um terceiro dialeto.
KILLS = "kills"
TOWERS = "towers"
DRAGONS = "dragons"
BARONS = "barons"
INHIBITORS = "inhibitors"
FIRSTBLOOD = "firstblood"
ELDER = "elder"
VENCEDOR = "ml"
PLACAR = "placar"

# A FORMA da aposta: o que se esta perguntando sobre a metrica.
TOTAL = "total"              # acima/abaixo de N
TOTAL_TIME = "total_time"    # acima/abaixo de N, so de um time
HANDICAP = "handicap"
PRIMEIRO = "primeiro"        # quem faz o primeiro (baron, inibidor, sangue)
CORRIDA = "corrida"          # quem chega primeiro a N abates
AMBOS = "ambos"              # os dois times fazem?
ALGUM = "algum"              # algum time faz?
PAR_IMPAR = "par_impar"
VENCE = "vence"

_MAPA = re.compile(r"^\(Map\s+(\d+)\)\s*(.+)$", re.IGNORECASE)
_MAPA_PT = re.compile(r"\bmapa\s+(\d+)\b", re.IGNORECASE)
_CORRIDA = re.compile(r"^1st\s+To\s+(\d+)\s+Kills?$", re.IGNORECASE)

# Nome cru da casa (sem o "(Map N)") -> (metrica, forma, nome em portugues).
# Escrito na mao, um por um, a partir do que REALMENTE apareceu no historico.
# Uma tabela e mais chata que uma regra esperta, e e exatamente por isso que
# ela nao inventa mercado que ninguem viu.
_TABELA = {
    "total turrets destroyed": (TOWERS, TOTAL, "Torres - total"),
    "total barons killed": (BARONS, TOTAL, "Baroes - total"),
    "total elemental dragons killed": (DRAGONS, TOTAL, "Dragoes - total"),
    "total kills odd/even": (KILLS, PAR_IMPAR, "Abates - par ou impar"),
    "1st inhibitor": (INHIBITORS, PRIMEIRO, "Primeiro inibidor"),
    "1st baron": (BARONS, PRIMEIRO, "Primeiro barao"),
    "1st blood": (FIRSTBLOOD, PRIMEIRO, "Primeiro sangue"),
    "both teams to kill an elemental dragon?": (DRAGONS, AMBOS,
                                                "Os dois times matam dragao?"),
    "both teams to kill a baron?": (BARONS, AMBOS, "Os dois times matam barao?"),
    "both teams to destroy an inhibitor?": (INHIBITORS, AMBOS,
                                            "Os dois times derrubam inibidor?"),
    "will either team slay the elder dragon?": (ELDER, ALGUM,
                                                "Algum time mata o Elder?"),
    # DOTA (vistos ao vivo em 23/09/2026, PGL Wallachia). O motor do Dota e o
    # mesmo do LoL e fala a lingua dele: Roshan mora em "barons" e barraca em
    # "inhibitors" (ver dota_totals_scorecard.py no repositorio do Dota).
    # Este arquivo e identico nos dois repositorios.
    "1st roshan": (BARONS, PRIMEIRO, "Primeiro Roshan"),
    "1st barracks": (INHIBITORS, PRIMEIRO, "Primeira barraca"),
}

# Os nomes que o PROPRIO coletor monta (ver tools/coletar_odds.rotulo_da_linha)
# quando a casa nao da descricao. Esses ja vem em portugues.
_NOSSOS = (
    ("total do time - kills", (KILLS, TOTAL_TIME)),
    ("handicap - kills", (KILLS, HANDICAP)),
    ("vencedor - kills", (KILLS, VENCE)),
    ("total - kills", (KILLS, TOTAL)),
    ("handicap", (VENCEDOR, HANDICAP)),
    ("vencedor", (VENCEDOR, VENCE)),
    ("total", (VENCEDOR, TOTAL)),
)


def _vazio(nome: Any) -> Dict[str, Any]:
    return {"nome_casa": str(nome or ""), "metrica": None, "forma": None,
            "mapa": None, "alvo": None, "nome_pt": str(nome or ""),
            "entendido": False}


def entender_mercado(nome: Any) -> Dict[str, Any]:
    """O que este mercado e, em campos que o programa entende.

    Devolve sempre o `nome_casa` intacto. `metrica` fica None quando o nome
    nao bate com nada conhecido -- e ai quem chama sabe que nao sabe.

    `mapa` e o numero do mapa (1..5) ou 0 para a serie inteira; None quando o
    nome nao diz.
    """
    texto = str(nome or "").strip()
    if not texto:
        return _vazio(nome)
    fora = _vazio(texto)

    mapa: Optional[int] = None
    resto = texto
    m = _MAPA.match(texto)
    if m:
        try:
            mapa = int(m.group(1))
        except ValueError:
            mapa = None
        resto = m.group(2).strip()
    else:
        mp = _MAPA_PT.search(texto)
        if mp:
            try:
                mapa = int(mp.group(1))
            except ValueError:
                mapa = None
        elif re.search(r"\bs[eé]rie\b", texto, re.IGNORECASE):
            mapa = 0
    fora["mapa"] = mapa

    baixo = resto.lower().strip()

    corrida = _CORRIDA.match(resto.strip())
    if corrida:
        try:
            alvo = int(corrida.group(1))
        except ValueError:
            alvo = None
        fora.update({"metrica": KILLS, "forma": CORRIDA, "alvo": alvo,
                     "entendido": True,
                     "nome_pt": "Primeiro a %s abates" % (alvo if alvo is not None else "?")})
        return _com_mapa(fora)

    if baixo in _TABELA:
        met, forma, pt = _TABELA[baixo]
        fora.update({"metrica": met, "forma": forma, "nome_pt": pt,
                     "entendido": True})
        return _com_mapa(fora)

    if baixo == "correct score":
        fora.update({"metrica": PLACAR, "forma": VENCE, "mapa": 0,
                     "nome_pt": "Placar exato da serie", "entendido": True})
        return fora

    # Os rotulos que nos mesmos montamos ("Vencedor - mapa 1", "Total - Kills
    # - mapa 3"). A ordem importa: "total do time - kills" antes de "total".
    for prefixo, (met, forma) in _NOSSOS:
        if baixo.startswith(prefixo):
            fora.update({"metrica": met, "forma": forma, "entendido": True,
                         "nome_pt": texto})
            return fora

    return fora


def _com_mapa(d: Dict[str, Any]) -> Dict[str, Any]:
    if d.get("mapa"):
        d["nome_pt"] = "%s - mapa %d" % (d["nome_pt"], int(d["mapa"]))
    return d


def resumir(nomes) -> Dict[str, int]:
    """Quantos mercados foram entendidos e quantos nao. Para o diario."""
    total = entendidos = 0
    for n in nomes or []:
        total += 1
        if entender_mercado(n).get("entendido"):
            entendidos += 1
    return {"total": total, "entendidos": entendidos,
            "sem_entender": total - entendidos}


def nao_entendidos(nomes) -> Dict[str, int]:
    """Os nomes que ninguem soube ler, com quantas vezes cada um apareceu.

    E daqui que sai o trabalho da proxima vez: a casa cria mercado novo, ele
    aparece aqui, e a tabela ganha uma linha. Nunca um chute.
    """
    fora: Dict[str, int] = {}
    for n in nomes or []:
        if not entender_mercado(n).get("entendido"):
            chave = str(n or "")
            fora[chave] = fora.get(chave, 0) + 1
    return fora
