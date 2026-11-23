# -*- coding: utf-8 -*-
"""DRAFT COLADO (23/11/2026): ele cola os herois, o app acha quem joga cada um e diz POR QUE o draft
mexe no preco. Ele: "como que voce vai decidir que aquele draft e melhor para o time A, atraves de X
player, [...] de um boneco que ganha todo jogo? atraves do que?"

Resposta MEDIDA (23/11/2026, as cegas, teste 2026): no Dota so mexem no preco a forca recente do heroi
e o jogador naquele heroi (conforto e pratica) -- a camada de app_core/camada_draft_core.py. Counter,
dupla, time no heroi, ultimo pick, ban no heroi-assinatura e estilo NAO melhoraram a previsao (por isso
nao aparecem como motivo).

Motivos somam EXATO: logit(p_pos) - logit(p_pre) = lado + soma das partes de cada jogador; cada parte
vira pontos percentuais na proporcao da sua fatia do logit. Arquivo identico nos dois repositorios.
"""
from __future__ import annotations

import itertools
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app_core import camada_draft_core as CD

# Apelidos comuns (chave normalizada -> nome oficial). O resto casa pelo nome normalizado.
APELIDOS_DOTA = {
    "am": "Anti-Mage", "antimage": "Anti-Mage", "fv": "Faceless Void", "void": "Faceless Void",
    "wr": "Windranger", "wd": "Witch Doctor", "cm": "Crystal Maiden", "es": "Earthshaker",
    "et": "Elder Titan", "ds": "Dark Seer", "dk": "Dragon Knight", "ck": "Chaos Knight",
    "pa": "Phantom Assassin", "pl": "Phantom Lancer", "ta": "Templar Assassin", "sf": "Shadow Fiend",
    "ss": "Shadow Shaman", "sd": "Shadow Demon", "qop": "Queen of Pain", "qope": "Queen of Pain",
    "np": "Nature's Prophet", "furion": "Nature's Prophet", "prophet": "Nature's Prophet",
    "od": "Outworld Devourer", "outworlddestroyer": "Outworld Devourer", "obsidian": "Outworld Devourer",
    "ld": "Lone Druid", "sb": "Spirit Breaker", "bara": "Spirit Breaker", "bh": "Bounty Hunter",
    "bs": "Bloodseeker", "bb": "Bristleback", "cent": "Centaur Warrunner", "centaur": "Centaur Warrunner",
    "tb": "Terrorblade", "ns": "Night Stalker", "ls": "Lifestealer",
    "naix": "Lifestealer", "lc": "Legion Commander", "kotl": "Keeper of the Light", "aa": "Ancient Apparition",
    "ww": "Winter Wyvern", "wk": "Wraith King", "sk": "Sand King", "ok": "Ogre Magi", "ogre": "Ogre Magi",
    "abba": "Abaddon", "alch": "Alchemist", "bm": "Beastmaster", "brew": "Brewmaster", "clock": "Clockwerk",
    "cw": "Clockwerk", "doom": "Doom", "ench": "Enchantress", "gyro": "Gyrocopter", "invo": "Invoker",
    "jug": "Juggernaut", "jugg": "Juggernaut", "kunkka": "Kunkka", "lesh": "Leshrac", "lina": "Lina",
    "lion": "Lion", "luna": "Luna", "mag": "Magnus", "magnataur": "Magnus", "mk": "Monkey King",
    "morph": "Morphling", "nyx": "Nyx Assassin", "pango": "Pangolier", "pudge": "Pudge", "riki": "Riki",
    "sniper": "Sniper", "spec": "Spectre", "slark": "Slark", "storm": "Storm Spirit", "ember": "Ember Spirit",
    "earth": "Earth Spirit", "void spirit": "Void Spirit", "voidspirit": "Void Spirit", "tink": "Tinker",
    "treant": "Treant Protector", "tusk": "Tusk", "veno": "Venomancer", "viper": "Viper", "visage": "Visage",
    "weaver": "Weaver", "zeus": "Zeus", "io": "Io", "wisp": "Io", "marci": "Marci", "muerta": "Muerta",
    "primal": "Primal Beast", "pb": "Primal Beast", "dawn": "Dawnbreaker", "dawnbreaker": "Dawnbreaker",
    "hood": "Hoodwink", "snap": "Snapfire", "grim": "Grimstroke", "rubick": "Rubick", "techies": "Techies",
    "timber": "Timbersaw", "tiny": "Tiny", "troll": "Troll Warlord", "undying": "Undying", "ursa": "Ursa",
    "vs": "Vengeful Spirit", "venge": "Vengeful Spirit", "dazzle": "Dazzle", "disruptor": "Disruptor",
    "naga": "Naga Siren", "medusa": "Medusa", "dusa": "Medusa", "meepo": "Meepo", "huskar": "Huskar",
    "razor": "Razor", "sven": "Sven", "skywrath": "Skywrath Mage", "sky": "Skywrath Mage",
    "silencer": "Silencer", "jakiro": "Jakiro", "jak": "Jakiro", "batrider": "Batrider", "bat": "Batrider",
    "chen": "Chen", "enigma": "Enigma", "mirana": "Mirana", "potm": "Mirana", "puck": "Puck",
    "phoenix": "Phoenix", "oracle": "Oracle", "lich": "Lich", "lycan": "Lycan", "mars": "Mars",
    "kez": "Kez", "ringmaster": "Ringmaster",
}

_SEP = re.compile(r"[,;|\t/]+|\s{2,}")


def catalogo_dota(herois: Mapping[str, str]) -> Dict[str, str]:
    """herois_dota.json {id: nome} -> {chave normalizada: id}, com os apelidos."""
    por_nome = {CD.chave_campeao(n): str(i) for i, n in herois.items()}
    out = dict(por_nome)
    for ap, nome in APELIDOS_DOTA.items():
        i = por_nome.get(CD.chave_campeao(nome))
        if i:
            out.setdefault(CD.chave_campeao(ap), i)
    return out


def _tokens(linha: str) -> List[str]:
    if ":" in linha:  # "Radiant: Puck, Mars" / "Team Spirit: ..."
        linha = linha.split(":", 1)[1]
    return [t.strip() for t in _SEP.split(linha) if t.strip()]


def ler_draft(texto: str, catalogo: Mapping[str, str]) -> Tuple[Optional[Tuple[List[str], List[str]]], List[str], str]:
    """Pura. Texto colado -> ((5 ids lado 1, 5 ids lado 2) ou None, nao reconhecidos, aviso).
    Aceita: duas linhas (uma por time) ou os 10 numa linha so (5 primeiros = lado 1)."""
    linhas = [_tokens(l) for l in str(texto or "").splitlines()]
    linhas = [l for l in linhas if l]
    nao: List[str] = []

    def ids(toks):
        out = []
        for t in toks:
            i = catalogo.get(CD.chave_campeao(t)) or (t if t.isdigit() else None)
            if i:
                out.append(str(i))
            else:
                nao.append(t)
        return out

    if len(linhas) == 2:
        a, b = ids(linhas[0]), ids(linhas[1])
    else:
        todos = ids([t for l in linhas for t in l])
        a, b = todos[:5], todos[5:]
    if nao:
        return None, nao, "Não reconheci: " + ", ".join(nao) + ". Confira o nome do herói."
    if len(a) != 5 or len(b) != 5:
        return None, [], f"Preciso de 5 heróis de cada lado; li {len(a)} + {len(b)}."
    if len(set(a + b)) != 10:
        return None, [], "Há herói repetido no draft."
    return (a, b), [], ""


def atribuir(contas: Sequence[str], herois: Sequence[str], estado: CD.EstadoDraft) -> List[Optional[str]]:
    """Pura. Quem joga cada heroi: a combinacao em que os 5 jogadores somam MAIS jogos nos herois
    (log, para um especialista nao engolir o resto); empate -> mais vitorias. Sem elenco: Nones."""
    cs = [str(c) for c in contas][:5]
    if len(cs) < 5:
        return [None] * len(herois)
    melhor, chave = None, None
    for perm in itertools.permutations(cs):
        jogos = vit = 0.0
        for c, h in zip(perm, herois):
            v, n = estado.jogos_no_campeao(CD.chave_jogador(c), CD.chave_campeao(h))
            jogos += math.log1p(n)
            vit += v
        k = (round(jogos, 9), vit)
        if chave is None or k > chave:
            melhor, chave = perm, k
    return list(melhor)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def explicar(estado: CD.EstadoDraft, coef: Mapping[str, float], azul: Sequence[Tuple[str, str]],
             vermelho: Sequence[Tuple[str, str]], quando: datetime, p_pre_azul: float) -> Dict[str, Any]:
    """Pura. azul/vermelho = [(conta ou '', heroi)] (azul = Radiant). Devolve p_pos e as partes:
    cada jogador com a fatia do heroi e a fatia dele no heroi, em logit e em pontos percentuais
    (positivo = a favor do time DELE). A soma das partes + lado = p_pos - p_pre, exato."""
    chaves = lambda lado: [(CD.chave_jogador(j), CD.chave_campeao(h)) for j, h in lado]  # noqa: E731
    ka, kv = chaves(azul), chaves(vermelho)
    s, faltam = estado.sinais_no_app(ka, kv, quando)
    p_pos = CD.p_pos_draft(p_pre_azul, s, coef)
    usa_jogador = not faltam
    c_ch, c_cf, c_pr = (float(coef.get(k, 0.0)) for k in ("campeao", "conforto", "pratica"))
    # cada parte e medida contra a MEDIA DOS 10 deste jogo (o duelo): um lado soma +X e o outro -X;
    # a diferenca entre os lados -- o que entra no preco -- nao muda.
    def _cf(j, h):
        v, n = estado.jogos_no_campeao(j, h)
        return (v + 0.5 * CD.K_JOGADOR) / (n + CD.K_JOGADOR)
    todos = ka + kv
    wr_med = sum(estado.aproveitamento_campeao(h, quando) for _, h in todos) / 10.0
    cf_med = sum(_cf(j, h) for j, h in todos) / 10.0
    pr_med = sum(math.log1p(estado.jogos_no_campeao(j, h)[1]) for j, h in todos) / 10.0
    partes, z_soma = [], 0.0
    for nome_lado, lado, sinal in (("azul", ka, 1.0), ("vermelho", kv, -1.0)):
        for (j, h) in lado:
            wr_h = estado.aproveitamento_campeao(h, quando)
            v, n = estado.jogos_no_campeao(j, h)
            z_h = c_ch * (wr_h - wr_med) / 5.0
            z_j = 0.0
            if usa_jogador:
                z_j = (c_cf * (_cf(j, h) - cf_med) + c_pr * (math.log1p(n) - pr_med)) / 5.0
            z_soma += sinal * (z_h + z_j)
            partes.append({"lado": nome_lado, "conta": j, "heroi": h, "na_base": estado.conhece(j),
                           "wr_heroi": wr_h, "jogos": int(round(n)), "vitorias": int(round(v)),
                           "z_heroi": z_h, "z_jogador": z_j})
    z_lado = float(coef.get("intercepto", 0.0))
    dz = _logit(p_pos) - _logit(p_pre_azul)          # = z_lado + z_soma
    dp = p_pos - p_pre_azul
    pm = (p_pos + p_pre_azul) / 2.0
    fator = dp / dz if abs(dz) > 1e-9 else pm * (1 - pm)
    for x in partes:
        x["pp_heroi"] = 100.0 * fator * x["z_heroi"]
        x["pp_jogador"] = 100.0 * fator * x["z_jogador"]
        x["pp"] = x["pp_heroi"] + x["pp_jogador"]
    return {"p_pre": p_pre_azul, "p_pos": p_pos, "pp_lado_azul": 100.0 * fator * z_lado,
            "partes": partes, "faltam": faltam, "usa_jogador": usa_jogador}
