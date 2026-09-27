# -*- coding: utf-8 -*-
"""Quao VELHO e o dado de cada time, mostrado na hora de escolher.

ELE PEDIU (21/09/2026), depois de conferirmos 16 apelidos um a um: "vao ter
jogos que os dados vao ser bem velhos, e seria legal avisar em algum lugar na
mesma escolha mesmo ao lado assim 'dado mais recente - (e aqui a data)' e uma
cor diferente para ficar facil identificar".

O CASO QUE MOTIVOU. Entre os apelidos conferidos, "Bilibili Junior" da
Pinnacle e mesmo o "Bilibili Gaming Junior" do CSV -- a Leaguepedia confirma.
Mas o ultimo jogo dele no CSV e de 08/10/2025, quase um ano. O apelido esta
CERTO e a analise vai sair FRACA, e essas duas coisas nao se distinguiam na
tela: os dois apareciam como um nome qualquer na lista.

Outro caso da mesma conferencia, ao contrario: "T1 Academy" tinha dois
candidatos, e o que jogou MAIS RECENTE ("T1 Esports Academy Rookies", 20/09)
era o ERRADO -- quem disputa a WSCI e o "T1 Esports Academy". Por isso a data
aqui e informacao, nunca criterio automatico: ela ajuda a decidir, nao decide.

O SIMBOLO E A COR. O seletor do Streamlit nao pinta opcao individual, entao a
cor vem de um simbolo no comeco do rotulo. E legivel de relance e sobrevive a
copiar e colar.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Em LoL um time em atividade joga toda semana. Tres semanas sem jogo ja e
# pausa de calendario; tres meses e outro campeonato, outra escalacao, e
# muitas vezes outro time com o mesmo nome.
DIAS_FRESCO = 21
DIAS_VELHO = 90

VERDE = "\U0001F7E2"      # em atividade
AMARELO = "\U0001F7E1"    # parado ha algumas semanas
VERMELHO = "\U0001F534"   # dado velho: a analise sai fraca
BRANCO = "⚪"         # sem jogo nenhum no CSV


def _como_data(v: Any) -> Optional[date]:
    """Aceita date, datetime, Timestamp do pandas ou texto ISO."""
    if v is None:
        return None
    # O NaT do pandas E um datetime, e `.date()` nele devolve NaT de novo --
    # uma ausencia que passeia pelo sistema com cara de data. Um teste pegou
    # isso na primeira rodada (PRINCIPIO IV).
    if v != v:
        return None
    if isinstance(v, datetime):
        d = v.date()
        return d if isinstance(d, date) and d == d else None
    if isinstance(v, date):
        return v
    texto = str(v).strip()
    if not texto:
        return None
    try:
        return datetime.fromisoformat(texto[:19].replace("Z", "")).date()
    except ValueError:
        pass
    for formato in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m"):
        try:
            return datetime.strptime(texto[:10], formato).date()
        except ValueError:
            continue
    return None


def ultimo_jogo_por_time(team_games: Any) -> Dict[str, date]:
    """{nome do time no CSV: data do jogo mais recente dele}.

    Le o quadro de jogos por time. Um time que nunca aparece simplesmente nao
    entra no dicionario -- ausencia fica ausencia, nao vira data antiga.
    """
    fora: Dict[str, date] = {}
    if team_games is None:
        return fora
    try:
        colunas = set(getattr(team_games, "columns", []))
        if not {"team", "date"} <= colunas:
            return fora
        for time, quando in zip(team_games["team"], team_games["date"]):
            nome = str(time or "").strip()
            if not nome:
                continue
            d = _como_data(quando)
            if d is None:
                continue
            atual = fora.get(nome)
            if atual is None or d > atual:
                fora[nome] = d
    except (TypeError, KeyError, ValueError):
        return fora
    return fora


def dias_parado(ultimo: Optional[date], hoje: Optional[date] = None) -> Optional[int]:
    if ultimo is None:
        return None
    hoje = hoje or date.today()
    return int((hoje - ultimo).days)


def faixa(dias: Optional[int]) -> str:
    if dias is None:
        return "sem_dado"
    if dias <= DIAS_FRESCO:
        return "fresco"
    if dias <= DIAS_VELHO:
        return "parado"
    return "velho"


def simbolo(dias: Optional[int]) -> str:
    return {"fresco": VERDE, "parado": AMARELO,
            "velho": VERMELHO, "sem_dado": BRANCO}[faixa(dias)]


def texto_do_frescor(ultimo: Optional[date], hoje: Optional[date] = None) -> str:
    """"dado mais recente: 20/09 (ontem)" -- o que ele pediu, nessas palavras."""
    if ultimo is None:
        return "sem jogo no CSV"
    dias = dias_parado(ultimo, hoje)
    quando = ultimo.strftime("%d/%m/%Y")
    if dias is None:
        return "dado mais recente: %s" % quando
    if dias <= 0:
        desde = "hoje"
    elif dias == 1:
        desde = "ontem"
    elif dias < 30:
        desde = "ha %d dias" % dias
    elif dias < 365:
        desde = "ha %d meses" % max(1, round(dias / 30.0))
    else:
        desde = "ha mais de um ano"
    return "dado mais recente: %s (%s)" % (quando, desde)


def rotulo(time: str, ultimos: Dict[str, date],
           hoje: Optional[date] = None) -> str:
    """O texto que aparece na lista de escolha, com o aviso ao lado."""
    nome = str(time or "")
    if not nome:
        return ""
    ultimo = ultimos.get(nome)
    return "%s %s  -  %s" % (simbolo(dias_parado(ultimo, hoje)), nome,
                             texto_do_frescor(ultimo, hoje))


def opcoes(times: Sequence[str], ultimos: Dict[str, date],
           hoje: Optional[date] = None) -> Tuple[List[str], Dict[str, str]]:
    """(rotulos para mostrar, {rotulo: nome de verdade do time}).

    A volta pelo dicionario existe porque o rotulo NAO e o nome do time --
    gravar o rotulo como apelido salvaria "Team Liquid - dado mais recente:
    ..." no lugar de "Team Liquid", e nada casaria depois.
    """
    rotulos: List[str] = []
    de_volta: Dict[str, str] = {}
    for t in times:
        nome = str(t or "")
        if not nome:
            continue
        r = rotulo(nome, ultimos, hoje)
        # Dois times com o mesmo rotulo nao deveriam existir, mas se
        # existirem o primeiro nao pode ser sobrescrito em silencio.
        if r in de_volta:
            r = "%s  [%s]" % (r, nome)
        rotulos.append(r)
        de_volta[r] = nome
    return rotulos, de_volta


def legenda() -> str:
    return ("%s em atividade (ate %d dias)  ·  %s parado (ate %d dias)  ·  "
            "%s dado velho: a analise sai fraca  ·  %s sem jogo no CSV"
            % (VERDE, DIAS_FRESCO, AMARELO, DIAS_VELHO, VERMELHO, BRANCO))


def avisar_se_velho(time: str, ultimos: Dict[str, date],
                    hoje: Optional[date] = None) -> str:
    """Aviso para depois da escolha. Vazio quando o dado esta em dia.

    Separado do rotulo de proposito: o rotulo ajuda a ESCOLHER, este texto
    avisa depois de escolhido, quando o numero ja vai aparecer na tela.
    """
    ultimo = ultimos.get(str(time or ""))
    f = faixa(dias_parado(ultimo, hoje))
    if f == "fresco":
        return ""
    if f == "sem_dado":
        return ("%s %s nao tem nenhum jogo no CSV. Nao ha o que analisar."
                % (BRANCO, time))
    return ("%s %s: %s. O modelo tem pouco dado recente deste time, entao a "
            "analise sai mais fraca do que o numero sugere."
            % (simbolo(dias_parado(ultimo, hoje)), time,
               texto_do_frescor(ultimo, hoje)))
