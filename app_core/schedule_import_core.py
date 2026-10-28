"""Parsing and alias helpers for agenda text import."""

from __future__ import annotations

import difflib
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List, MutableMapping, Tuple

from app_core.schedule_state_core import (
    ensure_schedule_alias_state,
    save_schedule_aliases as save_schedule_alias_state,
)


# Letras que a decomposicao Unicode NAO separa em letra + acento (23/10/2026).
_LETRAS_SEM_DECOMPOSICAO = {"ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "ł": "l",
                            "đ": "d", "ð": "d", "þ": "th", "ı": "i"}


def _chave_antiga(value: str) -> str:
    """A normalizacao de ate 23/10/2026, mantida so para achar apelidos salvos
    com ela. As trocas que ela tinha eram texto UTF-8 lido como latin-1 e nunca
    casavam: letra acentuada era simplesmente APAGADA."""
    text = (value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_schedule_team_key(value: str) -> str:
    """Chave de comparacao de nome de time: sem acento, sem espaco, minuscula.

    23/10/2026: "LØS" (CSV) e "LOS" (agenda) davam "ls" e "los" -- o Ø era
    apagado -- e o Mundial pedia para casar o time na mao, toda vez."""
    text = (value or "").strip().lower()
    text = "".join(_LETRAS_SEM_DECOMPOSICAO.get(c, c) for c in text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text)


def dedup_schedule_repeat_token(value: str) -> str:
    text = html.unescape((value or "").strip())
    compact = re.sub(r"\s+", "", text)
    size = len(compact)
    if size > 0 and size % 2 == 0:
        half = size // 2
        if compact[:half].lower() == compact[half:].lower():
            return compact[:half].strip()
    return text


def extract_schedule_league_header(line: str) -> str:
    text = html.unescape((line or "").strip())
    if not text:
        return ""
    text = text.replace("League Icon", " ").replace("Flag Icon", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("Flag Icon", " ").strip()
    if len(text) < 2:
        return ""
    return text


def _is_schedule_separator(line: str) -> bool:
    text = html.unescape((line or "").strip())
    if not text:
        return True
    if text in {"-", "–", "—", "â€“", "â€”"}:
        return True
    if not re.search(r"[A-Za-z0-9À-ÿ]", text):
        return True
    return text.strip("-–— ") == ""


def parse_schedule_text(text_in: str) -> list[dict[str, Any]]:
    lines = [html.unescape(line).strip() for line in (text_in or "").splitlines()]
    lines = [line for line in lines if line]
    games: list[dict[str, Any]] = []
    cur_date = ""
    cur_league = ""
    date_re = re.compile(r"(\b\d{2}\/\d{2}\b)")
    time_re = re.compile(r"(\b\d{2}\:\d{2}\b)")
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        m_date = date_re.search(line)
        if m_date:
            cur_date = m_date.group(1)
            idx += 1
            continue

        m_time = time_re.search(line)
        if m_time:
            cur_time = m_time.group(1)
            j = idx + 1
            block_groups: list[list[str]] = []
            current_group: list[str] = []

            while j < len(lines):
                probe = lines[j]
                if date_re.search(probe) or time_re.search(probe):
                    break
                if _is_schedule_separator(probe):
                    if current_group:
                        block_groups.append(current_group)
                        current_group = []
                    j += 1
                    continue
                token = dedup_schedule_repeat_token(probe)
                if token and (not current_group or token != current_group[-1]):
                    current_group.append(token)
                j += 1

            if current_group:
                block_groups.append(current_group)

            team_tokens: list[str] = [group[0] for group in block_groups if group]
            if len(team_tokens) < 2:
                flat_tokens: list[str] = []
                for group in block_groups:
                    for token in group:
                        if token and (not flat_tokens or token != flat_tokens[-1]):
                            flat_tokens.append(token)
                team_tokens = flat_tokens

            if len(team_tokens) >= 2:
                games.append(
                    {
                        "date": cur_date,
                        "time": cur_time,
                        "league": cur_league,
                        "teamA_raw": team_tokens[0],
                        "teamB_raw": team_tokens[1],
                    }
                )
            idx = j
            continue

        league_candidate = extract_schedule_league_header(line)
        if league_candidate:
            cur_league = league_candidate
        idx += 1

    return games


def load_schedule_aliases(session_state: MutableMapping[str, Any], alias_path: Path) -> dict[str, str]:
    cached = session_state.get("team_aliases_user")
    if isinstance(cached, dict) and cached:
        return {str(k): str(v) for k, v in cached.items()}
    aliases = ensure_schedule_alias_state(session_state)
    if alias_path.exists():
        try:
            data = json.loads(alias_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                aliases = save_schedule_alias_state(session_state, data)
        except Exception:
            aliases = save_schedule_alias_state(session_state, aliases)
    return dict(aliases)


def save_schedule_aliases(session_state: MutableMapping[str, Any], alias_path: Path, aliases: dict[str, Any]) -> dict[str, str]:
    normalized = save_schedule_alias_state(session_state, aliases)
    alias_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    # Na nuvem o disco nao sobrevive a reinicio: manda para o GitHub tambem.
    try:
        from app_core.viewer_mode_core import is_viewer_mode, publicar_apelidos
        if is_viewer_mode():
            # O resultado fica na sessao e a tela mostra (23/10/2026): um envio
            # que falhava calado deixou ele achando que o apelido estava salvo.
            ok, msg = publicar_apelidos(dict(normalized))
            session_state["_apelido_envio"] = {"ok": bool(ok), "msg": str(msg)}
    except Exception as exc:
        try:
            session_state["_apelido_envio"] = {"ok": False, "msg": f"{type(exc).__name__}: {exc}"}
        except Exception:
            pass
    return dict(normalized)


def upsert_schedule_alias(
    session_state: MutableMapping[str, Any],
    alias_path: Path,
    raw: str,
    canonical_team: str,
) -> dict[str, str]:
    aliases = load_schedule_aliases(session_state, alias_path)
    key = normalize_schedule_team_key(raw)
    if key and canonical_team:
        aliases[key] = str(canonical_team)
    return save_schedule_aliases(session_state, alias_path, aliases)


def resolve_schedule_team(raw: str, teams_all: Iterable[str], aliases: dict[str, str]) -> Tuple[str, bool]:
    team = (raw or "").strip()
    teams = [str(v) for v in teams_all if str(v or "").strip()]
    if not team:
        return "", False
    key = normalize_schedule_team_key(team)
    # APELIDO ANTES do nome exato (27/10/2026): no Dota a Pinnacle diz "PVISION",
    # que existe no CSV -- mas e o nome ANTIGO do PARIVISION (mesmo team_id, ultimo
    # jogo 19/07). Com o nome exato primeiro, o apelido ensinado nunca era lido.
    for k in (key, _chave_antiga(team)):
        if k in aliases and str(aliases[k]) in teams:
            return str(aliases[k]), True
    if team in teams:
        return team, True
    for known in teams:
        if normalize_schedule_team_key(known) == key:
            return known, True
    # PREFIXO "TEAM" (27/10/2026): a casa diz "Liquid", "Spirit", "Yandex"; o CSV,
    # "Team Liquid"... Casa so se houver UMA opcao -- na duvida, nao chuta.
    alt = key[4:] if key.startswith("team") else "team" + key
    if alt:
        cands = sorted({known for known in teams if normalize_schedule_team_key(known) == alt})
        if len(cands) == 1:
            return cands[0], True
    # SUFIXO "GAMING"/"ESPORTS" (28/10/2026): a OddsPapi diz "Lyon Gaming", o CSV "LYON".
    # Mesma trava do prefixo: tira o sufixo dos DOIS lados e casa so com UMA opcao.
    base = _sem_sufixo(key)
    if base:
        cands = sorted({known for known in teams if _sem_sufixo(normalize_schedule_team_key(known)) == base})
        if len(cands) == 1:
            return cands[0], True
    return team, False


_SUFIXOS = ("esports", "esport", "gaming", "club", "gg")


def _sem_sufixo(key: str) -> str:
    for suf in _SUFIXOS:
        if key.endswith(suf) and len(key) > len(suf) + 1:
            return key[: -len(suf)]
    return key


def suggest_schedule_teams(raw: str, teams_all: Iterable[str], *, limit: int = 60,
                           prioridade: Iterable[str] | None = None) -> List[str]:
    """Palpites de qual time do CSV e esse nome da agenda.

    `prioridade` sao os times que jogam na MESMA LIGA do confronto. Eles sobem
    para o topo da lista sem sair dela -- a ordem muda, nada e escondido.

    POR QUE ISSO EXISTE (16/09/2026, achado depois de ele estranhar uma
    analise): num jogo do Hitpoint Masters (Tchequia) a agenda trazia "UP2U
    Meavedron". A lista ofereceu em PRIMEIRO "UP2UMEDIA Cebulaki" -- time
    POLONES, da Rift Legends, com 4 jogos de historico -- so porque o nome
    comeca parecido. O time certo, "Meavedron", com 63 jogos, ficou embaixo.
    Ele escolheu o primeiro, e a analise inteira saiu sobre o time errado.

    Nomes parecidos entre ligas nao sao coincidencia: sao o mesmo
    patrocinador em paises diferentes. A liga desempata.
    """
    teams = [str(v) for v in teams_all if str(v or "").strip()]
    team = (raw or "").strip()
    if not team:
        return teams[:limit]
    key = normalize_schedule_team_key(team)
    pref = key[:2] if len(key) >= 2 else key
    starts = [item for item in teams if pref and normalize_schedule_team_key(item).startswith(pref)]
    fuzzy = difflib.get_close_matches(team, teams, n=20, cutoff=0.25)
    contains = [item for item in teams if pref and pref in normalize_schedule_team_key(item)]
    out: list[str] = []
    for group in (starts, fuzzy, contains):
        for item in group:
            if item not in out:
                out.append(item)
    out = out[:limit] if out else teams[:limit]

    da_liga = {str(x) for x in (prioridade or []) if str(x or "").strip()}
    if not da_liga:
        return out
    # Os da liga do confronto primeiro, na ordem em que ja estavam. Os outros
    # continuam logo abaixo: quem manda na escolha e ele, nao a ordenacao.
    return ([x for x in out if x in da_liga] + [x for x in out if x not in da_liga])
