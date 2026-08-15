"""Página "Radar" — a leitura de apostar com o DRAFT EM MÃOS (design 2026-07-04).

Momento único de uso (B1): o pick&ban fechou, faltam minutos pro mapa. Uma tela:
1) sinais H2H×Liga×Forma dos mercados que ele gira nesse momento (régua B2);
2) odd justa do ML (mesma fonte do entre-mapas);
3) montador de draft (grid por rota + colar) com players e pool FEARLESS
   acumulado por série;
4) leitura do draft: arquétipo (iTero), veredito 3-linhas com disclaimer ~10%,
   fit time×estilo (sempre com n) e ponte draft→mercados (comp_compare Fase 2).

Sem veredito de aposta: fatos verbalizados, o usuário julga (padrão HLTV).
"""
from __future__ import annotations

import difflib
import math
import re
from typing import Any, Mapping, Optional

import pandas as pd

from app_core.champion_knowledge_core import load_champion_knowledge_registry
from app_core.comp_archetype_core import build_comp_profile, compare_comp_profiles
from app_core.live_between_maps_core import resolve_p_base
from app_core.radar_confronto_core import (
    ARCHETYPES,
    RADAR_MARKETS,
    ROLE_ORDER,
    RadarSignal,
    build_radar_signals,
    build_team_windows,
    champion_meta_presence,
    classify_comp_archetype,
    latest_roster,
    map_pattern_signals,
    player_champion_familiarity,
    team_style_history,
    towers_draft_delta,
    towers_draft_signal,
)
from app_core.tab_ui_core import render_tab_hero, render_tab_section
from app_core.team_logos_core import team_logo_html

_COR_A = "#2daac4"
_COR_B = "#8b5cf6"
_ROLE_LABELS = {"top": "TOP", "jng": "JNG", "mid": "MID", "bot": "ADC", "sup": "SUP"}
_FAMILIARITY_OK = 15  # iTero: 0→14 jogos de palco importa, depois satura


def _norm_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


# ----------------------------------------------------------------------------
# Registro de campeões / opções por rota
# ----------------------------------------------------------------------------

def _get_registry(session_state: Any) -> Mapping[str, Any]:
    reg = session_state.get("_radar_champ_registry")
    if not isinstance(reg, dict) or not reg.get("champions"):
        reg = load_champion_knowledge_registry()
        session_state["_radar_champ_registry"] = reg
    return reg


def _role_champion_options(registry: Mapping[str, Any]) -> dict[str, list[str]]:
    """Por rota: campeões ordenados por presença observada na rota (mais comuns
    primeiro), com o resto do pool no fim (flex/fearless sempre possível)."""
    champs = registry.get("champions") or {}
    all_names = sorted({str(e.get("name") or k) for k, e in champs.items()})
    out: dict[str, list[str]] = {}
    for role in ROLE_ORDER:
        scored: list[tuple[int, str]] = []
        rest: list[str] = []
        for k, e in champs.items():
            name = str(e.get("name") or k)
            obs = e.get("observed_roles") or {}
            try:
                cnt = int(obs.get(role, 0) or 0)
            except (TypeError, ValueError):
                cnt = 0
            if cnt > 0:
                scored.append((cnt, name))
            else:
                rest.append(name)
        scored.sort(key=lambda t: (-t[0], t[1]))
        out[role] = [name for _, name in scored] + sorted(rest)
    if not champs:
        out = {role: list(all_names) for role in ROLE_ORDER}
    return out


def _match_champion(raw: str, registry: Mapping[str, Any]) -> Optional[str]:
    """Nome digitado/colado → nome canônico do registro (alias + fuzzy)."""
    champs = registry.get("champions") or {}
    aliases = registry.get("aliases") or {}
    n = _norm_key(raw)
    if not n:
        return None
    key = str(aliases.get(n) or n)
    entry = champs.get(key)
    if isinstance(entry, Mapping):
        return str(entry.get("name") or raw)
    norm_to_name = {_norm_key(e.get("name") or k): str(e.get("name") or k) for k, e in champs.items()}
    close = difflib.get_close_matches(n, list(norm_to_name), n=1, cutoff=0.6)
    return norm_to_name[close[0]] if close else None


# ----------------------------------------------------------------------------
# Série / fearless (session)
# ----------------------------------------------------------------------------

_SERIES_FILE = None


def _series_file():
    """ops/radar_series.json — a série sobrevive a restart do app (BO5 da EWC
    não pode morrer com o Streamlit)."""
    global _SERIES_FILE
    if _SERIES_FILE is None:
        from pathlib import Path
        _SERIES_FILE = Path(__file__).resolve().parents[1] / "ops" / "radar_series.json"
    return _SERIES_FILE


def _series_disk_load() -> dict[str, Any]:
    import json
    try:
        p = _series_file()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _series_disk_save(all_series: Mapping[str, Any]) -> None:
    import json
    import time
    try:
        p = _series_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        # poda: guarda só séries tocadas nos últimos 3 dias (série é coisa de UM dia)
        cutoff = time.time() - 3 * 86400
        pruned = {k: v for k, v in all_series.items()
                  if isinstance(v, dict) and float(v.get("updated_at", 0) or 0) >= cutoff}
        p.write_text(json.dumps(pruned, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def _series_store(session_state: Any, team_a: str, team_b: str) -> dict[str, Any]:
    key = f"{team_a}|{team_b}"
    store = session_state.get("radar_series")
    if not isinstance(store, dict) or str(store.get("matchup")) != key:
        # tenta recuperar do disco (app pode ter reiniciado no meio da série)
        disk = _series_disk_load().get(key)
        if isinstance(disk, dict) and isinstance(disk.get("maps"), dict):
            store = {"matchup": key, "maps": dict(disk["maps"])}
        else:
            store = {"matchup": key, "maps": {}}
        session_state["radar_series"] = store
    return store


def _series_persist(store: Mapping[str, Any]) -> None:
    import time
    all_series = _series_disk_load()
    all_series[str(store.get("matchup"))] = {
        "maps": dict(store.get("maps") or {}),
        "updated_at": time.time(),
    }
    _series_disk_save(all_series)


def _fearless_used(store: Mapping[str, Any], current_map: int) -> set[str]:
    used: set[str] = set()
    for map_no, drafts in (store.get("maps") or {}).items():
        try:
            if int(map_no) >= int(current_map):
                continue
        except (TypeError, ValueError):
            continue
        for side in ("A", "B"):
            for champ in (drafts or {}).get(side) or []:
                if str(champ).strip():
                    used.add(str(champ).strip())
    return used


# ----------------------------------------------------------------------------
# HTML dos sinais (padrão HLTV: fato seco + números, sem veredito)
# ----------------------------------------------------------------------------

def _signal_chip_html(sig: RadarSignal) -> str:
    from html import escape
    if sig.kind == "conflito":
        base = sig.conflict_with
        liga_txt = escape(base.text) if base is not None else ""
        return (
            "<div style='border-left:3px solid #f59e0b;background:rgba(245,158,11,0.10);"
            "padding:6px 10px;border-radius:6px;margin:4px 0;font-size:0.92em;'>"
            "<span style='color:#f59e0b;font-weight:800;'>⚠ CONFLITO</span> "
            f"<b>{escape(sig.market_label)}</b>: {escape(sig.text)} "
            f"<span style='opacity:0.8;'>— mas {liga_txt}</span>"
            "</div>"
        )
    positive = sig.direction in ("win", "over")
    cor = "#22c55e" if positive else "#ef4444"
    return (
        f"<div style='border-left:3px solid {cor};background:rgba(255,255,255,0.03);"
        "padding:6px 10px;border-radius:6px;margin:4px 0;font-size:0.92em;'>"
        f"<b>{sig.market_label}</b>: {sig.text} "
        f"<span style='color:{cor};font-weight:700;'>{sig.pct:.0f}%</span>"
        "</div>"
    )


_ICON_DIR = None


def _champ_icon_uri(champ_id: str) -> Optional[str]:
    """Data-URI do ícone do campeão (champ_icons/<id>.png, cache em módulo)."""
    global _ICON_DIR
    if _ICON_DIR is None:
        from pathlib import Path
        _ICON_DIR = Path(__file__).resolve().parents[1] / "champ_icons"
    cache = getattr(_champ_icon_uri, "_cache", None)
    if cache is None:
        cache = {}
        _champ_icon_uri._cache = cache
    cid = str(champ_id or "").strip()
    if not cid:
        return None
    if cid not in cache:
        p = _ICON_DIR / f"{cid}.png"
        if p.exists():
            import base64
            cache[cid] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
        else:
            cache[cid] = None
    return cache[cid]


def _short_name(name: str) -> str:
    return name if len(name) <= 9 else name[:8] + "…"


def _draft_header_bar_html(blue_team: str, red_team: str) -> str:
    from html import escape
    return (
        "<div style='display:flex;border-radius:6px;overflow:hidden;margin:6px 0 10px 0;"
        "font-weight:800;letter-spacing:0.06em;font-size:0.95em;'>"
        f"<div style='flex:1;background:#3b82f6;color:white;padding:7px 14px;'>"
        f"BLUE SIDE · {escape(str(blue_team)).upper()}</div>"
        f"<div style='flex:1;background:#ef4444;color:white;padding:7px 14px;text-align:right;'>"
        f"{escape(str(red_team)).upper()} · RED SIDE</div>"
        "</div>"
    )


def _slot_html(*, role: str, player: str, champ: str, champ_id: str,
               tone: str, is_active: bool) -> str:
    from html import escape
    uri = _champ_icon_uri(champ_id) if champ else None
    if uri:
        icon = f"<img src='{uri}' style='width:40px;height:40px;border-radius:6px;flex:0 0 auto;'/>"
    else:
        icon = ("<div style='width:40px;height:40px;border-radius:6px;background:rgba(255,255,255,0.06);"
                "display:flex;align-items:center;justify-content:center;opacity:0.5;flex:0 0 auto;'>—</div>")
    border = f"2px solid {tone}" if is_active else "1px solid rgba(255,255,255,0.10)"
    who = escape(_ROLE_LABELS.get(role, role)) + (f" · {escape(player)}" if player else "")
    champ_txt = escape(champ) if champ else "<span style='opacity:0.45;'>vazio</span>"
    return (
        f"<div style='display:flex;align-items:center;gap:8px;border:{border};"
        f"border-left:4px solid {tone};border-radius:8px;padding:4px 8px;margin:3px 0;'>"
        f"{icon}"
        f"<div style='min-width:0;'>"
        f"<div style='font-size:0.72em;opacity:0.6;text-transform:uppercase;letter-spacing:0.04em;"
        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{who}</div>"
        f"<div style='font-weight:700;font-size:0.95em;white-space:nowrap;'>{champ_txt}</div>"
        f"</div></div>"
    )


def _grid_icon_html(name: str, champ_id: str, tier: Optional[str] = None) -> str:
    from html import escape
    uri = _champ_icon_uri(champ_id)
    if not uri:
        return "<div style='height:48px;'></div>"
    badge = ""
    if tier:
        from app_core.meta_patch_core import TIER_COLORS
        cor = TIER_COLORS.get(str(tier), "#6b7280")
        badge = (f"<span style='position:absolute;top:-4px;right:-6px;background:{cor};"
                 f"color:#0b0f14;font-size:0.62em;font-weight:800;border-radius:5px;"
                 f"padding:0 4px;'>{escape(str(tier))}</span>")
    return (
        f"<div style='text-align:center;'><span style='position:relative;display:inline-block;'>"
        f"<img src='{uri}' alt='{escape(name)}' style='width:48px;height:48px;border-radius:8px;'/>"
        f"{badge}</span></div>"
    )


def _team_header_html(team: str, color: str) -> str:
    from html import escape
    return (
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>"
        f"{team_logo_html(team, size=28)}"
        f"<span style='font-weight:800;color:{color};font-size:1.05em;'>{escape(str(team))}</span>"
        "</div>"
    )


# ----------------------------------------------------------------------------
# Render principal
# ----------------------------------------------------------------------------

# Sufixo do jogo ativo no Radar multi-jogo (2026-07-10, pedido do usuário:
# "vários jogos rolando ao mesmo tempo"). "" = Jogo 1 (compatível com estado
# salvo antes do multi-jogo); "_g2".."_g10" = demais abas. Os keys de widget/
# sessão por-jogo deste módulo embutem {_S}; caches de campeões/meta e o store
# da série (por dupla de times, no disco) continuam compartilhados de propósito.
_S = ""
_MAX_SLOTS = 10


def render_radar_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    """Maestro do multi-jogo: abas 'Jogo N' + adicionar/fechar (máx 10)."""
    global _S
    session_state = st.session_state
    slots = session_state.get("radar_slots")
    if not isinstance(slots, list) or not slots:
        slots = [1]
        session_state["radar_slots"] = slots

    c_add, c_close, _sp = st.columns([2, 2, 8])
    with c_add:
        if len(slots) < _MAX_SLOTS and st.button(
                f"➕ Adicionar jogo ({len(slots)}/{_MAX_SLOTS})", key="radar_slots_add"):
            slots.append(max(slots) + 1)
            session_state["radar_slots"] = slots
            st.rerun()
    with c_close:
        if len(slots) > 1 and st.button("✕ Fechar último jogo", key="radar_slots_close"):
            slots.pop()
            session_state["radar_slots"] = slots
            st.rerun()

    if len(slots) == 1:
        _S = ""
        try:
            _render_radar_slot(st=st, ctx=ctx)
        finally:
            _S = ""
        return

    tabs = st.tabs([f"Jogo {i}" for i in slots])
    for tab, slot in zip(tabs, slots):
        with tab:
            _S = "" if slot == 1 else f"_g{slot}"
            try:
                _render_radar_slot(st=st, ctx=ctx)
            finally:
                _S = ""


def _render_radar_slot(*, st: Any, ctx: Mapping[str, Any]) -> None:
    render_tab_hero(
        st,
        title="Radar",
        subtitle="Draft em mãos: sinais do confronto, leitura do draft, fearless e a série ao vivo.",
        chips=["Sinais H2H×Liga", "Draft por estilo", "Fearless", "Entre mapas", "Durante o mapa"],
    )

    team_games = ctx["team_games"]
    csv_sig = ctx["csv_sig"]
    load_player_rows_fn = ctx["load_player_rows_fn"]
    format_num_fn = ctx["format_num_fn"]
    artifact_path = str(ctx.get("artifact_path") or "")
    session_state = st.session_state

    if team_games is None or team_games.empty:
        st.warning("Sem jogos carregados (CSV vazio?).")
        return

    # ------------------------- Confronto -------------------------
    years = sorted({int(y) for y in pd.to_numeric(team_games.get("year"), errors="coerce").dropna().unique()})
    year = int(years[-1]) if years else 0
    df_year = team_games[pd.to_numeric(team_games["year"], errors="coerce") == year].copy()
    teams = sorted({str(t) for t in df_year["team"].dropna().astype(str) if str(t).strip()})

    with st.container(border=True):
        render_tab_section(st, title="Confronto", subtitle=f"Season {year} · o recorte segue a liga de cada time.")
        c1, c2 = st.columns(2)
        team_a_seed = str(session_state.get(f"radar{_S}_team_a") or session_state.get("teamA") or "")
        team_b_seed = str(session_state.get(f"radar{_S}_team_b") or session_state.get("teamB") or "")
        opts = ["", *teams]
        # Semear SÓ via session_state (sem index=) — as duas coisas juntas geram
        # o warning "created with a default value but also had its value set".
        if str(session_state.get(f"radar{_S}_team_a_sel", "") or "") not in opts:
            session_state[f"radar{_S}_team_a_sel"] = team_a_seed if team_a_seed in opts else ""
        if str(session_state.get(f"radar{_S}_team_b_sel", "") or "") not in opts:
            session_state[f"radar{_S}_team_b_sel"] = team_b_seed if team_b_seed in opts else ""
        with c1:
            team_a = st.selectbox("Time 1", opts, key=f"radar{_S}_team_a_sel")
        with c2:
            team_b = st.selectbox("Time 2", opts, key=f"radar{_S}_team_b_sel")
        session_state[f"radar{_S}_team_a"] = team_a
        session_state[f"radar{_S}_team_b"] = team_b

    if not team_a or not team_b or team_a == team_b:
        st.info("Escolha os dois times pra abrir o radar.")
        return

    from app_core.line_analysis_tab_core import _augment_with_total_columns
    df_year_aug = _augment_with_total_columns(df_year)

    def _team_df(team: str) -> pd.DataFrame:
        return df_year_aug[df_year_aug["team"].astype(str) == str(team)].copy()

    def _own_league(df_team: pd.DataFrame) -> str:
        if df_team.empty or "league" not in df_team.columns or "date" not in df_team.columns:
            return ""
        d = df_team.copy()
        d["_dt"] = pd.to_datetime(d["date"], errors="coerce")
        return str(d.sort_values("_dt")["league"].iloc[-1])

    df_a_all, df_b_all = _team_df(team_a), _team_df(team_b)
    league_a, league_b = _own_league(df_a_all), _own_league(df_b_all)

    # Regra de escopo (window_scope_core): cross-league → janela longa = ANO
    # TODO (todas as ligas) e L10/L5 = campeonato mais recente; mesma liga →
    # tudo na liga do recorte. Mantém Radar == Resumo == Análise da Linha.
    # Cross = liga DOMINANTE (doméstica) diferente — a do último jogo não serve
    # (num MSI os dois times têm "MSI" como última liga e o cross nunca dispara).
    from app_core.window_scope_core import dominant_league as _dom_lg
    _dom_a, _dom_b = _dom_lg(df_a_all), _dom_lg(df_b_all)
    _cross = bool(_dom_a and _dom_b and _dom_a != _dom_b)

    def _windows_for(team: str, df_team_all: pd.DataFrame, league: str, opponent: str) -> dict[str, pd.DataFrame]:
        wins_all = build_team_windows(df_team_all, opponent)
        if _cross:
            from app_core.window_scope_core import recent_championship_df
            recent = build_team_windows(recent_championship_df(df_team_all), opponent)
            return {"ano": wins_all["liga"], "l10": recent["l10"],
                    "l5": recent["l5"], "h2h": wins_all["h2h"]}
        df_liga = df_team_all[df_team_all["league"].astype(str) == league] if league else df_team_all
        wins = build_team_windows(df_liga, opponent)
        wins["h2h"] = wins_all["h2h"]  # H2H atravessa ligas (MSI/EWC contam)
        return wins

    windows_a = _windows_for(team_a, df_a_all, league_a, team_b)
    windows_b = _windows_for(team_b, df_b_all, league_b, team_a)

    # ------------------------- Linhas de referência -------------------------
    with st.expander("Linhas de referência (abre nas padrão; ajuste pra linha da casa)", expanded=False):
        lc = st.columns(4)
        lines: dict[str, float] = {}
        for i, mk in enumerate([m for m in RADAR_MARKETS if m["line"] is not None]):
            with lc[i % 4]:
                lines[str(mk["key"])] = float(st.number_input(
                    f"{mk['label']}", value=float(mk["line"]), step=0.5,
                    key=f"radar{_S}_line_{mk['key']}",
                ))

    # ------------------------- Sinais (radar) -------------------------
    with st.container(border=True):
        render_tab_section(
            st, title="Sinais do confronto",
            subtitle="Só o que está quente ou em conflito — o resto fica de fora.",
        )
        sa, sb = st.columns(2)
        for col, team, windows, color, league in (
            (sa, team_a, windows_a, _COR_A, league_a),
            (sb, team_b, windows_b, _COR_B, league_b),
        ):
            with col:
                st.markdown(_team_header_html(team, color), unsafe_allow_html=True)
                if league:
                    st.caption(f"Liga do recorte: {league}")
                sigs = build_radar_signals(team=team, windows=windows, lines=lines)
                if not sigs:
                    st.caption("Sem sinais fortes na régua (nada ≥70% com amostra).")
                for sig in sigs:
                    st.markdown(_signal_chip_html(sig), unsafe_allow_html=True)

        # Odd justa do ML (mesma fonte do entre-mapas: análise principal → artefato)
        p_base, fonte = resolve_p_base(
            session_state=session_state, team_a=team_a, team_b=team_b, artifact_path=artifact_path,
        )
        if p_base is not None:
            st.markdown(
                f"**Odd justa (mapa)** · {team_a}: **{format_num_fn(1.0 / p_base, 2)}** "
                f"({100.0 * p_base:.1f}%) · {team_b}: **{format_num_fn(1.0 / (1.0 - p_base), 2)}** "
                f"({100.0 * (1.0 - p_base):.1f}%)  \n"
                f"<span style='opacity:0.6;font-size:0.85em;'>fonte: {fonte}</span>",
                unsafe_allow_html=True,
            )

    # ------------------------- Draft -------------------------
    registry = _get_registry(session_state)
    role_options = _role_champion_options(registry)
    players_df = None
    try:
        players_df = load_player_rows_fn(csv_sig)
    except Exception:
        players_df = None

    roster_a = latest_roster(players_df=players_df, team=team_a) if players_df is not None else {}
    roster_b = latest_roster(players_df=players_df, team=team_b) if players_df is not None else {}

    store = _series_store(session_state, team_a, team_b)

    with st.container(border=True):
        render_tab_section(
            st, title="Draft do mapa",
            subtitle="Clique nos bonecos na ordem do draft, ou cole os 10 nomes. Mapas salvos riscam o pool fearless.",
        )
        cf1, cf2, cf3 = st.columns([1.2, 1, 2.2])
        with cf1:
            fearless = st.checkbox("Fearless", value=True, key=f"radar{_S}_fearless",
                                   help="Campeões usados nos mapas anteriores da série ficam fora do pool.")
        with cf2:
            map_no = int(st.number_input("Mapa", 1, 5, step=1, key=f"radar{_S}_map_no"))
        used = _fearless_used(store, map_no) if fearless else set()
        with cf3:
            saved_maps = sorted(int(k) for k in (store.get("maps") or {}) if str(k).isdigit())
            if saved_maps:
                st.caption(f"Série salva: mapa(s) {', '.join(str(m) for m in saved_maps)} · "
                           f"{len(used)} campeões fora do pool deste mapa" if used else
                           f"Série salva: mapa(s) {', '.join(str(m) for m in saved_maps)}")
            if used:
                st.caption("Fora (fearless): " + ", ".join(sorted(used)))

        # Padrões por número do mapa (medidos; gate ACCEPT 2026-07-04 —
        # dragões mapa-2 foi REJECT e por isso não aparece).
        for pat in map_pattern_signals(map_no):
            st.markdown(
                "<div style='border-left:3px solid #f59e0b;background:rgba(245,158,11,0.08);"
                "padding:5px 10px;border-radius:6px;margin:3px 0;font-size:0.88em;'>"
                f"📐 {pat}</div>",
                unsafe_allow_html=True,
            )

        pick_key = lambda side, role: f"radar_pick_{side}_{role}_m{map_no}"  # noqa: E731

        # Chegou com a série andando? Pode selecionar o mapa 1, colar/salvar,
        # depois o 2, ... — o pool fearless reconstrói igual. Os slots de um
        # mapa JÁ SALVO pré-carregam do que foi salvo (pode editar e re-salvar).
        saved_this_map = (store.get("maps") or {}).get(str(map_no)) or {}
        for side in ("A", "B"):
            saved_picks = list(saved_this_map.get(side) or [])
            for i, role in enumerate(ROLE_ORDER):
                key = pick_key(side, role)
                if not str(session_state.get(key) or "") and i < len(saved_picks) and str(saved_picks[i]).strip():
                    session_state[key] = str(saved_picks[i])
        st.caption("Série já em andamento? Selecione o mapa 1, cole o draft e salve; repita até o mapa atual — o fearless reconstrói sozinho.")

        # Colagem única no esquema do usuário: 10 campeões (tab/espaço), os 5
        # PRIMEIROS são do time BLUE SIDE, na ordem TOP JNG MID ADC SUP.
        from app_core.live_bet_shared_core import parse_compact_draft_tokens
        pc1, pc2, pc3 = st.columns([3.2, 1.2, 1])
        with pc1:
            paste_all = st.text_input(
                "Colar o draft (10 campeões — os 5 primeiros são do blue side)",
                key=f"radar{_S}_paste_all_m{map_no}",
                placeholder="ex: Gnar Trundle Taliyah Yunara Lulu  Rumble Kindred Syndra Ashe Seraphine",
            )
        with pc2:
            blue_team = st.radio("Blue side", [team_a, team_b], horizontal=True,
                                 key=f"radar{_S}_blue_side_m{map_no}")
        with pc3:
            st.markdown("<div style='height:1.9em;'></div>", unsafe_allow_html=True)
            if st.button("Aplicar", key=f"radar{_S}_paste_btn_m{map_no}"):
                champs = registry.get("champions") or {}
                vocab = [str(e.get("name") or k) for k, e in champs.items()]
                tokens = parse_compact_draft_tokens(str(paste_all or ""), vocab, max_picks=10)
                resolved = [(_match_champion(t, registry) or "") for t in tokens]
                blue_picks, red_picks = resolved[:5], resolved[5:10]
                side_of = {"A": blue_picks, "B": red_picks} if str(blue_team) == str(team_a) \
                    else {"A": red_picks, "B": blue_picks}
                for side, plist in side_of.items():
                    for i, role in enumerate(ROLE_ORDER):
                        if i < len(plist) and plist[i]:
                            session_state[pick_key(side, role)] = plist[i]
                st.rerun()

        # ---------------- montador visual (estilo pro draft) ----------------
        blue_side = "A" if str(blue_team) == str(team_a) else "B"
        red_side = "B" if blue_side == "A" else "A"
        team_of = {"A": team_a, "B": team_b}
        roster_of = {"A": roster_a, "B": roster_b}
        name2id = session_state.get("_radar_champ_name2id")
        if not isinstance(name2id, dict) or not name2id:
            name2id = {str(e.get("name") or k): str(e.get("id") or "")
                       for k, e in (registry.get("champions") or {}).items()}
            session_state["_radar_champ_name2id"] = name2id

        st.markdown(_draft_header_bar_html(team_of[blue_side], team_of[red_side]), unsafe_allow_html=True)

        def _filled(side: str, role: str) -> bool:
            return bool(str(session_state.get(pick_key(side, role)) or "").strip())

        # Ordem REAL de picks do competitivo (não é por rota): B R R B B R R B B R.
        # Sem slot escolhido à mão, o clique no grid segue essa sequência e o
        # campeão cai na MELHOR rota vazia dele (observed_roles) no time da vez.
        _PICK_SEQ = [blue_side, red_side, red_side, blue_side, blue_side,
                     red_side, red_side, blue_side, blue_side, red_side]
        n_picked = sum(1 for s in ("A", "B") for r in ROLE_ORDER if _filled(s, r))
        seq_side = _PICK_SEQ[n_picked] if n_picked < 10 else None

        manual = session_state.get(f"radar{_S}_active_slot")
        if not (isinstance(manual, tuple) and len(manual) == 2
                and manual[0] in ("A", "B") and manual[1] in ROLE_ORDER):
            manual = None
        # slot manual: alvo exato do grid; se estiver PREENCHIDO, é modo MOVER
        active = manual
        move_from = manual if (manual is not None and _filled(*manual)) else None

        def _best_empty_role(side: str, champ_name: str) -> Optional[str]:
            empty = [r for r in ROLE_ORDER if not _filled(side, r)]
            if not empty:
                return None
            entry = None
            champs_map = registry.get("champions") or {}
            k = _norm_key(champ_name)
            aliases = registry.get("aliases") or {}
            entry = champs_map.get(str(aliases.get(k) or k))
            if isinstance(entry, Mapping):
                obs = entry.get("observed_roles") or {}
                scored = sorted(empty, key=lambda r: -int(obs.get(r, 0) or 0))
                if int(obs.get(scored[0], 0) or 0) > 0:
                    return scored[0]
                prim = str(entry.get("primary_role") or "")
                if prim in empty:
                    return prim
            return empty[0]

        col_blue, col_grid, col_red = st.columns([1.05, 2.1, 1.05])
        for col, side, tone in ((col_blue, blue_side, "#3b82f6"), (col_red, red_side, "#ef4444")):
            with col:
                for role in ROLE_ORDER:
                    champ = str(session_state.get(pick_key(side, role)) or "")
                    player = str(roster_of[side].get(role, "") or "")
                    is_active = (active == (side, role))
                    sc1, sc2, sc3 = st.columns([3.4, 0.8, 0.8])
                    with sc1:
                        st.markdown(
                            _slot_html(role=role, player=player, champ=champ,
                                       champ_id=name2id.get(champ, ""), tone=tone,
                                       is_active=is_active),
                            unsafe_allow_html=True,
                        )
                    with sc2:
                        lbl = "↔" if champ else ("●" if is_active else "○")
                        hlp = ("Selecionar pra MOVER (depois clique no slot de destino)"
                               if champ else "Escolher este slot pro próximo boneco do grid")
                        if st.button(lbl, key=f"radar{_S}_slot_sel_{side}_{role}_m{map_no}", help=hlp):
                            here = (side, role)
                            if move_from is not None and move_from != here:
                                # troca/move: destino recebe o boneco selecionado
                                a_key, b_key = pick_key(*move_from), pick_key(side, role)
                                session_state[a_key], session_state[b_key] = \
                                    str(session_state.get(b_key) or ""), str(session_state.get(a_key) or "")
                                session_state[f"radar{_S}_active_slot"] = None
                            elif manual == here:
                                session_state[f"radar{_S}_active_slot"] = None
                            else:
                                session_state[f"radar{_S}_active_slot"] = here
                            st.rerun()
                    with sc3:
                        if champ and st.button("✕", key=f"radar{_S}_slot_clr_{side}_{role}_m{map_no}",
                                               help="Limpar este slot"):
                            session_state[pick_key(side, role)] = ""
                            session_state[f"radar{_S}_active_slot"] = None
                            st.rerun()

        with col_grid:
            q = st.text_input("Buscar campeão", key=f"radar{_S}_grid_q_m{map_no}",
                              placeholder="digite pra filtrar (ex: gnar)", label_visibility="collapsed")
            picked_now = {str(session_state.get(pick_key(s, r)) or "").strip()
                          for s in ("A", "B") for r in ROLE_ORDER}
            excluded = set(used) | (picked_now - {""})
            if manual is not None:
                pool = list(role_options.get(manual[1], []))
            else:
                pool = sorted(name2id)
            qn = _norm_key(q)
            if qn:
                subs = [n for n in pool if qn in _norm_key(n)]
                if not subs:
                    # Tolerância a erro de digitação ("tundre"→Trundle,
                    # "nafiri"→Naafiri): sem match literal, cai no fuzzy.
                    norm_map = {_norm_key(n): n for n in pool}
                    close = difflib.get_close_matches(qn, list(norm_map), n=8, cutoff=0.5)
                    subs = [norm_map[c] for c in close]
                pool = subs
            pool = [n for n in pool if n not in excluded][:18]

            if move_from is not None:
                _mc = str(session_state.get(pick_key(*move_from)) or "")
                st.caption(f"Movendo **{_mc}** ({team_of[move_from[0]]} · {_ROLE_LABELS[move_from[1]]}) — "
                           "clique no ↔/○ do slot de DESTINO pra trocar, ou num boneco pra substituir.")
            elif manual is not None:
                _r, _p = manual[1], str(roster_of[manual[0]].get(manual[1], "") or "")
                st.caption(f"Escolhendo: {team_of[manual[0]]} · {_ROLE_LABELS[_r]}"
                           + (f" ({_p})" if _p else ""))
            elif seq_side is not None:
                st.caption(f"Ordem do draft: vez do **{team_of[seq_side]}** (pick {n_picked + 1}/10) — "
                           "clica no boneco e ele cai na rota certa sozinho.")
            elif not q:
                st.caption("Draft completo — ↔ num slot pra mover/trocar, ✕ pra limpar.")

            # tiers da aba Meta (badge no canto do ícone)
            meta_lu = session_state.get("_radar_meta_lu")
            if not (isinstance(meta_lu, tuple) and len(meta_lu) == 2 and meta_lu[0] == str(csv_sig)):
                try:
                    from app_core.meta_patch_core import meta_lookup_for_radar
                    from app_core.meta_tab_core import _get_draft_rows
                    meta_lu = (str(csv_sig), meta_lookup_for_radar(
                        _get_draft_rows(session_state, csv_sig), registry=registry))
                except Exception:
                    meta_lu = (str(csv_sig), {})
                session_state["_radar_meta_lu"] = meta_lu
            meta_map = meta_lu[1]

            grid_cols = st.columns(6)
            for i, name in enumerate(pool):
                with grid_cols[i % 6]:
                    _tier = (meta_map.get(name) or {}).get("tier") if meta_map else None
                    st.markdown(_grid_icon_html(name, name2id.get(name, ""), tier=_tier),
                                unsafe_allow_html=True)
                    if st.button(_short_name(name), key=f"radar{_S}_gp_{name2id.get(name, name)}_m{map_no}",
                                 use_container_width=True, help=name):
                        if manual is not None:
                            session_state[pick_key(*manual)] = name
                            session_state[f"radar{_S}_active_slot"] = None
                            st.rerun()
                        elif seq_side is not None:
                            role_dest = _best_empty_role(seq_side, name)
                            if role_dest:
                                session_state[pick_key(seq_side, role_dest)] = name
                                st.rerun()

        picks: dict[str, list[str]] = {
            s: [str(session_state.get(pick_key(s, r)) or "") for r in ROLE_ORDER]
            for s in ("A", "B")
        }

        bc1, bc2 = st.columns([1.4, 1])
        with bc1:
            if st.button(f"💾 Salvar mapa {map_no} na série (libera o fearless do próximo)",
                         key=f"radar{_S}_save_map_{map_no}"):
                store.setdefault("maps", {})[str(map_no)] = {"A": list(picks["A"]), "B": list(picks["B"])}
                session_state["radar_series"] = store
                _series_persist(store)
                st.rerun()
        with bc2:
            if st.button("♻ Resetar série", key=f"radar{_S}_reset_series"):
                fresh = {"matchup": f"{team_a}|{team_b}", "maps": {}}
                session_state["radar_series"] = fresh
                _series_persist(fresh)
                st.rerun()

        saved_map_nos = sorted(int(k) for k in (store.get("maps") or {}) if str(k).isdigit())
        if saved_map_nos:
            with st.expander(f"Mapas salvos da série ({len(saved_map_nos)})", expanded=False):
                for m in saved_map_nos:
                    drafts = (store.get("maps") or {}).get(str(m)) or {}
                    da_txt = ", ".join(c for c in (drafts.get("A") or []) if str(c).strip()) or "—"
                    db_txt = ", ".join(c for c in (drafts.get("B") or []) if str(c).strip()) or "—"
                    mc1, mc2 = st.columns([5, 1])
                    with mc1:
                        st.markdown(f"**Mapa {m}** · {team_a}: {da_txt}  \n"
                                    f"{'&nbsp;' * 14} {team_b}: {db_txt}", unsafe_allow_html=True)
                    with mc2:
                        if st.button("Remover", key=f"radar{_S}_rm_map_{m}"):
                            (store.get("maps") or {}).pop(str(m), None)
                            session_state["radar_series"] = store
                            _series_persist(store)
                            st.rerun()

    complete_a = all(str(c).strip() for c in picks["A"])
    complete_b = all(str(c).strip() for c in picks["B"])
    compare: Optional[Mapping[str, Any]] = None
    if complete_a and complete_b:
        # ------------------------- Leitura do draft -------------------------
        compare = _render_draft_reading(
            st,
            session_state=session_state,
            registry=registry,
            players_df=players_df,
            team_games=team_games,
            team_a=team_a, team_b=team_b,
            picks_a=picks["A"], picks_b=picks["B"],
            roster_a=roster_a, roster_b=roster_b,
            league_a=league_a, league_b=league_b,
            year=year, csv_sig=csv_sig,
        )
    else:
        st.info("Complete os 10 picks pra abrir a leitura do draft.")

    # ---------------- Ao vivo (ex-Live-Bet, fundido aqui a pedido) ----------------
    with st.container(border=True):
        render_tab_section(
            st, title="Entre mapas (série ao vivo)",
            subtitle="Placar da série → odd justa do próximo mapa e de fechar a série.",
        )
        try:
            from app_core.live_between_maps_core import render_between_maps_block
            render_between_maps_block(
                st=st, session_state=session_state, team_a=team_a, team_b=team_b,
                artifact_path=artifact_path, format_num_fn=format_num_fn,
            )
        except Exception as _exc_lbm:
            _notify(st, _exc_lbm, "Falha no bloco entre-mapas")

    with st.container(border=True):
        render_tab_section(
            st, title="Durante o mapa (estado → vitória)",
            subtitle="Ouro/torres/objetivos/tempo do mapa em andamento → P(vitória) calibrada.",
        )
        try:
            from app_core.live_state_panel_ui_core import render_live_state_panel
            render_live_state_panel(
                st=st, session_state=session_state,
                team_a=team_a, team_b=team_b,
                p_map_pre=p_base, league=league_a,
                comp_compare=compare,
                format_num_fn=format_num_fn,
            )
        except Exception as _exc_lsp:
            _notify(st, _exc_lsp, "Falha na predição calibrada (durante o mapa)")


def _notify(st: Any, exc: Exception, context: str) -> None:
    try:
        from app_core.ui_notify_core import notify_exception
        notify_exception(st, exc, context=context)
    except Exception:
        st.warning(f"{context}: {exc}")


def _render_draft_reading(
    st: Any,
    *,
    session_state: Any,
    registry: Mapping[str, Any],
    players_df: Optional[pd.DataFrame],
    team_games: Optional[pd.DataFrame],
    team_a: str, team_b: str,
    picks_a: list[str], picks_b: list[str],
    roster_a: Mapping[str, str], roster_b: Mapping[str, str],
    league_a: str, league_b: str,
    year: int, csv_sig: Any,
) -> Mapping[str, Any]:
    comp_a = build_comp_profile(registry=registry, picks=picks_a, role_order=list(ROLE_ORDER))
    comp_b = build_comp_profile(registry=registry, picks=picks_b, role_order=list(ROLE_ORDER))
    arch_a, _scores_a = classify_comp_archetype(comp_a)
    arch_b, _scores_b = classify_comp_archetype(comp_b)
    compare = compare_comp_profiles(comp_a=comp_a, comp_b=comp_b, team_a=team_a, team_b=team_b)

    # ---- fatores mensuráveis com os dados que temos ----
    # Meta vem da fonte da aba META (presença pick+ban do patch atual);
    # fallback: pick% da liga (quando o lookup não cobre os picks).
    _meta_lu = session_state.get("_radar_meta_lu")
    meta_map: dict = _meta_lu[1] if (isinstance(_meta_lu, tuple) and len(_meta_lu) == 2) else {}

    def _meta_avg(picks: list[str], league: str) -> float:
        vals = [float(meta_map[c]["presence"]) for c in picks if c in meta_map]
        if len(vals) >= 3:
            return sum(vals) / len(vals)
        if players_df is None or players_df.empty:
            return float("nan")
        scope = players_df
        if "year" in scope.columns:
            scope = scope[pd.to_numeric(scope["year"], errors="coerce") == int(year)]
        if league and "league" in scope.columns:
            scope = scope[scope["league"].astype(str) == str(league)]
        vals = [champion_meta_presence(players_df_scope=scope, champion=c)[0] for c in picks]
        vals = [v for v in vals if pd.notna(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def _familiarity(picks: list[str], roster: Mapping[str, str]) -> list[tuple[str, str, int]]:
        out = []
        for i, role in enumerate(ROLE_ORDER):
            player = str(roster.get(role, "") or "")
            games = player_champion_familiarity(players_df=players_df, player=player, champion=picks[i]) \
                if players_df is not None else 0
            out.append((player or _ROLE_LABELS[role], picks[i], games))
        return out

    meta_a, meta_b = _meta_avg(picks_a, league_a), _meta_avg(picks_b, league_b)
    fam_a, fam_b = _familiarity(picks_a, roster_a), _familiarity(picks_b, roster_b)
    fam_avg_a = sum(min(g, _FAMILIARITY_OK) for _, _, g in fam_a) / (5.0 * _FAMILIARITY_OK)
    fam_avg_b = sum(min(g, _FAMILIARITY_OK) for _, _, g in fam_b) / (5.0 * _FAMILIARITY_OK)

    # ---- veredito 3-linhas (fórmula fixa; pesos na ordem iTero, sem IA) ----
    comp_pts = 0.0
    for key, w in (("early_team", 1.0), ("mid_team", 1.0), ("late_team", 1.0),
                   ("objective_team", 0.5), ("towers_team", 0.5)):
        who = str(compare.get(key) or "")
        comp_pts += w if who == team_a else (-w if who == team_b else 0.0)
    meta_pts = 0.0
    if pd.notna(meta_a) and pd.notna(meta_b):
        # escala de PRESENÇA (pick+ban, aba Meta): diffs típicos ~5-15pp
        meta_pts = max(-1.5, min(1.5, (meta_a - meta_b) / 10.0))
    fam_pts = max(-1.0, min(1.0, (fam_avg_a - fam_avg_b) * 2.0))
    score = 1.0 * meta_pts + 0.9 * comp_pts / 4.0 * 2.0 + 0.6 * fam_pts
    if score >= 1.2:
        lean_team, lean_label, lean_color = team_a, "VANTAGEM CLARA", _COR_A
    elif score >= 0.45:
        lean_team, lean_label, lean_color = team_a, "LEVE VANTAGEM", _COR_A
    elif score <= -1.2:
        lean_team, lean_label, lean_color = team_b, "VANTAGEM CLARA", _COR_B
    elif score <= -0.45:
        lean_team, lean_label, lean_color = team_b, "LEVE VANTAGEM", _COR_B
    else:
        lean_team, lean_label, lean_color = "", "DRAFT EQUILIBRADO", "#9ca3af"

    with st.container(border=True):
        render_tab_section(st, title="Leitura do draft",
                           subtitle="Comps, veredito e o que esse draft costuma fazer com os mercados.")
        ca, cb = st.columns(2)
        for col, team, arch, comp, color, fam, meta in (
            (ca, team_a, arch_a, comp_a, _COR_A, fam_a, meta_a),
            (cb, team_b, arch_b, comp_b, _COR_B, fam_b, meta_b),
        ):
            with col:
                st.markdown(_team_header_html(team, color), unsafe_allow_html=True)
                st.markdown(f"**Comp: {arch}** · janela {comp.get('window_label', '-')} · "
                            f"{comp.get('primary_style', '-')}")
                if pd.notna(meta):
                    st.caption(f"Meta: presença média dos picks {meta:.0f}%")
                for player, champ, games in fam:
                    if games < _FAMILIARITY_OK:
                        tag = (f"<span style='color:#f59e0b;font-weight:700;'>"
                               f"{games}j de palco ⚠</span>")
                    else:
                        tag = f"<span style='opacity:0.7;'>{games}j de palco</span>"
                    st.markdown(
                        f"<div style='font-size:0.86em;margin:2px 0;'>"
                        f"<b>{player}</b> — {champ}: {tag}</div>",
                        unsafe_allow_html=True,
                    )
        # Veredito em DESTAQUE (feedback 2026-07-04: "meio perdido... ou está
        # pequeno, ou não tem") — banner grande + barra de duelo.
        from html import escape as _esc
        share_a = 50.0 + max(-2.5, min(2.5, score)) * 8.0
        banner_txt = f"DRAFT: {lean_label}" + (f" · {_esc(str(lean_team).upper())}" if lean_team else "")
        st.markdown(
            f"<div style='margin:10px 0 4px 0;padding:12px 16px;border-radius:10px;"
            f"background:linear-gradient(90deg, {lean_color}26, transparent);"
            f"border-left:6px solid {lean_color};display:flex;align-items:center;"
            f"justify-content:space-between;gap:10px;flex-wrap:wrap;'>"
            f"<span style='font-size:1.35em;font-weight:800;color:{lean_color};'>"
            f"🏆 {banner_txt}</span>"
            f"</div>"
            f"<div style='display:flex;height:12px;border-radius:6px;overflow:hidden;margin:2px 0 10px 0;'>"
            f"<div style='width:{share_a:.0f}%;background:{_COR_A};'></div>"
            f"<div style='flex:1;background:{_COR_B};'></div>"
            f"</div>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.82em;"
            f"opacity:0.75;margin:-6px 0 8px 0;'>"
            f"<span>{_esc(str(team_a))} {share_a:.0f}%</span>"
            f"<span>{100.0 - share_a:.0f}% {_esc(str(team_b))}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Quem leva cada fase (early / mid / late): "
                   f"{compare.get('early_team', '-')} / {compare.get('mid_team', '-')} / "
                   f"{compare.get('late_team', '-')}")

        # ---- ponte draft→mercados (motor da Fase 2, leitura descritiva) ----
        st.markdown("**Draft → mercados**")
        calls = [
            ("Kills", str(compare.get("kills_call") or "Neutro")),
            ("Torres (total)", str(compare.get("tower_total_call") or "Neutro")),
            ("Dragões (total)", str(compare.get("dragons_total_call") or "Neutro")),
            ("Tempo", str(compare.get("time_call") or "Neutro")),
        ]
        chips = " · ".join(f"{lbl}: **{val}**" for lbl, val in calls)
        st.markdown(chips)
        for reason_key in ("tower_total_reason", "dragons_total_reason"):
            reason = str(compare.get(reason_key) or "").strip()
            if reason:
                st.caption(reason)

        # Linha de TORRES ajustada pelo draft (único mercado que passou o gate
        # cego 2026; ver DRAFT_TOWERS_ADJUST). A base já é do confronto — o
        # time que não bate torre tem base baixa; o draft desloca a partir dela.
        try:
            if team_games is not None and not team_games.empty:
                from mlcore.lines_engine import build_profile, matchup_expected_totals
                from mlcore.stats_teams import Filters as _Filters
                pA = build_profile(team_games, team=str(team_a), filters=_Filters(), half_life_days=180.0)
                pB = build_profile(team_games, team=str(team_b), filters=_Filters(), half_life_days=180.0)
                tot = matchup_expected_totals(pA, pB).get("towers")
                if tot is not None and math.isfinite(float(tot.mean)):
                    sig = towers_draft_signal(comp_a.get("macro_scores") or {},
                                              comp_b.get("macro_scores") or {})
                    delta = towers_draft_delta(sig)
                    base_m = float(tot.mean)
                    cor = "#22c55e" if delta > 0.15 else ("#ef4444" if delta < -0.15 else "#9ca3af")
                    st.markdown(
                        f"<div style='border-left:3px solid {cor};background:rgba(255,255,255,0.03);"
                        f"padding:6px 10px;border-radius:6px;margin:6px 0;'>"
                        f"🗼 <b>Torres com este draft:</b> base do confronto {base_m:.1f} → "
                        f"<b style='color:{cor};'>{base_m + delta:.1f}</b> "
                        f"<span style='opacity:0.7;'>({delta:+.1f} torre)</span></div>",
                        unsafe_allow_html=True,
                    )
        except Exception:
            pass

    # ------------------------- Fit time×estilo -------------------------
    with st.container(border=True):
        render_tab_section(st, title="Fit time × estilo",
                           subtitle="Como cada time joga com o tipo de comp que draftou hoje (season atual).")
        fit_cols = st.columns(2)
        for col, team, arch, color in ((fit_cols[0], team_a, arch_a, _COR_A),
                                       (fit_cols[1], team_b, arch_b, _COR_B)):
            with col:
                st.markdown(_team_header_html(team, color), unsafe_allow_html=True)
                cache_key = f"_radar_fit_{team}_{year}_{str(csv_sig)}"
                hist = session_state.get(cache_key)
                if not isinstance(hist, dict):
                    hist = team_style_history(players_df=players_df, team=team,
                                              registry=registry, year=year) \
                        if players_df is not None else {a: {"wins": 0, "n": 0} for a in ARCHETYPES}
                    session_state[cache_key] = hist
                for label in ARCHETYPES:
                    rec = hist.get(label) or {"wins": 0, "n": 0}
                    wins, n = int(rec.get("wins", 0)), int(rec.get("n", 0))
                    is_today = (label == arch)
                    pct = (100.0 * wins / n) if n else float("nan")
                    strong = n >= 5 and pd.notna(pct) and (pct >= 70.0 or pct <= 30.0)
                    mark = "🎯 " if is_today else ""
                    if n == 0:
                        line = f"{mark}{label}: sem jogos"
                    else:
                        line = f"{mark}{label}: <b>{wins}-{n - wins}</b> ({pct:.0f}%, n={n})"
                    style = ""
                    if is_today:
                        style = f"border-left:3px solid {color};padding-left:8px;"
                    if strong and is_today:
                        cor = "#22c55e" if pct >= 70.0 else "#ef4444"
                        line += f" <span style='color:{cor};font-weight:800;'>← sinal</span>"
                    elif is_today and 0 < n < 5:
                        line += " <span style='opacity:0.55;'>(amostra curta)</span>"
                    st.markdown(f"<div style='{style}margin:3px 0;'>{line}</div>", unsafe_allow_html=True)

    return compare
