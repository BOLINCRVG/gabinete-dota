# -*- coding: utf-8 -*-
"""Gera gabinete_dota_app.py = clone do plays_app.py do LoL + patches Dota.

O clone NAO e mantido a mao: este script re-copia o plays_app.py atual e aplica
os patches abaixo. Se um patch nao ancorar (o LoL mudou), ERRO ALTO — ajustar o
patch aqui. Assim o app do Dota herda evolucoes do LoL com 1 comando.

Patches (redesign decidido 2026-07-08):
  1. sys.path aponta pro repo LoL (imports app_core/mlcore).
  2. page_title/hero: GABINETE DOTA.
  3. CSS vermelho (dota_ui) injetado DEPOIS do css do LoL (override do acento).
  4. Nav sem Radar/Meta (dependem de herois -> P3).
Live-Bet: fica para patch posterior (fica dentro das tabs de resultado).

Uso: python build_clone.py
"""
from __future__ import annotations

from pathlib import Path

from dota_paths import lol_repo_path

LOL_APP = lol_repo_path() / "plays_app.py"
OUT = Path(__file__).with_name("gabinete_dota_app.py")

PATCHES: list[tuple[str, str]] = [
    # 1. bootstrap de imports (primeira linha depois do encoding/docstring)
    (
        "from __future__ import annotations",
        "from __future__ import annotations\n\n"
        "# Portatil (2026-07-10): o usuario manda as pastas num .rar pro amigo —\n"
        "# o repo do LoL e descoberto por vizinhanca (dota_paths), nunca por\n"
        "# caminho absoluto com username.\n"
        "import sys as _dota_sys\n"
        "_dota_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))\n"
        "from dota_paths import ensure_lol_on_path as _dota_lol_path\n"
        "_dota_lol_path()\n"
        "import dota_overrides  # noqa: F401  Baroes->Roshans etc. ANTES de qualquer app_core derivar listas",
    ),
    # 2a. titulo da pagina
    (
        'st.set_page_config(page_title="Plays (ML + Totais)", layout="wide")',
        'st.set_page_config(page_title="Gabinete Dota", page_icon="🛡️", layout="wide")',
    ),
    # 3. css vermelho por cima do css do LoL + overrides de rotulos
    (
        "_inject_global_ui_css(st)",
        "_inject_global_ui_css(st)\n"
        "from dota_ui import inject_global_ui_css as _dota_inject_css\n"
        "_dota_inject_css(st)",
    ),
    # 2b. hero
    (
        '<div class="title">GABINETE</div>',
        '<div class="title">GABINETE <span style="color:#c94a38;">DOTA</span></div>',
    ),
    # 4. nav sem Radar/Meta (P3: herois)
    (
        "_nav_options = list(NAV_TOP_NAV_OPTIONS)",
        "_nav_options = [o for o in NAV_TOP_NAV_OPTIONS if o not in (\"Radar\", \"Meta\")]",
    ),
    # 5b. DIARIA AUTOMATICA: ao entrar no app, carrega sozinho os jogos das
    #     proximas 36h (menos os que ja comecaram), resolvendo nomes por alias;
    #     desconhecidos vao pra previa de mapeamento no expander (minimizado).
    (
        'with st.expander("Jogos (agenda) - colar / adicionar", expanded=False):',
        '''import time as _dota_time
if _dota_time.time() - float(st.session_state.get("dota_diaria_auto_ts", 0.0) or 0.0) > 900.0:
    st.session_state["dota_diaria_auto_ts"] = _dota_time.time()
    try:
        from pandascore_client import upcoming_matches as _dota_up
        from zoneinfo import ZoneInfo as _dota_zi
        _fx = [f for f in _dota_up(hours_ahead=48) if f.get("tier") in ("s", "a", "b")]
        _fx.sort(key=lambda f: str(f.get("scheduled_at") or ""))
        _als = core_load_schedule_aliases(st.session_state, APP_ROOT / "user_team_aliases.json")
        from app_core.schedule_state_core import save_schedule_games as _dota_save_ag
        _games_now = list(core_get_schedule_games(st.session_state))
        _auto, _pend = [], []
        _updated = 0
        _mutated = False
        for _f in _fx:
            _w = pd.Timestamp(_f["scheduled_at"])
            _w = (_w.tz_localize("UTC") if _w.tzinfo is None else _w)
            # jogos que ja comecaram FICAM (usuario consulta info com jogo rolando)
            _wl = _w.tz_convert(_dota_zi("America/Sao_Paulo"))
            _a, _ok_a = core_resolve_schedule_team(str(_f["team_a"]), teams, _als)
            _b, _ok_b = core_resolve_schedule_team(str(_f["team_b"]), teams, _als)
            _psid = str(_f.get("match_id") or "")
            _g = {"date": _wl.strftime("%d/%m"), "time": _wl.strftime("%H:%M"),
                  "league": str(_f.get("league") or ""), "teamA": _a, "teamB": _b,
                  "bo": int(_f.get("bo") or 3) or 3, "lines": "", "ps_id": _psid}
            if not (_ok_a and _ok_b):
                _pend.append({"date": _g["date"], "time": _g["time"], "league": _g["league"],
                              "teamA_raw": str(_f["team_a"]), "teamB_raw": str(_f["team_b"])})
                continue
            # Jogo ADIADO (pedido 13/07): rastreia pelo id do PandaScore —
            # horario mudou -> ATUALIZA a entrada (a hora antiga sai), em vez
            # de duplicar. Rebatida legitima do mesmo par no dia = id novo.
            _hit = None
            if _psid:
                _hit = next((x for x in _games_now
                             if str(x.get("ps_id") or "") == _psid), None)
            if _hit is None:
                _hit = next((x for x in _games_now
                             if not str(x.get("ps_id") or "")
                             and str(x.get("date")) == _g["date"]
                             and str(x.get("teamA")) == _g["teamA"]
                             and str(x.get("teamB")) == _g["teamB"]), None)
            if _hit is not None:
                if (str(_hit.get("time")) != _g["time"] or str(_hit.get("bo")) != str(_g["bo"])
                        or str(_hit.get("ps_id") or "") != _psid):
                    if str(_hit.get("time")) != _g["time"]:
                        _updated += 1
                    _hit["time"], _hit["bo"], _hit["ps_id"] = _g["time"], _g["bo"], _psid
                    _mutated = True
                # limpa duplicatas sem id do mesmo confronto/dia (horas antigas)
                _dups = [x for x in _games_now if x is not _hit
                         and not str(x.get("ps_id") or "")
                         and str(x.get("date")) == _g["date"]
                         and str(x.get("teamA")) == _g["teamA"]
                         and str(x.get("teamB")) == _g["teamB"]]
                if _dups:
                    _games_now = [x for x in _games_now if x not in _dups]
                    _mutated = True
            else:
                _auto.append(_g)
        if _mutated:
            _dota_save_ag(st.session_state, _games_now)
        if _auto:
            core_append_schedule_games(st.session_state, _auto)
        if _pend:
            _pp = core_get_pending_schedule_import(st.session_state)
            _seen = {(str(p.get("date", "")), str(p.get("time", "")), str(p.get("teamA_raw", "")), str(p.get("teamB_raw", ""))) for p in _pp}
            _pp.extend([p for p in _pend if (str(p["date"]), str(p["time"]), str(p["teamA_raw"]), str(p["teamB_raw"])) not in _seen])
            core_save_pending_schedule_import(st.session_state, _pp)
        _parts = []
        if _auto:
            _parts.append(f"{len(_auto)} jogo(s) carregados na agenda")
        if _updated:
            _parts.append(f"{_updated} horário(s) atualizado(s)")
        if _pend:
            _parts.append(f"{len(_pend)} aguardando mapeamento de nomes (abra 'Jogos (agenda)')")
        st.session_state["dota_diaria_auto_msg"] = ("Diária automática: " + "; ".join(_parts) + ".") if _parts else ""
    except Exception as _dota_exc:
        st.session_state["dota_diaria_auto_msg"] = f"Diária automática indisponível: {_dota_exc}"
_dota_msg = str(st.session_state.get("dota_diaria_auto_msg", "") or "")
if _dota_msg:
    (st.warning if "indisponível" in _dota_msg else st.info)(_dota_msg)

# --- painel DIÁRIA: seletor de dia + jogos do dia (nada some ao começar) ---
# Botão SEMPRE visível (pedido 13/07: "imprescindível conseguir ver os
# jogos") — força a busca na hora, ignorando o intervalo do auto-fetch.
_cbt, _cinfo = st.columns([1.4, 5])
with _cbt:
    if st.button("🔄 Procurar jogos", key="dota_diaria_force"):
        st.session_state["dota_diaria_auto_ts"] = 0.0
        st.rerun()
with _cinfo:
    _ts_last = float(st.session_state.get("dota_diaria_auto_ts", 0.0) or 0.0)
    if _ts_last > 0:
        _min_atras = max(0, int((_dota_time.time() - _ts_last) / 60))
        st.caption(f"Agenda buscada há {_min_atras} min (auto a cada 15 min e ao abrir o app).")
    else:
        st.caption("Agenda ainda não buscada nesta sessão.")

_dota_games_all = core_get_schedule_games(st.session_state)
if not _dota_games_all:
    st.info("Sem jogos T1/T2 na agenda — clique em 🔄 Procurar jogos ou aguarde a "
            "próxima janela (busca cobre 48h à frente).")
if _dota_games_all:
    from app_core.schedule_state_core import set_schedule_pending_load as _dota_set_pending
    from app_core.schedule_state_core import save_schedule_games as _dota_save_games
    from zoneinfo import ZoneInfo as _dota_zi2
    _tz_now = pd.Timestamp.now(tz=_dota_zi2("America/Sao_Paulo"))
    _hoje_d = _tz_now.normalize()

    def _dota_parse_ddmm(s):
        try:
            _dd, _mm = str(s).strip().split("/")[:2]
            cand = pd.Timestamp(year=_tz_now.year, month=int(_mm), day=int(_dd), tz=str(_tz_now.tz))
            # virada de ano: dd/mm sem ano — muito no futuro = era do ano passado
            if (cand - _hoje_d).days > 200:
                cand = cand.replace(year=_tz_now.year - 1)
            elif (_hoje_d - cand).days > 200:
                cand = cand.replace(year=_tz_now.year + 1)
            return cand
        except Exception:
            return None

    # Pedido 13/07: dia que PASSOU sai da agenda; seletor mostra só hoje +48h
    # (jogos de hoje seguem visíveis mesmo já iniciados — decisão anterior).
    _keep = []
    for _g in _dota_games_all:
        _dt_g = _dota_parse_ddmm(_g.get("date"))
        if _dt_g is not None and _dt_g < _hoje_d:
            continue
        _keep.append(_g)
    if len(_keep) != len(_dota_games_all):
        _dota_save_games(st.session_state, _keep)
        _dota_games_all = _keep

    _dota_dates = []
    for _g in _dota_games_all:
        _d = str(_g.get("date", "") or "")
        _dt_g = _dota_parse_ddmm(_d)
        if _dt_g is None or _dt_g > _hoje_d + pd.Timedelta(days=2):
            continue
        if _d and _d not in _dota_dates:
            _dota_dates.append(_d)
    _dota_dates.sort(key=lambda s: _dota_parse_ddmm(s) or _hoje_d)
    _hoje = _tz_now.strftime("%d/%m")
if _dota_games_all and _dota_dates:
    _c1, _c2, _c3 = st.columns([1, 3, 1])
    with _c1:
        if str(st.session_state.get("dota_diaria_day", "") or "") not in _dota_dates:
            st.session_state["dota_diaria_day"] = _dota_dates[_dota_dates.index(_hoje) if _hoje in _dota_dates else 0]
        _day = st.selectbox("📅 Dia", _dota_dates, key="dota_diaria_day")
    _day_games = [g for g in _dota_games_all if str(g.get("date", "")) == _day]
    _labels = [f"{g.get('time','')}  {g.get('teamA','')} x {g.get('teamB','')} (BO{g.get('bo','?')})"
               for g in _day_games]
    with _c2:
        _pick = st.selectbox("Jogos do dia", ["", *_labels], key="dota_diaria_pick")
    with _c3:
        st.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)
        _load_clicked = st.button("Carregar", key="dota_diaria_load", width='stretch')
    if _load_clicked and _pick:
        _g = _day_games[_labels.index(_pick)]
        _payload = {"_reset_flow": True}
        if _g.get("teamA") in teams:
            _payload["teamA"] = _g.get("teamA")
        if _g.get("teamB") in teams:
            _payload["teamB"] = _g.get("teamB")
        if str(_g.get("bo")) in ("1", "2", "3", "5"):
            _payload["bo"] = int(_g.get("bo"))
        _dota_set_pending(st.session_state, _payload)
        st.rerun()

with st.expander("Jogos (agenda) - colar / adicionar", expanded=False):''',
    ),
    # 6. BO2 nos seletores de import do proprio plays_app (o principal vem de
    #    matchup_selection_core.BO_OPTIONS, sobrescrito em dota_overrides)
    (
        'default_bo = st.selectbox("BO padrão (para os jogos importados)", [1, 3, 5], key="plays_schedule_default_bo")',
        'default_bo = st.selectbox("BO padrão (para os jogos importados)", [1, 2, 3, 5], key="plays_schedule_default_bo")',
    ),
    (
        'm_bo = st.selectbox("BO", [1,3,5], key="plays_manual_bo")',
        'm_bo = st.selectbox("BO", [1, 2, 3, 5], key="plays_manual_bo")',
    ),
    # 5. DIARIA: botao que puxa a agenda do PandaScore e alimenta o fluxo
    #    NATIVO de importacao (Processar texto -> aliases -> Adicionar).
    (
        '    _alias_path = APP_ROOT / "user_team_aliases.json"',
        '''    if st.button("🗓️ Buscar jogos do dia (Diária/PandaScore)", key="dota_diaria_fetch"):
        try:
            from pandascore_client import upcoming_matches as _dota_up
            from zoneinfo import ZoneInfo as _dota_zi
            import pandas as _dota_pd
            _fx = [f for f in _dota_up(hours_ahead=48) if f.get("tier") in ("s", "a", "b")]
            _fx.sort(key=lambda f: str(f.get("scheduled_at") or ""))
            _ls, _prev = [], ""
            for _f in _fx:
                _w = _dota_pd.Timestamp(_f["scheduled_at"])
                _w = (_w.tz_localize("UTC") if _w.tzinfo is None else _w).tz_convert(_dota_zi("America/Sao_Paulo"))
                _d = _w.strftime("%d/%m")
                if _d != _prev:
                    _ls.append(_d); _prev = _d
                _ls.extend([_w.strftime("%H:%M"), str(_f["team_a"]), "-", str(_f["team_b"])])
            if _ls:
                st.session_state["plays_schedule_text"] = "\\n".join(_ls)
                st.success(f"{len(_fx)} jogo(s) das próximas 36h (tier S/A/B). Agora clique em 'Processar texto'.")
            else:
                st.info("Sem jogos T1/T2 nas próximas 36h.")
        except Exception as _dota_exc:
            st.error(f"Diária indisponível: {_dota_exc}")

    _alias_path = APP_ROOT / "user_team_aliases.json"''',
    ),
]


def main() -> int:
    src = LOL_APP.read_text(encoding="utf-8")
    for old, new in PATCHES:
        if old not in src:
            raise SystemExit(f"PATCH NAO ANCOROU (LoL mudou?): {old[:70]!r}")
        if src.count(old) != 1:
            raise SystemExit(f"PATCH AMBIGUO ({src.count(old)}x): {old[:70]!r}")
        src = src.replace(old, new)
    header = (
        "# -*- coding: utf-8 -*-\n"
        "# GERADO por build_clone.py — NAO EDITAR A MAO (edite os patches).\n"
        f"# Fonte: {LOL_APP}\n"
    )
    OUT.write_text(header + src, encoding="utf-8")
    print(f"OK: {OUT} ({len(src)} chars, {len(PATCHES)} patches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
