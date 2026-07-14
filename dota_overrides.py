# -*- coding: utf-8 -*-
"""Overrides de rotulos/termos para o clone Dota (importado pelo clone ANTES
dos modulos de UI renderizarem — mutacao IN-PLACE, so neste processo; o app do
LoL nunca ve isso).

Decisao do usuario (2026-07-08): Baroes->Roshans, Inibidores->Barracks,
Dragoes fica na UI (vazio) ate ele mandar tirar.
"""
from __future__ import annotations

RENAMES = {
    "Barões": "Roshans",
    "Barão": "Roshan",
    "Inibidores": "Barracks",
    "Inibidor": "Barrack",
}


def _rename_text(s):
    if not isinstance(s, str):
        return s
    out = s
    for old, new in RENAMES.items():
        out = out.replace(old, new)
    return out


def apply() -> None:
    from dota_paths import ensure_lol_on_path
    ensure_lol_on_path()
    from app_core import ui_labels, stats_config

    # BO2 existe no Dota (ligas usam muito; ha mercado de empate 1-1).
    # No LoL o seletor segue [1, 3, 5]; aqui incluimos o 2.
    from app_core import matchup_selection_core
    matchup_selection_core.BO_OPTIONS[:] = [1, 2, 3, 5]

    # lista canonica de tabs (mutacao in-place: quem importou ve a mudanca)
    for i, name in enumerate(ui_labels.MARKET_TAB_NAMES):
        ui_labels.MARKET_TAB_NAMES[i] = _rename_text(name)
    for k, v in list(ui_labels.METRIC_PT.items()):
        ui_labels.METRIC_PT[k] = _rename_text(v)

    # registro central de mercados (label/v3_label e afins)
    for cfg in stats_config.STATS_MARKET_CONFIG.values():
        for field, val in list(cfg.items()):
            if isinstance(val, str):
                cfg[field] = _rename_text(val)

    # --- Dragoes -> Megacreeps (pedido dele 2026-07-10) ---
    # Dota nao tem dragoes; tem MEGACREEPS (um lado derruba as 6 rax -> creeps
    # viram megas; mercado sim/nao nas casas). Modelado como total 0/1 com
    # linha 0.5: as janelas Hist mostram "% de jogos com megas".
    cfg_reg = stats_config.STATS_MARKET_CONFIG
    dragoes_key = next((k for k, c in cfg_reg.items()
                        if str(c.get("metric_key")) == "dragons"), None)
    mega_entry = {
        "metric_key": "megacreeps",
        "combined_col": "total_megacreeps",
        "threshold_key": "sc_th_megacreeps",
        "default_thresholds": "0.5",
        "default_line": 0.5,
        "pct_prefix": "% Megacreeps > ",
        "v3_label": "Megacreeps", "v3_key": "megacreeps",
        "mv2_col": "mv2_total_megacreeps",
        "for_col": "megacreeps_for", "against_col": "megacreeps_against",
        "is_time": False,
    }
    # reconstroi o dict NA MESMA ordem, trocando Dragoes por Megacreeps
    # (mutacao in-place: quem importou o dict ve a mudanca)
    items = [(k, v) for k, v in cfg_reg.items()]
    cfg_reg.clear()
    for k, v in items:
        if k == dragoes_key:
            cfg_reg["Megacreeps"] = mega_entry
        else:
            cfg_reg[k] = v
    if dragoes_key is None:
        cfg_reg["Megacreeps"] = mega_entry

    # listas/labels de UI
    for nome in ("Dragões", "Dragões Elementais"):
        if nome in ui_labels.MARKET_TAB_NAMES:
            ui_labels.MARKET_TAB_NAMES.remove(nome)
    if "Megacreeps" not in ui_labels.MARKET_TAB_NAMES:
        try:
            ix = ui_labels.MARKET_TAB_NAMES.index("Torres") + 1
        except ValueError:
            ix = len(ui_labels.MARKET_TAB_NAMES)
        ui_labels.MARKET_TAB_NAMES.insert(ix, "Megacreeps")
    ui_labels.METRIC_PT.pop("dragons", None)
    ui_labels.METRIC_PT["megacreeps"] = "Megacreeps"

    # --- LINHAS DEFAULT NA ESCALA DO DOTA (pedido dele 2026-07-10) ---
    # As defaults do registro sao do LoL (kills 27.5...). Aqui recalculamos das
    # MEDIAS da season atual (CSV_DIARIO) a cada boot — zero manutencao: a
    # linha padrao acompanha o meta (ex.: kills ~52.5, torres ~11.5).
    try:
        import math as _math
        from pathlib import Path as _Path
        import pandas as _pd
        _csvs = sorted((_Path(__file__).resolve().parent / "CSV_DIARIO").glob("*.csv"))
        if _csvs:
            _df = _pd.read_csv(_csvs[-1], usecols=lambda c: c in (
                "gameid", "kills", "towers", "barons", "inhibitors",
                "gamelength", "megacreeps"), low_memory=False)
            _g = _df.groupby("gameid")

            def _line_of(mean_val):
                if mean_val is None or not _math.isfinite(float(mean_val)):
                    return None
                return _math.floor(float(mean_val)) + 0.5

            _means = {
                "kills": _line_of(_g["kills"].sum(min_count=2).mean()),
                "towers": _line_of(_g["towers"].sum(min_count=2).mean()),
                "barons": _line_of(_g["barons"].sum(min_count=2).mean()),      # Roshans
                "inhibitors": _line_of(_g["inhibitors"].sum(min_count=2).mean()),  # Barracks
                "time": _line_of(_df["gamelength"].mean() / 60.0),
                "megacreeps": 0.5,
            }
            _spread = {"kills": 2.0, "time": 2.0, "towers": 1.0,
                       "barons": 0.5, "inhibitors": 1.0, "megacreeps": 0.0}
            for cfg in stats_config.STATS_MARKET_CONFIG.values():
                mk = str(cfg.get("metric_key"))
                ln = _means.get(mk)
                if ln is None:
                    continue
                cfg["default_line"] = float(ln)
                sp = _spread.get(mk, 1.0)
                if sp > 0:
                    cfg["default_thresholds"] = f"{ln - sp:g},{ln:g},{ln + sp:g}"
                else:
                    cfg["default_thresholds"] = f"{ln:g}"

            # lista V2 legada (defaults hardcoded do LoL) — mesma escala nova;
            # dragons sai, megacreeps entra
            from app_core import line_analysis_tab_core as _la
            _v2 = _la._MARKETS_V2_DEFAULT
            _v2[:] = [m for m in _v2 if m.get("key") != "dragons"]
            _v2_by_key = {m.get("key"): m for m in _v2}
            for k, ln in (("kills", _means["kills"]), ("towers", _means["towers"]),
                          ("time", _means["time"])):
                if k in _v2_by_key and ln is not None:
                    _v2_by_key[k]["default_line"] = float(ln)
            if "megacreeps" not in _v2_by_key:
                _v2.insert(2, {"label": "Megacreeps", "key": "megacreeps",
                               "metric_col": "mv2_total_megacreeps",
                               "direction": "Over", "default_line": 0.5})

            # entradas extras HARDCODED do V3 (fora do registro): ML+Totais
            # usa a linha de kills; Kills p/Time = metade do total do jogo;
            # Megacreeps vira MARCAÇÃO sim/não (pedido 11/07); First X ganha
            # os objetivos do DOTA (blood/torre/Roshan — sem dragão/arauto).
            _k = _means.get("kills")
            for entry in _la._V3_MARKETS:
                if entry.get("key") == "ml_totals" and _k is not None:
                    entry["default_line"] = float(_k)
                elif entry.get("key") == "kills_team" and _k is not None:
                    entry["default_line"] = _math.floor(float(_k) / 2.0) + 0.5
                elif entry.get("key") == "megacreeps":
                    entry["binary_market"] = True
                elif entry.get("key") == "first_x":
                    entry["first_x_opts"] = {
                        "First Blood": "firstblood",
                        "First Tower": "firsttower",
                        "First Roshan": "firstbaron",
                    }
    except Exception:
        pass  # defaults do LoL ficam como fallback; nunca derrubar o boot

    # MARKET_TAB_NAMES tambem e re-exportado com outro nome em alguns modulos;
    # como a LISTA e o mesmo objeto, a mutacao acima cobre todos.


apply()
