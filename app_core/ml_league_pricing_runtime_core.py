from __future__ import annotations

"""Inferencia de liga/tier e precificacao ML v2 — fatiado de plays_app.py em
16/08/2026 (refatoracao A, corte 3).

Por que aqui e nao la: das 20 dependencias do bloco, 17 JA vinham do app_core —
ele estava no plays_app por inercia historica, nao por necessidade. As outras 3
sao injetadas por configurar_runtime(): APP_ROOT e as duas funcoes que continuam
no plays_app por terem cache do Streamlit amarrado ao ciclo do app.

Os imports abaixo foram copiados LITERALMENTE do plays_app, nao reescritos: ao
reconstrui-los a mao eu errei dois apelidos (team_meta_from_artifact e
MAJOR_LEAGUES sao importados COM ) e a aba quebrou com ImportError. Quando o
codigo ja existe, copiar vence reescrever.

O plays_app RE-IMPORTA tudo daqui: o padrao globals()->ctx exige os nomes
visiveis naquele namespace, senao as abas recebem None e quebram longe.
"""

from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

from app_core.artifact_loaders_core import (
    _artifact_sig,
    _file_sig,
    _load_artifact_cached,
)
from app_core.identity_registry_core import (
    load_identity_registry as core_load_identity_registry,
    team_identity_candidates as core_team_identity_candidates,
    team_identity_league_candidates as core_team_identity_league_candidates,
)
from app_core.ml_adjust_helpers_core import (
    _p_from_diff_points,
    _strength_breakdown,
)
from app_core.ml_artifact_pricing_core import (
    mlcore_p_map_v2 as core_mlcore_p_map_v2,
    team_meta_from_artifact as _team_meta_from_artifact,
)
from app_core.ml_league_inference_core import (
    infer_match_league_from_csv as core_infer_match_league_from_csv,
    infer_team_league_from_csv as core_infer_team_league_from_csv,
)
from app_core.ml_league_runtime_core import (
    MAJOR_LEAGUES as CORE_ML_MAJOR_LEAGUES,
    norm_league_key as core_norm_league_key,
)
from app_core.ml_microseg_core import (
    build_ml_microseg_table as _build_ml_microseg_table_core,
)
from app_core.ml_probability_adjustments_core import (
    apply_ml_microseg_calibration as _apply_ml_microseg_calibration_core,
)
from mlcore.utils import (
    clip_prob,
)


APP_ROOT: Path = Path(".")
# Injetados por configurar_runtime(). Mantidos com o MESMO nome do original para
# o bloco abaixo nao precisar ser reescrito — renomear por texto ja me falhou
# hoje (a substituicao pegou so as chamadas com parenteses).
_load_team_games: Any = None
_ach_layer_for_artifact: Any = None


def configurar_runtime(*, app_root: Path, load_team_games: Any, ach_layer_for_artifact: Any) -> None:
    """Injeta o que continua no plays_app. Chamar UMA vez, antes de usar."""
    global APP_ROOT, _load_team_games, _ach_layer_for_artifact
    APP_ROOT = Path(app_root)
    _load_team_games = load_team_games
    _ach_layer_for_artifact = ach_layer_for_artifact


_MAJOR_LEAGUES = {
    *CORE_ML_MAJOR_LEAGUES,
}


@st.cache_data(show_spinner=False)
def _load_identity_registry_cached(_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    path = APP_ROOT / "identity_registry.json"
    return dict(core_load_identity_registry(path))


def _identity_registry_sig() -> Tuple[str, int, int]:
    path = APP_ROOT / "identity_registry.json"
    if not path.exists():
        return ("", 0, 0)
    return _file_sig(str(path))


def _identity_team_name_candidates(team: str) -> list[str]:
    registry = _load_identity_registry_cached(_identity_registry_sig())
    return list(core_team_identity_candidates(registry, team))


def _identity_team_league_candidates(team: str) -> list[str]:
    registry = _load_identity_registry_cached(_identity_registry_sig())
    out: list[str] = []
    for value in list(core_team_identity_league_candidates(registry, team)):
        text = str(value or "").strip()
        if not text:
            continue
        canon = str(core_norm_league_key(text) or text).strip()
        if canon and canon not in out:
            out.append(canon)
    return out


@st.cache_data(show_spinner=False)
def _league_inference_csv_sigs(
    _primary_sig: Tuple[str, int, int],
    year: Optional[int],
) -> list[Tuple[str, int, int]]:
    out: list[Tuple[str, int, int]] = []
    seen: set[str] = set()

    def _push(path: Path) -> None:
        if not path.exists():
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        out.append(_file_sig(str(path)))

    primary_raw = str((_primary_sig or ("", 0, 0))[0] or "").strip()
    if primary_raw:
        _push(Path(primary_raw))

    candidates: list[Path] = []
    for base in [APP_ROOT / "CSV_DIARIO", APP_ROOT / "CSV_HISTORICO"]:
        if not base.exists():
            continue
        candidates.extend(base.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))

    def _year_score(path: Path) -> int:
        _prefix = str(path.name).split("_", 1)[0].strip()
        return int(_prefix) if _prefix.isdigit() else 0

    try:
        target_year = int(year) if year is not None else 0
    except (TypeError, ValueError):
        target_year = 0
    candidates = sorted(
        candidates,
        key=lambda path: (abs(_year_score(path) - target_year) if target_year else 0, -_year_score(path), path.name),
    )
    for path in candidates:
        _push(path)
    return out


@st.cache_data(show_spinner=False)
def _infer_team_league_from_csv(_csv_sig: Tuple[str, int, int], team: str, year: Optional[int]) -> str:
    aliases = _identity_team_name_candidates(team)
    for _sig in _league_inference_csv_sigs(_csv_sig, year):
        out = core_infer_team_league_from_csv(
            _sig,
            team,
            year,
            norm_league_key_fn=core_norm_league_key,
            major_leagues=_MAJOR_LEAGUES,
            team_aliases=aliases,
            fallback_any_year=True,
        )
        if str(out or "").strip():
            return str(out)
    for league in _identity_team_league_candidates(team):
        if str(league or "").strip():
            return str(league)
    return ""


@st.cache_data(show_spinner=False)
def _infer_match_league_from_csv(_csv_sig: Tuple[str, int, int], teamA: str, teamB: str, year: Optional[int]) -> str:
    aliases_a = _identity_team_name_candidates(teamA)
    aliases_b = _identity_team_name_candidates(teamB)
    for _sig in _league_inference_csv_sigs(_csv_sig, year):
        out = core_infer_match_league_from_csv(
            _sig,
            teamA,
            teamB,
            year,
            norm_league_key_fn=core_norm_league_key,
            major_leagues=_MAJOR_LEAGUES,
            load_team_games_fn=_load_team_games,
            infer_team_league_fn=lambda sig, team, y: core_infer_team_league_from_csv(
                sig,
                team,
                y,
                norm_league_key_fn=core_norm_league_key,
                major_leagues=_MAJOR_LEAGUES,
                team_aliases=_identity_team_name_candidates(team),
                fallback_any_year=True,
            ),
            team_aliases_a=aliases_a,
            team_aliases_b=aliases_b,
            fallback_any_year=True,
        )
        if str(out or "").strip():
            return str(out)
    return ""





def _mlcore_p_map_v2(
    artifact_path: str,
    team_blue: str,
    team_red: str,
    as_of: date,
    ach_override_path: Optional[str] = None,
    league_hint_blue: Optional[str] = None,
    league_hint_red: Optional[str] = None,
    tier_hint_blue: Optional[str] = None,
    tier_hint_red: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return core_mlcore_p_map_v2(
        artifact_path=artifact_path,
        team_blue=team_blue,
        team_red=team_red,
        as_of=as_of,
        ach_override_path=ach_override_path,
        league_hint_blue=league_hint_blue,
        league_hint_red=league_hint_red,
        tier_hint_blue=tier_hint_blue,
        tier_hint_red=tier_hint_red,
        clip_prob_fn=clip_prob,
        artifact_sig_fn=_artifact_sig,
        load_artifact_cached_fn=_load_artifact_cached,
        team_meta_from_artifact_fn=_team_meta_from_artifact,
        ach_layer_for_artifact_fn=_ach_layer_for_artifact,
        strength_breakdown_fn=_strength_breakdown,
        p_from_diff_points_fn=_p_from_diff_points,
    )




@st.cache_data(show_spinner=False)
def _build_ml_microseg_table(
    csv_sig: Tuple[str, int, int],
    art_sig: Tuple[str, int, int],
    lookback_games: int = 1200,
    shrink_k: float = 90.0,
) -> Dict[str, Dict[str, float]]:
    return _build_ml_microseg_table_core(
        _load_team_games(csv_sig),
        _load_artifact_cached(art_sig),
        lookback_games=lookback_games,
        shrink_k=shrink_k,
    )


def _apply_ml_microseg_calibration(
    p_map_cal: float,
    *,
    league: str,
    split: str,
    side: str,
    bo: str,
    phase: str,
    table,
    min_n: int,
    strength: float,
    cap_pp: float,
    hierarchical: bool = True,
    hierarchy_decay: float = 0.80,
):
    return _apply_ml_microseg_calibration_core(
        p_map_cal,
        clip_prob_fn=clip_prob,
        league=league,
        split=split,
        side=side,
        bo=bo,
        phase=phase,
        table=table,
        min_n=min_n,
        strength=strength,
        cap_pp=cap_pp,
        hierarchical=hierarchical,
        hierarchy_decay=hierarchy_decay,
    )
