from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import re
import unicodedata

import numpy as np
import pandas as pd

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline

    SKLEARN_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - fallback path
    CalibratedClassifierCV = None  # type: ignore[assignment]
    ColumnTransformer = None  # type: ignore[assignment]
    ExtraTreesClassifier = None  # type: ignore[assignment]
    SimpleImputer = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    brier_score_loss = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    SKLEARN_IMPORT_ERROR = str(exc)


STAT_COLS: Tuple[str, ...] = (
    "result",
    "diff_kills",
    "diff_towers",
    "diff_roshans",
    "diff_barracks",
    "teamkills",
    "oppkills",
    "time_min",
)
ROLLING_WINDOWS: Tuple[int, ...] = (5, 10)
STD_STATS: Tuple[str, ...] = ("result", "diff_kills", "diff_towers", "time_min")
EWM_SPAN = 5
TRAIN_MIN_ROWS = 200
TRAIN_MIN_MATCHES = 120
HISTORY_MIN_GAMES = 3


@dataclass(frozen=True)
class WinnerMLBundle:
    ready: bool
    reason: str
    estimator: object | None
    numeric_features: Tuple[str, ...]
    categorical_features: Tuple[str, ...]
    model_label: str
    train_rows: int
    holdout_rows: int
    train_matches: int
    holdout_matches: int
    holdout_auc: Optional[float]
    holdout_brier: Optional[float]
    baseline_brier: Optional[float]


def _strip_accents(text: str) -> str:
    raw = str(text or "")
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", raw)
        if not unicodedata.combining(ch)
    )


def norm_team(text: str) -> str:
    base = _strip_accents(str(text or "")).lower().strip()
    for suffix in [" esports", " e-sports", " esport", " gaming", " team", " club"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    base = base.replace(".", " ").replace("_", " ")
    return re.sub(r"\s+", " ", base).strip()


def _safe_mean(df: pd.DataFrame, col: str, window: Optional[int] = None) -> float:
    if col not in df.columns or df.empty:
        return math.nan
    series = pd.to_numeric(df[col], errors="coerce")
    if window is not None:
        series = series.head(window)
    series = series.dropna()
    if series.empty:
        return math.nan
    return float(series.mean())


def _safe_std(df: pd.DataFrame, col: str, window: int) -> float:
    if col not in df.columns or df.empty:
        return math.nan
    series = pd.to_numeric(df[col], errors="coerce").head(window).dropna()
    if len(series) < 2:
        return math.nan
    return float(series.std())


def _safe_ewm_mean(df: pd.DataFrame, col: str, span: int) -> float:
    if col not in df.columns or df.empty:
        return math.nan
    frame = df.sort_values("date", ascending=True) if "date" in df.columns else df.iloc[::-1]
    series = pd.to_numeric(frame[col], errors="coerce").dropna()
    if series.empty:
        return math.nan
    return float(series.ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])


def _current_result_streak(df: pd.DataFrame) -> float:
    if df.empty or "result" not in df.columns:
        return 0.0
    series = pd.to_numeric(df["result"], errors="coerce").dropna()
    if series.empty:
        return 0.0
    streak = 0
    for value in series.head(25):
        is_win = int(value > 0.5)
        if is_win:
            streak = streak + 1 if streak >= 0 else 1
        else:
            streak = streak - 1 if streak <= 0 else -1
    return float(streak)


def _prior_expanding_mean(series: pd.Series) -> pd.Series:
    shifted = pd.to_numeric(series, errors="coerce").shift(1)
    return shifted.expanding(min_periods=1).mean()


def _prior_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    shifted = pd.to_numeric(series, errors="coerce").shift(1)
    return shifted.rolling(window=window, min_periods=1).mean()


def _prior_ewm_mean(series: pd.Series, span: int) -> pd.Series:
    shifted = pd.to_numeric(series, errors="coerce").shift(1)
    return shifted.ewm(span=span, adjust=False, min_periods=1).mean()


def _prior_rolling_std(series: pd.Series, window: int) -> pd.Series:
    shifted = pd.to_numeric(series, errors="coerce").shift(1)
    return shifted.rolling(window=window, min_periods=2).std()


def _prior_result_streak(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int).tolist()
    out = []
    current = 0
    for value in values:
        out.append(float(current))
        if value == 1:
            current = current + 1 if current >= 0 else 1
        else:
            current = current - 1 if current <= 0 else -1
    return pd.Series(out, index=series.index, dtype="float64")


def _safe_side_wr(df: pd.DataFrame, side: str) -> float:
    if df is None or df.empty or "side" not in df.columns or "result" not in df.columns:
        return math.nan
    mask = df["side"].str.lower().str.strip() == side
    sub = df[mask]
    if sub.empty:
        return math.nan
    return float(pd.to_numeric(sub["result"], errors="coerce").dropna().mean())


def _numeric_feature_names() -> Tuple[str, ...]:
    names = [
        "team_hist_games",
        "opp_hist_games",
        "delta_hist_games",
        "team_league_games",
        "opp_league_games",
        "delta_league_games",
        "h2h_games",
        "team_result_streak",
        "opp_result_streak",
        "delta_result_streak",
        "team_radiant_wr",
        "opp_radiant_wr",
        "delta_radiant_wr",
        "team_dire_wr",
        "opp_dire_wr",
        "delta_dire_wr",
    ]
    for stat in STAT_COLS:
        names.extend(
            [
                f"team_hist_{stat}_mean",
                f"opp_hist_{stat}_mean",
                f"delta_hist_{stat}_mean",
                f"team_league_{stat}_mean",
                f"opp_league_{stat}_mean",
                f"delta_league_{stat}_mean",
                f"team_h2h_{stat}_mean",
                f"opp_h2h_{stat}_mean",
                f"delta_h2h_{stat}_mean",
                f"team_ewm{EWM_SPAN}_{stat}_mean",
                f"opp_ewm{EWM_SPAN}_{stat}_mean",
                f"delta_ewm{EWM_SPAN}_{stat}_mean",
                f"delta_recent5_{stat}_momentum",
            ]
        )
        for window in ROLLING_WINDOWS:
            names.extend(
                [
                    f"team_recent{window}_{stat}_mean",
                    f"opp_recent{window}_{stat}_mean",
                    f"delta_recent{window}_{stat}_mean",
                ]
            )
    for stat in STD_STATS:
        names.extend(
            [
                f"team_recent5_{stat}_std",
                f"opp_recent5_{stat}_std",
                f"delta_recent5_{stat}_std",
            ]
        )
    return tuple(names)


NUMERIC_FEATURES = _numeric_feature_names()
CATEGORICAL_FEATURES: Tuple[str, ...] = ()


def _prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    required = {"match_id", "date", "league", "teamname", "opp_teamname", "result"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"CSV sem colunas obrigatorias para ML: {missing}")
    base["date"] = pd.to_datetime(base["date"], errors="coerce", utc=True).dt.tz_convert(None)
    base["match_uid"] = base.get("match_uid", base["match_id"]).astype(str)
    base["league_token"] = base["league"].fillna("Sem campeonato").astype(str).str.strip()
    base["team_token"] = base["teamname"].apply(norm_team)
    base["opp_token"] = base["opp_teamname"].apply(norm_team)
    for col in [
        "result",
        "teamkills",
        "oppkills",
        "towers",
        "opp_towers",
        "roshans",
        "opp_roshans",
        "barracks",
        "opp_barracks",
        "duration",
    ]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")
    if "diff_kills" not in base.columns and {"teamkills", "oppkills"}.issubset(base.columns):
        base["diff_kills"] = pd.to_numeric(base["teamkills"], errors="coerce") - pd.to_numeric(base["oppkills"], errors="coerce")
    if "diff_towers" not in base.columns and {"towers", "opp_towers"}.issubset(base.columns):
        base["diff_towers"] = pd.to_numeric(base["towers"], errors="coerce") - pd.to_numeric(base["opp_towers"], errors="coerce")
    if "diff_roshans" not in base.columns and {"roshans", "opp_roshans"}.issubset(base.columns):
        base["diff_roshans"] = pd.to_numeric(base["roshans"], errors="coerce") - pd.to_numeric(base["opp_roshans"], errors="coerce")
    if "diff_barracks" not in base.columns and {"barracks", "opp_barracks"}.issubset(base.columns):
        base["diff_barracks"] = pd.to_numeric(base["barracks"], errors="coerce") - pd.to_numeric(base["opp_barracks"], errors="coerce")
    if "time_min" not in base.columns and "duration" in base.columns:
        base["time_min"] = pd.to_numeric(base["duration"], errors="coerce") / 60.0
    base = base.sort_values(["date", "match_uid", "team_token", "opp_token"], ascending=True).reset_index(drop=True)
    base = base.drop_duplicates(subset=["match_uid", "team_token"], keep="last").reset_index(drop=True)
    return base


def _build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    base = _prepare_base_frame(df)
    base["team_hist_games"] = base.groupby("team_token", sort=False).cumcount()
    base["team_league_games"] = base.groupby(["team_token", "league_token"], sort=False).cumcount()
    base["h2h_games"] = base.groupby(["team_token", "opp_token"], sort=False).cumcount()
    base["team_result_streak"] = base.groupby("team_token", sort=False)["result"].transform(_prior_result_streak)

    if "side" in base.columns:
        def _side_wr_group(grp: pd.DataFrame) -> pd.DataFrame:
            is_rad = (grp["side"].str.lower().str.strip() == "radiant").astype(float)
            wins = pd.to_numeric(grp["result"], errors="coerce").fillna(0.0)
            rad_wins = (is_rad * wins).shift(1).expanding(min_periods=1).sum()
            rad_games = is_rad.shift(1).expanding(min_periods=1).sum()
            dire_wins = ((1.0 - is_rad) * wins).shift(1).expanding(min_periods=1).sum()
            dire_games = (1.0 - is_rad).shift(1).expanding(min_periods=1).sum()
            return pd.DataFrame({
                "team_radiant_wr": rad_wins / rad_games.replace(0.0, float("nan")),
                "team_dire_wr": dire_wins / dire_games.replace(0.0, float("nan")),
            }, index=grp.index)

        _side_wr_df = base.groupby("team_token", sort=False, group_keys=False).apply(_side_wr_group)
        base = base.join(_side_wr_df)
    else:
        base["team_radiant_wr"] = float("nan")
        base["team_dire_wr"] = float("nan")

    for stat in STAT_COLS:
        base[f"team_hist_{stat}_mean"] = base.groupby("team_token", sort=False)[stat].transform(_prior_expanding_mean)
        base[f"team_league_{stat}_mean"] = base.groupby(["team_token", "league_token"], sort=False)[stat].transform(_prior_expanding_mean)
        base[f"team_h2h_{stat}_mean"] = base.groupby(["team_token", "opp_token"], sort=False)[stat].transform(_prior_expanding_mean)
        base[f"team_ewm{EWM_SPAN}_{stat}_mean"] = base.groupby("team_token", sort=False)[stat].transform(
            lambda series, span=EWM_SPAN: _prior_ewm_mean(series, span)
        )
        for window in ROLLING_WINDOWS:
            base[f"team_recent{window}_{stat}_mean"] = base.groupby("team_token", sort=False)[stat].transform(
                lambda series, size=window: _prior_rolling_mean(series, size)
            )
    for stat in STD_STATS:
        base[f"team_recent5_{stat}_std"] = base.groupby("team_token", sort=False)[stat].transform(
            lambda series: _prior_rolling_std(series, 5)
        )

    team_feature_cols = [
        col
        for col in base.columns
        if col.startswith("team_hist_")
        or col.startswith("team_league_")
        or col.startswith("team_h2h_")
        or col.startswith("team_recent")
        or col.startswith(f"team_ewm{EWM_SPAN}_")
        or col in {"team_hist_games", "team_league_games", "team_result_streak", "team_radiant_wr", "team_dire_wr"}
    ]
    opp_lookup = base[["match_uid", "team_token"] + team_feature_cols].rename(
        columns={
            "team_token": "opp_token",
            **{col: col.replace("team_", "opp_", 1) for col in team_feature_cols},
        }
    )
    base = base.merge(opp_lookup, on=["match_uid", "opp_token"], how="left", validate="many_to_one")

    base["delta_hist_games"] = base["team_hist_games"] - base["opp_hist_games"]
    base["delta_league_games"] = base["team_league_games"] - base["opp_league_games"]
    base["delta_result_streak"] = base["team_result_streak"] - base["opp_result_streak"]
    for _side in ("radiant_wr", "dire_wr"):
        t_col, o_col = f"team_{_side}", f"opp_{_side}"
        base[f"delta_{_side}"] = (
            pd.to_numeric(base[t_col], errors="coerce") - pd.to_numeric(base[o_col], errors="coerce")
            if t_col in base.columns and o_col in base.columns
            else float("nan")
        )
    for stat in STAT_COLS:
        base[f"delta_hist_{stat}_mean"] = base[f"team_hist_{stat}_mean"] - base[f"opp_hist_{stat}_mean"]
        base[f"delta_league_{stat}_mean"] = base[f"team_league_{stat}_mean"] - base[f"opp_league_{stat}_mean"]
        base[f"delta_h2h_{stat}_mean"] = base[f"team_h2h_{stat}_mean"] - base[f"opp_h2h_{stat}_mean"]
        base[f"delta_ewm{EWM_SPAN}_{stat}_mean"] = base[f"team_ewm{EWM_SPAN}_{stat}_mean"] - base[f"opp_ewm{EWM_SPAN}_{stat}_mean"]
        for window in ROLLING_WINDOWS:
            base[f"delta_recent{window}_{stat}_mean"] = base[f"team_recent{window}_{stat}_mean"] - base[f"opp_recent{window}_{stat}_mean"]
        base[f"delta_recent5_{stat}_momentum"] = base[f"delta_recent5_{stat}_mean"] - base[f"delta_hist_{stat}_mean"]
    for stat in STD_STATS:
        base[f"delta_recent5_{stat}_std"] = base[f"team_recent5_{stat}_std"] - base[f"opp_recent5_{stat}_std"]

    base["result"] = pd.to_numeric(base["result"], errors="coerce")
    return base


def _build_pipeline() -> object:
    return Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        (
                            "num",
                            SimpleImputer(strategy="median"),
                            list(NUMERIC_FEATURES),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=6,
                    random_state=7,
                    n_jobs=1,
                ),
            ),
        ]
    )


@lru_cache(maxsize=4)
def winner_ml_bundle_for_path(path_str: str, mtime: float) -> WinnerMLBundle:
    del mtime
    model_label = "ExtraTrees numerico"
    if SKLEARN_IMPORT_ERROR is not None:
        return WinnerMLBundle(
            ready=False,
            reason=f"scikit-learn indisponivel: {SKLEARN_IMPORT_ERROR}",
            estimator=None,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            model_label=model_label,
            train_rows=0,
            holdout_rows=0,
            train_matches=0,
            holdout_matches=0,
            holdout_auc=None,
            holdout_brier=None,
            baseline_brier=None,
        )

    source = Path(path_str)
    if not source.exists():
        return WinnerMLBundle(
            ready=False,
            reason=f"CSV nao encontrado: {source}",
            estimator=None,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            model_label=model_label,
            train_rows=0,
            holdout_rows=0,
            train_matches=0,
            holdout_matches=0,
            holdout_auc=None,
            holdout_brier=None,
            baseline_brier=None,
        )

    raw = pd.read_csv(source, low_memory=False)
    frame = _build_training_frame(raw)
    frame = frame[(frame["team_hist_games"] >= HISTORY_MIN_GAMES) & (frame["opp_hist_games"] >= HISTORY_MIN_GAMES)].copy()
    frame = frame[frame["result"].isin([0, 1])].copy()
    if frame.empty:
        return WinnerMLBundle(
            ready=False,
            reason="Sem amostra suficiente para treinar o ML.",
            estimator=None,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            model_label=model_label,
            train_rows=0,
            holdout_rows=0,
            train_matches=0,
            holdout_matches=0,
            holdout_auc=None,
            holdout_brier=None,
            baseline_brier=None,
        )

    match_order = frame[["match_uid", "date"]].drop_duplicates().sort_values(["date", "match_uid"], ascending=True)
    if len(match_order) < TRAIN_MIN_MATCHES:
        return WinnerMLBundle(
            ready=False,
            reason="Poucas partidas para um holdout temporal confiavel.",
            estimator=None,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            model_label=model_label,
            train_rows=0,
            holdout_rows=0,
            train_matches=0,
            holdout_matches=0,
            holdout_auc=None,
            holdout_brier=None,
            baseline_brier=None,
        )

    split_idx = max(int(len(match_order) * 0.8), 1)
    train_ids = set(match_order.iloc[:split_idx]["match_uid"].astype(str).tolist())
    holdout_ids = set(match_order.iloc[split_idx:]["match_uid"].astype(str).tolist())
    train = frame[frame["match_uid"].astype(str).isin(train_ids)].copy()
    holdout = frame[frame["match_uid"].astype(str).isin(holdout_ids)].copy()
    if len(train) < TRAIN_MIN_ROWS or len(holdout) < max(40, TRAIN_MIN_ROWS // 4):
        return WinnerMLBundle(
            ready=False,
            reason="Amostra insuficiente apos split temporal do ML.",
            estimator=None,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            model_label=model_label,
            train_rows=int(len(train)),
            holdout_rows=int(len(holdout)),
            train_matches=int(len(train_ids)),
            holdout_matches=int(len(holdout_ids)),
            holdout_auc=None,
            holdout_brier=None,
            baseline_brier=None,
        )

    if train["result"].nunique(dropna=True) < 2 or holdout["result"].nunique(dropna=True) < 2:
        return WinnerMLBundle(
            ready=False,
            reason="Split temporal sem variacao de classe suficiente.",
            estimator=None,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            model_label=model_label,
            train_rows=int(len(train)),
            holdout_rows=int(len(holdout)),
            train_matches=int(len(train_ids)),
            holdout_matches=int(len(holdout_ids)),
            holdout_auc=None,
            holdout_brier=None,
            baseline_brier=None,
        )

    # Split train into fit (80%) + calibration (20%) by time
    cal_split_idx = max(int(split_idx * 0.8), 1)
    fit_ids = set(match_order.iloc[:cal_split_idx]["match_uid"].astype(str).tolist())
    cal_ids = set(match_order.iloc[cal_split_idx:split_idx]["match_uid"].astype(str).tolist())
    train_fit = frame[frame["match_uid"].astype(str).isin(fit_ids)].copy()
    train_cal = frame[frame["match_uid"].astype(str).isin(cal_ids)].copy()

    if len(train_fit) < max(TRAIN_MIN_ROWS // 2, 40):
        return WinnerMLBundle(
            ready=False,
            reason="Amostra insuficiente para treino apos reserva de calibracao.",
            estimator=None,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            model_label=model_label,
            train_rows=int(len(train)),
            holdout_rows=int(len(holdout)),
            train_matches=int(len(train_ids)),
            holdout_matches=int(len(holdout_ids)),
            holdout_auc=None,
            holdout_brier=None,
            baseline_brier=None,
        )

    model = _build_pipeline()
    model.fit(train_fit[list(NUMERIC_FEATURES)], train_fit["result"].astype(int))

    calibrated = False
    if (
        CalibratedClassifierCV is not None
        and len(train_cal) >= 30
        and train_cal["result"].nunique(dropna=True) >= 2
    ):
        try:
            cal_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            cal_model.fit(train_cal[list(NUMERIC_FEATURES)], train_cal["result"].astype(int))
            model = cal_model
            calibrated = True
        except Exception:
            pass

    holdout_proba = model.predict_proba(holdout[list(NUMERIC_FEATURES)])[:, 1]
    baseline_prob = float(train["result"].mean())
    baseline = np.full(shape=len(holdout), fill_value=baseline_prob, dtype=float)
    holdout_auc = float(roc_auc_score(holdout["result"].astype(int), holdout_proba))
    holdout_brier = float(brier_score_loss(holdout["result"].astype(int), holdout_proba))
    baseline_brier = float(brier_score_loss(holdout["result"].astype(int), baseline))

    final_label = "ExtraTrees + calibração" if calibrated else "ExtraTrees numérico"
    return WinnerMLBundle(
        ready=True,
        reason="ok",
        estimator=model,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        model_label=final_label,
        train_rows=int(len(train)),
        holdout_rows=int(len(holdout)),
        train_matches=int(len(train_ids)),
        holdout_matches=int(len(holdout_ids)),
        holdout_auc=holdout_auc,
        holdout_brier=holdout_brier,
        baseline_brier=baseline_brier,
    )


def _h2h_stat_block(h2h_games: pd.DataFrame, perspective: str) -> Dict[str, float]:
    if h2h_games.empty:
        return {stat: math.nan for stat in STAT_COLS}
    frame = h2h_games.copy()
    result_t1 = pd.to_numeric(frame.get("result_t1"), errors="coerce")
    diff_kills = pd.to_numeric(frame.get("diff_kills"), errors="coerce")
    diff_towers = pd.to_numeric(frame.get("diff_towers"), errors="coerce")
    diff_roshans = pd.to_numeric(frame.get("diff_roshans"), errors="coerce")
    diff_barracks = pd.to_numeric(frame.get("diff_barracks"), errors="coerce")
    kills_t1 = pd.to_numeric(frame.get("kills_t1"), errors="coerce")
    kills_t2 = pd.to_numeric(frame.get("kills_t2"), errors="coerce")
    time_min = pd.to_numeric(frame.get("time_min"), errors="coerce")
    if perspective == "team2":
        result = 1.0 - result_t1
        teamkills = kills_t2
        oppkills = kills_t1
        diff_kills = -diff_kills
        diff_towers = -diff_towers
        diff_roshans = -diff_roshans
        diff_barracks = -diff_barracks
    else:
        result = result_t1
        teamkills = kills_t1
        oppkills = kills_t2
    out = {
        "result": float(result.dropna().mean()) if not result.dropna().empty else math.nan,
        "diff_kills": float(diff_kills.dropna().mean()) if not diff_kills.dropna().empty else math.nan,
        "diff_towers": float(diff_towers.dropna().mean()) if not diff_towers.dropna().empty else math.nan,
        "diff_roshans": float(diff_roshans.dropna().mean()) if not diff_roshans.dropna().empty else math.nan,
        "diff_barracks": float(diff_barracks.dropna().mean()) if not diff_barracks.dropna().empty else math.nan,
        "teamkills": float(teamkills.dropna().mean()) if not teamkills.dropna().empty else math.nan,
        "oppkills": float(oppkills.dropna().mean()) if not oppkills.dropna().empty else math.nan,
        "time_min": float(time_min.dropna().mean()) if not time_min.dropna().empty else math.nan,
    }
    return out


def _feature_block(
    hist_games: pd.DataFrame,
    league_games: pd.DataFrame,
    h2h_block: Dict[str, float],
    h2h_games: int,
    prefix: str,
) -> Dict[str, object]:
    block: Dict[str, object] = {
        f"{prefix}_hist_games": int(len(hist_games)),
        f"{prefix}_league_games": int(len(league_games)),
        f"{prefix}_result_streak": _current_result_streak(hist_games),
        f"{prefix}_radiant_wr": _safe_side_wr(hist_games, "radiant"),
        f"{prefix}_dire_wr": _safe_side_wr(hist_games, "dire"),
    }
    for stat in STAT_COLS:
        block[f"{prefix}_hist_{stat}_mean"] = _safe_mean(hist_games, stat)
        block[f"{prefix}_league_{stat}_mean"] = _safe_mean(league_games, stat)
        block[f"{prefix}_h2h_{stat}_mean"] = float(h2h_block.get(stat, math.nan))
        block[f"{prefix}_ewm{EWM_SPAN}_{stat}_mean"] = _safe_ewm_mean(hist_games, stat, EWM_SPAN)
        for window in ROLLING_WINDOWS:
            block[f"{prefix}_recent{window}_{stat}_mean"] = _safe_mean(hist_games, stat, window=window)
        block[f"{prefix}_recent5_{stat}_momentum"] = math.nan
    for stat in STD_STATS:
        block[f"{prefix}_recent5_{stat}_std"] = _safe_std(hist_games, stat, 5)
    if prefix == "team":
        block["h2h_games"] = int(h2h_games)
    return block


def _matchup_row(
    team1_hist_games: pd.DataFrame,
    team2_hist_games: pd.DataFrame,
    team1_league_games: pd.DataFrame,
    team2_league_games: pd.DataFrame,
    h2h_games: pd.DataFrame,
) -> pd.DataFrame:
    h2h_count = int(len(h2h_games)) if h2h_games is not None else 0
    h2h_team1 = _h2h_stat_block(h2h_games, "team1")
    h2h_team2 = _h2h_stat_block(h2h_games, "team2")
    row: Dict[str, object] = {}
    row.update(_feature_block(team1_hist_games, team1_league_games, h2h_team1, h2h_count, "team"))
    row.update(_feature_block(team2_hist_games, team2_league_games, h2h_team2, h2h_count, "opp"))
    row["delta_hist_games"] = float(row["team_hist_games"]) - float(row["opp_hist_games"])
    row["delta_league_games"] = float(row["team_league_games"]) - float(row["opp_league_games"])
    row["delta_result_streak"] = float(row["team_result_streak"]) - float(row["opp_result_streak"])
    for stat in STAT_COLS:
        left_hist = row[f"team_hist_{stat}_mean"]
        right_hist = row[f"opp_hist_{stat}_mean"]
        row[f"delta_hist_{stat}_mean"] = float(left_hist) - float(right_hist) if math.isfinite(float(left_hist)) and math.isfinite(float(right_hist)) else math.nan
        left_league = row[f"team_league_{stat}_mean"]
        right_league = row[f"opp_league_{stat}_mean"]
        row[f"delta_league_{stat}_mean"] = float(left_league) - float(right_league) if math.isfinite(float(left_league)) and math.isfinite(float(right_league)) else math.nan
        left_h2h = row[f"team_h2h_{stat}_mean"]
        right_h2h = row[f"opp_h2h_{stat}_mean"]
        row[f"delta_h2h_{stat}_mean"] = float(left_h2h) - float(right_h2h) if math.isfinite(float(left_h2h)) and math.isfinite(float(right_h2h)) else math.nan
        left_ewm = row[f"team_ewm{EWM_SPAN}_{stat}_mean"]
        right_ewm = row[f"opp_ewm{EWM_SPAN}_{stat}_mean"]
        row[f"delta_ewm{EWM_SPAN}_{stat}_mean"] = float(left_ewm) - float(right_ewm) if math.isfinite(float(left_ewm)) and math.isfinite(float(right_ewm)) else math.nan
        for window in ROLLING_WINDOWS:
            left_recent = row[f"team_recent{window}_{stat}_mean"]
            right_recent = row[f"opp_recent{window}_{stat}_mean"]
            row[f"delta_recent{window}_{stat}_mean"] = float(left_recent) - float(right_recent) if math.isfinite(float(left_recent)) and math.isfinite(float(right_recent)) else math.nan
        row[f"delta_recent5_{stat}_momentum"] = (
            float(row[f"delta_recent5_{stat}_mean"]) - float(row[f"delta_hist_{stat}_mean"])
            if math.isfinite(float(row[f"delta_recent5_{stat}_mean"])) and math.isfinite(float(row[f"delta_hist_{stat}_mean"]))
            else math.nan
        )
    for stat in STD_STATS:
        left_std = row[f"team_recent5_{stat}_std"]
        right_std = row[f"opp_recent5_{stat}_std"]
        row[f"delta_recent5_{stat}_std"] = float(left_std) - float(right_std) if math.isfinite(float(left_std)) and math.isfinite(float(right_std)) else math.nan
    for _side in ("radiant_wr", "dire_wr"):
        left_s = row.get(f"team_{_side}", math.nan)
        right_s = row.get(f"opp_{_side}", math.nan)
        row[f"delta_{_side}"] = (
            float(left_s) - float(right_s)
            if math.isfinite(float(left_s)) and math.isfinite(float(right_s))
            else math.nan
        )
    return pd.DataFrame([row], columns=list(NUMERIC_FEATURES))


def predict_winner_matchup(
    bundle: WinnerMLBundle,
    team1_hist_games: pd.DataFrame,
    team2_hist_games: pd.DataFrame,
    team1_league_games: pd.DataFrame,
    team2_league_games: pd.DataFrame,
    h2h_games: pd.DataFrame,
) -> dict:
    if not bundle.ready or bundle.estimator is None:
        return {
            "ready": False,
            "reason": bundle.reason,
            "model_label": bundle.model_label,
            "t1_prob": None,
            "t2_prob": None,
            "h2h_games_used": int(len(h2h_games)) if h2h_games is not None else 0,
            "train_rows": bundle.train_rows,
            "holdout_rows": bundle.holdout_rows,
            "train_matches": bundle.train_matches,
            "holdout_matches": bundle.holdout_matches,
            "holdout_auc": bundle.holdout_auc,
            "holdout_brier": bundle.holdout_brier,
            "baseline_brier": bundle.baseline_brier,
        }
    row = _matchup_row(team1_hist_games, team2_hist_games, team1_league_games, team2_league_games, h2h_games)
    probability = float(bundle.estimator.predict_proba(row[list(NUMERIC_FEATURES)])[:, 1][0])
    probability = max(0.001, min(0.999, probability))
    return {
        "ready": True,
        "reason": "ok",
        "model_label": bundle.model_label,
        "t1_prob": probability,
        "t2_prob": 1.0 - probability,
        "h2h_games_used": int(len(h2h_games)) if h2h_games is not None else 0,
        "train_rows": bundle.train_rows,
        "holdout_rows": bundle.holdout_rows,
        "train_matches": bundle.train_matches,
        "holdout_matches": bundle.holdout_matches,
        "holdout_auc": bundle.holdout_auc,
        "holdout_brier": bundle.holdout_brier,
        "baseline_brier": bundle.baseline_brier,
    }
