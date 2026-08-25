from __future__ import annotations

"""Incremental updater for ML artifacts (fast daily updates).

Goal:
- Avoid full retraining every time the OracleElixir CSV is updated.
- Apply Elo updates only for *new* games since the artifact's last processed date.
- Keep calibrator unchanged (retrain_check can recommend full retrain/calibration when needed).

Usage:
  python -m mlcore.update_incremental --artifact ml_artifact.json --csv CSV_DIARIO\2026_....csv
  python -m mlcore.update_incremental --artifact ml_artifact.json --csv 2014.csv --csv 2015.csv --csv diario.csv

Notes:
- We assume OracleElixir provides stable `gameid`. We use it to dedupe.
- To handle multiple updates within the same date, we also store a tail of recently
  processed gameids and a fingerprint of the last N games.
"""

import os
import shutil
import sys

# Console cp1252 no Windows derruba qualquer print com acento/emoji — e o estouro
# vem no fim, depois do trabalho todo feito. Ver tools/find_silent_failures.py.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

# Allow running as a script from repo root
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "mlcore"

import argparse
import inspect
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List
import hashlib
import json

import pandas as pd

from .artifact import MLArtifact
from .io_oracle import load_oracle_elixir_games
from .tiers import canonical_league, tier_for_league, tier_priority, is_international_league


def _parse_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _resolve_csv_path(p: Path) -> Path:
    """Accept a CSV file path OR a directory. If directory, picks newest *.csv."""
    try:
        p = Path(p)
    except TypeError:
        return p
    if p.exists() and p.is_dir():
        cands = sorted(p.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"No .csv files found in directory: {p}")
        chosen = cands[0]
        print(f"[update_incremental] csv_path is a directory; using newest CSV: {chosen}")
        return chosen
    return p


def _safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # backup
    if path.exists():
        try:
            bak = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, bak)
        except OSError:
            pass
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            f.flush()
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _date_to_iso_or_str(v: Any) -> str:
    d = pd.to_datetime(v, errors="coerce")
    if pd.isna(d):
        return str(v)
    return d.date().isoformat()


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return int(default)


def _should_skip_artifact_write(new_n: int, backfill_suspected: bool) -> bool:
    """True quando NAO se deve reescrever o artifact: 0 jogos novos e sem
    backfill. Reescrever so para bumpar o mtime fazia o params_fitted.json
    parecer 'desatualizado vs artifact' para sempre (a manutencao pula o params
    quando new_games==0, mas antes ainda regravava o artifact -> mtime maior ->
    alerta que nunca limpava)."""
    return int(new_n) == 0 and not bool(backfill_suspected)


def _extract_model_team_meta(model: Any) -> Dict[str, Dict[str, str]]:
    raw = getattr(model, "team_meta", None)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


def _tier_priority_value(v: Any) -> int | None:
    s = str(v or "").strip()
    if not s:
        return None
    try:
        return int(tier_priority(s))
    except (TypeError, ValueError):
        return None


def _reset_team_history(model: Any, team: str) -> None:
    ratings = getattr(model, "ratings", None)
    if isinstance(ratings, dict):
        ratings[str(team)] = 0.0
    games_played = getattr(model, "games_played", None)
    if isinstance(games_played, dict):
        games_played[str(team)] = 0


def _apply_promotion_reset(model: Any, *, team: str, prev_tier: str, new_tier: str) -> bool:
    prev_rank = _tier_priority_value(prev_tier)
    new_rank = _tier_priority_value(new_tier)
    if prev_rank is None or new_rank is None:
        return False
    if str(prev_tier) == str(new_tier):
        return False
    if int(new_rank) < int(prev_rank):
        _reset_team_history(model, str(team))
        return True
    return False


def _tail_fingerprint(games: pd.DataFrame, n: int) -> tuple[str, list[str]]:
    if games.empty or n <= 0:
        return "", []
    tail = games.tail(int(n)).copy()
    # Stable, compact representation (include outcome & league to catch corrections)
    parts: List[str] = []
    gids: List[str] = []
    for r in tail.itertuples(index=False):
        gid = str(r.gameid)
        gids.append(gid)
        dt_s = _date_to_iso_or_str(r.date)
        lg = str(getattr(r, "league", ""))
        b = str(getattr(r, "blue", ""))
        rr = str(getattr(r, "red", ""))
        bw = _safe_int(getattr(r, "blue_win", 0), 0)
        parts.append(f"{gid}|{dt_s}|{lg}|{b}|{rr}|{bw}")
    payload = "\n".join(parts).encode("utf-8", errors="ignore")
    h = hashlib.sha256(payload).hexdigest()
    return h, gids


def _fingerprint_for_gameids(games: pd.DataFrame, gameids: list[str]) -> str:
    """Fingerprint a fixed list of gameids using the same compact row representation.

    This is used to detect *corrections/backfill* to already-seen recent games without
    triggering on normal appends of new games (which naturally change the 'tail').
    """
    if games.empty or not gameids:
        return ""
    if "gameid" not in games.columns:
        return ""

    # After drop_duplicates, gameid should be unique, but be defensive.
    gidx = games.set_index("gameid", drop=False)
    parts: List[str] = []
    for gid in gameids:
        gid = str(gid)
        if gid in gidx.index:
            row = gidx.loc[gid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            dt_s = _date_to_iso_or_str(row.get("date"))
            lg = str(row.get("league", "") or "")
            b = str(row.get("blue", "") or "")
            rr = str(row.get("red", "") or "")
            bw = _safe_int(row.get("blue_win", 0), 0)
            parts.append(f"{gid}|{dt_s}|{lg}|{b}|{rr}|{bw}")
        else:
            # If a previously-seen gid vanished, treat as a change.
            parts.append(f"MISSING|{gid}")

    payload = "\n".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def _infer_new_games(games: pd.DataFrame, *, last_dt: datetime, last_tail_gids: set[str]) -> pd.DataFrame:
    if games.empty:
        return games
    gdt = pd.to_datetime(games["date"], errors="coerce")
    last_date = last_dt.date()
    # Prefer day-level comparison like train_offline meta stores.
    newer = gdt.dt.date > last_date
    same_day = gdt.dt.date == last_date
    if "gameid" in games.columns and last_tail_gids:
        same_day_new = same_day & (~games["gameid"].astype(str).isin(last_tail_gids))
        mask = newer | same_day_new
    else:
        mask = newer
    return games.loc[mask].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast incremental update of ml_artifact.json (apply new games only).")
    ap.add_argument("--artifact", required=True, help="Path to ml_artifact.json to update in-place")
    ap.add_argument("--csv", required=True, action="append", help="OracleElixir match_data CSV (repeatable). Usually pass only the daily CSV.")
    ap.add_argument(
        "--keep-playoffs",
        choices=["auto", "yes", "no", "all"],
        default="auto",
        help="Filter playoffs rows. 'auto' (default) behaves like 'all' but keeps compatibility with older scripts.",
    )
    ap.add_argument("--tail-n", type=int, default=500, help="How many last games to fingerprint/store for same-day updates (default: 500).")
    ap.add_argument(
        "--players",
        default=None,
        help="Path to players_artifact.json to update incrementally. Overrides the path stored in the artifact meta "
             "(which can point to a stale/temp file). Strongly recommended so the players artifact the APP loads is "
             "the one that gets the new games.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write artifact; just print what would change.")
    args = ap.parse_args()

    art_path = Path(args.artifact).expanduser()
    if not art_path.exists():
        raise SystemExit(f"Artifact não encontrado: {art_path}")

    art = MLArtifact.load_json(str(art_path))

    # Ate onde os RATINGS realmente foram. Preferimos ratings_max_date porque
    # data_max_date grava o fim de TODOS os dados, incluindo o holdout que o
    # treino NAO aplicou aos ratings (achado 16/08/2026). Enquanto so existia
    # data_max_date este script respondia "0 jogos novos" com o modelo 30 dias
    # atrasado — Iron Wing tinha 16 jogos em 2026 e games_played=0.
    # Artefato antigo nao tem a chave nova: cai em data_max_date, igual a antes.
    _meta = art.meta or {}
    data_max = _parse_dt(_meta.get("ratings_max_date")) or _parse_dt(_meta.get("data_max_date"))
    trained_at = _parse_dt(_meta.get("trained_at"))
    if data_max is None:
        data_max = trained_at or datetime.fromtimestamp(art_path.stat().st_mtime, tz=timezone.utc)
    if data_max.tzinfo is None:
        data_max = data_max.replace(tzinfo=timezone.utc)

    # Load CSVs (mtime order so newest wins in dedupe)
    raw_paths = list(args.csv or [])
    csv_paths: List[str] = []
    for raw in raw_paths:
        try:
            resolved = _resolve_csv_path(Path(raw))
            csv_paths.append(str(resolved))
        except Exception:
            csv_paths.append(str(raw))
    try:
        # newest last (so it wins in dedupe)
        csv_paths = sorted(csv_paths, key=lambda p: os.path.getmtime(p))
    except Exception:
        pass

    # Resolve keep_playoffs
    keep_playoffs = None
    if args.keep_playoffs == "yes":
        keep_playoffs = True
    elif args.keep_playoffs == "no":
        keep_playoffs = False
    else:
        # 'auto'/'all' -> honor artifact meta if present
        kp = (art.meta or {}).get("keep_playoffs")
        if kp in (True, False):
            keep_playoffs = kp
        else:
            keep_playoffs = None

    dfs: List[pd.DataFrame] = []
    for p in csv_paths:
        df = load_oracle_elixir_games(str(p), keep_playoffs=keep_playoffs)
        if not df.empty:
            df["_src_csv"] = str(p)
            dfs.append(df)

    games = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if games.empty:
        raise SystemExit("Nenhum jogo carregado dos CSVs informados.")

    # Deduplicate by gameid and sort
    if "gameid" in games.columns:
        games["gameid"] = games["gameid"].astype(str)
        games = games.drop_duplicates(subset=["gameid"], keep="last").reset_index(drop=True)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"]).reset_index(drop=True)
    games = games.sort_values(["date", "gameid"]).reset_index(drop=True)


    # Tail info from artifact
    last_tail = (art.meta or {}).get("tail") or {}
    last_tail_gids_list = [str(x) for x in (last_tail.get("gameids") or [])]
    last_tail_gids = set(last_tail_gids_list)
    last_fp = str(last_tail.get("fingerprint") or "").strip()
    tail_n = int(last_tail.get("n") or int(args.tail_n) or 500)

    # Detect backfill/corrections in already-seen recent games.
    # Compare fingerprint of the *previously stored* tail gameids, so new games don't trigger false alarms.
    cur_tail_fp, cur_tail_gids = _tail_fingerprint(games, tail_n)
    cur_prev_fp = _fingerprint_for_gameids(games, last_tail_gids_list) if last_tail_gids_list else ""
    backfill_suspected = bool(last_fp) and bool(last_tail_gids_list) and (cur_prev_fp != last_fp)

    new_df = _infer_new_games(games, last_dt=data_max, last_tail_gids=last_tail_gids)
    new_n = len(new_df)

    # Team metadata (league/tier). Keep it *persistent* across runs and update it when new games arrive.
    # NOTE: plays_app prefers art.meta['team_meta'] when present, so we must keep this dict in sync.
    team_meta: Dict[str, Dict[str, str]] = dict((art.meta or {}).get("team_meta") or {})

    # Apply updates
    if new_n > 0:
        # Detect update() signature once (compat with older artifacts)
        try:
            sig = inspect.signature(art.model.update)
            params = set(sig.parameters.keys())
        except (TypeError, ValueError):
            params = set()

        use_new = ("blue_league" in params) or ("red_league" in params)
        use_legacy = ("league" in params) or ("tier" in params)

        def _home(team: str, fallback_league: str) -> tuple[str, str]:
            m = team_meta.get(team) or {}
            stored_lg = str(canonical_league(m.get("league") or "") or "").strip()
            fb_lg = str(canonical_league(fallback_league or "") or "").strip()

            # Prefer the match league if it's a *domestic* league (fixes teams that changed league/season).
            if fb_lg and (not is_international_league(fb_lg)):
                lg = fb_lg
            else:
                lg = stored_lg or fb_lg or "OTHER"

            tr = str(tier_for_league(lg))
            return lg, tr

        for r in new_df.itertuples(index=False):
            blue = str(r.blue)
            red = str(r.red)
            bw = int(r.blue_win)
            lg_match = str(canonical_league(getattr(r, "league", "") or "OTHER") or "OTHER")

            bl, bt = _home(blue, lg_match)
            rl, rt = _home(red, lg_match)
            # Promotion reset: if a team moved to a *stronger* tier, reset residual evidence.
            prev_bt = str((team_meta.get(blue) or {}).get("tier") or "")
            prev_rt = str((team_meta.get(red) or {}).get("tier") or "")
            _apply_promotion_reset(art.model, team=blue, prev_tier=prev_bt, new_tier=bt)
            _apply_promotion_reset(art.model, team=red, prev_tier=prev_rt, new_tier=rt)

            # Update team_meta: liga = a atual; TIER = o MELHOR já visto.
            # BUG CRÍTICO corrigido 24/07 (mesma raiz do train_offline): gravar o
            # tier do jogo atual fazia LCK -> KeSPA Cup (MINOR) -> LCK parecer
            # "promoção" e zerar o histórico do time. Os 10 times da LCK ficaram
            # com 2-14 jogos em vez de 300-500. Rebaixar o tier num torneio
            # pontual não pode apagar o passado do time.
            def _keep_best_tier(_team: str, _new_tier: str) -> str:
                _prev = str((team_meta.get(_team) or {}).get("tier") or "")
                if not _prev:
                    return str(_new_tier)
                try:
                    return str(_new_tier) if tier_priority(str(_new_tier)) < tier_priority(_prev) else _prev
                except (TypeError, ValueError):
                    return str(_new_tier)

            team_meta[str(blue)] = {"league": str(bl), "tier": _keep_best_tier(str(blue), bt)}
            team_meta[str(red)] = {"league": str(rl), "tier": _keep_best_tier(str(red), rt)}

            # MOV (diff de kills) para o termo MOVDA — inerte quando o artefato
            # tem mov_lambda=0 (artefatos antigos) ou quando kills faltam.
            _mov_val = None
            try:
                _bk = float(getattr(r, "blue_teamkills"))
                _rk = float(getattr(r, "red_teamkills"))
                if _bk == _bk and _rk == _rk:  # not-NaN
                    _mov_val = _bk - _rk
            except (TypeError, ValueError, AttributeError):
                _mov_val = None

            if use_new:
                art.model.update(blue, red, bw, blue_league=bl, red_league=rl, blue_tier=bt, red_tier=rt, mov=_mov_val)
            elif use_legacy:
                art.model.update(blue, red, bw, league=lg_match, tier=tier_for_league(lg_match))
            else:
                art.model.update(blue, red, bw)

        # Sync model.team_meta (newer artifacts) and recompute rank caps if supported
        try:
            if hasattr(art.model, "set_team_meta"):
                art.model.set_team_meta(team_meta)
            if hasattr(art.model, "compute_rank_caps"):
                art.model.rank_caps_by_tier = art.model.compute_rank_caps(team_meta)
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            art.model.finalize_meta()
        except AttributeError:
            pass

    # Optional: update players_layer artifact using the same new games
    players_update_report: Dict[str, Any] = {"enabled": False}
    try:
        pl_meta = dict((art.meta or {}).get("players_layer") or {})
        # Caminho explícito (--players) SEMPRE vence o do meta: o meta pode
        # apontar p/ um arquivo temporário/antigo (ex.: tmp_sos800_players.json
        # de um experimento), fazendo os jogos novos irem p/ o arquivo errado
        # enquanto o app carrega o players_artifact.json. Resultado: players
        # "em drift" para sempre. Passe --players <o que o app usa>.
        p_path = str(args.players or pl_meta.get("players_artifact") or "")
        players_enabled = bool(args.players) or bool(pl_meta.get("enabled"))
        if players_enabled and p_path and new_n > 0:
            from .players_layer import PlayersArtifact, load_match_player_rows, lineup_for_game

            if os.path.exists(p_path):
                p_art = PlayersArtifact.load_json(p_path)
                new_gids = set(new_df["gameid"].astype(str).tolist())
                pr_new = load_match_player_rows([str(p) for p in csv_paths], keep_playoffs=keep_playoffs, gameids=new_gids)

                used = 0
                for r in new_df.itertuples(index=False):
                    gid = str(r.gameid)
                    d = r.date
                    d = d.date() if isinstance(d, datetime) else (d if isinstance(d, date) else None)
                    if d is None:
                        continue
                    league = str(getattr(r, "league", "") or "")
                    bl, rl = lineup_for_game(pr_new, gid)
                    if not bl or not rl:
                        continue
                    p_art.update_game(game_date=d, league=league, blue_lineup=bl, red_lineup=rl, blue_win=int(r.blue_win))
                    used += 1

                p_art.save_json(p_path)
                players_update_report = {
                    "enabled": True,
                    "players_artifact": p_path,
                    "new_games": int(new_n),
                    "games_used": int(used),
                    "updated_at": p_art.updated_at,
                }
    except Exception as e:
        players_update_report = {"enabled": False, "error": str(e)}

    # Update artifact meta
    meta = dict(art.meta or {})
    meta["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["data_min_date"] = pd.to_datetime(games["date"]).min().date().isoformat()
    meta["data_max_date"] = pd.to_datetime(games["date"]).max().date().isoformat()
    # O incremental aplica TODO jogo novo direto nos ratings — nao tem holdout —
    # entao aqui os dois carimbos coincidem: o modelo esta mesmo em dia.
    meta["ratings_max_date"] = meta["data_max_date"]
    meta["n_games_total"] = len(games)
    # Persist updated team_meta (league/tier) back to the artifact meta.
    # If the artifact didn't have meta.team_meta (older artifacts), fall back to model.team_meta.
    if (not team_meta) and hasattr(art.model, "team_meta") and isinstance(art.model.team_meta, dict):
        team_meta = _extract_model_team_meta(art.model)
    if team_meta:
        meta["team_meta"] = team_meta

    meta["tail"] = {
        "n": int(tail_n),
        "fingerprint": cur_tail_fp,
        "gameids": cur_tail_gids,
    }
    meta["incremental"] = {
        "last_run_at": meta["trained_at"],
        "added_games": int(new_n),
        "csv_used": [str(p) for p in csv_paths],
        "keep_playoffs": keep_playoffs,
        "backfill_suspected": bool(backfill_suspected),
        "note": "If backfill_suspected=True, consider full retrain (CSV had corrections in recent games).",
    }

    if players_update_report.get("enabled") or players_update_report.get("error"):
        meta["players_layer_update"] = players_update_report
    art.meta = meta

    # Report
    print(f"[update_incremental] artifact: {art_path}")
    print(f"[update_incremental] last data_max_date: {data_max.date().isoformat()}")
    print(f"[update_incremental] csv games total: {len(games)}")
    print(f"[update_incremental] new games applied: {new_n}")
    if backfill_suspected:
        print("[update_incremental] WARNING: possible backfill/corrections detected in recent games (tail fingerprint changed).")
        print("[update_incremental] Recommendation: run full offline retrain to fully incorporate corrections.")

    if args.dry_run:
        print("[update_incremental] dry-run: artifact not written.")
        return

    # Sem jogos novos e sem backfill: NÃO reescrever o artifact (preserva mtime).
    # Ver _should_skip_artifact_write para o porquê (alerta de params que nunca
    # limpava por causa do bump gratuito de mtime).
    if _should_skip_artifact_write(new_n, backfill_suspected):
        print("[update_incremental] no new games; artifact unchanged (mtime preserved).")
        return

    _safe_write_json(art_path, art.to_dict())
    print("[update_incremental] saved.")


if __name__ == "__main__":
    main()
