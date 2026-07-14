from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from .elo import EloModel
from .calibration import PlattCalibrator


def _parse_ver(v: str) -> Tuple[int, int, int]:
    try:
        parts = str(v).split(".")
        a = int(parts[0]) if len(parts) > 0 else 0
        b = int(parts[1]) if len(parts) > 1 else 0
        c = int(parts[2]) if len(parts) > 2 else 0
        return (a, b, c)
    except Exception:
        return (0, 0, 0)


@dataclass
class MLArtifact:
    """Versioned artifact containing everything needed for predictions."""

    version: str
    model: EloModel
    calibrator: PlattCalibrator
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model": self.model.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MLArtifact":
        version = str(d.get("version", "0.1.0"))
        model = EloModel.from_dict(d.get("model") or {})
        cal = PlattCalibrator.from_dict(d.get("calibrator") or {})
        meta = dict(d.get("meta") or {})

        # Backward-compat: artifacts trained before v0.4.0 implicitly depended on
        # hardcoded tier/league baselines and did not use shrink.
        if _parse_ver(version) < (0, 4, 0):
            try:
                from .tiers import LEAGUE_BASE_POINTS, TIER_BASE_POINTS  # legacy dicts
                # replicate old behaviour by injecting baselines into learned-strength slots
                model.league_strength = {str(k): float(v) for k, v in (LEAGUE_BASE_POINTS or {}).items()}
                model.tier_strength = {str(k): float(v) for k, v in (TIER_BASE_POINTS or {}).items()}
                # disable shrink so residual_eff == residual_raw
                try:
                    model.params.alpha = 0.0
                except Exception:
                    pass
                # ranking caps did not exist
                model.rank_caps_by_tier = {}
            except Exception:
                pass

        return cls(version=version, model=model, calibrator=cal, meta=meta)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "MLArtifact":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
