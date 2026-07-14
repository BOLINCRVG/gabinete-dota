"""Backward-compat re-export shim.

Implementations live in:
  - app_core.stats_config        → STATS_MARKET_CONFIG
  - app_core.stats_scope_core    → scope resolution functions
  - app_core.stats_builders_core → table building and metric computation
"""
from __future__ import annotations

from app_core.stats_builders_core import (
    apply_scope_filters,
    build_championship_stats_table,
    build_stats_table_view,
    build_threshold_lines_by_metric,
    compute_team_metric_pct,
    enrich_resumo_rows_with_stats_detail,
    format_matchup_stats_detail,
)
from app_core.stats_config import STATS_MARKET_CONFIG
from app_core.stats_scope_core import (
    build_championship_scope_frame,
    build_relevant_championship_options,
    scope_label_from_meta,
)

__all__ = [
    "STATS_MARKET_CONFIG",
    "apply_scope_filters",
    "build_championship_scope_frame",
    "build_championship_stats_table",
    "build_relevant_championship_options",
    "build_stats_table_view",
    "build_threshold_lines_by_metric",
    "compute_team_metric_pct",
    "enrich_resumo_rows_with_stats_detail",
    "format_matchup_stats_detail",
    "scope_label_from_meta",
]
