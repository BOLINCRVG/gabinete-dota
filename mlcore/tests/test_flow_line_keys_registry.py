# -*- coding: utf-8 -*-
"""Equivalência da fase 3 do registro central de mercados (2026-07-17).

As flow line keys ("lines_kills"..."lines_time") passaram a DERIVAR de
STATS_MARKET_CONFIG (flow_line_keys) em lines_state, flow_runtime_core,
schedule_state_core, analysis_runtime_state_core e matchup_selection_core.
Este teste trava os valores HISTÓRICOS exatos — inclusive a ORDEM dentro das
assinaturas de seed, onde qualquer troca mudaria o seed de simulação.
(Mesma metodologia dos passos 1-2: test_stats_config_registry.py e
test_v3_markets_registry.py.)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Snapshot LITERAL das chaves como eram antes da derivação (2026-07-17).
_EXPECTED_CORE_KEYS = (
    "lines_kills",
    "lines_towers",
    "lines_dragons",
    "lines_barons",
    "lines_inhib",
    "lines_time",
)

_EXPECTED_LINE_KEY_BY_LABEL = {
    "Kills": "lines_kills",
    "Torres": "lines_towers",
    "Dragões": "lines_dragons",
    "Nashors": "lines_barons",
    "Inibidores": "lines_inhib",
    "Tempo": "lines_time",
}

_EXPECTED_LINE_FIELDS_DEFAULTS = {
    "lines_kills": "",
    "lines_towers": "",
    "lines_dragons": "",
    "lines_barons": "",
    "lines_inhib": "",
    "lines_time": "",
    "teamA_kills_line_text": "",
    "teamB_kills_line_text": "",
    "hc_kills_text": "",
    "hc_kills_team": "",
    "hc_towers_text": "",
    "hc_dragons_text": "",
    "combo_kills_text": "",
    "combo_time_text": "",
}

_EXPECTED_FLOW_LINE_KEYS = [
    "lines_kills",
    "lines_towers",
    "lines_dragons",
    "lines_barons",
    "lines_inhib",
    "lines_time",
    "teamA_kills_line_text",
    "teamB_kills_line_text",
    "hc_kills_text",
    "hc_kills_team",
    "hc_towers_text",
    "hc_dragons_text",
    "combo_kills_text",
    "combo_time_text",
]

_EXPECTED_SCHEDULE_LINE_FIELDS = (
    "lines_kills",
    "lines_towers",
    "lines_dragons",
    "lines_barons",
    "lines_inhib",
    "lines_time",
    "teamA_kills_line_text",
    "teamB_kills_line_text",
    "hc_kills_team",
    "hc_kills_text",
    "hc_towers_text",
    "hc_dragons_text",
    "combo_kills_text",
    "combo_time_text",
)

# analysis_runtime NÃO inclui hc_kills_team (histórico; o seed nunca o usou).
_EXPECTED_ANALYSIS_LINE_KEYS = (
    "lines_kills",
    "lines_towers",
    "lines_dragons",
    "lines_barons",
    "lines_inhib",
    "lines_time",
    "teamA_kills_line_text",
    "teamB_kills_line_text",
    "hc_kills_text",
    "hc_towers_text",
    "hc_dragons_text",
    "combo_kills_text",
    "combo_time_text",
)

_EXPECTED_AGENDA_INPUT_LABELS = {
    "lines_kills": "Kills (ex: 26.5, 27.5)",
    "lines_towers": "Torres (ex: 10.5)",
    "lines_dragons": "Dragões (ex: 4.5)",
    "lines_barons": "Barões (ex: 0.5)",
    "lines_inhib": "Inibidores (ex: 1.5)",
    "lines_time": "Tempo (ex: 32, 32.5)",
}

# Valores-sentinela distintos por chave: se a ordem dentro do seed trocar,
# a fatia capturada diverge.
_SENTINEL_LINE_TEXTS = {
    "lines_kills": "K27.5",
    "lines_towers": "T11.5",
    "lines_dragons": "D4.5",
    "lines_barons": "B1.5",
    "lines_inhib": "I1.5",
    "lines_time": "M32.5",
}


class TestFlowLineKeysRegistry(unittest.TestCase):
    def test_registry_helper_matches_historical_keys(self) -> None:
        from app_core.stats_config import flow_line_keys
        self.assertEqual(flow_line_keys(), _EXPECTED_CORE_KEYS)

    def test_line_key_by_label_matches_historical(self) -> None:
        from app_core.stats_config import line_key_by_label
        derived = line_key_by_label()
        self.assertEqual(derived, _EXPECTED_LINE_KEY_BY_LABEL)
        self.assertEqual(list(derived.keys()), list(_EXPECTED_LINE_KEY_BY_LABEL.keys()))

    def test_lines_state_defaults_literal(self) -> None:
        from app_core.lines_state import LINE_FIELDS_DEFAULTS
        self.assertEqual(LINE_FIELDS_DEFAULTS, _EXPECTED_LINE_FIELDS_DEFAULTS)
        self.assertEqual(list(LINE_FIELDS_DEFAULTS.keys()), list(_EXPECTED_LINE_FIELDS_DEFAULTS.keys()))

    def test_flow_runtime_keys_literal(self) -> None:
        from app_core.flow_runtime_core import _FLOW_LINE_KEYS
        self.assertEqual(list(_FLOW_LINE_KEYS), _EXPECTED_FLOW_LINE_KEYS)

    def test_schedule_state_fields_literal(self) -> None:
        from app_core.schedule_state_core import _SCHEDULE_LINE_FIELDS
        self.assertEqual(tuple(_SCHEDULE_LINE_FIELDS), _EXPECTED_SCHEDULE_LINE_FIELDS)

    def test_analysis_runtime_keys_literal(self) -> None:
        from app_core.analysis_runtime_state_core import _LINE_KEYS
        self.assertEqual(tuple(_LINE_KEYS), _EXPECTED_ANALYSIS_LINE_KEYS)

    def test_agenda_input_labels_cover_all_core_keys(self) -> None:
        from app_core.matchup_selection_core import _LINE_INPUT_LABELS
        self.assertEqual(_LINE_INPUT_LABELS, _EXPECTED_AGENDA_INPUT_LABELS)

    def _capture_seed_args(self, fn, **kwargs):
        captured = []

        def recording_seed_fn(*args):
            captured.append(list(args))
            return 1337

        fn(
            {},
            stable_seed_fn=recording_seed_fn,
            sanitize_marginal_band_coeffs_fn=lambda _x: {},
            season_label="2026",
            team_a="T1",
            team_b="GEN",
            bo=3,
            year_opt="2026",
            split_opt="Todos",
            playoffs_opt="Todos",
            league_mode="Auto",
            fixed_league="",
            window_opt="Ano todo",
            weight_mode="uniform",
            half_life_days=None,
            n_sims=1000,
            totals_mode="Auto",
            map_mode_ui="Todos",
            line_texts=dict(_SENTINEL_LINE_TEXTS),
            **kwargs,
        )
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_analysis_signature_line_order_unchanged(self) -> None:
        """As 6 linhas core entram no seed nas posições 16..21, na ordem histórica."""
        from app_core.analysis_runtime_state_core import build_analysis_signature
        args = self._capture_seed_args(build_analysis_signature, ml_mode="Auto")
        expected = [_SENTINEL_LINE_TEXTS[k] for k in _EXPECTED_CORE_KEYS]
        self.assertEqual(args[16:22], expected)
        # E os extras seguem imediatamente, vazios ("" para chaves ausentes).
        self.assertEqual(args[22:29], [""] * 7)

    def test_consistency_signature_line_order_unchanged(self) -> None:
        """Sem ml_mode no seed de consistência: linhas core nas posições 15..20."""
        from app_core.analysis_runtime_state_core import build_consistency_analysis_signature
        args = self._capture_seed_args(build_consistency_analysis_signature)
        expected = [_SENTINEL_LINE_TEXTS[k] for k in _EXPECTED_CORE_KEYS]
        self.assertEqual(args[15:21], expected)
        self.assertEqual(args[21:28], [""] * 7)


if __name__ == "__main__":
    unittest.main()
