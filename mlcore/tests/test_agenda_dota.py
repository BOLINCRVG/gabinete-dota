# -*- coding: utf-8 -*-
"""Agenda do Dota pela Liquipedia (17/11/2026): leitura do HTML de Liquipedia:Matches."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import agenda_dota as A  # noqa: E402


def _bloco(ts, a, b, bo=3, pagina="European Pro League/41", rot="EPL Season 41 - November 17"):
    def op(nome, esq):
        cls = "match-info-header-opponent match-info-header-opponent-left" if esq else "match-info-header-opponent"
        return (f'<div class="{cls}"><div class="block-team"><span class="name">'
                f'<a href="/dota2/x" title="{nome}">{nome[:3]}</a></span></div></div>')
    return (f'<div class="match-info"><span class="timer-object" data-timestamp="{ts}">x</span>'
            f'<div class="match-info-header">{op(a, True)}<span>(Bo{bo})</span>{op(b, False)}</div>'
            f'<div class="match-info-tournament"><span><a href="/dota2/{pagina}" title="{pagina}#November_17">'
            f'<span>{rot}</span></a></span></div></div>')


def _cartao(ts, a, b, vencedor=False, bo=3):
    def op(nome, win):
        cls = "match-ticker-card&#95;&#95;opponent" + (" winner" if win else "")
        return (f'<div class="{cls}"><div class="block-team"><span class="name team-name-dynamic">'
                f'<a href="/dota2/x" title="{nome}"><div class="team-shortname">X</div></a></span></div></div>')
    return (f'<div class="match-ticker-card"><div class="match-ticker-card&#95;&#95;header"><span class="timer-object" '
            f'data-timestamp="{ts}">x</span><div>November 24 - Bo{bo}</div></div>'
            f'<div class="match-ticker-card&#95;&#95;opponents">{op(a, vencedor)}{op(b, False)}</div>'
            f'<div class="match-ticker-card-tournament"><span class="match-ticker-card-tournament-name">'
            f'<a href="/dota2/European_Pro_League/41#November_24" title="European Pro League/41">EPL Season 41</a>'
            f'</span></div></div>')


class TestAgendaDotaCartoes(unittest.TestCase):
    """24/11/2026: a Liquipedia trocou para cartoes; a agenda ficou vazia de 20 a 24/11."""

    def test_le_cartao_com_e_sem_placar(self):
        h = _cartao(1795503600, "Angry Birds", "Team Synapse") + _cartao(1795503600, "Aurora Gaming", "LGD Gaming", True)             + _cartao(1795503600, "TBD", "Team Spirit")
        js = A.ler(h)
        self.assertEqual([(j["time_a"], j["time_b"]) for j in js],
                         [("Angry Birds", "Team Synapse"), ("Aurora Gaming", "LGD Gaming")])
        self.assertEqual((js[0]["bo"], js[0]["torneio"], js[0]["pagina"]), (3, "EPL Season 41", "European Pro League/41"))


class TestAgendaDota(unittest.TestCase):
    def test_le_partidas(self):
        h = _bloco(1794909600, "Power Rangers", "Team Synapse (page does not exist)") + \
            _bloco(1794920400, "LEGION (team)", "LGD Gaming", bo=2)
        js = A.ler(h)
        self.assertEqual(len(js), 2)
        self.assertEqual(js[0]["time_b"], "Team Synapse")
        self.assertEqual(js[0]["inicio_utc"], "2026-11-17 10:00:00+00")
        self.assertEqual((js[0]["bo"], js[0]["pagina"]), (3, "European Pro League/41"))
        self.assertEqual(js[1]["time_a"], "LEGION")

    def test_ignora_tbd_e_repetidos(self):
        h = _bloco(1794909600, "TBD", "Team Synapse") + _bloco(1794909600, "A", "B") + _bloco(1794909600, "A", "B")
        self.assertEqual(len(A.ler(h)), 1)


if __name__ == "__main__":
    unittest.main()
