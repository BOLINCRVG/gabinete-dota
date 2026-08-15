# -*- coding: utf-8 -*-
"""Testes do motor da aba Meta (app_core/meta_patch_core.py).

Pilar 2 da força-tarefa (23/07): o módulo tinha lógica pura importante
(presença = pick%+ban%, WR shrinkado k=20, tiers por limiar, delta entre
patches) e ZERO teste. Aqui travamos a matemática com um DataFrame sintético
e valores calculados na mão."""
import math
import unittest

import pandas as pd

from app_core.meta_patch_core import (
    WR_SHRINK_K,
    _norm_key,
    _patch_sort_key,
    compute_meta_table,
    patches_by_recency,
    tier_for_presence,
)


class TierPorPresencaTests(unittest.TestCase):
    """S ≥60 · A ≥35 · B ≥15 · C ≥5 · D <5 — travar as bordas exatas."""

    def test_bordas_dos_tiers(self):
        self.assertEqual(tier_for_presence(60.0), "S")
        self.assertEqual(tier_for_presence(59.99), "A")
        self.assertEqual(tier_for_presence(35.0), "A")
        self.assertEqual(tier_for_presence(34.99), "B")
        self.assertEqual(tier_for_presence(15.0), "B")
        self.assertEqual(tier_for_presence(14.99), "C")
        self.assertEqual(tier_for_presence(5.0), "C")
        self.assertEqual(tier_for_presence(4.99), "D")
        self.assertEqual(tier_for_presence(0.0), "D")


class PatchSortTests(unittest.TestCase):

    def test_sort_key_ordena_por_major_minor(self):
        self.assertEqual(_patch_sort_key("14.1"), (14, 1))
        self.assertEqual(_patch_sort_key("14.10"), (14, 10))
        self.assertGreater(_patch_sort_key("14.10"), _patch_sort_key("14.9"))
        self.assertGreater(_patch_sort_key("15.1"), _patch_sort_key("14.20"))

    def test_sort_key_lixo_vai_pro_fim(self):
        self.assertEqual(_patch_sort_key("nan"), (-1, -1))
        self.assertEqual(_patch_sort_key(None), (-1, -1))


class NormKeyTests(unittest.TestCase):

    def test_normaliza_removendo_nao_alfanumerico_e_caixa(self):
        self.assertEqual(_norm_key("Dr. Mundo"), "drmundo")
        self.assertEqual(_norm_key("Kai'Sa"), "kaisa")
        self.assertEqual(_norm_key("  LeBlanc "), "leblanc")
        self.assertEqual(_norm_key(None), "")


def _draft_scenario() -> pd.DataFrame:
    """2 mapas no patch 14.1 (+1 no 14.0 para o delta). Valores à mão nos testes."""
    rows = [
        # patch 14.1, g1
        {"gameid": "g1", "patch": "14.1", "result": 1, "pick1": "Ahri", "pick2": "Zed", "ban1": "Yasuo"},
        {"gameid": "g1", "patch": "14.1", "result": 0, "pick1": "Lux", "pick2": None, "ban1": "Ahri"},
        # patch 14.1, g2
        {"gameid": "g2", "patch": "14.1", "result": 0, "pick1": "Ahri", "pick2": "Lux", "ban1": "Zed"},
        {"gameid": "g2", "patch": "14.1", "result": 1, "pick1": "Zed", "pick2": None, "ban1": "Ahri"},
        # patch 14.0, g0 (para delta): Ahri presença menor
        {"gameid": "g0", "patch": "14.0", "result": 1, "pick1": "Ahri", "pick2": None, "ban1": None},
    ]
    return pd.DataFrame(rows)


class ComputeMetaTableTests(unittest.TestCase):

    def setUp(self):
        self.df = _draft_scenario()
        self.tbl = compute_meta_table(self.df, patch="14.1")
        self.by = {r["champion"]: r for _, r in self.tbl.iterrows()}

    def test_n_maps_conta_jogos_unicos(self):
        self.assertTrue((self.tbl["n_maps"] == 2).all())

    def test_presenca_e_pick_ban_pct(self):
        ahri = self.by["Ahri"]
        # Ahri: 2 picks / 2 mapas = 100%; 2 bans / 2 = 100%; presença 200
        self.assertAlmostEqual(ahri["pick_pct"], 100.0)
        self.assertAlmostEqual(ahri["ban_pct"], 100.0)
        self.assertAlmostEqual(ahri["presence"], 200.0)
        self.assertEqual(ahri["tier"], "S")

    def test_wr_shrinkado_k20_para_50(self):
        # Ahri: wins=1 em 2 picks -> 100*(1 + 0.5*20)/(2+20) = 50.0
        self.assertAlmostEqual(self.by["Ahri"]["wr_shrunk"], 50.0)
        # Zed: wins=2 em 2 picks -> 100*(2+10)/22 = 54.5454...
        self.assertAlmostEqual(self.by["Zed"]["wr_shrunk"], 100.0 * 12.0 / 22.0, places=4)
        self.assertEqual(WR_SHRINK_K, 20.0)

    def test_campeao_so_banido_tem_wr_nan_e_presenca_de_ban(self):
        yasuo = self.by["Yasuo"]
        self.assertEqual(int(yasuo["picks"]), 0)
        self.assertAlmostEqual(yasuo["presence"], 50.0)  # 1 ban / 2 mapas = 50%
        self.assertEqual(yasuo["tier"], "A")
        self.assertTrue(math.isnan(yasuo["wr_shrunk"]))

    def test_early_pick_share(self):
        # Zed: picks em pick2(g1) e pick1(g2) -> ambos early (i<2) = 100%
        self.assertAlmostEqual(self.by["Zed"]["early_pick_share"], 100.0)

    def test_ordena_por_presenca_desc(self):
        pres = list(self.tbl["presence"])
        self.assertEqual(pres, sorted(pres, reverse=True))

    def test_delta_vs_patch_anterior(self):
        tbl = compute_meta_table(self.df, patch="14.1", prev_patch="14.0")
        by = {r["champion"]: r for _, r in tbl.iterrows()}
        # Ahri no 14.0: 1 pick / 1 mapa = 100% presença (sem ban) ; no 14.1 = 200
        self.assertAlmostEqual(by["Ahri"]["presence_prev"], 100.0)
        self.assertAlmostEqual(by["Ahri"]["presence_delta"], 100.0)

    def test_patch_inexistente_da_tabela_vazia(self):
        self.assertTrue(compute_meta_table(self.df, patch="99.9").empty)

    def test_df_vazio_nao_quebra(self):
        self.assertTrue(compute_meta_table(pd.DataFrame(), patch="14.1").empty)


class PatchesByRecencyTests(unittest.TestCase):

    def test_filtra_min_maps_e_ordena_recencia(self):
        rows = []
        # 14.1 tem 3 mapas, 14.2 tem 1 mapa
        for g in ("a", "b", "c"):
            rows.append({"gameid": g, "patch": "14.1"})
        rows.append({"gameid": "d", "patch": "14.2"})
        df = pd.DataFrame(rows)
        out = patches_by_recency(df, min_maps=2)
        self.assertEqual(out, [("14.1", 3)])  # 14.2 filtrado (só 1 mapa)

    def test_df_vazio(self):
        self.assertEqual(patches_by_recency(pd.DataFrame()), [])


if __name__ == "__main__":
    unittest.main()
