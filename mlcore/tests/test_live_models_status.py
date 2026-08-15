"""Tests for app_core/live_models_status_core.py."""
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from app_core.live_models_status_core import (
    assess_freshness,
    build_retrain_command,
    csv_signatures,
    is_retrain_running,
    manifest_path,
    read_manifest,
    retrain_status,
    update_manifest_after_training,
    write_manifest,
)


def _make_fake_csv(tmpdir: Path, name: str, n_games: int = 100) -> Path:
    p = tmpdir / name
    rows = [f"g{i},{i},XXX,2025-01-{1+i%28:02d},1,1,1,team,1,30" for i in range(n_games)]
    header = "gameid,participantid,league,date,year,split,playoffs,teamname,result,gamelength"
    p.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")
    return p


class TestCSVSignatures(unittest.TestCase):

    def test_returns_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=50)
            sigs = csv_signatures([csv], count_games=True)
            self.assertEqual(len(sigs), 1)
            self.assertEqual(sigs[0]["n_games"], 50)
            self.assertIsNotNone(sigs[0]["size"])

    def test_missing_file_no_size(self):
        sigs = csv_signatures(["__nonexistent__.csv"])
        self.assertEqual(len(sigs), 1)
        self.assertIsNone(sigs[0]["size"])


class TestManifestIO(unittest.TestCase):

    def test_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            payload = {"version": "live_models_manifest_v1", "n_games_at_train": 1000}
            write_manifest(md, payload)
            self.assertEqual(read_manifest(md)["n_games_at_train"], 1000)

    def test_missing_manifest_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_manifest(tmp), {})

    def test_manifest_path_format(self):
        p = manifest_path("/some/dir")
        self.assertTrue(str(p).endswith("manifest.json"))


class TestUpdateManifest(unittest.TestCase):

    def test_updates_manifest_with_signatures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=80)
            md = tmp_p / "live_models"
            md.mkdir()
            manifest = update_manifest_after_training(md, [csv])
            self.assertEqual(manifest["n_games_at_train"], 80)
            self.assertIn("last_trained_at", manifest)


class TestAssessFreshness(unittest.TestCase):

    def test_no_models_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp) / "missing_subdir"
            fs = assess_freshness(md, [])
            self.assertEqual(fs.state, "no_models")

    def test_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            fs = assess_freshness(md, [])
            self.assertEqual(fs.state, "no_manifest")

    def test_fresh_after_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=100)
            md = tmp_p / "live_models"
            md.mkdir()
            update_manifest_after_training(md, [csv])
            fs = assess_freshness(md, [csv])
            self.assertEqual(fs.state, "fresh")
            self.assertEqual(fs.n_new_games, 0)

    def test_stale_after_new_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=100)
            md = tmp_p / "live_models"
            md.mkdir()
            update_manifest_after_training(md, [csv])
            # Recreate CSV with more games
            _make_fake_csv(tmp_p, "data.csv", n_games=500)
            fs = assess_freshness(md, [csv], stale_n_games=200)
            self.assertIn(fs.state, ("stale", "very_stale"))

    def test_very_stale_after_huge_new_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=100)
            md = tmp_p / "live_models"
            md.mkdir()
            update_manifest_after_training(md, [csv])
            _make_fake_csv(tmp_p, "data.csv", n_games=1500)
            fs = assess_freshness(md, [csv], stale_n_games=200, very_stale_n_games=1000)
            self.assertEqual(fs.state, "very_stale")


class TestBuildRetrainCommand(unittest.TestCase):

    def test_command_contains_required_args(self):
        # 2026-07: o retreino aponta pro ORQUESTRADOR (fases 6/7/8/10 + manifest),
        # nao mais pro run_phase6_live_gate isolado (--save-models-dir).
        cmd = build_retrain_command(val_cutoff="2026-04-30")
        self.assertIn("--val-cutoff", cmd)
        self.assertIn("2026-04-30", cmd)
        self.assertTrue(any("retrain_live_models_all" in str(part) for part in cmd))
        self.assertIn("--models-root", cmd)


class TestRetrainStatus(unittest.TestCase):

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = retrain_status(tmp)
            self.assertIn("running", s)
            self.assertIn("log_exists", s)
            self.assertFalse(s["running"])


class TestPidFantasmaSeAutoCura(unittest.TestCase):
    """Bug recorrente achado com ele (23/07): o retreino detached escreve o
    retrain.pid mas NAO o apaga ao morrer; o app confiava no arquivo e dizia
    "em retreino" pra sempre. Agora is_retrain_running checa liveness e apaga o
    pid morto sozinho -- nunca mais limpar na mao."""

    def test_pid_morto_retorna_false_e_apaga_o_arquivo(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "retrain.pid"
            # PID que com certeza nao existe
            pid_file.write_text("999999999", encoding="utf-8")
            self.assertFalse(is_retrain_running(tmp))
            self.assertFalse(pid_file.exists(), "pid morto deveria ter sido apagado")

    def test_pid_ilegivel_retorna_false_e_apaga(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "retrain.pid"
            pid_file.write_text("nao_e_um_numero", encoding="utf-8")
            self.assertFalse(is_retrain_running(tmp))
            self.assertFalse(pid_file.exists())

    def test_sem_pid_file_retorna_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(is_retrain_running(tmp))

    def test_pid_vivo_retorna_true_e_mantem(self):
        import os as _os
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "retrain.pid"
            pid_file.write_text(str(_os.getpid()), encoding="utf-8")
            # o proprio processo de teste esta vivo
            self.assertTrue(is_retrain_running(tmp))
            self.assertTrue(pid_file.exists(), "pid vivo NAO deve ser apagado")


class TestAmbienteNuvem(unittest.TestCase):
    """A nuvem (Streamlit Cloud, 1GB de RAM) nao pode treinar -> is_training_capable_env
    False la, e o painel esconde retreino. Detecta por HOME=/home/appuser ou /mount/src."""

    def test_windows_nunca_e_nuvem(self):
        import app_core.live_models_status_core as m
        with unittest.mock.patch.object(m.os, "name", "nt"):
            self.assertFalse(m.is_streamlit_cloud())
            self.assertTrue(m.is_training_capable_env())

    def test_linux_home_appuser_e_nuvem(self):
        import app_core.live_models_status_core as m
        with unittest.mock.patch.object(m.os, "name", "posix"), \
             unittest.mock.patch.dict(m.os.environ, {"HOME": "/home/appuser"}, clear=False):
            self.assertTrue(m.is_streamlit_cloud())
            self.assertFalse(m.is_training_capable_env())

    def test_linux_mount_src_e_nuvem(self):
        import app_core.live_models_status_core as m
        with unittest.mock.patch.object(m.os, "name", "posix"), \
             unittest.mock.patch.dict(m.os.environ, {"HOME": "/root"}, clear=False), \
             unittest.mock.patch.object(m.os, "getcwd", return_value="/mount/src/gabinete-lol"):
            self.assertTrue(m.is_streamlit_cloud())

    def test_linux_comum_nao_e_nuvem(self):
        import app_core.live_models_status_core as m
        with unittest.mock.patch.object(m.os, "name", "posix"), \
             unittest.mock.patch.dict(m.os.environ, {"HOME": "/home/bruno"}, clear=False), \
             unittest.mock.patch.object(m.os, "getcwd", return_value="/home/bruno/proj"):
            self.assertFalse(m.is_streamlit_cloud())
            self.assertTrue(m.is_training_capable_env())


class TestPidAliveMatchExato(unittest.TestCase):
    """O match antigo era substring (str(pid) in stdout): 1844 casava com 18440
    ou "1,844 K". Agora e comparacao exata da coluna PID."""

    def test_pid_proprio_vivo(self):
        import os as _os
        from app_core.live_models_status_core import _pid_alive
        self.assertTrue(_pid_alive(_os.getpid()))

    def test_pid_inexistente_morto(self):
        from app_core.live_models_status_core import _pid_alive
        self.assertFalse(_pid_alive(999999999))


if __name__ == "__main__":
    unittest.main()
