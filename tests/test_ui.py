"""Uji asap halaman Streamlit memakai AppTest (config + model sintetis terisolasi via QOS_CONFIG)."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from qos.capture.demo import write_demo_pcap
from qos.capture.sniffer import iter_pcap
from qos.config import load_config
from qos.model.train import save_artifacts, train
from qos.monitor.engine import InferenceEngine, MonitoringSession, load_artifacts

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [
    "Home.py",
    "pages/1_Dashboard_Overview.py",
    "pages/2_Analisis_Evaluasi_Klaster.py",
    "pages/3_Pengaturan_Parameter.py",
    "pages/4_Rekomendasi_Ekspor.py",
]


def _train_synthetic(cfg):
    rng = np.random.default_rng(0)
    parts = []
    for mean_us, std_us, pkts in ((1000.0, 500.0, 100), (10_000.0, 8_000.0, 300), (400_000.0, 90_000.0, 800)):
        fwd = rng.normal(pkts, pkts * 0.02, 80).round()
        parts.append(
            pd.DataFrame(
                {
                    "iat_mean_us": rng.normal(mean_us, mean_us * 0.03, 80),
                    "iat_std_us": rng.normal(std_us, std_us * 0.03, 80),
                    "fwd_packets_count": fwd,
                    "bwd_packets_count": (fwd * rng.normal(0.9, 0.02, 80)).round(),
                }
            )
        )
    save_artifacts(train(pd.concat(parts, ignore_index=True), cfg), cfg)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui")
    cfg = copy.deepcopy(load_config(use_user=False))
    cfg["paths"].update(
        {"artifacts_dir": str(root / "artifacts"), "reports_dir": str(root / "reports"), "logs_dir": str(root / "logs")}
    )
    cfg["hdbscan"]["grid_search"].update({"min_cluster_size": [5, 10], "min_samples": [2, 5]})
    _train_synthetic(cfg)
    conf_dir = root / "conf"
    conf_dir.mkdir()
    conf = conf_dir / "default_config.yaml"
    conf.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    pcap = write_demo_pcap(root / "demo.pcap", duration_sec=12.0)
    return cfg, conf, pcap


@pytest.fixture(autouse=True)
def app_env(env, monkeypatch):
    _, conf, _ = env
    monkeypatch.setenv("QOS_CONFIG", str(conf))
    monkeypatch.syspath_prepend(str(APP))
    for name in [m for m in sys.modules if m == "ui"]:
        del sys.modules[name]  # impor ulang agar membaca QOS_CONFIG yang baru


def run(page):
    at = AppTest.from_file(str(APP / page), default_timeout=60)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_halaman_berjalan_tanpa_error(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_tanpa_model_menampilkan_error_jelas(tmp_path, monkeypatch, env):
    cfg, _, _ = env
    c = copy.deepcopy(cfg)
    c["paths"]["artifacts_dir"] = str(tmp_path / "kosong")
    conf = tmp_path / "default_config.yaml"
    conf.write_text(yaml.safe_dump(c), encoding="utf-8")
    monkeypatch.setenv("QOS_CONFIG", str(conf))
    at = run("pages/1_Dashboard_Overview.py")
    assert any("Tidak ada model aktif" in e.value for e in at.error)


def test_halaman_2_menampilkan_skor_dan_profil(env):
    cfg, _, _ = env
    at = run("pages/2_Analisis_Evaluasi_Klaster.py")
    ev = load_artifacts(cfg).profile["evaluation"]
    values = {m.label: m.value for m in at.metric}
    assert values["Silhouette"] == f"{ev['silhouette']:.4f}"
    assert values["Noise Ratio"] == f"{ev['noise_ratio']:.2%}"
    assert at.dataframe  # tabel profil + sampel


def test_dashboard_menampilkan_kartu_dari_sesi_replay(env):
    cfg, _, pcap = env
    c = copy.deepcopy(cfg)
    session = MonitoringSession(c, InferenceEngine(load_artifacts(c), c), lambda stop: iter_pcap(pcap, 0, stop), "uji")
    session.start()
    session.wait(60)
    at = AppTest.from_file(str(APP / "pages/1_Dashboard_Overview.py"), default_timeout=60)
    at.session_state["session"] = session
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    cards = {m.label: m.value for m in at.metric}
    assert cards["Total Aliran Trafik"] == "3"
    assert sum(int(cards[k]) for k in ("QoS Baik", "QoS Cukup", "QoS Buruk", "Anomali")) == 3
    assert len(at.session_state["live_rows"]) == session.n_windows_rows
    session.stop()


def test_pengaturan_save_dan_reset(env):
    _, conf, _ = env
    user = conf.parent / "user_config.yaml"
    user.unlink(missing_ok=True)
    at = run("pages/3_Pengaturan_Parameter.py")
    at.slider[0].set_value(12)
    at.button[0].click().run()
    assert user.is_file()
    assert yaml.safe_load(user.read_text("utf-8"))["hdbscan"]["min_cluster_size"] == 12
    at2 = run("pages/3_Pengaturan_Parameter.py")
    assert at2.slider[0].value == 12  # user_config.yaml dimuat
    at2.button[1].click().run()
    assert not user.exists()


def test_pengaturan_menolak_delay_baik_melebihi_maks(env):
    _, conf, _ = env
    (conf.parent / "user_config.yaml").unlink(missing_ok=True)
    at = run("pages/3_Pengaturan_Parameter.py")
    at.number_input[0].set_value(500.0).run()  # delay baik 500 > maks 400
    assert any("Delay baik" in e.value for e in at.error)
    assert at.button[0].disabled


def test_riwayat_kosong_dan_terisi(env):
    cfg, _, pcap = env
    at = run("pages/4_Rekomendasi_Ekspor.py")
    assert not at.exception
    c = copy.deepcopy(cfg)
    s = MonitoringSession(c, InferenceEngine(load_artifacts(c), c), lambda stop: iter_pcap(pcap, 0, stop), "riwayat")
    s.start()
    s.wait(60)
    s.stop()
    at = run("pages/4_Rekomendasi_Ekspor.py")
    assert not at.exception, [e.value for e in at.exception]
    assert any(s.session_id in str(df.value.to_dict()) for df in at.dataframe)


def _click(at, label):
    next(b for b in at.sidebar.button if b.label == label).click().run()


def test_live_tanpa_tshark_menampilkan_error_dan_tidak_memulai(env, monkeypatch):
    from qos.capture import sniffer

    monkeypatch.setattr(sniffer, "find_tshark", lambda p=None: None)
    monkeypatch.setattr(sniffer, "npcap_installed", lambda: True)
    monkeypatch.setattr(sniffer, "list_interfaces", lambda cfg: ["Wi-Fi"])
    at = run("pages/1_Dashboard_Overview.py")
    at.sidebar.radio(key="mode").set_value("Live").run()
    at.sidebar.selectbox(key="backend").set_value("pyshark").run()
    _click(at, "Start")
    assert any("tshark tidak ditemukan" in e.value for e in at.sidebar.error)
    assert "session" not in at.session_state or at.session_state["session"] is None


def test_live_scapy_memulai_sesi_lewat_ui(env, monkeypatch):
    from qos.capture import sniffer

    _, _, pcap = env
    monkeypatch.setattr(sniffer, "check_live_environment", lambda cfg, backend: sniffer.EnvReport(backend))
    monkeypatch.setattr(sniffer, "list_interfaces", lambda cfg: ["Wi-Fi"])
    monkeypatch.setattr(
        sniffer, "open_source", lambda cfg, backend, pcap_path=None, stop=None, interface=None: iter_pcap(pcap, 0, stop)
    )
    at = run("pages/1_Dashboard_Overview.py")
    at.sidebar.radio(key="mode").set_value("Live").run()
    at.sidebar.selectbox(key="backend").set_value("scapy").run()
    at.sidebar.selectbox(key="interface").set_value("Wi-Fi").run()
    _click(at, "Start")
    session = at.session_state["session"]
    assert session is not None and session.session_id.startswith("scapy_Wi-Fi_")
    session.wait(60)
    session.stop()
    assert session.error is None and session.n_windows_rows > 0


def test_halaman_4_rekomendasi_dan_pdf(env):
    cfg, _, pcap = env
    c = copy.deepcopy(cfg)
    s = MonitoringSession(c, InferenceEngine(load_artifacts(c), c), lambda stop: iter_pcap(pcap, 0, stop), "pdf")
    s.start()
    s.wait(60)
    s.stop()
    at = run("pages/4_Rekomendasi_Ekspor.py")
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox[0].set_value(s.session_id).run()
    # saran muncul bila ada label bermasalah; selalu ada status sesi
    boxes = [*at.error, *at.warning, *at.success, *at.info]
    text = " ".join(m.value for m in at.markdown) + " ".join(e.value for e in boxes)
    assert "Status sesi" in text or "Kritis" in text or "Tidak ada aliran" in text
    next(b for b in at.button if b.label == "Buat PDF ringkasan").click().run()
    assert not at.exception, [e.value for e in at.exception]
    name, data = at.session_state["pdf_ready"][1], at.session_state["pdf_ready"][2]
    assert name.startswith("laporan_") and name.endswith(".pdf") and data[:5] == b"%PDF-"
    assert (Path(cfg["paths"]["reports_dir"]) / "exports" / name).is_file()


# ---- M8: latih ulang, Terapkan Ulang Pelabelan, versi model ------------------------------------------
def _btn(at, label):
    return next(b for b in at.button if b.label == label)


def _log_csv(cfg, name="uiuji", n=300):
    from tests.test_m8 import log_rows, write_log

    return write_log(cfg, name, log_rows(n))


@pytest.fixture
def restore_active(env):
    cfg, conf, _ = env
    current = Path(cfg["paths"]["artifacts_dir"]) / "current.txt"
    before = current.read_text("utf-8")
    user = conf.parent / "user_config.yaml"
    yield cfg, conf
    current.write_text(before, encoding="utf-8")
    user.unlink(missing_ok=True)
    for f in (Path(cfg["paths"]["reports_dir"]) / "riwayat").glob("jendela_uiuji*.csv"):
        f.unlink()


def test_ui_terapkan_ulang_pelabelan_membuat_versi_tanpa_mengaktifkan(restore_active):
    cfg, _ = restore_active
    current = Path(cfg["paths"]["artifacts_dir"]) / "current.txt"
    active = current.read_text("utf-8")
    at = run("pages/3_Pengaturan_Parameter.py")
    at.number_input[4].set_value(1000.0).run()  # loss maks sangat longgar
    _btn(at, "Terapkan Ulang Pelabelan").click().run()
    assert not at.exception, [e.value for e in at.exception]
    res = at.session_state["relabel_result"]
    assert res["version"] != active and res["changes"]["berubah"].any()
    assert current.read_text("utf-8") == active  # belum aktif
    _btn(at, "Aktifkan model ini").click().run()
    assert current.read_text("utf-8") == res["version"]


def test_ui_latih_ulang_dari_log_tidak_mengaktifkan(restore_active):
    cfg, conf = restore_active
    (conf.parent / "user_config.yaml").write_text(yaml.safe_dump({"retrain": {"log_source": {"min_rows": 100}}}), encoding="utf-8")
    _log_csv(cfg)
    current = Path(cfg["paths"]["artifacts_dir"]) / "current.txt"
    active = current.read_text("utf-8")
    at = run("pages/3_Pengaturan_Parameter.py")
    at.radio(key="retrain_source").set_value("Log monitoring (reports/riwayat)").run()
    assert any("mencukupi" in s.value for s in at.success)
    assert _btn(at, "Latih Ulang Model").disabled  # butuh konfirmasi dulu
    at.checkbox(key="confirm_retrain").check().run()
    _btn(at, "Latih Ulang Model").click().run()
    assert not at.exception, [e.value for e in at.exception]
    res = at.session_state["retrain_result"]
    assert res["version"] != active and current.read_text("utf-8") == active
    assert any(m.label == "Silhouette" for m in at.metric)
    assert res["version"] in list(__import__("qos.model.versions", fromlist=["x"]).list_versions(cfg)["versi"])


def test_ui_log_tidak_cukup_menonaktifkan_latih_ulang(restore_active):
    cfg, _ = restore_active
    _log_csv(cfg, n=30)  # < min_rows default 200
    at = run("pages/3_Pengaturan_Parameter.py")
    at.radio(key="retrain_source").set_value("Log monitoring (reports/riwayat)").run()
    assert any("Data log tidak cukup" in w.value for w in at.warning)
    assert _btn(at, "Latih Ulang Model").disabled


def test_ui_latih_ulang_dan_aktivasi_dinonaktifkan_saat_monitoring(restore_active):
    from types import SimpleNamespace

    at = AppTest.from_file(str(APP / "pages/3_Pengaturan_Parameter.py"), default_timeout=60)
    at.session_state["session"] = SimpleNamespace(running=True, error=None)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert _btn(at, "Latih Ulang Model").disabled
    assert any("Monitoring sedang berjalan" in w.value for w in at.warning)


# ---- Kompatibilitas versi lama (profil tanpa loss_good_pct) ---------------------------------------
def _versi_lama_terisolasi(tmp_path, monkeypatch, env, source: Path | None = None):
    """Config terisolasi dengan satu versi aktif bergaya 2 level lama (tanpa loss_good_pct)."""
    import json
    import shutil

    cfg, _, _ = env
    c = copy.deepcopy(cfg)
    c["paths"].update({"artifacts_dir": str(tmp_path / "artifacts"), "reports_dir": str(tmp_path / "reports")})
    root = Path(c["paths"]["artifacts_dir"])
    if source is not None:
        version = source.name
        shutil.copytree(source, root / version)
    else:
        src_root = Path(cfg["paths"]["artifacts_dir"])
        version = (src_root / "current.txt").read_text("utf-8").strip()
        shutil.copytree(src_root / version, root / version)
    (root / "current.txt").write_text(version, encoding="utf-8")
    prof_path = root / version / "cluster_profile.json"
    prof = json.loads(prof_path.read_text("utf-8"))
    prof["thresholds"].pop("loss_good_pct", None)
    prof_path.write_text(json.dumps(prof), encoding="utf-8")
    c["thresholds"]["loss_good_pct"] = c["thresholds"]["loss_max_pct"]  # tampilan default_th lama
    conf = tmp_path / "default_config.yaml"
    conf.write_text(yaml.safe_dump(c), encoding="utf-8")
    monkeypatch.setenv("QOS_CONFIG", str(conf))
    return c


@pytest.mark.parametrize("page", PAGES)
def test_versi_lama_tanpa_loss_good_semua_halaman_jalan(page, tmp_path, monkeypatch, env):
    _versi_lama_terisolasi(tmp_path, monkeypatch, env)
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_versi_nyata_20261019_204042_semua_halaman_jalan(page, tmp_path, monkeypatch, env):
    real = Path(__file__).resolve().parents[1] / "artifacts" / "20261019_204042"
    if not real.is_dir():
        pytest.skip("artefak 20261019_204042 tidak ada (artifacts/ tidak di-commit)")
    _versi_lama_terisolasi(tmp_path, monkeypatch, env, source=real)
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_ui_latih_ulang_butuh_konfirmasi(restore_active):
    at = run("pages/3_Pengaturan_Parameter.py")
    assert at.checkbox(key="confirm_retrain").label == "Saya mengerti ini membuat versi model baru"
    assert _btn(at, "Latih Ulang Model").disabled
    at.checkbox(key="confirm_retrain").check().run()
    assert not _btn(at, "Latih Ulang Model").disabled
