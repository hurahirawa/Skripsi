"""Uji mesin inferensi dan sesi pemantauan (integrasi: latih sintetis → replay PCAP → CSV → ringkasan)."""
import copy

import numpy as np
import pandas as pd
import pytest

from qos.capture.demo import write_demo_pcap
from qos.capture.flow_table import WindowFlow
from qos.capture.sniffer import iter_pcap
from qos.config import load_config
from qos.model.train import save_artifacts, train
from qos.monitor.engine import (
    INSUFFICIENT,
    SUMMARY_COLUMNS,
    WINDOW_COLUMNS,
    InferenceEngine,
    ModelNotFoundError,
    MonitoringSession,
    card_category,
    load_artifacts,
    overall_status,
    safe_name,
)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Config terisolasi + model sintetis aktif (fitur default Varian C) + PCAP demo."""
    root = tmp_path_factory.mktemp("env")
    cfg = copy.deepcopy(load_config(use_user=False))
    cfg["paths"].update({"artifacts_dir": str(root / "artifacts"), "reports_dir": str(root / "reports")})
    cfg["hdbscan"]["grid_search"].update({"min_cluster_size": [5, 10], "min_samples": [2, 5]})
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
    pcap = write_demo_pcap(root / "demo.pcap", duration_sec=12.0)
    return cfg, pcap


def make_session(cfg, pcap, mode="window", interface="eth0 test"):
    c = copy.deepcopy(cfg)
    c["monitoring"]["feature_mode"] = mode
    engine = InferenceEngine(load_artifacts(c), c)
    return MonitoringSession(c, engine, lambda stop: iter_pcap(pcap, 0, stop), interface)


def test_load_artifacts_tanpa_model(tmp_path):
    cfg = copy.deepcopy(load_config(use_user=False))
    cfg["paths"]["artifacts_dir"] = str(tmp_path / "kosong")
    with pytest.raises(ModelNotFoundError, match="model aktif"):
        load_artifacts(cfg)


def test_load_artifacts_tidak_lengkap(env):
    cfg, _ = env
    with pytest.raises(ModelNotFoundError, match="tidak lengkap"):
        load_artifacts(cfg, version="20000101_000000")


def test_predict_kolom_dan_lookup_label(env):
    cfg, _ = env
    art = load_artifacts(cfg)
    flows = [
        WindowFlow("f1", 1000.0, 1005.0, 17, 250, 250, 500, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("f2", 1000.0, 1005.0, 17, 100, 90, 190, 4.9e6, 1_000.0, 500.0),
    ]
    out = InferenceEngine(art, cfg).predict(flows, "sesi1")
    assert list(out.columns) == list(WINDOW_COLUMNS)
    assert out["id_sesi"].tolist() == ["sesi1", "sesi1"]
    assert out["delay_ms"].tolist() == pytest.approx([10.0, 1.0])
    assert out["jitter_ms"].tolist() == pytest.approx([8.0, 0.5])
    assert out["fai_pct"].tolist() == pytest.approx([0.0, 10.0])
    assert set(out["label"]) <= set(art.labels.values())
    assert ((out["membership_prob"] >= 0) & (out["membership_prob"] <= 1)).all()
    # label murni lookup dari cluster_profile
    assert all(art.labels[int(c)] == lb for c, lb in zip(out["cluster_id"], out["label"]))


def test_predict_impute_median_mon5(env):
    cfg, _ = env
    art = load_artifacts(cfg)
    flows = [WindowFlow("nan", 0.0, 5.0, 17, 100, 90, 190, 1e6, float("nan"), float("nan"))]
    out = InferenceEngine(art, cfg).predict(flows, "s")
    med = art.meta["median_imputasi"]
    assert out["delay_ms"].iloc[0] == pytest.approx(med["iat_mean_us"] / 1000)
    assert out["jitter_ms"].iloc[0] == pytest.approx(med["iat_std_us"] / 1000)
    assert out["label"].iloc[0] != INSUFFICIENT


def test_data_belum_cukup_dicatat_tanpa_prediksi(env):
    cfg, _ = env
    eng = InferenceEngine(load_artifacts(cfg), cfg)
    flows = [
        WindowFlow("cukup", 0.0, 5.0, 17, 250, 250, 500, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("bwd_kurang", 0.0, 5.0, 17, 100, 4, 104, 1e6, 100.0, 10.0),
        WindowFlow("fwd_kurang", 0.0, 5.0, 17, 4, 100, 104, 1e6, 100.0, 10.0),
        WindowFlow("fwd_nol", 0.0, 5.0, 17, 0, 10, 10, 1e6, 100.0, 10.0),
    ]
    out = eng.predict(flows, "s").set_index("flow_id")
    assert eng.n_insufficient == 3
    assert out.loc["cukup", "label"] != INSUFFICIENT and not pd.isna(out.loc["cukup", "cluster_id"])
    for fid in ("bwd_kurang", "fwd_kurang", "fwd_nol"):
        row = out.loc[fid]
        assert row["label"] == INSUFFICIENT and row["kategori_tindakan"] == "-"
        assert pd.isna(row["cluster_id"]) and pd.isna(row["membership_prob"]) and pd.isna(row["pc1"])
    assert out.loc["bwd_kurang", "fai_pct"] == pytest.approx(96.0)  # nilai mentah tetap dicatat
    assert pd.isna(out.loc["fwd_nol", "fai_pct"])


def test_min_session_live_nonaktif_tetap_buang_fwd_nol(env):
    cfg, _ = env
    c = copy.deepcopy(cfg)
    c["monitoring"]["min_session_live"] = {"fwd": None, "bwd": None}
    flows = [WindowFlow("a", 0.0, 5.0, 17, 3, 0, 3, 1e6, 100.0, 10.0), WindowFlow("b", 0.0, 5.0, 17, 0, 3, 3, 1e6, 1.0, 1.0)]
    out = InferenceEngine(load_artifacts(c), c).predict(flows, "s").set_index("flow_id")
    assert out.loc["a", "label"] != INSUFFICIENT  # tanpa batas, fwd ≥ 1 diprediksi
    assert out.loc["b", "label"] == INSUFFICIENT  # fwd = 0: FAI tak terdefinisi


def test_predict_kosong(env):
    cfg, _ = env
    out = InferenceEngine(load_artifacts(cfg), cfg).predict([], "s")
    assert out.empty and list(out.columns) == list(WINDOW_COLUMNS)


def test_sesi_replay_menulis_csv_dan_ringkasan(env):
    cfg, pcap = env
    s = make_session(cfg, pcap)
    s.start()
    assert s.wait(60)
    path = s.stop()
    assert s.error is None
    rows = pd.read_csv(s.window_csv)  # MON-7
    assert list(rows.columns) == list(WINDOW_COLUMNS)
    assert rows["flow_id"].nunique() == 3 and len(rows) >= 3 * 2  # 3 panggilan RTP, ≥ 2 jendela 5 detik
    assert (rows["id_sesi"] == s.session_id).all()

    summary = pd.read_csv(path)
    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    ev = load_artifacts(cfg).profile["evaluation"]
    assert summary["total_aliran"].iloc[0] == 3
    assert summary["jumlah_klaster"].iloc[0] == ev["n_clusters"]
    assert summary["skor_silhouette"].iloc[0] == pytest.approx(ev["silhouette"])
    assert summary["lokasi_laporan"].iloc[0] == str(s.window_csv)
    assert s.stop() == path  # idempoten


def test_queue_dan_kartu(env):
    cfg, pcap = env
    s = make_session(cfg, pcap)
    s.start()
    s.wait(60)
    s.stop()
    batches = s.drain()
    assert batches and sum(len(b) for b in batches) == s.n_windows_rows
    cards = s.card_counts()
    assert cards["Total Aliran Trafik"] == 3
    assert sum(cards[k] for k in ("QoS Baik", "QoS Cukup", "QoS Buruk", "Anomali")) == 3
    assert s.drain() == []


def test_mode_cumulative_menghasilkan_baris(env):
    cfg, pcap = env
    s = make_session(cfg, pcap, mode="cumulative")
    s.start()
    s.wait(60)
    s.stop()
    rows = pd.read_csv(s.window_csv)
    assert rows["flow_id"].nunique() >= 3
    assert (rows.groupby("flow_id")["fwd"].apply(lambda x: x.is_monotonic_increasing)).all()  # kumulatif naik


def test_stop_dini_menghentikan_worker(env):
    cfg, pcap = env
    c = copy.deepcopy(cfg)
    engine = InferenceEngine(load_artifacts(c), c)
    s = MonitoringSession(c, engine, lambda stop: iter_pcap(pcap, 0.05, stop), "lambat")  # sangat lambat
    s.start()
    assert s.running
    path = s.stop()
    assert not s.running and path.exists()


def test_error_sumber_ditangkap(env):
    cfg, _ = env
    engine = InferenceEngine(load_artifacts(cfg), cfg)
    s = MonitoringSession(cfg, engine, lambda stop: iter_pcap("tidak_ada.pcap", 0, stop), "x")
    s.start()
    s.wait(10)
    s.stop()
    assert s.error and "CaptureError" in s.error


def test_overall_status_dan_kategori():
    assert card_category("QoS Baik") == "QoS Baik"
    assert card_category("QoS Buruk - Jitter Melebihi Batas") == "QoS Buruk"
    assert card_category("Anomali - Delay Ekstrem") == "Anomali"
    labels = {"a": "QoS Buruk - X Melebihi Batas", "b": "Anomali", "c": "QoS Baik", "d": "QoS Baik"}
    assert overall_status(labels, 0.5) == "QoS Baik"  # 50% tidak > 0,5 → label mayoritas
    assert overall_status(labels, 0.4) == "Kritis"
    assert overall_status({}, 0.5) == "Tidak ada data"
    assert card_category(INSUFFICIENT) == INSUFFICIENT


def test_data_belum_cukup_tidak_masuk_kritis_ratio():
    base = {"a": "QoS Buruk - X Melebihi Batas", "b": "QoS Baik", "c": "QoS Baik"}  # 1/3 buruk → tidak kritis
    many = {**base, **{f"x{i}": INSUFFICIENT for i in range(10)}}
    assert overall_status(many, 0.5) == overall_status(base, 0.5) == "QoS Baik"
    only_bad = {"a": "Anomali", **{f"x{i}": INSUFFICIENT for i in range(10)}}
    assert overall_status(only_bad, 0.5) == "Kritis"  # penyebut hanya flow yang dinilai
    assert overall_status({f"x{i}": INSUFFICIENT for i in range(3)}, 0.5) == INSUFFICIENT


def test_id_sesi_punya_sufiks_uuid_6_karakter(env):
    import re

    cfg, pcap = env
    a, b = make_session(cfg, pcap), make_session(cfg, pcap)  # dibuat pada detik yang sama
    assert re.fullmatch(r"eth0_test_\d{8}_\d{6}_[0-9a-f]{6}", a.session_id)
    assert a.session_id != b.session_id and a.window_csv != b.window_csv


def test_default_feature_mode_cumulative():
    assert load_config(use_user=False)["monitoring"]["feature_mode"] == "cumulative"
    assert load_config(use_user=False)["monitoring"]["min_session_live"] == {"fwd": 5, "bwd": 5}


def test_safe_name():
    assert safe_name("Wi-Fi 2/eth0") == "Wi-Fi_2_eth0"
    assert safe_name("///") == "sesi"
