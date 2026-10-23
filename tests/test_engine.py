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


# ---- ringkasan per panggilan (MON-9) ----
from qos.capture.flow_table import PacketRecord  # noqa: E402
from qos.monitor.calls import CALL_COLUMNS, WINDOW_MODE_NOTE, CallTracker, call_title, worst_label  # noqa: E402

CA, CB = ("10.1.0.1", 4000), ("10.1.0.2", 4000)


def call_packets(a, b, t0, t1, step=0.1, both=True):
    """Paket RTP dua arah (atau searah) antara a dan b dari t0 sampai t1."""
    out, t = [], t0
    while t < t1:
        out.append(PacketRecord(t, a[0], b[0], a[1], b[1], 17, 160))
        if both:
            out.append(PacketRecord(t + 0.01, b[0], a[0], b[1], a[1], 17, 160))
        t += step
    return out


def call_session(cfg, packets, mode="cumulative", hold=False):
    c = copy.deepcopy(cfg)
    c["monitoring"]["feature_mode"] = mode
    engine = InferenceEngine(load_artifacts(c), c)

    def source(stop):
        yield from packets
        if hold:
            stop.wait(30)  # sumber tetap terbuka sampai Stop

    return MonitoringSession(c, engine, source, "panggilan")


def test_panggilan_selesai_karena_idle(env):
    cfg, _ = env
    other = ("10.1.0.9", 4100), ("10.1.0.8", 4100)
    pkts = sorted(call_packets(CA, CB, 0, 10) + call_packets(*other, 0, 40), key=lambda p: p.ts)
    s = call_session(cfg, pkts)
    s.start()
    assert s.wait(60)
    s.stop()
    calls = pd.read_csv(s.calls_csv)
    assert list(calls.columns) == list(CALL_COLUMNS)
    assert len(calls) == 2
    first = calls.iloc[0]  # selesai karena idle (call_end_idle_sec = 10) sebelum sumber habis
    assert first["flow_id"] == "10.1.0.1:4000-10.1.0.2:4000-17"
    assert first["durasi_s"] == pytest.approx(9.91, abs=0.1)
    assert first["jumlah_jendela"] >= 1 and first["total_fwd"] >= 90 and first["total_bwd"] >= 90
    last_row = pd.read_csv(s.window_csv).query("flow_id == @first.flow_id").iloc[-1]
    assert first["label_akhir"] == last_row["label"]  # label_akhir = label jendela kumulatif terakhir
    pct = calls[[c for c in CALL_COLUMNS if c.startswith("persen_")]].sum(axis=1)
    assert pct.tolist() == pytest.approx([100.0, 100.0])
    assert calls["catatan"].isna().all()  # kosong pada mode cumulative
    assert s.n_calls == 2 and sum(len(b) for b in s.drain_calls()) == 2


def test_panggilan_selesai_karena_stop(env):
    cfg, _ = env
    s = call_session(cfg, call_packets(CA, CB, 0, 10), hold=True)
    s.start()
    s.wait(1.5)
    assert s.running and not s.calls_csv.exists()  # belum idle, belum Stop → belum selesai
    path = s.stop()
    calls = pd.read_csv(s.calls_csv)
    assert len(calls) == 1
    summary = pd.read_csv(path)
    assert summary["jumlah_panggilan"].iloc[0] == 1
    final_cols = ["panggilan_baik", "panggilan_cukup", "panggilan_buruk", "panggilan_anomali", "panggilan_belum_cukup"]
    assert summary[final_cols].sum(axis=1).iloc[0] == 1


def test_panggilan_data_belum_cukup_tetap_dicatat(env):
    cfg, _ = env
    s = call_session(cfg, call_packets(CA, CB, 0, 10, both=False))  # searah: bwd = 0
    s.start()
    s.wait(60)
    s.stop()
    (call,) = pd.read_csv(s.calls_csv).to_dict("records")
    assert call["label_akhir"] == INSUFFICIENT and call["label_terburuk"] == INSUFFICIENT
    assert call["persen_belum_cukup"] == 100.0 and call["kategori_tindakan"] == "-"
    summary = pd.read_csv(s.stop())
    assert summary["panggilan_belum_cukup"].iloc[0] == 1


def test_panggilan_mode_window_diberi_catatan(env):
    cfg, _ = env
    s = call_session(cfg, call_packets(CA, CB, 0, 10), mode="window")
    s.start()
    s.wait(60)
    s.stop()
    (call,) = pd.read_csv(s.calls_csv).to_dict("records")
    assert call["label_akhir"]  # tetap dicatat
    assert call["catatan"] == WINDOW_MODE_NOTE and "cumulative" in call["catatan"]


def test_tracker_persen_dan_label_terburuk():
    from qos.capture.flow_table import FlowEnd

    def win(label, fwd):
        return {"flow_id": "f", "label": label, "fwd": fwd, "bwd": fwd, "delay_ms": 1.0, "jitter_ms": 1.0,
                "fai_pct": 0.0, "cluster_id": 0, "kategori_tindakan": "-", "metrik_penyebab": ""}

    labels = ["QoS Baik", "QoS Buruk - Delay", "QoS Buruk - Jitter", "QoS Buruk - Jitter", INSUFFICIENT]
    t = CallTracker("cumulative")
    t.update(pd.DataFrame([win(lab, i) for i, lab in enumerate(labels)]))
    (row,) = t.finish([FlowEnd("f", 0.0, 12.0), FlowEnd("tidak_ada", 0.0, 1.0)]).to_dict("records")
    assert row["jumlah_jendela"] == 5 and row["durasi_s"] == 12.0 and row["total_fwd"] == 4
    assert (row["persen_baik"], row["persen_buruk"], row["persen_belum_cukup"]) == (20.0, 60.0, 20.0)
    assert row["label_akhir"] == INSUFFICIENT and row["label_terburuk"] == "QoS Buruk - Jitter"
    assert t.finish([FlowEnd("f", 0.0, 1.0)]).empty  # state sudah dibuang


def test_worst_label_dan_call_title():
    from collections import Counter

    assert worst_label(Counter()) == INSUFFICIENT
    assert worst_label(Counter({"QoS Baik": 9, "Anomali - X": 1})) == "Anomali - X"
    assert call_title("192.168.1.25:16384-192.168.1.10:16384-17") == "192.168.1.25 ↔ 192.168.1.10"
    assert call_title("aneh") == "aneh"


# ---- flow sinyal SIP ----
from qos.capture.flow_table import flow_ports  # noqa: E402
from qos.monitor.calls import SIGNALING, call_toast_message  # noqa: E402


def test_flow_ports():
    assert flow_ports("10.0.0.1:5060-10.0.0.2:16384-17") == (5060, 16384)
    assert flow_ports("aneh") == (0, 0)


def test_predict_sinyal_sip_tidak_diprediksi(env):
    cfg, _ = env
    eng = InferenceEngine(load_artifacts(cfg), cfg)
    flows = [
        WindowFlow("10.0.0.1:5060-10.0.0.2:5060-17", 1000.0, 1005.0, 17, 50, 50, 100, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("10.0.0.1:16384-10.0.0.2:5061-17", 1000.0, 1005.0, 17, 50, 50, 100, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("10.0.0.1:16384-10.0.0.2:16384-17", 1000.0, 1005.0, 17, 250, 250, 500, 4.9e6, 10_000.0, 8_000.0),
    ]
    out = eng.predict(flows, "s").set_index("flow_id")
    assert list(out["label"].iloc[:2]) == [SIGNALING, SIGNALING]
    assert out["cluster_id"].iloc[:2].isna().all() and out["kategori_tindakan"].iloc[0] == "-"
    assert out["label"].iloc[2] != SIGNALING and not pd.isna(out["cluster_id"].iloc[2])
    assert eng.n_insufficient == 0  # sinyal bukan "Data Belum Cukup"
    assert card_category(SIGNALING) == SIGNALING


def test_sesi_sinyal_sip_dicatat_di_jendela_saja(env):
    cfg, _ = env
    sip_a, sip_b = ("10.1.0.1", 5060), ("10.1.0.2", 5060)
    pkts = sorted(call_packets(CA, CB, 0, 10) + call_packets(sip_a, sip_b, 0, 10), key=lambda p: p.ts)
    s = call_session(cfg, pkts)
    s.start()
    s.wait(60)
    path = s.stop()
    rows = pd.read_csv(s.window_csv)
    assert (rows.loc[rows["flow_id"].str.contains(":5060"), "label"] == SIGNALING).all()
    assert (rows["flow_id"].str.contains(":5060")).any()  # tetap dicatat di CSV jendela
    cards = s.card_counts()
    assert cards["Total Aliran Trafik"] == 1 and cards[SIGNALING] == 1
    calls = pd.read_csv(s.calls_csv)
    assert list(calls["flow_id"]) == ["10.1.0.1:4000-10.1.0.2:4000-17"]
    summary = pd.read_csv(path)
    assert summary["total_aliran"].iloc[0] == 1 and summary["jumlah_panggilan"].iloc[0] == 1


def test_toast_gabungan_dan_tunggal():
    one = pd.DataFrame({"flow_id": ["192.168.1.25:1-192.168.1.10:2-17"], "label_akhir": ["QoS Baik"]})
    assert call_toast_message(one) == "Panggilan 192.168.1.25 ↔ 192.168.1.10 selesai: QoS Baik"
    many = pd.DataFrame({"flow_id": ["a"] * 3, "label_akhir": ["QoS Baik", "QoS Baik", "Anomali - Delay"]})
    assert call_toast_message(many) == "3 panggilan selesai: 2 QoS Baik, 1 Anomali"


# ---- status live, panggilan aktif, sesi terputus ----
import os  # noqa: E402
import time  # noqa: E402

from qos.capture.sniffer import default_interface_index  # noqa: E402
from qos.monitor.engine import STATUS_ABNORMAL, connection_status  # noqa: E402
from qos.monitor.recover import recover_orphan_sessions  # noqa: E402


def test_status_koneksi_label_terparah_dan_menunggu():
    mk = lambda kat, label: {"kategori": kat, "label": label}  # noqa: E731
    assert connection_status([]) == ("Menunggu data", "Menunggu data")
    assert connection_status([mk(INSUFFICIENT, INSUFFICIENT)]) == (INSUFFICIENT, INSUFFICIENT)
    active = [mk("QoS Baik", "QoS Baik"), mk("QoS Buruk", "QoS Buruk - Jitter"), mk(INSUFFICIENT, INSUFFICIENT)]
    assert connection_status(active) == ("QoS Buruk - Jitter", "QoS Buruk")
    assert connection_status(active + [mk("Anomali", "Anomali - X")])[1] == "Anomali"


def test_panggilan_aktif_dan_status_capture(env):
    cfg, _ = env
    sip = ("10.1.0.9", 5060), ("10.1.0.8", 5060)
    pkts = sorted(call_packets(CA, CB, 0, 10) + call_packets(*sip, 0, 10), key=lambda p: p.ts)
    s = call_session(cfg, pkts, hold=True)  # sumber tetap terbuka → panggilan masih aktif
    assert s.capture_stats()["paket"] == 0 and s.capture_stats()["paket_terakhir"] is None
    s.start()
    deadline = time.time() + 20
    while time.time() < deadline and not s.active_calls():
        time.sleep(0.1)
    (call,) = s.active_calls()  # sinyal SIP tidak ikut
    assert call["flow_id"] == "10.1.0.1:4000-10.1.0.2:4000-17" and call["durasi_s"] > 0
    assert call["riwayat"] and call["kategori"] == card_category(call["label"])
    stats = s.capture_stats()
    assert stats["paket"] == len(pkts) and stats["flow_aktif"] == 2 and stats["detik_sejak_paket"] >= 0
    s.stop()
    assert s.active_calls() == []  # Stop menyelesaikan panggilan


def test_stop_tidak_normal_lewat_atexit(env):
    cfg, _ = env
    s = call_session(cfg, call_packets(CA, CB, 0, 10), hold=True)
    s.start()
    s.wait(1.5)
    s._on_exit()  # simulasi aplikasi ditutup tanpa Stop
    summary = pd.read_csv(s.summary_path)
    assert summary["status_keseluruhan"].iloc[0] == STATUS_ABNORMAL
    assert summary["jumlah_panggilan"].iloc[0] == 1  # panggilan aktif ikut diselesaikan
    assert s.stop() == s.summary_path  # Stop sesudahnya idempoten


def test_error_sumber_menulis_ringkasan_abnormal(env):
    cfg, _ = env

    def source(stop):
        yield from call_packets(CA, CB, 0, 10)
        raise RuntimeError("capture putus")

    c = copy.deepcopy(cfg)
    s = MonitoringSession(c, InferenceEngine(load_artifacts(c), c), source, "putus")
    s.start()
    s.wait(30)
    assert s.error and pd.read_csv(s.summary_path)["status_keseluruhan"].iloc[0] == STATUS_ABNORMAL


def test_pulihkan_sesi_terputus(env, tmp_path):
    cfg, _ = env
    c = copy.deepcopy(cfg)
    c["paths"]["reports_dir"] = str(tmp_path / "reports")
    folder = tmp_path / "reports" / "riwayat"
    folder.mkdir(parents=True)
    rows = pd.DataFrame(
        {"waktu_jendela": ["2026-10-23T10:00:00"] * 2, "flow_id": ["a", "b"], "label": ["QoS Baik", SIGNALING]}
    )
    old, fresh = folder / "jendela_lama.csv", folder / "jendela_baru.csv"
    rows.to_csv(old, index=False)
    rows.to_csv(fresh, index=False)
    past = time.time() - c["monitoring"]["orphan_after_sec"] - 60
    os.utime(old, (past, past))
    (path,) = recover_orphan_sessions(c)  # "baru" masih mungkin berjalan → dilewati
    summary = pd.read_csv(path)
    assert summary["id_sesi_pemantauan"].iloc[0] == "lama" and summary["total_aliran"].iloc[0] == 1  # sinyal tak dihitung
    assert summary["status_keseluruhan"].iloc[0] == STATUS_ABNORMAL
    assert recover_orphan_sessions(c) == []  # sudah punya ringkasan
    assert recover_orphan_sessions(c, active_ids=["baru"]) == []


def test_interface_bawaan_memilih_yang_aktif():
    names = ["Bluetooth Network Connection", "Ethernet", "Wi-Fi"]
    assert default_interface_index(names, {"Bluetooth Network Connection": False, "Ethernet": False, "Wi-Fi": True}) == 2
    assert default_interface_index(names, {}) == 0
