"""Uji mesin inferensi dan sesi pemantauan (integrasi: latih sintetis → replay PCAP → CSV → ringkasan), klasterisasi murni."""
import copy
import os
import re
import time

import numpy as np
import pandas as pd
import pytest

from qos.capture.demo import write_demo_pcap
from qos.capture.flow_table import FlowEnd, PacketRecord, WindowFlow, flow_ports
from qos.capture.sniffer import default_interface_index, iter_pcap
from qos.config import load_config
from qos.model.assignment import parse_distances
from qos.model.train import save_artifacts, train
from qos.monitor.calls import (
    CALL_COLUMNS,
    INSUFFICIENT,
    SIGNALING,
    STATUS_CLUSTERED,
    STATUS_OUTSIDE,
    WINDOW_MODE_NOTE,
    CallTracker,
    call_title,
    call_toast_message,
    format_percentages,
    parse_percentages,
    summarize_calls,
)
from qos.monitor.engine import (
    STATUS_ABNORMAL,
    SUMMARY_COLUMNS,
    WINDOW_COLUMNS,
    InferenceEngine,
    ModelNotFoundError,
    MonitoringSession,
    connection_status,
    load_artifacts,
    safe_name,
    state_key,
)
from qos.monitor.recover import recover_orphan_sessions


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Config terisolasi + model sintetis aktif (fitur default Varian C) + PCAP demo."""
    root = tmp_path_factory.mktemp("env")
    cfg = copy.deepcopy(load_config(use_user=False))
    cfg["paths"].update({"artifacts_dir": str(root / "artifacts"), "reports_dir": str(root / "reports")})
    cfg["hdbscan"]["grid_search"].update({"min_cluster_size": [5, 10], "min_samples": [2, 5]})
    cfg["hdbscan"]["stability"]["n_bootstrap"] = 8
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


# Flow di pusat kelompok sintetis 2 (delay 10 ms, jitter 8 ms, FAI 10%) dan flow yang sangat jauh dari semua klaster
INSIDE = WindowFlow("dalam", 1000.0, 1005.0, 17, 300, 270, 570, 4.9e6, 10_000.0, 8_000.0)
FAR = WindowFlow("jauh", 1000.0, 1005.0, 17, 300, 5, 305, 4.9e6, 9e8, 9e8)


def test_load_artifacts_tanpa_model(tmp_path):
    cfg = copy.deepcopy(load_config(use_user=False))
    cfg["paths"]["artifacts_dir"] = str(tmp_path / "kosong")
    with pytest.raises(ModelNotFoundError, match="model aktif"):
        load_artifacts(cfg)


def test_load_artifacts_tidak_lengkap(env):
    cfg, _ = env
    with pytest.raises(ModelNotFoundError, match="tidak lengkap"):
        load_artifacts(cfg, version="20000101_000000")


def test_load_artifacts_versi_lama_ditolak_dengan_pesan_jelas(env, tmp_path):
    import json

    cfg, _ = env
    c = copy.deepcopy(cfg)
    c["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    old = tmp_path / "artifacts" / "20200101_000000"
    old.mkdir(parents=True)
    (old / "cluster_profile.json").write_text(json.dumps({"label_map": {"0": "QoS Baik"}, "clusters": []}), "utf-8")
    (tmp_path / "artifacts" / "current.txt").write_text("20200101_000000", "utf-8")
    with pytest.raises(ModelNotFoundError, match="pelabelan QoS"):
        load_artifacts(c)


def test_artifacts_properti_klaster_dimulai_dari_1(env):
    cfg, _ = env
    art = load_artifacts(cfg)
    assert art.cluster_ids == list(range(1, len(art.cluster_ids) + 1))
    assert set(art.limits) == set(art.centroids)
    assert sorted(art.kmeans_centroids) == list(range(1, len(art.kmeans_centroids) + 1))
    assert all(v > 0 for v in art.limits.values()) and set(art.descriptions) == set(art.cluster_ids)


def test_predict_kolom_dan_penugasan_euclidean(env):
    cfg, _ = env
    art = load_artifacts(cfg)
    flows = [
        WindowFlow("f1", 1000.0, 1005.0, 17, 250, 250, 500, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("f2", 1000.0, 1005.0, 17, 100, 90, 190, 4.9e6, 1_000.0, 500.0),
    ]
    out = InferenceEngine(art, cfg).predict(flows, "sesi1")
    assert list(out.columns) == list(WINDOW_COLUMNS)
    assert not {"label", "kategori_tindakan", "membership_prob", "cluster_id", "metrik_penyebab"} & set(out.columns)
    assert out["id_sesi"].tolist() == ["sesi1", "sesi1"]
    assert out["delay_ms"].tolist() == pytest.approx([10.0, 1.0])
    assert out["jitter_ms"].tolist() == pytest.approx([8.0, 0.5])
    assert out["fai_pct"].tolist() == pytest.approx([0.0, 10.0])
    assert set(out["klaster_terdekat"]) <= set(art.cluster_ids)
    assert set(out["klaster_kmeans5"]) <= set(art.kmeans_centroids)
    for r in out.to_dict("records"):  # klaster terdekat = jarak minimum di antara semua centroid
        dists = parse_distances(r["jarak_ke_klaster"])
        assert set(dists) == set(art.cluster_ids)
        assert min(dists, key=dists.get) == r["klaster_terdekat"]
        assert r["jarak_terdekat"] == pytest.approx(min(dists.values()), abs=1e-3)
        assert r["status"] == (STATUS_OUTSIDE if r["di_luar_klaster"] else STATUS_CLUSTERED)


def test_di_luar_klaster_jika_melebihi_batas_semua_centroid(env):
    cfg, _ = env
    art = load_artifacts(cfg)
    out = InferenceEngine(art, cfg).predict([INSIDE, FAR], "s").set_index("flow_id")
    assert out.loc["dalam", "status"] == STATUS_CLUSTERED and not out.loc["dalam", "di_luar_klaster"]
    far = out.loc["jauh"]
    assert far["status"] == STATUS_OUTSIDE and far["di_luar_klaster"]
    assert far["klaster_terdekat"] in art.cluster_ids  # "Di luar klaster (terdekat: Klaster N)"
    dists = parse_distances(far["jarak_ke_klaster"])
    assert all(d > art.limits[c] for c, d in dists.items())  # melebihi batas SETIAP centroid


def test_predict_impute_median_mon5(env):
    cfg, _ = env
    art = load_artifacts(cfg)
    flows = [WindowFlow("nan", 0.0, 5.0, 17, 100, 90, 190, 1e6, float("nan"), float("nan"))]
    out = InferenceEngine(art, cfg).predict(flows, "s")
    med = art.meta["median_imputasi"]
    assert out["delay_ms"].iloc[0] == pytest.approx(med["iat_mean_us"] / 1000)
    assert out["jitter_ms"].iloc[0] == pytest.approx(med["iat_std_us"] / 1000)
    assert out["status"].iloc[0] in (STATUS_CLUSTERED, STATUS_OUTSIDE) and bool(out["diimputasi"].iloc[0])


def test_data_belum_cukup_dicatat_tanpa_penugasan(env):
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
    assert out.loc["cukup", "status"] != INSUFFICIENT and not pd.isna(out.loc["cukup", "klaster_terdekat"])
    for fid in ("bwd_kurang", "fwd_kurang", "fwd_nol"):
        row = out.loc[fid]
        assert row["status"] == INSUFFICIENT and row["jarak_ke_klaster"] == ""
        assert pd.isna(row["klaster_terdekat"]) and pd.isna(row["jarak_terdekat"]) and pd.isna(row["di_luar_klaster"])
        assert pd.isna(row["klaster_kmeans5"]) and pd.isna(row["pc1"])
    assert out.loc["bwd_kurang", "fai_pct"] == pytest.approx(96.0)  # nilai mentah tetap dicatat
    assert pd.isna(out.loc["fwd_nol", "fai_pct"])


def test_min_session_live_nonaktif_tetap_buang_fwd_nol(env):
    cfg, _ = env
    c = copy.deepcopy(cfg)
    c["monitoring"]["min_session_live"] = {"fwd": None, "bwd": None}
    flows = [WindowFlow("a", 0.0, 5.0, 17, 3, 0, 3, 1e6, 100.0, 10.0), WindowFlow("b", 0.0, 5.0, 17, 0, 3, 3, 1e6, 1.0, 1.0)]
    out = InferenceEngine(load_artifacts(c), c).predict(flows, "s").set_index("flow_id")
    assert out.loc["a", "status"] != INSUFFICIENT  # tanpa batas, fwd ≥ 1 ditugaskan
    assert out.loc["b", "status"] == INSUFFICIENT  # fwd = 0: FAI tak terdefinisi


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
    assert "status_keseluruhan" not in summary and "Kritis" not in summary.to_csv()
    ev = load_artifacts(cfg).profile["evaluation"]
    assert summary["total_aliran"].iloc[0] == 3 and summary["status_sesi"].iloc[0] == "Selesai"
    assert summary["jumlah_klaster"].iloc[0] == ev["n_clusters"]
    assert summary["skor_silhouette"].iloc[0] == pytest.approx(ev["silhouette"])
    assert summary["skor_dbcv"].iloc[0] == pytest.approx(ev["dbcv"])
    assert summary["jumlah_panggilan"].iloc[0] == 3
    per = sum(int(v.split(":")[1]) for v in str(summary["panggilan_per_klaster"].iloc[0]).split(";") if ":" in v)
    assert per + summary["panggilan_di_luar_klaster"].iloc[0] + summary["panggilan_belum_cukup"].iloc[0] == 3
    assert 0 <= summary["persen_di_luar_klaster"].iloc[0] <= 100
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
    k = len(s.engine.art.cluster_ids)
    assert cards["Total Aliran Trafik"] == 3
    assert [x for x in cards if x.startswith("Klaster ")] == [f"Klaster {i}" for i in range(1, k + 1)]
    assert sum(v for x, v in cards.items() if x.startswith("Klaster ") or x == "Di luar klaster") == 3
    assert not {"QoS Baik", "QoS Cukup", "QoS Buruk", "Anomali"} & set(cards)
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


def test_state_key():
    assert state_key(STATUS_CLUSTERED, 3) == "Klaster 3"
    assert state_key(STATUS_OUTSIDE, 3) == "Di luar klaster"  # kartu: di luar klaster tidak dihitung ke klaster terdekat
    assert state_key(INSUFFICIENT) == INSUFFICIENT and state_key(SIGNALING) == SIGNALING


def test_id_sesi_punya_sufiks_uuid_6_karakter(env):
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
    assert first["klaster_akhir"] == last_row["klaster_terdekat"]  # klaster_akhir = jendela kumulatif terakhir
    assert first["jarak_akhir"] == pytest.approx(last_row["jarak_terdekat"])
    assert first["status_akhir"] == last_row["status"]
    for _, c in calls.iterrows():
        total = sum(parse_percentages(c["persen_per_klaster"]).values()) + c["persen_di_luar_klaster"] + c["persen_belum_cukup"]
        assert total == pytest.approx(100.0, abs=0.2)  # pembulatan 1 desimal per klaster
    assert calls["catatan"].isna().all()  # kosong pada mode cumulative
    assert "label_terburuk" not in calls and s.n_calls == 2 and sum(len(b) for b in s.drain_calls()) == 2


def test_panggilan_selesai_karena_stop(env):
    cfg, _ = env
    s = call_session(cfg, call_packets(CA, CB, 0, 10), hold=True)
    s.start()
    s.wait(1.5)
    assert s.running and not s.calls_csv.exists()  # belum idle, belum Stop → belum selesai
    path = s.stop()
    assert len(pd.read_csv(s.calls_csv)) == 1
    summary = pd.read_csv(path)
    assert summary["jumlah_panggilan"].iloc[0] == 1
    per = str(summary["panggilan_per_klaster"].iloc[0])
    n_clustered = sum(int(v.split(":")[1]) for v in per.split(";")) if per != "nan" else 0
    assert n_clustered + summary["panggilan_di_luar_klaster"].iloc[0] == 1


def test_panggilan_data_belum_cukup_tetap_dicatat(env):
    cfg, _ = env
    s = call_session(cfg, call_packets(CA, CB, 0, 10, both=False))  # searah: bwd = 0
    s.start()
    s.wait(60)
    s.stop()
    (call,) = pd.read_csv(s.calls_csv).to_dict("records")
    assert call["status_akhir"] == INSUFFICIENT and pd.isna(call["klaster_akhir"]) and pd.isna(call["klaster_terbanyak"])
    assert call["persen_belum_cukup"] == 100.0 and call["pernah_di_luar_klaster"] in (False, "False")
    summary = pd.read_csv(s.stop())
    assert summary["panggilan_belum_cukup"].iloc[0] == 1


def test_panggilan_mode_window_diberi_catatan(env):
    cfg, _ = env
    s = call_session(cfg, call_packets(CA, CB, 0, 10), mode="window")
    s.start()
    s.wait(60)
    s.stop()
    (call,) = pd.read_csv(s.calls_csv).to_dict("records")
    assert call["status_akhir"]  # tetap dicatat
    assert call["catatan"] == WINDOW_MODE_NOTE and "cumulative" in call["catatan"]


def _win(status, cluster, dist, fwd, outside=None):
    return {"flow_id": "f", "status": status, "klaster_terdekat": cluster, "jarak_terdekat": dist, "fwd": fwd, "bwd": fwd,
            "delay_ms": 1.0, "jitter_ms": 1.0, "fai_pct": 0.0,
            "di_luar_klaster": (status == STATUS_OUTSIDE) if outside is None else outside}


def test_tracker_persen_klaster_terbanyak_dan_pernah_di_luar():
    wins = [
        _win(STATUS_CLUSTERED, 2, 0.1, 0), _win(STATUS_CLUSTERED, 2, 0.1, 1), _win(STATUS_CLUSTERED, 1, 0.2, 2),
        _win(STATUS_OUTSIDE, 1, 3.0, 3), _win(INSUFFICIENT, pd.NA, np.nan, 4, outside=pd.NA),
    ]
    t = CallTracker("cumulative")
    t.update(pd.DataFrame(wins))
    (row,) = t.finish([FlowEnd("f", 0.0, 12.0), FlowEnd("tidak_ada", 0.0, 1.0)]).to_dict("records")
    assert row["jumlah_jendela"] == 5 and row["durasi_s"] == 12.0 and row["total_fwd"] == 4
    assert parse_percentages(row["persen_per_klaster"]) == {1: 20.0, 2: 40.0}
    assert (row["persen_di_luar_klaster"], row["persen_belum_cukup"]) == (20.0, 20.0)
    assert row["klaster_terbanyak"] == 2 and row["pernah_di_luar_klaster"] is True
    assert row["status_akhir"] == INSUFFICIENT and pd.isna(row["klaster_akhir"])  # jendela terakhir menentukan *_akhir
    assert t.finish([FlowEnd("f", 0.0, 1.0)]).empty  # state sudah dibuang


def test_summarize_calls_per_klaster_di_luar_dan_mayoritas():
    calls = pd.DataFrame({
        "status_akhir": [STATUS_CLUSTERED, STATUS_CLUSTERED, STATUS_CLUSTERED, STATUS_OUTSIDE, INSUFFICIENT],
        "klaster_akhir": [2, 2, 1, 1, pd.NA],
    })
    s = summarize_calls(calls)
    assert s["jumlah_panggilan"] == 5 and s["panggilan_per_klaster"] == "1:1;2:2"
    assert s["panggilan_di_luar_klaster"] == 1 and s["persen_di_luar_klaster"] == pytest.approx(25.0)  # 1 dari 4 yang dinilai
    assert s["klaster_mayoritas"] == 2 and s["panggilan_belum_cukup"] == 1
    empty = summarize_calls(pd.DataFrame(columns=["status_akhir", "klaster_akhir"]))
    assert empty["jumlah_panggilan"] == 0 and pd.isna(empty["klaster_mayoritas"]) and empty["persen_di_luar_klaster"] == 0.0
    tie = pd.DataFrame({"status_akhir": [STATUS_CLUSTERED] * 2, "klaster_akhir": [3, 1]})
    assert summarize_calls(tie)["klaster_mayoritas"] == 1  # seri → nomor klaster kecil


def test_format_dan_parse_persen():
    assert format_percentages({2: 3, 1: 1}, 4) == "1:25.0;2:75.0"
    assert parse_percentages("1:25.0;2:75.0") == {1: 25.0, 2: 75.0} and parse_percentages(float("nan")) == {}
    assert format_percentages({}, 0) == ""


def test_call_title():
    assert call_title("192.168.1.25:16384-192.168.1.10:16384-17") == "192.168.1.25 ↔ 192.168.1.10"
    assert call_title("aneh") == "aneh"


# ---- flow sinyal SIP ----
def test_flow_ports():
    assert flow_ports("10.0.0.1:5060-10.0.0.2:16384-17") == (5060, 16384)
    assert flow_ports("aneh") == (0, 0)


def test_predict_sinyal_sip_tidak_ditugaskan(env):
    cfg, _ = env
    eng = InferenceEngine(load_artifacts(cfg), cfg)
    flows = [
        WindowFlow("10.0.0.1:5060-10.0.0.2:5060-17", 1000.0, 1005.0, 17, 50, 50, 100, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("10.0.0.1:16384-10.0.0.2:5061-17", 1000.0, 1005.0, 17, 50, 50, 100, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("10.0.0.1:16384-10.0.0.2:16384-17", 1000.0, 1005.0, 17, 250, 250, 500, 4.9e6, 10_000.0, 8_000.0),
    ]
    out = eng.predict(flows, "s").set_index("flow_id")
    assert list(out["status"].iloc[:2]) == [SIGNALING, SIGNALING]
    assert out["klaster_terdekat"].iloc[:2].isna().all() and (out["jarak_ke_klaster"].iloc[:2] == "").all()
    assert out["status"].iloc[2] != SIGNALING and not pd.isna(out["klaster_terdekat"].iloc[2])
    assert eng.n_insufficient == 0  # sinyal bukan "Data Belum Cukup"


def test_sesi_sinyal_sip_dicatat_di_jendela_saja(env):
    cfg, _ = env
    sip_a, sip_b = ("10.1.0.1", 5060), ("10.1.0.2", 5060)
    pkts = sorted(call_packets(CA, CB, 0, 10) + call_packets(sip_a, sip_b, 0, 10), key=lambda p: p.ts)
    s = call_session(cfg, pkts)
    s.start()
    s.wait(60)
    path = s.stop()
    rows = pd.read_csv(s.window_csv)
    assert (rows.loc[rows["flow_id"].str.contains(":5060"), "status"] == SIGNALING).all()
    assert (rows["flow_id"].str.contains(":5060")).any()  # tetap dicatat di CSV jendela
    cards = s.card_counts()
    assert cards["Total Aliran Trafik"] == 1 and cards[SIGNALING] == 1
    calls = pd.read_csv(s.calls_csv)
    assert list(calls["flow_id"]) == ["10.1.0.1:4000-10.1.0.2:4000-17"]
    summary = pd.read_csv(path)
    assert summary["total_aliran"].iloc[0] == 1 and summary["jumlah_panggilan"].iloc[0] == 1


def test_toast_gabungan_dan_tunggal():
    fid = "192.168.1.25:1-192.168.1.10:2-17"
    inside = pd.DataFrame({"flow_id": [fid], "status_akhir": [STATUS_CLUSTERED], "klaster_akhir": [2]})
    assert call_toast_message(inside) == "Panggilan 192.168.1.25 ↔ 192.168.1.10 selesai: paling dekat ke Klaster 2"
    outside = pd.DataFrame({"flow_id": [fid], "status_akhir": [STATUS_OUTSIDE], "klaster_akhir": [1]})
    assert call_toast_message(outside) == "Panggilan 192.168.1.25 ↔ 192.168.1.10 selesai: Di luar klaster (terdekat: Klaster 1)"
    many = pd.DataFrame({"flow_id": ["a"] * 4, "status_akhir": [STATUS_CLUSTERED, STATUS_CLUSTERED, STATUS_OUTSIDE, INSUFFICIENT],
                         "klaster_akhir": [1, 1, 2, pd.NA]})
    assert call_toast_message(many) == "4 panggilan selesai: 2 di Klaster 1, 1 di luar klaster, 1 data belum cukup"


# ---- status live, panggilan aktif, sesi terputus ----
def _active(status, cluster, dist):
    return {"status": status, "klaster": cluster, "jarak": dist}


def test_status_koneksi_klaster_terbanyak_dan_menunggu():
    assert connection_status([])["teks"] == "Menunggu data"
    ins = connection_status([_active(INSUFFICIENT, None, None)])
    assert ins["teks"] == INSUFFICIENT and ins["klaster"] is None
    one = connection_status([_active(STATUS_CLUSTERED, 2, 0.5)])
    assert one["teks"] == "paling dekat ke Klaster 2" and one["jarak"] == 0.5 and not one["di_luar"]
    active = [_active(STATUS_CLUSTERED, 1, 0.2), _active(STATUS_CLUSTERED, 3, 0.4), _active(STATUS_CLUSTERED, 3, 0.8),
              _active(INSUFFICIENT, None, None)]
    top = connection_status(active)
    assert top["klaster"] == 3 and top["jarak"] == pytest.approx(0.6) and top["n_aktif"] == 4 and top["n_kelompok"] == 2
    tie = connection_status([_active(STATUS_CLUSTERED, 2, 0.1), _active(STATUS_CLUSTERED, 1, 0.1)])
    assert tie["klaster"] == 1  # seri → nomor klaster kecil
    out = connection_status([_active(STATUS_OUTSIDE, 4, 9.0), _active(STATUS_OUTSIDE, 4, 7.0), _active(STATUS_CLUSTERED, 1, 0.1)])
    assert out["di_luar"] and out["teks"] == "Di luar klaster (terdekat: Klaster 4)" and out["jarak"] == pytest.approx(8.0)
    tie_out = connection_status([_active(STATUS_OUTSIDE, 4, 9.0), _active(STATUS_CLUSTERED, 1, 0.1)])
    assert not tie_out["di_luar"]  # seri: klaster menang atas di luar klaster


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
    assert call["riwayat"] and call["kunci"] == state_key(call["status"], call["klaster"])
    assert set(call["jarak_ke_klaster"]) == set(s.engine.art.cluster_ids) and call["jarak"] >= 0
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
    assert summary["status_sesi"].iloc[0] == STATUS_ABNORMAL
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
    assert s.error and pd.read_csv(s.summary_path)["status_sesi"].iloc[0] == STATUS_ABNORMAL


def test_pulihkan_sesi_terputus(env, tmp_path):
    cfg, _ = env
    c = copy.deepcopy(cfg)
    c["paths"]["reports_dir"] = str(tmp_path / "reports")
    folder = tmp_path / "reports" / "riwayat"
    folder.mkdir(parents=True)
    rows = pd.DataFrame(
        {"waktu_jendela": ["2026-10-23T10:00:00"] * 2, "flow_id": ["a", "b"], "status": [STATUS_CLUSTERED, SIGNALING]}
    )
    old, fresh = folder / "jendela_lama.csv", folder / "jendela_baru.csv"
    rows.to_csv(old, index=False)
    rows.to_csv(fresh, index=False)
    past = time.time() - c["monitoring"]["orphan_after_sec"] - 60
    os.utime(old, (past, past))
    (path,) = recover_orphan_sessions(c)  # "baru" masih mungkin berjalan → dilewati
    summary = pd.read_csv(path)
    assert summary["id_sesi_pemantauan"].iloc[0] == "lama" and summary["total_aliran"].iloc[0] == 1  # sinyal tak dihitung
    assert summary["status_sesi"].iloc[0] == STATUS_ABNORMAL
    assert recover_orphan_sessions(c) == []  # sudah punya ringkasan
    assert recover_orphan_sessions(c, active_ids=["baru"]) == []


def test_interface_bawaan_memilih_yang_aktif():
    names = ["Bluetooth Network Connection", "Ethernet", "Wi-Fi"]
    assert default_interface_index(names, {"Bluetooth Network Connection": False, "Ethernet": False, "Wi-Fi": True}) == 2
    assert default_interface_index(names, {}) == 0

