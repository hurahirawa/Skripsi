"""Uji M8: metrik_penyebab, latih ulang (CSV/log), versioning, dan Terapkan Ulang Pelabelan."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qos.model.retrain import (
    InsufficientDataError,
    MonitoringActiveError,
    RetrainError,
    apply_overrides,
    list_log_files,
    load_log_frame,
    retrain,
)
from qos.model.versions import (
    VersionError,
    activate_version,
    active_version,
    label_changes,
    list_versions,
    relabel_profile,
    relabel_version,
    validate_thresholds,
)
from qos.monitor.engine import INSUFFICIENT, WINDOW_COLUMNS, Artifacts, InferenceEngine, load_artifacts
from qos.capture.flow_table import WindowFlow
from qos.profiling.labeling import apply_labels, cause_metrics
from tests.conftest import synthetic_flows


def digest(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


def write_log(cfg, name, rows: pd.DataFrame) -> Path:
    folder = Path(cfg["paths"]["reports_dir"]) / "riwayat"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"jendela_{name}.csv"
    rows.to_csv(path, index=False)
    return path


def log_rows(n_flows=300, sesi="s1", label="QoS Buruk - Jitter Melebihi Batas", windows=1, mode="cumulative", imputed=False) -> pd.DataFrame:
    """Baris jendela mirip keluaran MonitoringSession (delay/jitter dalam ms)."""
    flows = synthetic_flows(n=n_flows // 3)
    rows = []
    for w in range(windows):
        for i, r in flows.iterrows():
            rows.append(
                {"id_sesi": sesi, "flow_id": f"f{i}", "delay_ms": r.iat_mean_us / 1000 * (1 + 0.01 * w),
                 "jitter_ms": r.iat_std_us / 1000, "fwd": r.fwd_packets_count + w, "bwd": r.bwd_packets_count + w,
                 "label": label, "feature_mode": mode, "diimputasi": imputed}
            )
    return pd.DataFrame(rows)


# ---- metrik_penyebab ----------------------------------------------------------------------------
def test_cause_metrics_semua_kombinasi():
    for d, j, l in itertools.product((1, 2, 3), (1, 3), (1, 3)):
        levels = {"delay": d, "jitter": j, "loss": l}
        t = max(levels.values())
        assert cause_metrics(levels) == ([] if t == 1 else [m for m in levels if levels[m] == t])
        assert cause_metrics(levels, noise=True) == [m for m in levels if levels[m] == 3]


def test_apply_labels_format_string_dan_ambang():
    th = {"delay_good_ms": 150.0, "delay_max_ms": 400.0, "jitter_max_ms": 50.0, "loss_max_pct": 0.1}
    out = apply_labels({"mean_delay_ms": 485.26, "p999_jitter_ms": 4.12, "plr_pct": 22.92}, th)
    assert out["metrik_penyebab"] == "delay;loss" and out["tingkat"]["klaster"] == 3
    assert out["label"] == "QoS Buruk - Delay dan Packet Loss Melebihi Batas"
    noise = apply_labels({"mean_delay_ms": 1183.45, "p999_jitter_ms": 24.09, "plr_pct": 47.11}, th, noise=True)
    assert noise["metrik_penyebab"] == "delay;loss" and noise["label"].startswith("Anomali")
    assert apply_labels({"mean_delay_ms": 10, "p999_jitter_ms": 1, "plr_pct": 0}, th)["metrik_penyebab"] == ""
    assert apply_labels({"mean_delay_ms": 200, "p999_jitter_ms": 1, "plr_pct": 0}, th)["metrik_penyebab"] == "delay"


def test_profil_dan_csv_jendela_punya_metrik_penyebab(active_cfg):
    art = load_artifacts(active_cfg)
    assert all("metrik_penyebab" in c for c in art.profile["clusters"])
    eng = InferenceEngine(art, active_cfg)
    flows = [
        WindowFlow("ok", 0.0, 5.0, 17, 250, 250, 500, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("kurang", 0.0, 5.0, 17, 100, 2, 102, 1e6, 100.0, 10.0),
    ]
    raw = eng.predict(flows, "s")
    assert "metrik_penyebab" in WINDOW_COLUMNS and list(raw.columns) == list(WINDOW_COLUMNS)
    out = raw.set_index("flow_id")
    assert out.loc["ok", "metrik_penyebab"] == art.causes[int(out.loc["ok", "cluster_id"])]
    assert out.loc["kurang", "metrik_penyebab"] == ""  # Data Belum Cukup


def test_profil_lama_tanpa_kolom_dihitung_dari_tingkat(active_cfg):
    art = load_artifacts(active_cfg)
    old = json.loads(json.dumps(art.profile))
    expected = art.causes
    for c in old["clusters"]:
        c.pop("metrik_penyebab")
    legacy = Artifacts(art.version, art.model, art.scaler, art.pca, old, art.meta)
    assert legacy.causes == expected


# ---- versioning & relabel -----------------------------------------------------------------------
def test_validate_thresholds():
    ok = {"delay_good_ms": 150, "delay_max_ms": 400, "jitter_max_ms": 50, "loss_max_pct": 0.1}
    assert validate_thresholds(ok)["loss_max_pct"] == 0.1
    for bad in ({**ok, "delay_good_ms": 500}, {**ok, "jitter_max_ms": -1}, {k: v for k, v in ok.items() if k != "loss_max_pct"}):
        with pytest.raises(VersionError):
            validate_thresholds(bad)


def test_relabel_profile_murni_dari_statistik_tersimpan(active_cfg):
    doc = load_artifacts(active_cfg).profile
    th = doc["thresholds"]
    same = relabel_profile(doc, th)
    assert [c["label"] for c in same["clusters"]] == [c["label"] for c in doc["clusters"]]
    assert same["evaluation"] == doc["evaluation"]
    # ambang super longgar → semua klaster non-noise menjadi QoS Baik, noise tetap Anomali (tanpa metrik penyebab)
    loose = relabel_profile(doc, {"delay_good_ms": 1e9, "delay_max_ms": 1e9, "jitter_max_ms": 1e9, "loss_max_pct": 1e9})
    for c in loose["clusters"]:
        assert c["label"] == ("Anomali" if c["id_klaster"] == -1 else "QoS Baik")
        assert c["metrik_penyebab"] == ""
    # ambang super ketat → semua non-noise Buruk karena semua metrik
    tight = relabel_profile(doc, {"delay_good_ms": 0, "delay_max_ms": 0, "jitter_max_ms": 0, "loss_max_pct": 0})
    assert all(c["label"].startswith("QoS Buruk") or c["id_klaster"] == -1 for c in tight["clusters"])
    # statistik, boxplot, dan jumlah anggota tidak berubah
    for old, new in zip(doc["clusters"], loose["clusters"]):
        for k in ("statistik", "boxplot", "statistik_pelabelan", "jumlah_anggota"):
            assert old[k] == new[k]
    assert loose["label_map"] == {str(c["id_klaster"]): c["label"] for c in loose["clusters"]}


def test_relabel_version_membuat_versi_baru_tanpa_refit(active_cfg):
    root = Path(active_cfg["paths"]["artifacts_dir"])
    old_name = active_cfg["_active"]
    before_old = digest(root / old_name)
    before_current = (root / "current.txt").read_text("utf-8")
    new_th = {"delay_good_ms": 150, "delay_max_ms": 400, "jitter_max_ms": 1e9, "loss_max_pct": 1e9}

    out = relabel_version(active_cfg, old_name, new_th)

    assert out.name != old_name and out.parent == root
    assert digest(root / old_name) == before_old  # versi lama tidak berubah
    assert (root / "current.txt").read_text("utf-8") == before_current  # tidak mengaktifkan
    new = digest(out)
    for name in ("hdbscan_model.joblib", "scaler.joblib", "pca.joblib", "training_sample.csv"):
        assert new[name] == before_old[name]  # model/scaler/PCA persis sama → tanpa refit
    doc = json.loads((out / "cluster_profile.json").read_text("utf-8"))
    assert doc["model_version"] == out.name and doc["thresholds"]["loss_max_pct"] == 1e9
    meta = json.loads((out / "train_meta.json").read_text("utf-8"))
    assert meta["relabeled_from"] == old_name and meta["thresholds_sebelum"]["loss_max_pct"] == 0.1
    old_doc = json.loads((root / old_name / "cluster_profile.json").read_text("utf-8"))
    assert label_changes(old_doc, doc)["berubah"].any()


def test_relabel_version_tolak_versi_tak_lengkap_dan_ambang_tak_valid(active_cfg):
    with pytest.raises(VersionError):
        relabel_version(active_cfg, "20000101_000000", active_cfg["thresholds"])
    with pytest.raises(VersionError):
        relabel_version(active_cfg, active_cfg["_active"], {**active_cfg["thresholds"], "delay_good_ms": 999})
    assert len(list_versions(active_cfg)) == 1  # tidak ada versi setengah jadi


def test_list_versions_dan_aktivasi(active_cfg):
    first = active_cfg["_active"]
    out = relabel_version(active_cfg, first, {**active_cfg["thresholds"], "loss_max_pct": 5.0})
    v = list_versions(active_cfg).set_index("versi")
    assert v.loc[first, "aktif"] and not v.loc[out.name, "aktif"]
    assert v.loc[out.name, "sumber"] == "relabel" and v.loc[out.name, "induk"] == first
    assert list(list_versions(active_cfg)["versi"]) == sorted([first, out.name], reverse=True)

    with pytest.raises(VersionError, match="monitoring"):
        activate_version(active_cfg, out.name, monitoring_active=True)
    assert active_version(active_cfg) == first
    activate_version(active_cfg, out.name)
    assert active_version(active_cfg) == out.name
    assert not (Path(active_cfg["paths"]["artifacts_dir"]) / "current.tmp").exists()  # tulis atomik
    with pytest.raises(VersionError, match="tidak ditemukan"):
        activate_version(active_cfg, "tidak_ada")
    (Path(active_cfg["paths"]["artifacts_dir"]) / "rusak").mkdir()
    with pytest.raises(VersionError, match="tidak lengkap"):
        activate_version(active_cfg, "rusak")
    assert "rusak" not in list(list_versions(active_cfg)["versi"])


# ---- sumber log ---------------------------------------------------------------------------------
def test_load_log_frame_pengecualian_dan_konversi(iso_cfg):
    iso_cfg["retrain"]["log_source"].update({"min_rows": 10})
    good = log_rows(60)
    insufficient = good.head(20).assign(label=INSUFFICIENT, flow_id=lambda d: "x" + d["flow_id"])
    anomaly = good.head(15).assign(label="Anomali - Delay Ekstrem", flow_id=lambda d: "y" + d["flow_id"])
    write_log(iso_cfg, "a", pd.concat([good, insufficient, anomaly], ignore_index=True))
    frame, counts = load_log_frame(iso_cfg)
    assert counts["baris_log"] == 95 and counts["setelah_buang_data_belum_cukup"] == 75
    assert counts["setelah_buang_anomali"] == 60 and counts["setelah_filter_sesi"] == 60
    assert list(frame.columns) == ["iat_mean_us", "iat_std_us", "fwd_packets_count", "bwd_packets_count"]
    assert frame["iat_mean_us"].iloc[0] == pytest.approx(good["delay_ms"].iloc[0] * 1000)

    iso_cfg["retrain"]["log_source"]["exclude_anomaly"] = False
    _, counts = load_log_frame(iso_cfg)
    assert counts["setelah_buang_anomali"] == 75  # opsi config menentukan


def test_load_log_frame_per_flow_last_vs_all(iso_cfg):
    iso_cfg["retrain"]["log_source"].update({"min_rows": 10})
    write_log(iso_cfg, "a", log_rows(30, windows=3))
    last, c1 = load_log_frame(iso_cfg)
    assert len(last) == 30 and c1["baris_log"] == 90
    iso_cfg["retrain"]["log_source"]["per_flow"] = "all"
    allr, _ = load_log_frame(iso_cfg)
    assert len(allr) == 90
    iso_cfg["retrain"]["log_source"]["per_flow"] = "acak"
    with pytest.raises(RetrainError, match="per_flow"):
        load_log_frame(iso_cfg)


def test_load_log_frame_per_flow_terpisah_per_sesi(iso_cfg):
    iso_cfg["retrain"]["log_source"].update({"min_rows": 10})
    write_log(iso_cfg, "a", log_rows(30, sesi="s1"))
    write_log(iso_cfg, "b", log_rows(30, sesi="s2"))  # flow_id sama, sesi berbeda → tetap dua baris
    assert len(load_log_frame(iso_cfg)[0]) == 60


def test_load_log_frame_filter_sesi_minimum(iso_cfg):
    iso_cfg["retrain"]["log_source"].update({"min_rows": 1})
    rows = log_rows(30)
    rows.loc[:9, "bwd"] = 2  # di bawah min_session.bwd = 5
    write_log(iso_cfg, "a", rows)
    frame, counts = load_log_frame(iso_cfg)
    assert counts["setelah_per_flow"] == 30 and counts["setelah_filter_sesi"] == 20 and len(frame) == 20


def test_min_rows_dicek_dan_pesan_menyebut_hitungan(iso_cfg):
    write_log(iso_cfg, "a", log_rows(30))
    iso_cfg["retrain"]["log_source"]["min_rows"] = 200
    with pytest.raises(InsufficientDataError, match=r"30 baris < min_rows \(200\)"):
        load_log_frame(iso_cfg)


def test_log_tidak_ada_dan_exclude_insufficient_wajib(iso_cfg):
    with pytest.raises(InsufficientDataError, match="Tidak ada log"):
        load_log_frame(iso_cfg)
    iso_cfg["retrain"]["log_source"]["exclude_insufficient"] = False
    write_log(iso_cfg, "a", log_rows(30))
    with pytest.raises(RetrainError, match="exclude_insufficient"):
        load_log_frame(iso_cfg)
    assert list_log_files(iso_cfg) and list_log_files(iso_cfg)[0].name.startswith("jendela_")


# ---- latih ulang --------------------------------------------------------------------------------
def test_apply_overrides_tidak_mengubah_config_asli(iso_cfg):
    out = apply_overrides(iso_cfg, {"min_cluster_size": 9, "min_samples": 3, "grid_enabled": False,
                                    "thresholds": {"loss_max_pct": 1.0}})
    assert (out["hdbscan"]["min_cluster_size"], out["hdbscan"]["min_samples"]) == (9, 3)
    assert out["hdbscan"]["grid_search"]["enabled"] is False and out["thresholds"]["loss_max_pct"] == 1.0
    assert iso_cfg["thresholds"]["loss_max_pct"] == 0.1 and iso_cfg["hdbscan"]["min_cluster_size"] == 5


def test_retrain_log_tidak_menimpa_dan_tidak_mengaktifkan(active_cfg):
    root = Path(active_cfg["paths"]["artifacts_dir"])
    old = active_cfg["_active"]
    write_log(active_cfg, "a", log_rows(300))
    active_cfg["retrain"]["log_source"]["min_rows"] = 100
    before = digest(root / old)
    cur_before = (root / "current.txt").read_text("utf-8")
    ticks = []

    res = retrain(active_cfg, "log", progress=lambda d, t: ticks.append((d, t)))

    assert res["version"] != old and (root / res["version"]).is_dir()
    assert digest(root / old) == before and (root / "current.txt").read_text("utf-8") == cur_before
    assert ticks and ticks[-1][0] == ticks[-1][1]  # progress sampai selesai
    meta = json.loads((root / res["version"] / "train_meta.json").read_text("utf-8"))
    assert meta["sumber_data"]["jenis"] == "log" and meta["sumber_data"]["jumlah_baris"]["setelah_filter_sesi"] == 300
    assert list_versions(active_cfg).set_index("versi").loc[res["version"], "sumber"] == "log"
    assert not list_versions(active_cfg).set_index("versi").loc[res["version"], "aktif"]
    assert res["evaluation"]["n_clusters"] >= 1


def test_retrain_ditolak_saat_monitoring_dan_data_kurang(active_cfg):
    root = Path(active_cfg["paths"]["artifacts_dir"])
    n_before = len(list(root.iterdir()))
    with pytest.raises(MonitoringActiveError):
        retrain(active_cfg, "log", monitoring_active=True)
    write_log(active_cfg, "a", log_rows(30))
    with pytest.raises(InsufficientDataError):
        retrain(active_cfg, "log")  # min_rows default 200
    with pytest.raises(RetrainError, match="tidak dikenal"):
        retrain(active_cfg, "ngawur")
    assert len(list(root.iterdir())) == n_before  # tidak ada versi setengah jadi


def test_retrain_parameter_dari_ui_dipakai(active_cfg):
    write_log(active_cfg, "a", log_rows(300))
    active_cfg["retrain"]["log_source"]["min_rows"] = 100
    res = retrain(active_cfg, "log", params={"grid_enabled": False, "min_cluster_size": 7, "min_samples": 3})
    assert res["params"]["min_cluster_size"] == 7 and res["params"]["grid_search"] is False
    assert active_cfg["hdbscan"]["min_cluster_size"] == 5  # config asli utuh
    assert active_version(active_cfg) == active_cfg["_active"]


def test_label_changes_tabel():
    old = {"clusters": [{"id_klaster": 0, "label": "A"}, {"id_klaster": 1, "label": "B"}]}
    new = {"clusters": [{"id_klaster": 0, "label": "A"}, {"id_klaster": 1, "label": "C"}]}
    t = label_changes(old, new)
    assert t["berubah"].tolist() == [False, True] and t.loc[1, "label_baru"] == "C"
    assert np.all(t.columns == ["klaster", "label_lama", "label_baru", "berubah"])


# ---- kolom feature_mode dan diimputasi (M8b) ----------------------------------------------------
def test_load_log_frame_csv_lama_tanpa_kolom_dikecualikan_dan_dilaporkan(iso_cfg):
    iso_cfg["retrain"]["log_source"]["min_rows"] = 10
    write_log(iso_cfg, "baru", log_rows(60))
    write_log(iso_cfg, "lama", log_rows(30, sesi="lama").drop(columns=["feature_mode", "diimputasi"]))
    frame, counts = load_log_frame(iso_cfg)
    assert counts["baris_log"] == 90 and counts["dikecualikan_mode_tidak_diketahui"] == 30
    assert counts["dikecualikan_mode_lain"] == 0 and len(frame) == 60


def test_load_log_frame_hanya_mode_config(iso_cfg):
    iso_cfg["retrain"]["log_source"]["min_rows"] = 10
    write_log(iso_cfg, "c", log_rows(45, sesi="c", mode="cumulative"))
    write_log(iso_cfg, "w", log_rows(30, sesi="w", mode="window"))
    _, counts = load_log_frame(iso_cfg)
    assert counts["dikecualikan_mode_lain"] == 30 and counts["setelah_filter_sesi"] == 45
    iso_cfg["retrain"]["log_source"]["feature_mode"] = "window"
    frame, counts = load_log_frame(iso_cfg)
    assert len(frame) == 30 and counts["dikecualikan_mode_lain"] == 45
    iso_cfg["retrain"]["log_source"]["feature_mode"] = None  # null = semua mode
    assert len(load_log_frame(iso_cfg)[0]) == 75
    iso_cfg["retrain"]["log_source"]["feature_mode"] = "acak"
    with pytest.raises(RetrainError, match="feature_mode"):
        load_log_frame(iso_cfg)


def test_load_log_frame_kecualikan_diimputasi_sesuai_opsi(iso_cfg):
    iso_cfg["retrain"]["log_source"]["min_rows"] = 10
    rows = pd.concat([log_rows(45, sesi="a"), log_rows(30, sesi="b", imputed=True)], ignore_index=True)
    write_log(iso_cfg, "a", rows)
    frame, counts = load_log_frame(iso_cfg)
    assert counts["dikecualikan_diimputasi"] == 30 and len(frame) == 45
    iso_cfg["retrain"]["log_source"]["exclude_imputed"] = False
    assert len(load_log_frame(iso_cfg)[0]) == 75


def test_pesan_min_rows_merinci_pengecualian_mode_dan_imputasi(iso_cfg):
    write_log(iso_cfg, "lama", log_rows(30).drop(columns=["feature_mode", "diimputasi"]))
    with pytest.raises(InsufficientDataError, match="mode tidak diketahui 30"):
        load_log_frame(iso_cfg)


def test_csv_jendela_punya_kolom_feature_mode_dan_diimputasi(active_cfg):
    eng = InferenceEngine(load_artifacts(active_cfg), active_cfg)
    flows = [
        WindowFlow("normal", 0.0, 5.0, 17, 250, 250, 500, 4.9e6, 10_000.0, 8_000.0),
        WindowFlow("nan", 0.0, 5.0, 17, 250, 250, 500, 4.9e6, float("nan"), 8_000.0),
        WindowFlow("kurang", 0.0, 5.0, 17, 100, 2, 102, 1e6, float("nan"), 10.0),
    ]
    out = eng.predict(flows, "s").set_index("flow_id")
    assert out["feature_mode"].tolist() == [active_cfg["monitoring"]["feature_mode"]] * 3
    assert out["diimputasi"].tolist() == [False, True, False]  # baris tanpa prediksi tidak diimputasi
    assert out["diimputasi"].dtype == bool
    assert {"feature_mode", "diimputasi"} <= set(WINDOW_COLUMNS)


def test_csv_jendela_nyata_dari_sesi_bisa_dibaca_loader(active_cfg, tmp_path):
    """Round-trip: CSV hasil MonitoringSession memuat kolom baru dan lolos pengecualian bila mode cocok."""
    from qos.capture.demo import write_demo_pcap
    from qos.capture.sniffer import iter_pcap
    from qos.monitor.engine import MonitoringSession

    pcap = write_demo_pcap(tmp_path / "d.pcap", duration_sec=12.0)
    active_cfg["monitoring"]["feature_mode"] = "cumulative"
    s = MonitoringSession(active_cfg, InferenceEngine(load_artifacts(active_cfg), active_cfg), lambda stop: iter_pcap(pcap, 0, stop), "rt")
    s.start()
    s.wait(60)
    s.stop()
    rows = pd.read_csv(s.window_csv)
    assert set(rows["feature_mode"]) == {"cumulative"} and not rows["diimputasi"].any()
    active_cfg["retrain"]["log_source"]["min_rows"] = 1
    frame, counts = load_log_frame(active_cfg, [s.window_csv])
    assert counts["dikecualikan_mode_tidak_diketahui"] == 0 and len(frame) >= 1
