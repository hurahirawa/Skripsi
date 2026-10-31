"""Uji unggah CSV latih ulang: deteksi format, filter per tahap, mode Tambah/Ganti, minimum baris, grid adaptif, data_latih.csv, pemetaan."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qos.model.dataset import (
    DATA_COLUMNS, DATA_FILE, FORMAT_PANGGILAN, DatasetError, backfill_data_latih, detect_upload_format, load_upload,
    prepare_dataset, stage_table,
)
from qos.model.retrain import InsufficientDataError, MonitoringActiveError, RetrainError, map_old_to_new, retrain_upload
from qos.model.train import adaptive_grid, save_artifacts, train
from qos.model.versions import active_version, list_versions
from tests.conftest import isolated_cfg, synthetic_flows

BASE_COLS = ["iat_mean_us", "iat_std_us", "fwd_packets_count", "bwd_packets_count"]


def digest(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


def flows_to_darknet(flows: pd.DataFrame, path: Path, app="VOIP", traffic="NonVPN") -> Path:
    """Tulis flow sebagai CSV Darknet (kolom Label = jenis traffic, Label.1 = kategori aplikasi)."""
    n = len(flows)
    flows = flows.reset_index(drop=True)
    df = pd.DataFrame(
        {
            "Flow ID": [f"f{i}" for i in range(n)], "Src IP": "1.1.1.1", "Src Port": 40000, "Dst IP": "2.2.2.2", "Dst Port": 20000,
            "Protocol": 17, "Timestamp": "x", "Flow Duration": (flows["iat_mean_us"] * 50).round(1),
            "Total Fwd Packet": flows["fwd_packets_count"], "Total Bwd packets": flows["bwd_packets_count"],
            "Flow IAT Mean": flows["iat_mean_us"], "Flow IAT Std": flows["iat_std_us"], "Label": traffic, "Label.1": app,
        }
    )
    df.to_csv(path, index=False)
    return path


def flows_to_panggilan(flows: pd.DataFrame, path: Path, status="Terklaster", catatan="") -> Path:
    n = len(flows)
    flows = flows.reset_index(drop=True)
    df = pd.DataFrame(
        {
            "flow_id": [f"10.0.0.1:{16000 + i}-10.0.0.2:{16000 + i}-17" for i in range(n)],
            "waktu_mulai": "2026-10-29T10:00:00", "waktu_selesai": "2026-10-29T10:01:00", "durasi_s": 60.0,
            "total_fwd": flows["fwd_packets_count"], "total_bwd": flows["bwd_packets_count"],
            "delay_ms": flows["iat_mean_us"] / 1000, "jitter_ms": flows["iat_std_us"] / 1000, "fai_pct": 0.0,
            "klaster_akhir": 1, "jarak_akhir": 0.1, "di_luar_klaster_akhir": False, "status_akhir": status, "jumlah_jendela": 12,
            "persen_per_klaster": "1:100.0", "persen_di_luar_klaster": 0.0, "persen_belum_cukup": 0.0, "klaster_terbanyak": 1,
            "pernah_di_luar_klaster": False, "catatan": catatan,
        }
    )
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def base(tmp_path):
    """Config terisolasi + model aktif dilatih dari 240 baris sintetis beserta data_latih.csv."""
    cfg = isolated_cfg(tmp_path)
    out = save_artifacts(train(synthetic_flows(n=80), cfg), cfg, activate=True)
    cfg["_active"] = out.name
    return cfg, tmp_path


def new_flows(seed, n=40):
    return synthetic_flows(seed=seed, n=n)


def version_dir(cfg, version=None) -> Path:
    return Path(cfg["paths"]["artifacts_dir"]) / (version or cfg["_active"])


# ---- data_latih.csv per versi -------------------------------------------------------------------------
def test_data_latih_csv_tersimpan_lengkap(base):
    cfg, _ = base
    folder = version_dir(cfg)
    data = pd.read_csv(folder / DATA_FILE)
    meta = json.loads((folder / "train_meta.json").read_text("utf-8"))
    assert list(data.columns) == list(DATA_COLUMNS)
    assert len(data) == meta["jumlah_baris_latih"] == 240  # seluruh baris latih, bukan sampel
    assert data["id_baris"].is_unique and data["waktu_ditambahkan"].notna().all()
    assert meta["data_latih"]["mode"] == "awal" and meta["data_latih"]["jumlah_total"] == 240
    assert meta["data_latih"]["versi_induk"] is None


def test_data_latih_lengkap_walau_training_sample_disubsampel(tmp_path):
    cfg = isolated_cfg(tmp_path)
    cfg["evaluation"]["pca_sample_size"] = 50
    out = save_artifacts(train(synthetic_flows(n=80), cfg), cfg, activate=True)
    assert len(pd.read_csv(out / "training_sample.csv")) == 50 and len(pd.read_csv(out / DATA_FILE)) == 240


def test_backfill_tanpa_refit_dan_tidak_menimpa(base):
    cfg, _ = base
    folder = version_dir(cfg)
    (folder / DATA_FILE).unlink()
    before = {k: v for k, v in digest(folder).items() if k != "train_meta.json"}
    data = pd.read_csv(backfill_data_latih(cfg, cfg["_active"], "Darknet.CSV"))
    assert len(data) == 240 and set(data["sumber"]) == {"Darknet.CSV"}
    after = {k: v for k, v in digest(folder).items() if k not in ("train_meta.json", DATA_FILE)}
    assert after == before  # model tidak disentuh
    meta = json.loads((folder / "train_meta.json").read_text("utf-8"))
    assert meta["data_latih"]["dibuat_tanpa_refit"] and meta["data_latih"]["file_sumber"] == ["Darknet.CSV"]
    with pytest.raises(DatasetError, match="sudah ada"):
        backfill_data_latih(cfg, cfg["_active"], "x")


def test_backfill_ditolak_bila_sampel_bukan_seluruh_data(base):
    cfg, _ = base
    folder = version_dir(cfg)
    (folder / DATA_FILE).unlink()
    pd.read_csv(folder / "training_sample.csv").head(10).to_csv(folder / "training_sample.csv", index=False)
    with pytest.raises(DatasetError, match="bukan seluruh data latih"):
        backfill_data_latih(cfg, cfg["_active"], "x")


# ---- deteksi format -------------------------------------------------------------------------------------
def test_deteksi_format_panggilan_dan_darknet(base):
    cfg, tmp = base
    f = new_flows(1)
    assert detect_upload_format(flows_to_panggilan(f, tmp / "panggilan_a.csv"), cfg["data"]) == FORMAT_PANGGILAN
    assert detect_upload_format(flows_to_darknet(f, tmp / "Darknet.csv"), cfg["data"]) == "darknet"


def test_format_salah_ditolak_dengan_kolom_yang_kurang(base):
    cfg, tmp = base
    bad = tmp / "salah.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, index=False)
    with pytest.raises(DatasetError, match="tidak dikenali") as exc:
        detect_upload_format(bad, cfg["data"])
    assert "durasi_s" in str(exc.value) and "Flow IAT Mean" in str(exc.value)
    partial = tmp / "setengah.csv"
    pd.DataFrame({"flow_id": ["x"], "total_fwd": [1], "delay_ms": [1.0]}).to_csv(partial, index=False)
    with pytest.raises(DatasetError, match="Format panggilan tidak lengkap.*total_bwd"):
        detect_upload_format(partial, cfg["data"])
    frame, rep = load_upload(partial, cfg)  # load_upload melaporkan, tidak melempar
    assert frame.empty and "total_bwd" in rep.error


def test_panggilan_format_lama_tanpa_status_akhir_ditolak(base):
    cfg, tmp = base
    old = pd.read_csv(flows_to_panggilan(new_flows(1), tmp / "p.csv")).drop(columns=["status_akhir"])
    old.to_csv(tmp / "lama.csv", index=False)
    with pytest.raises(DatasetError, match="status_akhir"):
        detect_upload_format(tmp / "lama.csv", cfg["data"])


# ---- filter per tahap -----------------------------------------------------------------------------------
def test_panggilan_filter_per_tahap_dan_konversi_ke_mikrodetik(base):
    cfg, tmp = base
    f = new_flows(1, n=7)  # 21 baris
    good = pd.read_csv(flows_to_panggilan(f, tmp / "g.csv"))
    sip = good.head(3).assign(status_akhir="Sinyal SIP", flow_id=lambda d: "s" + d["flow_id"])
    short = good.head(4).assign(status_akhir="Data Belum Cukup", flow_id=lambda d: "b" + d["flow_id"])
    window = good.head(5).assign(catatan="Mode window: ...", flow_id=lambda d: "w" + d["flow_id"])
    imputed = good.head(6).assign(diimputasi=True, flow_id=lambda d: "i" + d["flow_id"])
    good["diimputasi"] = False
    pd.concat([good, sip, short, window, imputed], ignore_index=True).to_csv(tmp / "p.csv", index=False)
    frame, rep = load_upload(tmp / "p.csv", cfg)
    stages = dict(rep.stages)
    assert stages["baris di berkas"] == 39 and stages["tanpa Sinyal SIP"] == 36 and stages["tanpa Data Belum Cukup"] == 32
    assert stages["tanpa mode window"] == 27 and stages["tanpa baris diimputasi"] == 21
    assert len(frame) == 21 and rep.error is None
    assert frame["iat_mean_us"].iloc[0] == pytest.approx(f["iat_mean_us"].iloc[0], rel=1e-9)  # ms x 1000 kembali ke µs
    assert (frame["duration_us"] == 60_000_000).all() and (frame["protocol"] == 17).all()


def test_flow_darknet_filter_voip_exclude_tor_sesi_dan_nilai_buruk(base):
    cfg, tmp = base
    f = new_flows(2, n=10)  # 30 baris
    f.loc[0, "bwd_packets_count"] = 2      # < min_session
    f.loc[1, "iat_std_us"] = -5.0          # negatif
    f.loc[2, "iat_mean_us"] = np.inf       # inf
    path = flows_to_darknet(f, tmp / "d.csv")
    raw = pd.read_csv(path)
    raw.loc[20:24, "Label.1"] = "Browsing"  # 5 baris bukan VOIP
    raw.loc[25:26, "Label"] = "Tor"         # 2 baris Tor
    raw.loc[27, "Flow IAT Mean"] = np.nan   # NaN
    raw.to_csv(path, index=False)
    out, rep = load_upload(path, cfg)
    s = dict(rep.stages)
    assert s["baris di berkas"] == 30 and s["setelah filter VoIP"] == 25 and s["tanpa Tor"] == 23
    assert s["setelah filter sesi minimum"] == 22
    assert s["setelah buang NaN/inf/negatif"] == 19 and len(out) == 19


def test_duplikat_dibuang_di_berkas_antarberkas_dan_laporan_per_tahap(base):
    cfg, tmp = base
    f = new_flows(5, n=20)  # 60 baris unik
    a = flows_to_darknet(pd.concat([f, f.head(5)], ignore_index=True), tmp / "a.csv")  # 5 duplikat di dalam berkas
    b = flows_to_darknet(f.head(8), tmp / "b.csv")                                      # semua sudah ada di a.csv
    prep = prepare_dataset(cfg, [(a, "a.csv"), (b, "b.csv")], "ganti", cfg["_active"])
    assert [r.accepted for r in prep.files] == [60, 0]
    table = stage_table(prep)
    assert list(table["berkas"]) == ["a.csv", "b.csv"] and "setelah buang duplikat" in table.columns


# ---- mode Tambah / Ganti ---------------------------------------------------------------------------------
def test_mode_tambah_200_ditambah_50_duplikat_tidak_dihitung_dua_kali(tmp_path):
    cfg = isolated_cfg(tmp_path)
    first = synthetic_flows(n=80).iloc[:200]  # tepat 200 baris; durasi/protokol sama seperti yang ditulis flows_to_darknet
    first = first.assign(duration_us=(first["iat_mean_us"] * 50).round(1), protocol=17)
    save_artifacts(train(first, cfg), cfg, activate=True)
    old = pd.read_csv(version_dir(cfg, active_version(cfg)) / DATA_FILE)
    assert len(old) == 200
    fresh = synthetic_flows(seed=9, n=50).iloc[:50]
    p_fresh = flows_to_darknet(fresh, tmp_path / "baru.csv")
    prep = prepare_dataset(cfg, [(p_fresh, "baru.csv")], "tambah", active_version(cfg))
    assert (prep.n_old, prep.n_new, prep.n_total) == (200, 50, 250) and prep.ok
    # berkas yang sama diunggah dua kali + 10 baris yang sudah ada di data lama: tidak bertambah
    p_dup = flows_to_darknet(pd.concat([fresh, old.head(10)[BASE_COLS]]).reset_index(drop=True), tmp_path / "dup.csv")
    dup = prepare_dataset(cfg, [(p_fresh, "baru.csv"), (p_dup, "dup.csv")], "tambah", active_version(cfg))
    assert dup.n_new == 50 and dup.n_total == 250 and dup.files[1].accepted == 0
    assert dup.frame["id_baris"].is_unique


def test_mode_ganti_hanya_data_baru(base):
    cfg, tmp = base
    path = flows_to_darknet(new_flows(7), tmp / "baru.csv")
    prep = prepare_dataset(cfg, [(path, "baru.csv")], "ganti", cfg["_active"])
    assert prep.n_old == 0 and prep.n_new == 120 and prep.n_total == 120 and prep.n_old_dropped == 240
    assert set(prep.frame["sumber"]) == {"baru.csv"}
    # data identik dengan data lama tetap lolos pada Ganti (duplikat terhadap data lama tidak relevan)
    old = pd.read_csv(version_dir(cfg) / DATA_FILE)
    same = flows_to_darknet(old[BASE_COLS], tmp / "sama.csv")
    assert prepare_dataset(cfg, [(same, "sama.csv")], "ganti", cfg["_active"]).n_new == 240


def test_tambah_tanpa_data_latih_versi_aktif_memberi_petunjuk(base):
    cfg, tmp = base
    (version_dir(cfg) / DATA_FILE).unlink()
    path = flows_to_darknet(new_flows(7), tmp / "b.csv")
    with pytest.raises(DatasetError, match="backfill_data_latih"):
        prepare_dataset(cfg, [(path, "b.csv")], "tambah", cfg["_active"])
    with pytest.raises(DatasetError, match="Mode tidak dikenal"):
        prepare_dataset(cfg, [(path, "b.csv")], "acak", cfg["_active"])


def test_minimum_baris_ditolak_dan_peringatan_di_bawah_100(base):
    cfg, tmp = base
    tiny = flows_to_darknet(new_flows(7, n=5).iloc[:12], tmp / "t.csv")
    prep = prepare_dataset(cfg, [(tiny, "t.csv")], "ganti", cfg["_active"])
    assert not prep.ok and any("minimum" in p for p in prep.problems)
    root = Path(cfg["paths"]["artifacts_dir"])
    n_dirs = len(list(root.iterdir()))
    with pytest.raises(InsufficientDataError, match="retrain.min_rows"):
        retrain_upload(cfg, [(tiny, "t.csv")], "ganti")
    assert len(list(root.iterdir())) == n_dirs  # tidak ada versi setengah jadi
    mid = flows_to_darknet(new_flows(8, n=20), tmp / "m.csv")  # 60 baris: lolos tetapi diperingatkan
    prep = prepare_dataset(cfg, [(mid, "m.csv")], "ganti", cfg["_active"])
    assert prep.ok and prep.warnings and "< 100" in prep.warnings[0]


def test_berkas_ditolak_menghalangi_latih(base):
    cfg, tmp = base
    bad = tmp / "salah.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    good = flows_to_darknet(new_flows(7), tmp / "ok.csv")
    prep = prepare_dataset(cfg, [(good, "ok.csv"), (bad, "salah.csv")], "tambah", cfg["_active"])
    assert not prep.ok and "salah.csv" in prep.problems[0]
    with pytest.raises(RetrainError, match="salah.csv"):
        retrain_upload(cfg, [(good, "ok.csv"), (bad, "salah.csv")], "tambah")
    assert not prepare_dataset(cfg, [], "tambah", cfg["_active"]).ok


# ---- grid adaptif ---------------------------------------------------------------------------------------
def test_adaptive_grid_membuang_nilai_di_atas_setengah_n():
    full = {"min_cluster_size": [3, 5, 8, 10, 15, 20, 30, 50], "min_samples": [1, 2, 3, 5, 8, 10, 15]}
    assert adaptive_grid(full, 50) == ([3, 5, 8, 10, 15, 20], [1, 2, 3, 5, 8, 10, 15])
    assert adaptive_grid(full, 12) == ([3, 5], [1, 2, 3, 5])
    assert adaptive_grid(full, 247)[0] == full["min_cluster_size"]  # data awal tidak berubah


def test_latih_50_baris_grid_adaptif_dan_kmeans_kmax(tmp_path):
    cfg = isolated_cfg(tmp_path)
    cfg["hdbscan"]["grid_search"].update({"min_cluster_size": [3, 5, 8, 30, 50], "min_samples": [2, 5, 40]})
    cfg["kmeans"]["k_max"] = 60
    res = train(synthetic_flows(n=17).iloc[:50], cfg)
    grid = res["grid"]
    assert grid["min_cluster_size"].max() <= 25 and grid["min_samples"].max() <= 25 and len(grid) == 3 * 2
    assert any("Grid disesuaikan" in w for w in res["peringatan"])
    assert res["kmeans_sweep"]["k"].max() == 49  # min(config, n-1)
    assert res["n_train_rows"] == 50


def test_tanpa_kandidat_lolos_pakai_fallback_dengan_peringatan(tmp_path):
    cfg = isolated_cfg(tmp_path)
    cfg["hdbscan"]["selection"]["min_ari_mean"] = 1.01  # mustahil
    res = train(synthetic_flows(n=17).iloc[:50], cfg)
    assert res["params"]["grid_fallback"] and any("Tidak ada kandidat grid" in w for w in res["peringatan"])
    cfg2 = isolated_cfg(tmp_path / "b")
    cfg2["hdbscan"]["grid_search"].update({"min_cluster_size": [60], "min_samples": [60]})  # semua kombinasi terbuang
    res2 = train(synthetic_flows(n=17).iloc[:50], cfg2)
    assert res2["params"]["grid_fallback"] and res2["grid"] is None
    assert any("Tidak ada kandidat" in w for w in res2["peringatan"])


# ---- latih ulang end-to-end -----------------------------------------------------------------------------
def test_retrain_upload_tambah_versi_baru_tidak_menimpa_dan_tidak_mengaktifkan(base):
    cfg, tmp = base
    root = Path(cfg["paths"]["artifacts_dir"])
    old = cfg["_active"]
    before, current = digest(root / old), (root / "current.txt").read_text("utf-8")
    path = flows_to_panggilan(new_flows(11), tmp / "panggilan_x.csv")
    ticks = []
    res = retrain_upload(cfg, [(path, "panggilan_x.csv")], "tambah", progress=lambda d, t: ticks.append((d, t)))
    assert res["version"] != old and digest(root / old) == before and (root / "current.txt").read_text("utf-8") == current
    assert ticks and ticks[-1][0] == ticks[-1][1]
    folder = root / res["version"]
    data = pd.read_csv(folder / DATA_FILE)
    assert len(data) == 360 and (data["sumber"] == "panggilan_x.csv").sum() == 120
    meta = json.loads((folder / "train_meta.json").read_text("utf-8"))["data_latih"]
    assert meta["mode"] == "tambah" and meta["versi_induk"] == old
    assert (meta["jumlah_baris_lama"], meta["jumlah_baris_baru"], meta["jumlah_total"]) == (240, 120, 360)
    assert "panggilan_x.csv" in meta["file_sumber"]
    assert not list_versions(cfg).set_index("versi").loc[res["version"], "aktif"]
    table = res["perbandingan"]
    assert list(table["metrik"])[:2] == ["Jumlah data latih", "Jumlah klaster"]
    assert table.iloc[0, 1] == 240 and table.iloc[0, 2] == 360


def test_retrain_upload_ganti_hanya_data_baru_dan_ditolak_saat_monitoring(base):
    cfg, tmp = base
    path = flows_to_darknet(new_flows(12), tmp / "baru.csv")
    with pytest.raises(MonitoringActiveError):
        retrain_upload(cfg, [(path, "baru.csv")], "ganti", monitoring_active=True)
    res = retrain_upload(cfg, [(path, "baru.csv")], "ganti", params={"grid_enabled": False, "min_cluster_size": 7, "min_samples": 3})
    folder = version_dir(cfg, res["version"])
    data = pd.read_csv(folder / DATA_FILE)
    assert len(data) == 120 and set(data["sumber"]) == {"baru.csv"}
    meta = json.loads((folder / "train_meta.json").read_text("utf-8"))["data_latih"]
    assert meta["mode"] == "ganti" and meta["jumlah_baris_lama"] == 0 and meta["jumlah_baris_lama_dibuang"] == 240
    assert res["params"]["min_cluster_size"] == 7 and cfg["hdbscan"]["min_cluster_size"] == 3


def test_pemetaan_klaster_lama_ke_baru(base):
    cfg, tmp = base
    path = flows_to_darknet(new_flows(13), tmp / "baru.csv")
    res = retrain_upload(cfg, [(path, "baru.csv")], "ganti")
    old_profile = json.loads((version_dir(cfg) / "cluster_profile.json").read_text("utf-8"))
    mp = res["pemetaan"]
    olds = sorted((c for c in old_profile["clusters"] if c["id_klaster"] > 0), key=lambda c: c["centroid_asli"]["delay_ms"])
    assert len(mp) == len(olds) and (mp["jarak"] >= 0).all()
    assert set(mp["klaster_baru"]) <= {c["nama"] for c in res["profile"]}
    # data baru berdistribusi sama → klaster lama berdelay terkecil dipetakan ke klaster baru berdelay terkecil
    new = {c["nama"]: c["centroid_asli"]["delay_ms"] for c in res["profile"] if c["id_klaster"] > 0}
    assert mp.set_index("klaster_lama").loc[olds[0]["nama"], "klaster_baru"] == min(new, key=new.get)


def test_pemetaan_ke_model_yang_sama_adalah_identitas(base):
    cfg, _ = base
    data = pd.read_csv(version_dir(cfg) / DATA_FILE)
    result = train(data, cfg)
    mp = map_old_to_new({"clusters": result["profile"]}, result, cfg)
    assert len(mp) >= 1 and (mp["klaster_lama"] == mp["klaster_baru"]).all() and (mp["jarak"] < 1e-6).all()
