"""Uji evaluasi, aturan seleksi, pelatihan klasterisasi murni, dan artefak (integrasi data sintetis)."""
import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from qos.config import load_config
from qos.model.assignment import assign
from qos.model.evaluate import evaluate_clustering, n_clusters, noise_status, silhouette_label
from qos.model.train import ARTIFACT_FILES, SCHEMA, TrainError, _jsonable, grid_search, save_artifacts, select_final, train

BANDS = [0.70, 0.50, 0.25]
SEL = {"max_noise_ratio": 0.20, "min_clusters": 2, "min_ari_mean": 0.90}


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config(use_user=False))
    c["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    c["hdbscan"]["grid_search"].update({"min_cluster_size": [5, 10], "min_samples": [2, 5]})
    c["hdbscan"]["stability"]["n_bootstrap"] = 8
    c["evaluation"]["pca_sample_size"] = 100
    # data sintetis tidak punya fai_pct mentah; uji dasar memakai 4 kolom mentah tanpa log
    c["features"]["cluster_columns"] = ["iat_mean_us", "iat_std_us", "fwd_packets_count", "bwd_packets_count"]
    c["features"]["log_transform"] = []
    return c


@pytest.fixture(scope="module")
def flows():
    """Tiga kelompok flow sintetis yang terpisah jelas."""
    rng = np.random.default_rng(0)
    parts = []
    for mean_us, std_us, pkts in ((1000.0, 500.0, 100), (50_000.0, 20_000.0, 300), (400_000.0, 90_000.0, 800)):
        n = 80
        fwd = rng.normal(pkts, pkts * 0.02, n).round()
        parts.append(
            pd.DataFrame(
                {
                    "iat_mean_us": rng.normal(mean_us, mean_us * 0.03, n),
                    "iat_std_us": rng.normal(std_us, std_us * 0.03, n),
                    "fwd_packets_count": fwd,
                    "bwd_packets_count": (fwd * rng.normal(0.9, 0.01, n)).round(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_silhouette_label_batas():
    assert silhouette_label(0.71, BANDS) == "Strong Structure"
    assert silhouette_label(0.70, BANDS) == "Reasonable Structure"
    assert silhouette_label(0.51, BANDS) == "Reasonable Structure"
    assert silhouette_label(0.50, BANDS) == "Weak Structure"
    assert silhouette_label(0.26, BANDS) == "Weak Structure"
    assert silhouette_label(0.25, BANDS) == "No Substantial Structure"
    assert silhouette_label(float("nan"), BANDS) == "Tidak terdefinisi"


def test_noise_status():
    assert noise_status(0.05, [0.01, 0.10], 0.20) == "Ideal"
    assert noise_status(0.15, [0.01, 0.10], 0.20) == "Di luar rentang ideal"
    assert noise_status(0.21, [0.01, 0.10], 0.20) == "Terlalu ketat"


def test_evaluate_clustering_tanpa_noise():
    X = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10], [100, -100]], dtype=float)
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    ev = evaluate_clustering(X, labels, sample_size=10000, random_state=42)
    assert ev["n_clusters"] == 2 and n_clusters(labels) == 2
    assert ev["noise_ratio"] == pytest.approx(1 / 7)
    assert ev["silhouette"] > 0.9  # titik noise tidak ikut dihitung
    assert ev["chi"] > 100 and ev["davies_bouldin"] < 0.2


def test_evaluate_clustering_satu_klaster_nan():
    ev = evaluate_clustering(np.zeros((4, 2)), np.array([0, 0, 0, -1]), 100, 42)
    assert all(np.isnan(ev[k]) for k in ("silhouette", "chi", "davies_bouldin", "dbcv")) and ev["n_clusters"] == 1


def test_evaluate_clustering_dbcv_subsampel_untuk_data_besar():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.3, (150, 2)), rng.normal(8, 0.3, (150, 2))])
    labels = np.repeat([0, 1], 150)
    full = evaluate_clustering(X, labels, 10000, 42, dbcv_max_points=10000)["dbcv"]
    sub = evaluate_clustering(X, labels, 10000, 42, dbcv_max_points=100)["dbcv"]
    assert full > 0.5 and sub > 0.5  # subsampel tetap menilai dua klaster terpisah


def _grid(**cols):
    base = {"min_cluster_size": [3, 5, 8, 10], "min_samples": [1, 1, 1, 1], "n_clusters": [5, 5, 1, 5],
            "noise_ratio": [0.1, 0.1, 0.1, 0.3], "silhouette": [0.8, 0.9, 0.9, 0.95],
            "dbcv": [0.7, 0.7, 0.99, 0.99], "ari_mean": [0.95, 0.95, 0.99, 0.99]}
    base.update(cols)
    return pd.DataFrame(base)


def test_select_final_filter_dan_tiebreak_silhouette():
    best, ok = select_final(_grid(), SEL)
    # baris 3: 1 klaster; baris 4: noise 30% -> gugur. Sisanya seri DBCV -> Silhouette lebih tinggi (mcs=5)
    assert len(ok) == 2 and best["min_cluster_size"] == 5
    best, _ = select_final(_grid(ari_mean=[0.5, 0.95, 0.99, 0.99]), SEL)
    assert best["min_cluster_size"] == 5  # ARI < 0,90 gugur
    assert select_final(_grid(ari_mean=[0.5] * 4), SEL)[0] is None


def test_select_final_dbcv_tertinggi_menang():
    best, _ = select_final(_grid(n_clusters=[5, 5, 5, 5], noise_ratio=[0.1] * 4, dbcv=[0.7, 0.6, 0.8, 0.5]), SEL)
    assert best["min_cluster_size"] == 8


def test_select_final_seri_penuh_mcs_ms_terkecil():
    grid = _grid(min_cluster_size=[10, 5, 8, 3], silhouette=[0.8] * 4, dbcv=[0.7] * 4, n_clusters=[5] * 4,
                 noise_ratio=[0.1] * 4, ari_mean=[0.95] * 4)
    assert select_final(grid, SEL)[0]["min_cluster_size"] == 3


def test_grid_search_kolom_dan_validitas(flows, cfg):
    from qos.features.preprocess import fit_scaler

    cols = cfg["features"]["cluster_columns"]
    X = fit_scaler(flows[cols]).transform(flows[cols])
    grid = grid_search(X, cfg)
    assert len(grid) == 4
    assert {"min_cluster_size", "min_samples", "silhouette", "davies_bouldin", "chi", "dbcv", "noise_ratio", "n_clusters",
            "ari_mean", "ari_sd", "valid"} <= set(grid)
    _, ok = select_final(grid, cfg["hdbscan"]["selection"])
    assert set(grid.index[grid["valid"]]) == set(ok.index)
    assert grid["ari_mean"].between(-1, 1).all()


def test_train_dan_artefak_bisa_dimuat(flows, cfg):
    result = train(flows, cfg, counts={"voip": len(flows)})
    assert result["evaluation"]["n_clusters"] >= 2
    out = save_artifacts(result, cfg)
    assert set(ARTIFACT_FILES) == {p.name for p in out.iterdir()}
    assert (out.parent / "current.txt").read_text("utf-8") == out.name

    scaler = joblib.load(out / "scaler.joblib")
    pca = joblib.load(out / "pca.joblib")
    assert joblib.load(out / "kmeans_n.joblib").n_clusters == result["evaluation"]["n_clusters"]
    assert joblib.load(out / "kmeans_opt.joblib").n_clusters == result["params"]["k_kmeans_optimal"]
    cols = cfg["features"]["cluster_columns"]
    X = scaler.transform(flows[cols].iloc[:5])
    assert pca.transform(X).shape == (5, 2)

    doc = json.loads((out / "cluster_profile.json").read_text("utf-8"))
    assert doc["schema"] == SCHEMA and doc["model_version"] == out.name
    assert doc["evaluation"]["silhouette_label"]
    assert {"hdbscan", "hdbscan_assigned", "kmeans_n", "kmeans_opt"} == set(doc["comparison"])
    for key in ("silhouette", "davies_bouldin", "chi", "dbcv", "noise_ratio", "ari_mean", "ari_sd"):
        assert key in doc["comparison"]["kmeans_n"] and key in doc["comparison"]["hdbscan"]
    assert doc["comparison"]["hdbscan_assigned"]["noise_ratio"] == 0
    assert "label_map" not in doc and all("label" not in c for c in doc["clusters"])
    ids = [c["id_klaster"] for c in doc["clusters"] if c["id_klaster"] != -1]
    assert ids == list(range(1, len(ids) + 1))  # nomor klaster 1..k
    for c in doc["clusters"]:
        if c["id_klaster"] != -1:
            assert c["batas_jarak"] > 0 and set(c["centroid_asli"]) >= {"delay_ms", "jitter_ms"}
            assert c["deskripsi"]["ringkas"] and set(c["boxplot"]) == {"delay_ms", "jitter_ms", "fai_pct"}
        else:
            assert c["nama"] == "Noise" and c["batas_jarak"] is None
    meta = json.loads((out / "train_meta.json").read_text("utf-8"))
    assert set(meta["median_imputasi"]) == set(cols)
    sample = pd.read_csv(out / "training_sample.csv")
    assert len(sample) <= cfg["evaluation"]["pca_sample_size"]
    assert {"pc1", "pc2", "cluster_id", "jarak_terdekat", "di_luar_klaster", "klaster_kmeans_n"} <= set(sample)
    assert "membership_prob" not in sample
    assert {"ari_mean", "dbcv", "valid"} <= set(pd.read_csv(out / "grid_results.csv"))


def test_assignment_json_konsisten_dengan_label_pelatihan(flows, cfg):
    result = train(flows, cfg)
    spec = result["assignment"]
    cents = {int(k): np.array(v) for k, v in spec["hdbscan"]["centroid"].items()}
    limits = {int(k): v for k, v in spec["hdbscan"]["batas"].items()}
    assert spec["aturan"] == cfg["assignment"]["limit_rule"] and spec["batas_global"] > 0
    cols = cfg["features"]["cluster_columns"]
    X = result["scaler"].transform(result["frame"][cols])
    labels = result["labels"]
    a = assign(X, cents, limits)
    inside = labels != -1
    assert (a.nearest[inside] == labels[inside]).mean() > 0.99  # anggota ditugaskan kembali ke klasternya
    for c, centre in cents.items():  # centroid = rata-rata anggota
        assert centre == pytest.approx(X[labels == c].mean(axis=0))
    assert set(spec["kmeans_n"]["centroid"]) == {str(i) for i in range(1, spec["kmeans_n"]["k"] + 1)}


def test_train_tanpa_grid_pakai_parameter_config(flows, cfg):
    result = train(flows, cfg, use_grid=False)
    assert result["grid"] is None
    assert result["params"]["min_cluster_size"] == cfg["hdbscan"]["min_cluster_size"]
    assert result["params"]["grid_search"] is False


def test_train_semua_noise_ditolak(flows, cfg):
    cfg["hdbscan"].update(min_cluster_size=200, min_samples=100)
    with pytest.raises(TrainError):
        train(flows, cfg, use_grid=False)


def test_train_reproducible(flows, cfg):
    a = train(flows, cfg)["labels"]
    b = train(flows, cfg)["labels"]
    assert (a == b).all()


def test_save_artifacts_aktivasi(flows, cfg):
    import time

    result = train(flows, cfg)
    first = save_artifacts(result, cfg)  # belum ada versi aktif → otomatis aktif
    current = first.parent / "current.txt"
    time.sleep(1.1)
    second = save_artifacts(result, cfg)  # sudah ada → tidak menimpa
    assert second.name != first.name
    assert current.read_text("utf-8") == first.name
    time.sleep(1.1)
    third = save_artifacts(result, cfg, activate=True)
    assert current.read_text("utf-8") == third.name


def test_jsonable_nan_jadi_null():
    out = _jsonable({"a": float("nan"), "b": np.float64(1.5), "c": [np.int64(2), float("inf")], "d": np.array([1.0, np.nan])})
    assert out == {"a": None, "b": 1.5, "c": [2, None], "d": [1.0, None]}
    json.dumps(out)


def test_train_fai_dan_log_transform_konsisten_dengan_artefak(flows, cfg):
    from qos.features.extract import derive_metrics
    from qos.features.preprocess import build_features

    cfg["features"]["cluster_columns"] = ["iat_mean_us", "iat_std_us", "fai_pct"]
    cfg["features"]["log_transform"] = ["iat_mean_us", "iat_std_us"]
    result = train(flows, cfg)
    out = save_artifacts(result, cfg, activate=False)
    assert not (out.parent / "current.txt").exists()  # activate=False tidak membuat/menimpa current.txt

    meta = json.loads((out / "train_meta.json").read_text("utf-8"))
    assert meta["log_transform"] == cfg["features"]["log_transform"]
    scaler = joblib.load(out / "scaler.joblib")
    # alur live: median imputasi → turunan → transformasi dari meta → scaler → centroid Euclidean
    raw = flows.iloc[:20].copy()
    live = derive_metrics(raw.fillna(meta["median_imputasi"]), "iat_mean_us", "iat_std_us", True)
    X = scaler.transform(build_features(live, meta["cluster_columns"], meta["log_transform"]))
    a = assign(X, {int(k): np.array(v) for k, v in result["assignment"]["hdbscan"]["centroid"].items()})
    assert len(a.nearest) == 20 and (a.nearest_dist >= 0).all()
    frame = result["frame"]
    X_train = scaler.transform(build_features(frame, meta["cluster_columns"], meta["log_transform"]))
    assert X_train.shape[0] == len(result["labels"])


def test_train_meta_mencatat_aturan_seleksi(flows, cfg):
    result = train(flows, cfg)
    out = save_artifacts(result, cfg, activate=False)
    meta = json.loads((out / "train_meta.json").read_text("utf-8"))
    assert meta["aturan_seleksi"] == cfg["hdbscan"]["selection"]
    assert "DBCV tertinggi" in meta["parameters"]["aturan_seleksi"]
    assert meta["penugasan"]["limit_rule"] == "max_cluster_global"
