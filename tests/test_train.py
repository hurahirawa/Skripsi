"""Uji evaluasi, grid search, pelatihan, dan artefak (integrasi data sintetis)."""
import copy
import json

import hdbscan
import joblib
import numpy as np
import pandas as pd
import pytest

from qos.config import load_config
from qos.model.evaluate import evaluate_clustering, n_clusters, noise_status, silhouette_label
from qos.model.train import ARTIFACT_FILES, _jsonable, grid_search, pick_best, save_artifacts, train

BANDS = [0.70, 0.50, 0.25]


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config(use_user=False))
    c["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    c["hdbscan"]["grid_search"].update({"min_cluster_size": [5, 10], "min_samples": [2, 5]})
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
    assert ev["chi"] > 100


def test_evaluate_clustering_satu_klaster_nan():
    ev = evaluate_clustering(np.zeros((4, 2)), np.array([0, 0, 0, -1]), 100, 42)
    assert np.isnan(ev["silhouette"]) and np.isnan(ev["chi"]) and ev["n_clusters"] == 1


def test_pick_best_tiebreak_chi_dan_validitas():
    res = pd.DataFrame(
        {
            "min_cluster_size": [5, 10, 15, 20],
            "min_samples": [1, 1, 1, 1],
            "silhouette": [0.9, 0.8, 0.8, 0.95],
            "chi": [10.0, 5.0, 50.0, 99.0],
            "valid": [True, True, True, False],
        }
    )
    assert pick_best(res)["min_cluster_size"] == 5  # 0.95 tidak valid
    res.loc[0, "valid"] = False
    assert pick_best(res)["min_cluster_size"] == 15  # silhouette sama → CHI lebih tinggi
    res["valid"] = False
    assert pick_best(res) is None


def test_grid_search_kolom_dan_validitas(flows, cfg):
    from qos.features.preprocess import fit_scaler

    cols = cfg["features"]["cluster_columns"]
    X = fit_scaler(flows[cols]).transform(flows[cols])
    grid = grid_search(X, cfg)
    assert len(grid) == 4
    assert {"min_cluster_size", "min_samples", "silhouette", "chi", "noise_ratio", "n_clusters", "valid"} <= set(grid)
    max_noise = cfg["hdbscan"]["grid_search"]["max_noise_ratio"]
    assert (grid["valid"] == ((grid["n_clusters"] >= 2) & (grid["noise_ratio"] <= max_noise))).all()


def test_train_dan_artefak_bisa_dimuat(flows, cfg):
    result = train(flows, cfg, counts={"voip": len(flows)})
    assert result["evaluation"]["n_clusters"] >= 2
    out = save_artifacts(result, cfg)
    assert set(ARTIFACT_FILES) == {p.name for p in out.iterdir()}
    assert (out.parent / "current.txt").read_text("utf-8") == out.name

    model = joblib.load(out / "hdbscan_model.joblib")
    scaler = joblib.load(out / "scaler.joblib")
    pca = joblib.load(out / "pca.joblib")
    cols = cfg["features"]["cluster_columns"]
    X = scaler.transform(flows[cols].iloc[:5])
    ids, prob = hdbscan.approximate_predict(model, X)
    assert len(ids) == 5 and len(prob) == 5
    assert pca.transform(X).shape == (5, 2)

    doc = json.loads((out / "cluster_profile.json").read_text("utf-8"))
    assert doc["model_version"] == out.name
    assert doc["evaluation"]["silhouette_label"]
    assert set(doc["label_map"]) == {str(c["id_klaster"]) for c in doc["clusters"]}
    meta = json.loads((out / "train_meta.json").read_text("utf-8"))
    assert set(meta["median_imputasi"]) == set(cols)
    sample = pd.read_csv(out / "training_sample.csv")
    assert len(sample) <= cfg["evaluation"]["pca_sample_size"]
    assert {"pc1", "pc2", "cluster_id", "membership_prob"} <= set(sample)


def test_train_tanpa_grid_pakai_parameter_config(flows, cfg):
    result = train(flows, cfg, use_grid=False)
    assert result["grid"] is None
    assert result["params"]["min_cluster_size"] == cfg["hdbscan"]["min_cluster_size"]
    assert result["params"]["grid_search"] is False


def test_train_reproducible(flows, cfg):
    a = train(flows, cfg)["labels"]
    b = train(flows, cfg)["labels"]
    assert (a == b).all()


def test_save_artifacts_aktivasi(flows, cfg):
    import time

    result = train(flows, cfg)
    current = None
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
    out = _jsonable({"a": float("nan"), "b": np.float64(1.5), "c": [np.int64(2), float("inf")]})
    assert out == {"a": None, "b": 1.5, "c": [2, None]}
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
    model = joblib.load(out / "hdbscan_model.joblib")
    # alur live: median imputasi → turunan → transformasi dari meta → scaler → approximate_predict
    raw = flows.iloc[:20].copy()
    live = derive_metrics(raw.fillna(meta["median_imputasi"]), "iat_mean_us", "iat_std_us", True)
    X = scaler.transform(build_features(live, meta["cluster_columns"], meta["log_transform"]))
    ids, prob = hdbscan.approximate_predict(model, X)
    assert len(ids) == 20 and ((prob >= 0) & (prob <= 1)).all()
    frame = result["frame"]
    X_train = scaler.transform(build_features(frame, meta["cluster_columns"], meta["log_transform"]))
    assert X_train.shape[0] == len(result["labels"])


def test_tie_break_kandidat_identik():
    from qos.model.train import TIE_BREAK_RULE, rank_candidates, tied_with_best

    res = pd.DataFrame(
        {
            "min_cluster_size": [25, 10, 5, 15],
            "min_samples": [20, 20, 20, 5],
            "silhouette": [0.8655, 0.8655, 0.8655, 0.80],
            "chi": [1523.61, 1523.61, 1523.61, 9999.0],
            "valid": [True, True, True, True],
        }
    )
    ranked = rank_candidates(res)
    assert ranked.iloc[0][["min_cluster_size", "min_samples"]].tolist() == [5, 20]  # seri → mcs terkecil
    assert tied_with_best(res) == [(5, 20), (10, 20), (25, 20)]
    assert "min_cluster_size terkecil" in TIE_BREAK_RULE


def test_train_meta_mencatat_aturan_tie_break(flows, cfg):
    result = train(flows, cfg)
    out = save_artifacts(result, cfg, activate=False)
    meta = json.loads((out / "train_meta.json").read_text("utf-8"))
    assert "min_cluster_size terkecil" in meta["parameters"]["tie_break_rule"]
    assert isinstance(meta["parameters"]["kandidat_seri_penuh"], list)
