"""Uji fungsi analisis klasterisasi murni: metrik, siku, pemilihan, centroid, penugasan jarak terdekat, cleaning."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from qos.analysis import clustering as cl
from qos.analysis.data import add_derived, clean_set, iqr_outliers, transform_features
from qos.config import PROJECT_ROOT

CLEAN_CFG = {"drop_duplicates": True, "drop_negative": True, "nan_strategy": "median", "iqr_factor": 1.5}


@pytest.fixture
def blobs():
    """Tiga gumpalan terpisah jelas di 2D + label benar."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([c + rng.normal(0, 0.3, (40, 2)) for c in centers])
    return X, np.repeat([0, 1, 2], 40)


def test_compute_metrics_gumpalan_terpisah(blobs):
    X, y = blobs
    m = cl.compute_metrics(X, y)
    assert m["n_clusters"] == 3 and m["noise_ratio"] == 0
    assert m["silhouette"] > 0.9 and m["davies_bouldin"] < 0.2 and m["calinski_harabasz"] > 1000
    assert m["dbcv"] > 0.5


def test_compute_metrics_noise_diabaikan_dan_rasio(blobs):
    X, y = blobs
    y2 = y.copy()
    y2[:6] = cl.NOISE
    m = cl.compute_metrics(X, y2)
    assert m["noise_ratio"] == pytest.approx(6 / 120) and m["n_clusters"] == 3
    ref = cl.compute_metrics(X[6:], y[6:])
    assert m["silhouette"] == pytest.approx(ref["silhouette"])  # metrik hanya pada titik non-noise


def test_compute_metrics_kurang_dari_dua_klaster_nan(blobs):
    X, _ = blobs
    m = cl.compute_metrics(X, np.zeros(len(X), dtype=int))
    assert np.isnan(m["silhouette"]) and np.isnan(m["dbcv"]) and m["n_clusters"] == 1


def test_kneedle_menemukan_siku():
    k = np.arange(2, 11)
    inertia = np.array([1000, 500, 200, 150, 130, 120, 112, 106, 100], dtype=float)  # siku di k=4
    assert cl.kneedle_elbow(k, inertia) == 4
    assert cl.kneedle_elbow(k, np.linspace(100, 10, 9)) is None  # garis lurus: tidak ada siku
    assert cl.kneedle_elbow(np.array([2, 3]), np.array([1.0, 0.5])) is None


def test_select_kmeans_k_peringkat_dan_seri_k_kecil():
    sweep = pd.DataFrame(
        {"k": [2, 3, 4], "silhouette": [0.5, 0.8, 0.8], "davies_bouldin": [1.0, 0.4, 0.4], "calinski_harabasz": [1, 2, 3]}
    )
    k, reason = cl.select_kmeans_k(sweep, ["silhouette", "davies_bouldin"])
    assert k == 3 and "k=3" in reason  # seri antara 3 dan 4 → k lebih kecil


def test_select_hdbscan_batas_dan_metrik():
    grid = pd.DataFrame(
        {
            "min_cluster_size": [5, 10, 25, 50], "min_samples": [1, 1, 1, 1], "n_clusters": [4, 1, 3, 3],
            "noise_ratio": [0.1, 0.1, 0.7, 0.2], "silhouette": [0.5, 0.9, 0.9, 0.4], "dbcv": [0.6, 0.99, 0.95, 0.3],
        }
    )
    best, reason = cl.select_hdbscan(grid, 2, 0.5, "dbcv")
    assert int(best["min_cluster_size"]) == 5 and "DBCV" in reason  # 10: hanya 1 klaster; 25: noise 70%
    relaxed, note = cl.select_hdbscan(grid[grid["noise_ratio"] > 0.5], 2, 0.5, "dbcv")
    assert int(relaxed["min_cluster_size"]) == 25 and "dilonggarkan" in note
    with pytest.raises(ValueError):
        cl.select_hdbscan(grid[grid["n_clusters"] < 2], 2, 0.5, "dbcv")


def test_centroid_rata_rata_anggota_tanpa_noise(blobs):
    X, y = blobs
    y2 = y.copy()
    y2[0] = cl.NOISE
    c = cl.centroids(X, y2)
    assert sorted(c) == [0, 1, 2]
    assert c[0] == pytest.approx(X[1:40].mean(axis=0))  # titik noise tidak ikut dihitung


def test_assign_nearest_euclidean():
    cents = {1: np.array([0.0, 0.0]), 2: np.array([10.0, 0.0])}
    labels, dist = cl.assign_nearest(np.array([[1.0, 0.0], [8.0, 0.0], [5.1, 0.0]]), cents)
    assert labels.tolist() == [1, 2, 2]
    assert dist[0].tolist() == pytest.approx([1.0, 9.0]) and dist[2, 0] == pytest.approx(5.1)
    one, _ = cl.assign_nearest(np.array([0.5, 0.0]), cents)  # satu titik (vektor 1D)
    assert one.tolist() == [1]


def test_konsistensi_dan_kmeans_cocok_penuh(blobs):
    X, y = blobs
    assert cl.consistency(X, y)["cocok_persen"] == 100.0
    y2 = y.copy()
    y2[:5] = cl.NOISE
    cons = cl.consistency(X, y2)
    assert cons["n_noise"] == 5 and cons["n_dinilai"] == 115


def test_renumber_by_size_noise_tetap():
    labels = np.array([7, 7, 7, 3, 3, -1, 9])
    mapping = cl.renumber_by_size(labels)
    assert mapping == {7: 1, 3: 2, 9: 3, -1: -1}
    assert cl.apply_mapping(labels, mapping).tolist() == [1, 1, 1, 2, 2, -1, 3]


def test_noise_di_assign_ke_centroid_terdekat():
    X = np.array([[0.0, 0], [0.1, 0], [10, 0], [10.1, 0], [9.0, 0]])
    out = cl.assign_noise_to_nearest(X, np.array([1, 1, 2, 2, cl.NOISE]))
    assert out.tolist() == [1, 1, 2, 2, 2]


def test_jarak_ke_centroid_sendiri(blobs):
    X, y = blobs
    d = cl.distance_to_own_centroid(X, y)
    assert list(d["klaster"]) == [0, 1, 2] and (d["median"] <= d["p95"]).all() and (d["p95"] <= d["maks"]).all()
    assert (d["maks"] < 2).all()


def test_stabilitas_ari_tinggi_pada_gumpalan_terpisah(blobs):
    X, y = blobs
    mean, std = cl.stability_ari(lambda Xs: cl.kmeans_fit(Xs, 3, 5, 42).labels_, X, y, 5, 0.8, 42)
    assert mean == pytest.approx(1.0) and std == pytest.approx(0.0)
    again = cl.stability_ari(lambda Xs: cl.kmeans_fit(Xs, 3, 5, 42).labels_, X, y, 5, 0.8, 42)
    assert again == (mean, std)  # random_state tetap → reprodusibel


def test_kmeans_dan_hdbscan_menemukan_tiga_gumpalan(blobs):
    X, _ = blobs
    assert cl.compute_metrics(X, cl.kmeans_fit(X, 3, 5, 42).labels_)["n_clusters"] == 3
    assert cl.hdbscan_fit(X, 10, 5, "eom").labels_.max() + 1 == 3
    sweep = cl.kmeans_sweep(X, [2, 3, 4], 5, 42)
    assert list(sweep["k"]) == [2, 3, 4] and sweep["inertia"].is_monotonic_decreasing


def _flows():
    return pd.DataFrame(
        {
            "iat_mean_us": [1000.0, 1000.0, 2000.0, np.inf, np.nan, 3000.0],
            "iat_std_us": [10.0, 10.0, 20.0, 30.0, 40.0, -5.0],
            "fwd_packets_count": [10, 10, 20, 30, 40, 50],
            "bwd_packets_count": [9, 9, 25, 0, 40, 50],
            "protocol": [17, 17, 17, 6, 17, 17],
            "traffic_type": ["NonVPN", "NonVPN", "Tor", "VPN", "Tor", "VPN"],
        }
    )


def test_clean_set_tahap_dan_laporan():
    cfg = {"exclude_traffic_types": [], "min_fwd": 1, "min_bwd": None}
    df, rep = clean_set(_flows(), cfg, CLEAN_CFG, True)
    stages = dict(rep["tahap"])
    assert stages["Buang duplikat"] == 5 and rep["duplikat"] == 1
    assert rep["negatif_per_kolom"]["iat_std_us"] == 1 and rep["baris_negatif_dibuang"] == 1 and len(df) == 4
    assert rep["inf_per_kolom"]["iat_mean_us"] == 1 and rep["nan_per_kolom"]["iat_mean_us"] == 1
    assert rep["baris_diimputasi"] == 2 and not df[["iat_mean_us", "iat_std_us"]].isna().any().any()
    assert (df["fai_pct"] >= 0).all()  # FAI negatif (25 > 20) di-clip ke 0
    no_tor, _ = clean_set(_flows(), {**cfg, "exclude_traffic_types": ["Tor"]}, CLEAN_CFG, True)
    assert "Tor" not in set(no_tor["traffic_type"])


def test_fitur_log1p_fai_tidak_di_log():
    df = add_derived(_flows().iloc[:2], True)
    out = transform_features(df, ["iat_mean_us", "fai_pct"], ["iat_mean_us"])
    assert out["iat_mean_us"].iloc[0] == pytest.approx(np.log1p(1000.0))
    assert out["fai_pct"].iloc[0] == pytest.approx(10.0)  # (10 - 9) / 10 * 100
    with pytest.raises(ValueError):
        transform_features(df, ["fai_pct"], ["iat_mean_us"])


def test_iqr_outliers():
    s = pd.Series([1.0] * 20 + [100.0, np.nan])
    r = iqr_outliers(s, 1.5)
    assert r["jumlah"] == 1 and r["persen"] == pytest.approx(100 / 21)


def test_config_analisis_lengkap_dan_default_tidak_berubah():
    acfg = yaml.safe_load(Path(PROJECT_ROOT / "config" / "analysis_config.yaml").read_text("utf-8"))["analysis"]
    assert acfg["random_state"] == 42 and acfg["stability"]["n_bootstrap"] == 50
    assert sum(1 for s in acfg["sets"] if s.get("primary")) == 1
    assert {v for s in acfg["sets"] for v in s["feature_variants"]} <= set(acfg["features"])
    default = yaml.safe_load(Path(PROJECT_ROOT / "config" / "default_config.yaml").read_text("utf-8"))
    assert "analysis" not in default  # blok analisis terpisah dari config default


# ---- Bagian F: ambang ITU, batas jarak, kandidat jumlah klaster ----
from qos.analysis.final import add_itu_column, centroids_original_units, distance_limits, far_from_all, itu_position  # noqa: E402

THRESH = {"delay_good_ms": 150.0, "delay_max_ms": 400.0, "jitter_max_ms": 50.0, "loss_good_pct": 1.0, "loss_max_pct": 3.0}


def test_select_hdbscan_max_clusters_membatasi_kandidat():
    grid = pd.DataFrame({
        "min_cluster_size": [3, 5, 8], "min_samples": [1, 1, 1], "n_clusters": [8.0, 5.0, 3.0],
        "noise_ratio": [0.1, 0.1, 0.1], "silhouette": [0.8, 0.7, 0.6], "dbcv": [0.9, 0.5, 0.4],
    })
    assert cl.select_hdbscan(grid, 2, 0.5, "dbcv")[0]["n_clusters"] == 8
    best, reason = cl.select_hdbscan(grid, 3, 0.5, "dbcv", max_clusters=5)
    assert best["n_clusters"] == 5 and "3–5" in reason
    assert cl.select_hdbscan(grid, 2, 0.5, "silhouette")[0]["silhouette"] == 0.8


@pytest.mark.parametrize(
    "delay,jitter,fai,expected",
    [
        (10, 6, 0.0, "delay ≤150 ms; jitter ≤50 ms; FAI ≤1 %"),
        (150, 50, 1.0, "delay ≤150 ms; jitter ≤50 ms; FAI ≤1 %"),   # batas inklusif
        (200, 51, 2.0, "delay 150–400 ms; jitter >50 ms; FAI 1–3 %"),
        (401, 10, 3.5, "delay >400 ms; jitter ≤50 ms; FAI >3 %"),
    ],
)
def test_itu_position(delay, jitter, fai, expected):
    assert itu_position(delay, jitter, fai, THRESH) == expected


def test_add_itu_column_memakai_median():
    prof = pd.DataFrame({"delay_median_ms": [10.0], "jitter_median_ms": [60.0], "fai_median_pct": [0.0]})
    assert add_itu_column(prof, THRESH)["posisi ITU (median)"].iloc[0] == "delay ≤150 ms; jitter >50 ms; FAI ≤1 %"


def test_distance_limits_dan_far_from_all(blobs):
    X, y = blobs
    limits = distance_limits(X, y, 95)
    cents = cl.centroids(X, y)
    own = np.linalg.norm(X[y == 0] - cents[0], axis=1)
    assert limits[0] == pytest.approx(np.percentile(own, 95))
    probe = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])  # dua di klaster, satu jauh dari semua
    assert far_from_all(probe, cents, limits).tolist() == [False, True, False]


def test_centroids_original_units_invers_transform():
    from sklearn.preprocessing import StandardScaler
    raw = pd.DataFrame({"iat_mean_us": [1000.0, 9000.0, 99000.0], "iat_std_us": [0.0, 2000.0, 8000.0], "fai_pct": [0.0, 5.0, 10.0]})
    feats = transform_features(raw, list(raw.columns), ["iat_mean_us", "iat_std_us"])
    scaler = StandardScaler().fit(feats)
    out = centroids_original_units({1: scaler.transform(feats).mean(axis=0)}, scaler, list(raw.columns), ["iat_mean_us", "iat_std_us"])
    assert out["delay_ms"].iloc[0] == pytest.approx(np.expm1(np.log1p(raw["iat_mean_us"]).mean()) / 1000)  # µs → ms
    assert out["fai_pct"].iloc[0] == pytest.approx(5.0)


from qos.analysis.final import distance_rules, recommend_rule, select_final_hdbscan  # noqa: E402


def test_select_final_hdbscan_filter_ari_dan_seri():
    grid = pd.DataFrame({
        "min_cluster_size": [3, 5, 8, 10], "min_samples": [1, 1, 1, 1], "n_clusters": [5.0, 5.0, 1.0, 5.0],
        "noise_ratio": [0.1, 0.1, 0.1, 0.1], "silhouette": [0.8, 0.9, 0.9, 0.7], "dbcv": [0.7, 0.7, 0.99, 0.95],
        "ari_mean": [0.95, 0.95, 0.99, 0.5],
    })
    sel = {"max_noise_ratio": 0.2, "min_clusters": 2, "min_ari_mean": 0.9}
    best, valid, _ = select_final_hdbscan(grid, sel)
    assert len(valid) == 2 and best["min_cluster_size"] == 5   # klaster=1 dan ARI rendah gugur; seri DBCV → Silhouette
    with pytest.raises(ValueError):
        select_final_hdbscan(grid, {**sel, "min_ari_mean": 0.999})


def test_distance_rules_dan_rekomendasi():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.2, (100, 2)), rng.normal(10, 2.0, (100, 2)), [[5.0, 5.0]]])
    labels = np.r_[np.zeros(100), np.ones(100), -1].astype(int)
    r = distance_rules(X, labels, 95)
    assert len(r) == 3 and (r["noise total"] == 1).all() and (r["noise ditandai"] == 1).all()
    c = r.iloc[2]   # aturan (c) = max(per klaster, global) tidak pernah lebih longgar menandai daripada (a)
    assert c["non-noise ditandai"] <= r.iloc[0]["non-noise ditandai"]
    assert recommend_rule(r) in (0, 1, 2)
