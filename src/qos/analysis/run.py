"""Satu putaran analisis (himpunan x varian fitur): sweep K-Means, grid HDBSCAN, perbandingan, stabilitas, centroid."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from qos.analysis import clustering as cl
from qos.analysis.data import PreparedData, protocol_name


@dataclass
class ConfigResult:
    """Satu konfigurasi algoritma beserta seluruh keluarannya (label sudah bernomor 1..n berurut ukuran; noise = -1)."""

    name: str
    description: str
    labels: np.ndarray
    metrics: dict[str, float]                       # K-Means: semua titik; HDBSCAN: titik non-noise
    metrics_assigned: dict[str, float] | None       # HDBSCAN versi (ii): noise di-assign ke centroid terdekat
    stability: tuple[float, float]
    fit_time: float
    predict_time_point: dict[str, float]            # waktu prediksi per titik (detik) per metode
    centroids: dict[int, np.ndarray]
    consistency: dict[str, float]
    own_distance: pd.DataFrame
    profile: pd.DataFrame
    model: Any = None


@dataclass
class RunResult:
    """Hasil lengkap satu putaran analisis."""

    run_name: str
    set_title: str
    variant: str
    prep: PreparedData
    sweep: pd.DataFrame
    elbow_k: int | None
    k_opt: int
    k_reason: str
    grid: pd.DataFrame
    hdb_best: pd.Series
    hdb_reason: str
    configs: dict[str, ConfigResult] = field(default_factory=dict)
    primary: bool = False


def cluster_profile(prep: PreparedData, labels: np.ndarray) -> pd.DataFrame:
    """Profil per klaster dalam satuan asli (delay/jitter ms, FAI %, paket), sebaran traffic type dan protokol."""
    df = prep.frame.assign(klaster=labels)
    rows = []
    for c in sorted(set(labels), key=lambda v: (v == cl.NOISE, v)):
        sub = df[df["klaster"] == c]
        rows.append(
            {
                "klaster": "Noise" if c == cl.NOISE else str(c),
                "n": len(sub),
                "delay_mean_ms": sub["delay_ms"].mean(), "delay_median_ms": sub["delay_ms"].median(),
                "jitter_mean_ms": sub["jitter_ms"].mean(), "jitter_median_ms": sub["jitter_ms"].median(),
                "fai_mean_pct": sub["fai_pct"].mean(), "fai_median_pct": sub["fai_pct"].median(),
                "paket_median": sub["total_packets"].median(),
                "jitter_nol_pct": (sub["iat_std_us"] == 0).mean() * 100,
                "bwd_nol_pct": (sub["bwd_packets_count"] == 0).mean() * 100,
                "traffic_type": _counts(sub["traffic_type"].fillna("?")) if "traffic_type" in sub else "-",
                "protokol": _counts(sub["protocol"].map(protocol_name)),
            }
        )
    return pd.DataFrame(rows)


def _counts(series: pd.Series) -> str:
    return ", ".join(f"{k}: {v}" for k, v in series.value_counts().items())


def _build_config(
    name: str, description: str, prep: PreparedData, raw_labels: np.ndarray, model: Any, fit_time: float,
    refit: Any, acfg: dict[str, Any], hdbscan_like: bool,
) -> ConfigResult:
    X = prep.X
    mapping = cl.renumber_by_size(raw_labels)
    labels = cl.apply_mapping(raw_labels, mapping)
    st = acfg["stability"]
    stability = cl.stability_ari(refit, X, labels, st["n_bootstrap"], st["fraction"], acfg["random_state"])
    cents = cl.centroids(X, labels)
    repeats = acfg["timing"]["repeats"]
    n = len(X)
    times: dict[str, float] = {}
    if hdbscan_like:
        times["approximate_predict"] = cl.median_time(lambda: hdbscan_approx(model, X), repeats)[0] / n
    else:
        times["kmeans.predict"] = cl.median_time(lambda: model.predict(X), repeats)[0] / n
    times["centroid_euclidean"] = cl.median_time(lambda: cl.assign_nearest(X, cents), repeats)[0] / n
    assigned = cl.assign_noise_to_nearest(X, labels) if hdbscan_like else None
    return ConfigResult(
        name=name, description=description, labels=labels,
        metrics=cl.compute_metrics(X, labels),
        metrics_assigned=cl.compute_metrics(X, assigned) if assigned is not None else None,
        stability=stability, fit_time=fit_time, predict_time_point=times, centroids=cents,
        consistency=cl.consistency(X, labels), own_distance=cl.distance_to_own_centroid(X, labels),
        profile=cluster_profile(prep, labels), model=model,
    )


def hdbscan_approx(model: Any, X: np.ndarray) -> np.ndarray:
    """Label prediksi titik baru dengan hdbscan.approximate_predict."""
    import hdbscan

    return hdbscan.approximate_predict(model, X)[0]


def analyze(prep: PreparedData, run_name: str, set_title: str, variant: str, acfg: dict[str, Any]) -> RunResult:
    """Jalankan B (penentuan jumlah klaster) dan C (perbandingan) untuk satu himpunan data."""
    X, rs = prep.X, acfg["random_state"]
    km, hd, sel = acfg["kmeans"], acfg["hdbscan"], acfg["selection"]
    ks = list(range(km["k_min"], min(km["k_max"], len(X) - 1) + 1))
    sweep = cl.kmeans_sweep(X, ks, km["n_init"], rs)
    elbow = cl.kneedle_elbow(sweep["k"].to_numpy(), sweep["inertia"].to_numpy())
    k_opt, k_reason = cl.select_kmeans_k(sweep, sel["kmeans_rank_metrics"])
    if elbow is not None:
        k_reason += f" Titik siku Elbow (kneedle) di k={elbow}" + (" (sejalan)." if elbow == k_opt else " (berbeda).")

    method = hd["cluster_selection_method"]
    grid = cl.hdbscan_grid(X, hd["min_cluster_size"], hd["min_samples"], method)
    best, hdb_reason = cl.select_hdbscan(grid, sel["hdbscan_min_clusters"], sel["hdbscan_max_noise_ratio"], sel["hdbscan_primary_metric"])
    mcs, ms = int(best["min_cluster_size"]), int(best["min_samples"])

    res = RunResult(run_name, set_title, variant, prep, sweep, elbow, k_opt, k_reason, grid, best, hdb_reason)
    repeats = acfg["timing"]["repeats"]

    fit_t, model = cl.median_time(lambda: cl.hdbscan_fit(X, mcs, ms, method), repeats)
    res.configs["hdbscan"] = _build_config(
        "HDBSCAN optimal", f"min_cluster_size={mcs}, min_samples={ms}, {method}", prep, model.labels_, model, fit_t,
        lambda Xs: cl.hdbscan_fit(Xs, mcs, ms, method).labels_, acfg, True,
    )
    n_hdb = int(best["n_clusters"])
    for key, k, title in (("kmeans_opt", k_opt, "K-Means k optimal"), ("kmeans_fair", n_hdb, "K-Means k = klaster HDBSCAN")):
        fit_t, kmodel = cl.median_time(lambda k=k: cl.kmeans_fit(X, k, km["n_init"], rs), repeats)
        res.configs[key] = _build_config(
            title, f"k={k}, n_init={km['n_init']}", prep, kmodel.labels_, kmodel, fit_t,
            lambda Xs, k=k: cl.kmeans_fit(Xs, k, km["n_init"], rs).labels_, acfg, False,
        )
    return res


def pca_coords(X: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    """Proyeksi PCA ke `n_components` dimensi."""
    return PCA(n_components=n_components, random_state=random_state).fit_transform(X)
