"""Metrik klaster, K-Means/HDBSCAN, pemilihan parameter, stabilitas, centroid, dan penugasan jarak terdekat."""
from __future__ import annotations

import time
import warnings
from typing import Any, Callable

import hdbscan
import numpy as np
import pandas as pd
from hdbscan.validity import validity_index
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

NOISE = -1
METRIC_COLUMNS: tuple[str, ...] = ("n_clusters", "noise_ratio", "silhouette", "davies_bouldin", "calinski_harabasz", "dbcv")


def dbcv(X: np.ndarray, labels: np.ndarray) -> float:
    """DBCV (hdbscan.validity.validity_index); noise diabaikan; NaN jika < 2 klaster atau gagal dihitung."""
    labels = np.asarray(labels)
    if len(set(labels[labels != NOISE])) < 2:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(validity_index(np.ascontiguousarray(X, dtype=np.float64), labels))
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return float("nan")


def compute_metrics(X: np.ndarray, labels: np.ndarray, with_dbcv: bool = True) -> dict[str, float]:
    """Metrik pada titik non-noise: n_clusters, noise_ratio, Silhouette, Davies-Bouldin, Calinski-Harabasz, DBCV."""
    labels = np.asarray(labels)
    mask = labels != NOISE
    n_clusters = len(set(labels[mask]))
    out = {"n_clusters": float(n_clusters), "noise_ratio": float((~mask).mean()) if len(labels) else 0.0}
    nan = float("nan")
    if n_clusters >= 2 and mask.sum() > n_clusters:
        Xm, lm = X[mask], labels[mask]
        out["silhouette"] = float(silhouette_score(Xm, lm))
        out["davies_bouldin"] = float(davies_bouldin_score(Xm, lm))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xm, lm))
    else:
        out.update(silhouette=nan, davies_bouldin=nan, calinski_harabasz=nan)
    out["dbcv"] = dbcv(X, labels) if with_dbcv else nan
    return out


def kneedle_elbow(x: np.ndarray, y: np.ndarray) -> int | None:
    """Titik siku kurva menurun-cembung (kneedle): x dan y dinormalisasi; siku = selisih terbesar terhadap diagonal.

    Kembalikan nilai x di siku, atau None bila kurva tidak melengkung.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    xn = (x - x.min()) / np.ptp(x)
    yn = (y.max() - y) / np.ptp(y)  # menurun → naik, cekung
    diff = yn - xn
    i = int(np.argmax(diff))
    return int(x[i]) if diff[i] > 0 else None


def kmeans_fit(X: np.ndarray, k: int, n_init: int, random_state: int) -> KMeans:
    """Fit K-Means dengan n_init dan random_state tetap."""
    return KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X)


def kmeans_sweep(X: np.ndarray, ks: list[int], n_init: int, random_state: int) -> pd.DataFrame:
    """Inertia dan metrik K-Means untuk tiap k."""
    rows = []
    for k in ks:
        model = kmeans_fit(X, k, n_init, random_state)
        rows.append({"k": k, "inertia": float(model.inertia_), **compute_metrics(X, model.labels_)})
    return pd.DataFrame(rows)


def hdbscan_fit(X: np.ndarray, min_cluster_size: int, min_samples: int, method: str) -> hdbscan.HDBSCAN:
    """Fit HDBSCAN (paket hdbscan) dengan prediction_data untuk approximate_predict."""
    return hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size), min_samples=int(min_samples), cluster_selection_method=method,
        prediction_data=True,
    ).fit(X)


def hdbscan_grid(X: np.ndarray, mcs_list: list[int], ms_list: list[int], method: str) -> pd.DataFrame:
    """Grid min_cluster_size x min_samples; metrik dihitung pada titik non-noise."""
    rows = []
    for mcs in mcs_list:
        for ms in ms_list:
            labels = hdbscan_fit(X, mcs, ms, method).labels_
            rows.append({"min_cluster_size": mcs, "min_samples": ms, **compute_metrics(X, labels)})
    return pd.DataFrame(rows)


def select_kmeans_k(sweep: pd.DataFrame, rank_metrics: list[str]) -> tuple[int, str]:
    """k optimal = rata-rata peringkat terbaik pada `rank_metrics` (silhouette: besar lebih baik; DBI: kecil); seri → k kecil."""
    higher_better = {"silhouette": True, "davies_bouldin": False, "calinski_harabasz": True}
    ranks = pd.DataFrame(
        {m: sweep[m].rank(ascending=not higher_better[m], method="min") for m in rank_metrics}
    )
    score = ranks.mean(axis=1)
    best = sweep.assign(skor_peringkat=score).sort_values(["skor_peringkat", "k"]).iloc[0]
    k = int(best["k"])
    reason = (
        f"k={k} memiliki rata-rata peringkat terbaik ({best['skor_peringkat']:.2f}) pada {', '.join(rank_metrics)}; "
        f"Silhouette={best['silhouette']:.4f}, DBI={best['davies_bouldin']:.4f}, CHI={best['calinski_harabasz']:,.0f}."
    )
    return k, reason


def select_hdbscan(
    grid: pd.DataFrame, min_clusters: int, max_noise: float, metric: str, max_clusters: int | None = None
) -> tuple[pd.Series, str]:
    """Parameter HDBSCAN: DBCV (atau `metric`) tertinggi di antara kombinasi dengan klaster >= min_clusters dan noise <= max_noise.

    Seri → Silhouette lebih tinggi. Jika tak ada kombinasi memenuhi batas, batas noise dilonggarkan (dicatat di alasan).
    """
    cap = grid["n_clusters"] <= (np.inf if max_clusters is None else max_clusters)
    ok = grid[(grid["n_clusters"] >= min_clusters) & cap & (grid["noise_ratio"] <= max_noise) & grid[metric].notna()]
    note = ""
    if ok.empty:
        ok = grid[(grid["n_clusters"] >= min_clusters) & cap & grid[metric].notna()]
        note = f" Tidak ada kombinasi dengan noise <= {max_noise:.0%}; batas noise dilonggarkan."
    if ok.empty:
        raise ValueError("Tidak ada kombinasi HDBSCAN dengan >= 2 klaster dan metrik valid")
    best = ok.sort_values([metric, "silhouette"], ascending=False).iloc[0]
    reason = (
        f"min_cluster_size={int(best['min_cluster_size'])}, min_samples={int(best['min_samples'])}: {metric.upper()} tertinggi "
        f"({best[metric]:.4f}) di antara {len(ok)} kombinasi dengan {f'>= {min_clusters}' if max_clusters is None else f'{min_clusters}–{max_clusters}'} klaster dan noise <= {max_noise:.0%}; "
        f"{int(best['n_clusters'])} klaster, noise {best['noise_ratio']:.1%}, Silhouette {best['silhouette']:.4f}.{note}"
    )
    return best, reason


def centroids(X: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Centroid = rata-rata anggota per klaster di ruang fitur ter-skala (noise tidak punya centroid)."""
    labels = np.asarray(labels)
    return {int(c): X[labels == c].mean(axis=0) for c in sorted(set(labels[labels != NOISE]))}


def distance_matrix(X: np.ndarray, cents: np.ndarray) -> np.ndarray:
    """Jarak Euclidean tiap titik (baris) ke tiap centroid (kolom)."""
    return np.linalg.norm(X[:, None, :] - cents[None, :, :], axis=2)


def assign_nearest(X: np.ndarray, cent_map: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Penugasan jarak terdekat: kembalikan (label klaster terdekat, matriks jarak) dengan urutan kolom = urutan label."""
    ids = sorted(cent_map)
    dist = distance_matrix(np.atleast_2d(X), np.vstack([cent_map[i] for i in ids]))
    return np.asarray(ids)[dist.argmin(axis=1)], dist


def renumber_by_size(labels: np.ndarray) -> dict[int, int]:
    """Pemetaan label asli → nomor 1..n berurut ukuran menurun (seri: label asli kecil dulu); noise tetap -1."""
    labels = np.asarray(labels)
    counts = pd.Series(labels[labels != NOISE]).value_counts()
    order = sorted(counts.index, key=lambda c: (-counts[c], c))
    return {**{int(c): i + 1 for i, c in enumerate(order)}, NOISE: NOISE}


def apply_mapping(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Terapkan pemetaan label."""
    return np.array([mapping[int(v)] for v in labels])


def assign_noise_to_nearest(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Versi HDBSCAN (ii): titik noise di-assign ke centroid terdekat; titik lain tetap."""
    labels = np.asarray(labels).copy()
    noise = labels == NOISE
    if noise.any() and (~noise).any():
        labels[noise], _ = assign_nearest(X[noise], centroids(X, labels))
    return labels


def distance_to_own_centroid(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Median, p95, dan maks jarak titik ke centroid klasternya sendiri, per klaster."""
    cents, rows = centroids(X, labels), []
    for c, centre in cents.items():
        d = np.linalg.norm(X[np.asarray(labels) == c] - centre, axis=1)
        rows.append({"klaster": c, "n": len(d), "median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "maks": float(d.max())})
    return pd.DataFrame(rows)


def consistency(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Persen titik non-noise yang jarak-terdekatnya ke centroid sama dengan label asli algoritma."""
    labels = np.asarray(labels)
    mask = labels != NOISE
    nearest, _ = assign_nearest(X[mask], centroids(X, labels))
    return {"cocok_persen": float((nearest == labels[mask]).mean() * 100), "n_dinilai": int(mask.sum()), "n_noise": int((~mask).sum())}


def stability_ari(
    fit_labels: Callable[[np.ndarray], np.ndarray], X: np.ndarray, full_labels: np.ndarray, n: int, fraction: float, random_state: int
) -> tuple[float, float]:
    """Bootstrap subsampel (tanpa pengembalian): ARI label hasil fit subsampel vs label data penuh pada titik yang sama."""
    rng = np.random.default_rng(random_state)
    size = max(2, int(round(len(X) * fraction)))
    scores = []
    for _ in range(n):
        idx = rng.choice(len(X), size=size, replace=False)
        scores.append(adjusted_rand_score(np.asarray(full_labels)[idx], fit_labels(X[idx])))
    return float(np.mean(scores)), float(np.std(scores))


def median_time(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    """Median waktu (detik) menjalankan `fn` sebanyak `repeats` kali; kembalikan juga hasil terakhir."""
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result
