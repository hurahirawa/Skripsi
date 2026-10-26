"""Evaluasi klaster: Silhouette, Davies-Bouldin, Calinski-Harabasz, DBCV, dan Noise Ratio (PRD TR-3)."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from qos.analysis.clustering import dbcv

NOISE_ID = -1


def noise_ratio(labels: np.ndarray) -> float:
    """Noise Ratio (Pers. 2.6) = jumlah titik berlabel -1 / total titik (0–1)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("labels kosong")
    return float((labels == NOISE_ID).sum() / labels.size)


def n_clusters(labels: np.ndarray) -> int:
    """Jumlah klaster tanpa menghitung noise."""
    return int(len(set(np.asarray(labels).tolist()) - {NOISE_ID}))


def silhouette_label(score: float, bands: list[float]) -> str:
    """Interpretasi Silhouette; `bands` = [strong, reasonable, weak] (batas bawah eksklusif)."""
    if np.isnan(score):
        return "Tidak terdefinisi"
    strong, reasonable, weak = bands
    if score > strong:
        return "Strong Structure"
    if score > reasonable:
        return "Reasonable Structure"
    if score > weak:
        return "Weak Structure"
    return "No Substantial Structure"


def noise_status(ratio: float, ideal: list[float], too_strict: float) -> str:
    """Penilaian Noise Ratio: ideal, terlalu ketat, atau di luar keduanya."""
    if ratio > too_strict:
        return "Terlalu ketat"
    if ideal[0] <= ratio <= ideal[1]:
        return "Ideal"
    return "Di luar rentang ideal"


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray, sample_size: int, random_state: int, dbcv_max_points: int = 5000
) -> dict[str, Any]:
    """Metrik pada titik non-noise: silhouette (sampel `sample_size`), davies_bouldin, chi, dbcv; NaN jika klaster < 2.

    DBCV O(n²): di atas `dbcv_max_points` titik dihitung pada subsampel acak. Juga: noise_ratio, n_clusters.
    """
    X, labels = np.asarray(X), np.asarray(labels)
    keep = labels != NOISE_ID
    Xc, lc = X[keep], labels[keep]
    k = n_clusters(labels)
    nan = float("nan")
    out = {"silhouette": nan, "davies_bouldin": nan, "chi": nan, "dbcv": nan, "noise_ratio": noise_ratio(labels), "n_clusters": k}
    if k >= 2 and len(lc) > k:
        out["silhouette"] = float(silhouette_score(Xc, lc, sample_size=min(sample_size, len(lc)), random_state=random_state))
        out["davies_bouldin"] = float(davies_bouldin_score(Xc, lc))
        out["chi"] = float(calinski_harabasz_score(Xc, lc))
        if len(X) > dbcv_max_points:
            idx = np.random.default_rng(random_state).choice(len(X), size=dbcv_max_points, replace=False)
            out["dbcv"] = dbcv(X[idx], labels[idx])
        else:
            out["dbcv"] = dbcv(X, labels)
    return out
