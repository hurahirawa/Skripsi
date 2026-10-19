"""Evaluasi klaster: Silhouette, Calinski-Harabasz, dan Noise Ratio (PRD TR-3)."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from qos.profiling.labeling import NOISE_ID


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
    X: np.ndarray, labels: np.ndarray, sample_size: int, random_state: int
) -> dict[str, Any]:
    """Silhouette (sample_size) dan CHI tanpa titik noise; NaN jika klaster < 2.

    Kembalian: silhouette, chi, noise_ratio, n_clusters.
    """
    X, labels = np.asarray(X), np.asarray(labels)
    keep = labels != NOISE_ID
    Xc, lc = X[keep], labels[keep]
    k = n_clusters(labels)
    sil = chi = float("nan")
    if k >= 2 and len(lc) > k:
        sil = float(silhouette_score(Xc, lc, sample_size=min(sample_size, len(lc)), random_state=random_state))
        chi = float(calinski_harabasz_score(Xc, lc))
    return {"silhouette": sil, "chi": chi, "noise_ratio": noise_ratio(labels), "n_clusters": k}
