"""Penugasan Euclidean ke centroid dan batas "di luar klaster" (aturan per klaster / global / max keduanya)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOISE_ID = -1
RULES: tuple[str, ...] = ("per_cluster", "global", "max_cluster_global")
RULE_TEXT: dict[str, str] = {
    "per_cluster": "persentil jarak per klaster",
    "global": "satu batas global (persentil jarak seluruh titik non-noise)",
    "max_cluster_global": "per klaster = max(persentil klaster, persentil global)",
}


def centroids_of(X: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Centroid = rata-rata anggota per klaster di ruang ter-skala; noise (-1) tidak punya centroid."""
    labels = np.asarray(labels)
    return {int(c): X[labels == c].mean(axis=0) for c in sorted(set(labels.tolist()) - {NOISE_ID})}


def distance_matrix(X: np.ndarray, cents: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(id klaster terurut, jarak Euclidean tiap titik (baris) ke tiap centroid (kolom, urut id))."""
    ids = np.asarray(sorted(cents), dtype=int)
    C = np.vstack([cents[int(i)] for i in ids])
    return ids, np.linalg.norm(np.atleast_2d(X)[:, None, :] - C[None, :, :], axis=2)


def distance_limits(
    X: np.ndarray, labels: np.ndarray, cents: dict[int, np.ndarray], percentile: float, rule: str
) -> tuple[dict[int, float], float]:
    """Batas jarak per klaster dan batas global dari jarak anggota ke centroid klasternya sendiri.

    per_cluster: persentil tiap klaster; global: satu nilai untuk semua; max_cluster_global: max(keduanya).
    """
    if rule not in RULES:
        raise ValueError(f"Aturan batas tidak dikenal: {rule!r} (pilihan: {list(RULES)})")
    labels = np.asarray(labels)
    own = {c: np.linalg.norm(X[labels == c] - centre, axis=1) for c, centre in cents.items()}
    per = {c: float(np.percentile(d, percentile)) for c, d in own.items()}
    glob = float(np.percentile(np.concatenate(list(own.values())), percentile))
    if rule == "per_cluster":
        return per, glob
    if rule == "global":
        return {c: glob for c in per}, glob
    return {c: max(v, glob) for c, v in per.items()}, glob


@dataclass
class Assignment:
    """Hasil penugasan: klaster terdekat, jaraknya, penanda di luar klaster, dan seluruh jarak."""

    ids: np.ndarray        # id klaster (urut kolom `dist`)
    nearest: np.ndarray    # id klaster terdekat tiap titik
    nearest_dist: np.ndarray
    outside: np.ndarray    # True bila jarak ke SETIAP centroid melebihi batas klaster itu
    dist: np.ndarray       # (n_titik, n_klaster)


def assign(X: np.ndarray, cents: dict[int, np.ndarray], limits: dict[int, float] | None = None) -> Assignment:
    """Klaster terdekat (Euclidean); `outside` bila di luar batas semua klaster (tanpa `limits`: selalu False)."""
    ids, dist = distance_matrix(X, cents)
    j = dist.argmin(axis=1)
    if limits is None:
        outside = np.zeros(len(dist), dtype=bool)
    else:
        outside = (dist > np.array([limits[int(i)] for i in ids])[None, :]).all(axis=1)
    return Assignment(ids, ids[j], dist[np.arange(len(dist)), j], outside, dist)


def format_distances(ids: np.ndarray, row: np.ndarray, digits: int = 3) -> str:
    """Jarak ke semua centroid sebagai teks "1:0.123;2:1.402" (titik desimal agar aman di CSV)."""
    return ";".join(f"{int(i)}:{v:.{digits}f}" for i, v in zip(ids, row))


def parse_distances(text: str) -> dict[int, float]:
    """Kebalikan format_distances; teks kosong/NaN → dict kosong."""
    if not isinstance(text, str) or not text.strip():
        return {}
    return {int(k): float(v) for k, v in (part.split(":") for part in text.split(";"))}
