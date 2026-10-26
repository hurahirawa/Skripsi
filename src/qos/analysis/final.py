"""Bagian F analisis klasterisasi: ambang ITU (deskriptif), batas jarak "jauh dari semua klaster", dan kandidat HDBSCAN."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qos.analysis import clustering as cl


def _band(value: float, good: float, limit: float | None) -> str:
    """Posisi nilai terhadap satu atau dua ambang: '<= good', 'good–limit', atau '> limit'."""
    if value <= good:
        return f"≤{good:g}"
    if limit is None or value <= limit:
        return f"{good:g}–{limit:g}" if limit is not None else f">{good:g}"
    return f">{limit:g}"


def itu_position(delay_ms: float, jitter_ms: float, fai_pct: float, thresholds: dict[str, float]) -> str:
    """Keterangan posisi median klaster terhadap ambang ITU (deskripsi, bukan label). FAI dipakai sebagai proksi loss."""
    t = thresholds
    return (
        f"delay {_band(delay_ms, t['delay_good_ms'], t['delay_max_ms'])} ms; "
        f"jitter {_band(jitter_ms, t['jitter_max_ms'], None)} ms; "
        f"FAI {_band(fai_pct, t['loss_good_pct'], t['loss_max_pct'])} %"
    )


def add_itu_column(profile: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Tambah kolom 'posisi ITU (median)' pada profil klaster."""
    out = profile.copy()
    out["posisi ITU (median)"] = [
        itu_position(r.delay_median_ms, r.jitter_median_ms, r.fai_median_pct, thresholds) for r in out.itertuples()
    ]
    return out


def distance_limits(X: np.ndarray, labels: np.ndarray, percentile: float) -> dict[int, float]:
    """Batas jarak per klaster = persentil `percentile` jarak anggota ke centroid klasternya sendiri."""
    cents = cl.centroids(X, labels)
    labels = np.asarray(labels)
    return {c: float(np.percentile(np.linalg.norm(X[labels == c] - centre, axis=1), percentile)) for c, centre in cents.items()}


def far_from_all(X: np.ndarray, cent_map: dict[int, np.ndarray], limits: dict[int, float]) -> np.ndarray:
    """True bila jarak titik ke SETIAP centroid melebihi batas klaster itu."""
    ids = sorted(cent_map)
    _, dist = cl.assign_nearest(X, cent_map)
    return (dist > np.array([limits[i] for i in ids])[None, :]).all(axis=1)


def centroids_original_units(cent_map: dict[int, np.ndarray], scaler, columns: list[str], log_columns: list[str]) -> pd.DataFrame:
    """Centroid ter-skala → satuan asli (invers StandardScaler lalu expm1 untuk kolom log1p); delay/jitter dalam ms."""
    ids = sorted(cent_map)
    raw = pd.DataFrame(scaler.inverse_transform(np.vstack([cent_map[i] for i in ids])), columns=columns, index=ids)
    for c in log_columns:
        raw[c] = np.expm1(raw[c])
    raw = raw.rename(columns={"iat_mean_us": "delay_ms", "iat_std_us": "jitter_ms"})
    for c in ("delay_ms", "jitter_ms"):
        if c in raw:
            raw[c] = raw[c] / 1000.0
    raw.index.name = "klaster"
    return raw.reset_index()
