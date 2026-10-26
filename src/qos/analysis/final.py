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


def select_final_hdbscan(grid: pd.DataFrame, sel: dict) -> tuple[pd.Series, pd.DataFrame, str]:
    """Aturan final: kandidat valid = noise <= max, klaster >= min, ARI mean >= min; pilih DBCV tertinggi (seri → Silhouette).

    `grid` harus punya kolom ari_mean. Kembalikan (baris terpilih, tabel kandidat valid terurut, alasan).
    """
    ok = grid[
        (grid["noise_ratio"] <= sel["max_noise_ratio"]) & (grid["n_clusters"] >= sel["min_clusters"])
        & (grid["ari_mean"] >= sel["min_ari_mean"]) & grid["dbcv"].notna()
    ].sort_values(["dbcv", "silhouette"], ascending=False)
    if ok.empty:
        raise ValueError("Tidak ada kandidat HDBSCAN yang memenuhi aturan final")
    best = ok.iloc[0]
    reason = (
        f"min_cluster_size={int(best['min_cluster_size'])}, min_samples={int(best['min_samples'])}: DBCV tertinggi ({best['dbcv']:.4f}) "
        f"di antara {len(ok)} kandidat valid (noise <= {sel['max_noise_ratio']:.0%}, klaster >= {sel['min_clusters']}, "
        f"ARI mean >= {sel['min_ari_mean']:g}); {int(best['n_clusters'])} klaster, noise {best['noise_ratio']:.1%}, "
        f"Silhouette {best['silhouette']:.4f}, ARI {best['ari_mean']:.3f}."
    )
    return best, ok, reason


def distance_rules(X: np.ndarray, labels: np.ndarray, percentile: float) -> pd.DataFrame:
    """Bandingkan tiga aturan batas "jauh dari semua klaster" (jarak ke SEMUA centroid > batas klaster itu).

    (a) p-persentil per klaster; (b) satu batas global = persentil seluruh jarak non-noise ke centroid sendiri;
    (c) per klaster = max(a, b). Kembalikan % noise dan % non-noise yang ditandai jauh per aturan.
    """
    labels = np.asarray(labels)
    cents = cl.centroids(X, labels)
    ids = sorted(cents)
    per = distance_limits(X, labels, percentile)
    mask = labels != cl.NOISE
    own = np.concatenate([np.linalg.norm(X[labels == c] - cents[c], axis=1) for c in ids])
    glob = float(np.percentile(own, percentile))
    rules = {
        f"(a) p{percentile:g} per klaster": per,
        f"(b) global p{percentile:g} ({glob:.3f})": {c: glob for c in ids},
        "(c) max(per klaster, global)": {c: max(per[c], glob) for c in ids},
    }
    rows = []
    for name, lim in rules.items():
        far_noise = far_from_all(X[~mask], cents, lim) if (~mask).any() else np.array([], dtype=bool)
        far_in = far_from_all(X[mask], cents, lim)
        rows.append({
            "aturan": name, "noise ditandai jauh (%)": float(far_noise.mean() * 100) if far_noise.size else float("nan"),
            "noise ditandai": int(far_noise.sum()), "noise total": int((~mask).sum()),
            "non-noise ditandai jauh (%)": float(far_in.mean() * 100), "non-noise ditandai": int(far_in.sum()), "non-noise total": int(mask.sum()),
            "salah tandai": int((~far_noise).sum() + far_in.sum()),
            "batas per klaster": ", ".join(f"K{c}: {lim[c]:.3f}" for c in ids),
        })
    return pd.DataFrame(rows)


def recommend_rule(rules: pd.DataFrame) -> int:
    """Indeks aturan dengan salah tandai paling sedikit (noise lolos + non-noise ditandai jauh); seri → noise ditandai lebih banyak."""
    return int(rules.sort_values(["salah tandai", "noise ditandai"], ascending=[True, False]).index[0])
