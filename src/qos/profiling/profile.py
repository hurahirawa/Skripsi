"""Profil klaster: statistik deskriptif, box plot, centroid, batas jarak, dan deskripsi ITU (PRD FR-PRF)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qos.profiling.describe import describe

PROFILE_METRICS: tuple[str, ...] = ("delay_ms", "jitter_ms", "fai_pct")
NOISE_ID = -1


def basic_stats(values: pd.Series) -> dict[str, float | None]:
    """mean, median, std (ddof=1), min, max; std = None jika anggota < 2."""
    v = values.to_numpy(dtype=float)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if len(v) > 1 else None,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def boxplot_stats(values: pd.Series, max_outliers: int) -> dict[str, Any]:
    """Q1/median/Q3 (np.percentile linear), IQR, pagar 1,5·IQR, whisker, dan pencilan (maks `max_outliers`).

    Whisker = nilai data terkecil/terbesar di dalam pagar. Jika pencilan melebihi batas,
    yang disimpan adalah yang terjauh dari pagar.
    """
    v = values.to_numpy(dtype=float)
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75], method="linear"))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lower) & (v <= upper)]
    outliers = v[(v < lower) | (v > upper)]
    if len(outliers) > max_outliers:
        dist = np.where(outliers < lower, lower - outliers, outliers - upper)
        outliers = outliers[np.argsort(-dist, kind="stable")[:max_outliers]]
    return {
        "q1": q1, "median": med, "q3": q3, "iqr": iqr,
        "lower_fence": lower, "upper_fence": upper,
        "whisker_low": float(inside.min()), "whisker_high": float(inside.max()),
        "outliers": sorted(float(x) for x in outliers),
    }


def percentile(values: pd.Series, pct: float) -> float:
    """Persentil linear (pct dalam 0–100), mis. 99,9 untuk jitter."""
    return float(np.percentile(values.to_numpy(dtype=float), pct, method="linear"))


def plr_pct(fwd: pd.Series, bwd: pd.Series) -> float:
    """PLR klaster (Pers. 2.12): Σ max(fwd − bwd, 0) / Σ fwd × 100."""
    total = float(fwd.sum())
    if total == 0:
        raise ValueError("Σ fwd = 0, PLR tidak terdefinisi")
    return float(np.maximum(fwd - bwd, 0).sum() / total * 100)


def extra_stats(
    df: pd.DataFrame, jitter_pct: float, fwd_col: str = "fwd_packets_count", bwd_col: str = "bwd_packets_count"
) -> dict[str, float]:
    """Statistik deskriptif tambahan satu klaster: mean delay, P(`jitter_pct`) jitter, dan PLR (bukan dasar klasterisasi)."""
    return {
        "mean_delay_ms": float(df["delay_ms"].mean()),
        "p999_jitter_ms": percentile(df["jitter_ms"], jitter_pct),
        "plr_pct": plr_pct(df[fwd_col], df[bwd_col]),
    }


def cluster_name(cid: int) -> str:
    """Nama tampilan: "Klaster N" (N mulai 1) atau "Noise"."""
    return "Noise" if cid == NOISE_ID else f"Klaster {cid}"


PROTOCOL_TCP, PROTOCOL_UDP = 6, 17
US_PER_SECOND = 1_000_000.0


def characteristics(df: pd.DataFrame, cluster_ids: np.ndarray | pd.Series) -> dict[int, dict[str, Any]]:
    """Karakter tiap klaster: median total paket (fwd+bwd), median durasi flow (detik), dan % UDP/TCP/lain.

    Durasi dan protokol bernilai None bila kolom `duration_us` / `protocol` tidak ada (mis. data latih dari log monitoring).
    """
    ids = pd.Series(np.asarray(cluster_ids), index=df.index)
    out: dict[int, dict[str, Any]] = {}
    for cid in sorted(ids.unique()):
        sub = df[ids == cid]
        k: dict[str, Any] = {"paket_median": float((sub["fwd_packets_count"] + sub["bwd_packets_count"]).median()),
                             "durasi_median_s": None, "persen_udp": None, "persen_tcp": None, "persen_lain": None}
        if "duration_us" in sub and sub["duration_us"].notna().any():
            k["durasi_median_s"] = float(sub["duration_us"].median() / US_PER_SECOND)
        if "protocol" in sub and sub["protocol"].notna().any():
            proto = sub["protocol"].dropna()
            k["persen_udp"] = float((proto == PROTOCOL_UDP).mean() * 100)
            k["persen_tcp"] = float((proto == PROTOCOL_TCP).mean() * 100)
            k["persen_lain"] = max(0.0, float(100 - k["persen_udp"] - k["persen_tcp"]))
        out[int(cid)] = k
    return out


def describe_stats(
    stats: dict[str, dict[str, Any]], thresholds: dict[str, float], nama: str | None = None, karakter: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Deskripsi ITU dari median delay/jitter/FAI pada `statistik` klaster (plus karakter bila ada)."""
    return describe(stats["delay_ms"]["median"], stats["jitter_ms"]["median"], stats["fai_pct"]["median"], thresholds, nama, karakter)


def build_cluster_profile(
    df: pd.DataFrame,
    cluster_ids: np.ndarray | pd.Series,
    cfg: dict[str, Any],
    centroids_raw: dict[int, dict[str, float]] | None = None,
    limits: dict[int, float] | None = None,
    own_distance: dict[int, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Profil tiap klaster (termasuk Noise = -1) dari `df` (berkolom delay_ms, jitter_ms, fai_pct).

    Berisi anggota, statistik dan box plot (satuan asli), centroid satuan asli, batas jarak, serta deskripsi
    posisi terhadap ambang ITU dari config (deskriptif, bukan label).
    """
    ids = pd.Series(np.asarray(cluster_ids), index=df.index)
    prof, th = cfg["profiling"], cfg["thresholds"]
    max_out = prof["max_outliers"]
    karakter = characteristics(df, ids)
    result = []
    for cid in sorted(ids.unique(), key=lambda v: (v == NOISE_ID, v)):  # Noise di urutan terakhir
        sub = df[ids == cid]
        stats = {m: basic_stats(sub[m]) for m in PROFILE_METRICS}
        desc = describe_stats(stats, th, cluster_name(int(cid)), karakter[int(cid)])
        result.append(
            {
                "id_klaster": int(cid),
                "nama": cluster_name(int(cid)),
                "jumlah_anggota": int(len(sub)),
                "statistik": stats,
                "boxplot": {m: boxplot_stats(sub[m], max_out) for m in PROFILE_METRICS},
                "statistik_tambahan": extra_stats(sub, prof["jitter_percentile"]),
                "karakter": karakter[int(cid)],
                "centroid_asli": (centroids_raw or {}).get(int(cid)),
                "batas_jarak": (limits or {}).get(int(cid)),
                "jarak_sendiri": (own_distance or {}).get(int(cid)),
                "deskripsi": desc,
                "metrik_melewati_ambang": desc["melewati_ambang"],
            }
        )
    return result


def redescribe(
    profile: list[dict[str, Any]], thresholds: dict[str, float], karakter: dict[int, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Hitung ulang deskripsi ITU dari statistik tersimpan dengan ambang baru (tanpa refit).

    `karakter` (id klaster → karakter) melengkapi profil lama yang belum punya; selain itu karakter tersimpan dipakai.
    """
    out = []
    for c in profile:
        kar = c.get("karakter") or (karakter or {}).get(int(c["id_klaster"]))
        c = {**c, "karakter": kar}
        desc = describe_stats(c["statistik"], thresholds, c.get("nama"), kar)
        out.append({**c, "deskripsi": desc, "metrik_melewati_ambang": desc["melewati_ambang"]})
    return out
