"""Penyiapan data analisis: himpunan VOIP, cleaning terlapor, dan matriks fitur ter-skala."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from qos.features.preprocess import compute_fai, exclude_types, session_mask

US_PER_MS = 1000.0
NONNEG_COLUMNS: tuple[str, ...] = ("iat_mean_us", "iat_std_us", "fwd_packets_count", "bwd_packets_count")


def add_derived(df: pd.DataFrame, clip_negative: bool) -> pd.DataFrame:
    """Tambah kolom turunan: fai_pct, total_packets, delay_ms, jitter_ms (satuan internal ms)."""
    out = df.copy()
    out["fai_pct"] = compute_fai(out["fwd_packets_count"], out["bwd_packets_count"], clip_negative)
    out["total_packets"] = out["fwd_packets_count"] + out["bwd_packets_count"]
    out["delay_ms"] = out["iat_mean_us"] / US_PER_MS
    out["jitter_ms"] = out["iat_std_us"] / US_PER_MS
    return out


def iqr_outliers(series: pd.Series, factor: float) -> dict[str, float]:
    """Batas dan jumlah/persen nilai di luar [Q1 - f*IQR, Q3 + f*IQR] (NaN diabaikan)."""
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    lo, hi = q1 - factor * (q3 - q1), q3 + factor * (q3 - q1)
    n = int(((s < lo) | (s > hi)).sum())
    return {"batas_bawah": float(lo), "batas_atas": float(hi), "jumlah": n, "persen": n / len(s) * 100 if len(s) else 0.0}


def transform_features(df: pd.DataFrame, columns: list[str], log_columns: list[str]) -> pd.DataFrame:
    """Matriks fitur: kolom `columns`, log1p pada `log_columns` (subset)."""
    unknown = [c for c in log_columns if c not in columns]
    if unknown:
        raise ValueError(f"log1p harus subset columns; tidak ada: {unknown}")
    out = df[columns].astype("float64").copy()
    for c in log_columns:
        out[c] = np.log1p(out[c])
    return out


def protocol_name(p: float) -> str:
    """Nama protokol IANA umum; selain itu angkanya."""
    if pd.isna(p):
        return "?"
    return {6: "TCP", 17: "UDP", 1: "ICMP"}.get(int(p), str(int(p)))


@dataclass
class PreparedData:
    """Hasil penyiapan satu himpunan: data bersih, fitur, matriks ter-skala, dan laporan cleaning."""

    name: str
    frame: pd.DataFrame            # data bersih lengkap (kolom mentah + turunan + label)
    features: pd.DataFrame         # fitur setelah transformasi (belum di-skala)
    X: np.ndarray                  # fitur ter-skala
    scaler: StandardScaler
    columns: list[str]
    log_columns: list[str]
    report: dict[str, Any] = field(default_factory=dict)


def clean_set(
    voip: pd.DataFrame, set_cfg: dict[str, Any], clean_cfg: dict[str, Any], clip_negative: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Terapkan himpunan + cleaning; kembalikan data bersih dan laporan (jumlah per tahap, NaN/inf/negatif, dst.)."""
    stages: list[tuple[str, int]] = [("VOIP awal (kategori VOIP)", len(voip))]
    df = exclude_types(voip, set_cfg["exclude_traffic_types"])
    if set_cfg["exclude_traffic_types"]:
        stages.append((f"Buang traffic type {set_cfg['exclude_traffic_types']}", len(df)))
    df = df[session_mask(df, set_cfg.get("min_fwd"), set_cfg.get("min_bwd"))]
    lo = lambda v: "tanpa batas" if v is None else f">= {v}"  # noqa: E731
    stages.append((f"Syarat fwd {lo(set_cfg.get('min_fwd'))}, bwd {lo(set_cfg.get('min_bwd'))}", len(df)))
    df = df.reset_index(drop=True)

    report: dict[str, Any] = {}
    report["nan_per_kolom"] = {c: int(df[c].isna().sum()) for c in NONNEG_COLUMNS}
    report["inf_per_kolom"] = {c: int(np.isinf(df[c]).sum()) for c in NONNEG_COLUMNS}
    report["negatif_per_kolom"] = {c: int((df[c] < 0).sum()) for c in NONNEG_COLUMNS}
    df[list(NONNEG_COLUMNS)] = df[list(NONNEG_COLUMNS)].replace([np.inf, -np.inf], np.nan)

    if clean_cfg["drop_duplicates"]:
        report["duplikat"] = int(df.duplicated().sum())
        df = df.drop_duplicates()
        stages.append(("Buang duplikat", len(df)))
    if clean_cfg["drop_negative"]:
        neg = (df[list(NONNEG_COLUMNS)] < 0).any(axis=1)
        report["baris_negatif_dibuang"] = int(neg.sum())
        df = df[~neg]
        stages.append(("Buang nilai negatif", len(df)))
    if clean_cfg["nan_strategy"] != "median":
        raise ValueError(f"nan_strategy tidak dikenal: {clean_cfg['nan_strategy']!r}")
    fill = {c: float(df[c].median()) for c in ("iat_mean_us", "iat_std_us")}
    report["median_imputasi"] = fill
    report["baris_diimputasi"] = int(df[["iat_mean_us", "iat_std_us"]].isna().any(axis=1).sum())
    df = df.fillna(fill)
    df = add_derived(df.reset_index(drop=True), clip_negative)
    stages.append(("Akhir (data bersih)", len(df)))

    report["tahap"] = stages
    report["ekstrem_iqr"] = {
        c: iqr_outliers(df[c], clean_cfg["iqr_factor"]) for c in ("iat_mean_us", "iat_std_us", "fai_pct", "total_packets")
    }
    report["persen_jitter_nol"] = float((df["iat_std_us"] == 0).mean() * 100)
    report["persen_bwd_nol"] = float((df["bwd_packets_count"] == 0).mean() * 100)
    report["per_traffic_type"] = df["traffic_type"].fillna("?").value_counts().to_dict() if "traffic_type" in df else {}
    report["per_protokol"] = df["protocol"].map(protocol_name).value_counts().to_dict()
    return df, report


def prepare(
    voip: pd.DataFrame, set_cfg: dict[str, Any], variant: dict[str, Any], clean_cfg: dict[str, Any], clip_negative: bool
) -> PreparedData:
    """Cleaning + transformasi fitur + StandardScaler untuk satu himpunan dan satu varian fitur."""
    df, report = clean_set(voip, set_cfg, clean_cfg, clip_negative)
    cols, logs = list(variant["columns"]), list(variant["log1p"])
    feats = transform_features(df, cols, logs)
    scaler = StandardScaler().fit(feats)
    report["korelasi_pearson"] = feats.corr(method="pearson")
    report["korelasi_spearman"] = feats.corr(method="spearman")
    return PreparedData(set_cfg["name"], df, feats, scaler.transform(feats), scaler, cols, logs, report)
