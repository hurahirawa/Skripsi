"""Praproses awal: filter sesi minimum, pengecualian traffic type, dan FAI."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def session_mask(df: pd.DataFrame, min_fwd: int | None, min_bwd: int | None) -> pd.Series:
    """True untuk flow dengan fwd ≥ min_fwd dan bwd ≥ min_bwd; batas None = tidak dipakai."""
    mask = pd.Series(True, index=df.index)
    if min_fwd is not None:
        mask &= df["fwd_packets_count"] >= min_fwd
    if min_bwd is not None:
        mask &= df["bwd_packets_count"] >= min_bwd
    return mask


def exclude_types(df: pd.DataFrame, types: list[str], group_col: str = "traffic_type") -> pd.DataFrame:
    """Buang flow yang traffic type-nya ada di `types` (tanpa peduli huruf besar/kecil)."""
    if not types:
        return df
    if group_col not in df.columns:
        raise ValueError(f"exclude_traffic_types diisi tetapi kolom {group_col!r} tidak ada di data")
    drop = {t.strip().casefold() for t in types}
    keep = ~df[group_col].astype("string").str.strip().str.casefold().isin(drop).fillna(False)
    return df[keep.astype(bool)]


def apply_session_filters(df: pd.DataFrame, data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Terapkan exclude_traffic_types lalu min_session dari config; log jumlah flow tiap tahap."""
    out = exclude_types(df, data_cfg["exclude_traffic_types"])
    logger.info("Exclude traffic type %s: %d → %d flow", data_cfg["exclude_traffic_types"], len(df), len(out))
    ms = data_cfg["min_session"]
    kept = out[session_mask(out, ms["fwd"], ms["bwd"])]
    logger.info("Filter sesi minimum fwd>=%s, bwd>=%s: %d → %d flow", ms["fwd"], ms["bwd"], len(out), len(kept))
    return kept.reset_index(drop=True)


def compute_fai(fwd: pd.Series, bwd: pd.Series, clip_negative: bool) -> pd.Series:
    """FAI (%) = (fwd − bwd) / fwd × 100 (Pers. 2.1); fwd = 0 → NaN; clip_negative: FAI < 0 jadi 0."""
    fai = (fwd - bwd) / fwd.where(fwd != 0) * 100
    return fai.clip(lower=0) if clip_negative else fai


def clean_features(
    df: pd.DataFrame, columns: list[str], max_rows: int, random_state: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Cleaning PRD §7.3: ±inf→NaN, drop duplikat, drop negatif pada `columns`, imputasi median, sampling.

    Kembalian: (data bersih, median tiap kolom untuk dipakai saat live). Median dihitung setelah
    pembuangan duplikat/negatif dan sebelum sampling.
    """
    out = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    out = out[~(out[columns] < 0).any(axis=1)].copy()
    medians = {c: float(out[c].median()) for c in columns}
    empty = [c for c, m in medians.items() if np.isnan(m)]
    if empty:
        raise ValueError(f"Kolom tanpa nilai valid, median tidak bisa dihitung: {empty}")
    out[columns] = impute_median(out[columns], medians)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state)
    logger.info("Cleaning: %d → %d baris, median=%s", len(df), len(out), medians)
    return out.reset_index(drop=True), medians


def impute_median(X: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Isi NaN dengan median (dari artefak latih); kolom lain tidak disentuh."""
    return X.fillna(value=medians)


def fit_scaler(X: pd.DataFrame | np.ndarray) -> StandardScaler:
    """Fit StandardScaler (Z-Score) pada data latih."""
    return StandardScaler().fit(X)


def build_features(df: pd.DataFrame, columns: list[str], log_columns: list[str]) -> pd.DataFrame:
    """Matriks fitur klaster: ambil `columns`, lalu log1p pada `log_columns` (subset `columns`).

    Dipakai sama persis saat pelatihan dan inferensi live; `columns` dan `log_columns` disimpan di artefak.
    """
    unknown = [c for c in log_columns if c not in columns]
    if unknown:
        raise ValueError(f"log_transform harus subset cluster_columns; tidak ada: {unknown}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom fitur tidak ada: {missing}")
    out = df[columns].astype("float64").copy()
    for c in log_columns:
        if (out[c] < 0).any():
            raise ValueError(f"log1p butuh nilai ≥ 0; kolom {c} punya nilai negatif")
        out[c] = np.log1p(out[c])
    return out
