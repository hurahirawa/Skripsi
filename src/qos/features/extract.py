"""Turunan fitur dalam satuan asli: delay_ms, jitter_ms, dan FAI (PRD §7.1)."""
from __future__ import annotations

import pandas as pd

from qos.features.preprocess import compute_fai

US_PER_MS = 1000.0


def us_to_ms(values: pd.Series) -> pd.Series:
    """Ubah µs ke ms."""
    return values / US_PER_MS


def derive_metrics(
    df: pd.DataFrame,
    delay_col: str,
    jitter_col: str,
    clip_negative: bool,
    fwd_col: str = "fwd_packets_count",
    bwd_col: str = "bwd_packets_count",
) -> pd.DataFrame:
    """Tambah kolom delay_ms, jitter_ms, fai_pct; baris dengan fwd = 0 dibuang.

    `delay_col` dan `jitter_col` (µs) dipilih pemanggil, mis. iat_mean_us dan iat_std_us.
    """
    out = df[df[fwd_col] != 0].copy()
    out["delay_ms"] = us_to_ms(out[delay_col])
    out["jitter_ms"] = us_to_ms(out[jitter_col])
    out["fai_pct"] = compute_fai(out[fwd_col], out[bwd_col], clip_negative)
    return out
