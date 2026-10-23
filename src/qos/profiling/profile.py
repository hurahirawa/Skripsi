"""Profil klaster: statistik, box plot, dan statistik pelabelan (PRD FR-PRF)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qos.profiling.labeling import NOISE_ID, apply_labels

PROFILE_METRICS: tuple[str, ...] = ("delay_ms", "jitter_ms", "fai_pct")


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


def labeling_stats(
    df: pd.DataFrame, jitter_pct: float, fwd_col: str = "fwd_packets_count", bwd_col: str = "bwd_packets_count"
) -> dict[str, float]:
    """Mean delay, P(`jitter_pct`) jitter, dan PLR untuk satu klaster."""
    return {
        "mean_delay_ms": float(df["delay_ms"].mean()),
        "p999_jitter_ms": percentile(df["jitter_ms"], jitter_pct),
        "plr_pct": plr_pct(df[fwd_col], df[bwd_col]),
    }


def build_cluster_profile(
    df: pd.DataFrame, cluster_ids: np.ndarray | pd.Series, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Profil tiap klaster (termasuk -1) dari `df` (berkolom delay_ms, jitter_ms, fai_pct, fwd, bwd).

    Ambang dan parameter dari `cfg`; label diturunkan dari statistik pelabelan.
    """
    ids = pd.Series(np.asarray(cluster_ids), index=df.index)
    prof, th = cfg["profiling"], cfg["thresholds"]
    result = []
    for cid in sorted(ids.unique()):
        sub = df[ids == cid]
        stats = labeling_stats(sub, prof["jitter_percentile"])
        labeled = apply_labels(stats, th, noise=cid == NOISE_ID)
        result.append(
            {
                "id_klaster": int(cid),
                "jumlah_anggota": int(len(sub)),
                "statistik": {m: basic_stats(sub[m]) for m in PROFILE_METRICS},
                "boxplot": {m: boxplot_stats(sub[m], prof["max_outliers"]) for m in PROFILE_METRICS},
                "statistik_pelabelan": stats,
                **labeled,
            }
        )
    return result


def label_map(profile: list[dict[str, Any]]) -> dict[str, str]:
    """Peta id_klaster (string) → label, seperti `label_map` di cluster_profile.json."""
    return {str(c["id_klaster"]): c["label"] for c in profile}
