"""Tingkat ITU per metrik (PRD FR-LBL, Pers. 2.13–2.14); ambang selalu dari config."""
from __future__ import annotations

from typing import Any

import pandas as pd

LEVEL_OK, LEVEL_MID, LEVEL_BAD = 1, 2, 3


def level_delay(delay_ms: pd.Series, thresholds: dict[str, Any]) -> pd.Series:
    """L_delay: 1 jika ≤ delay_good_ms; 2 jika ≤ delay_max_ms; 3 jika lebih. NaN tetap NaN."""
    lvl = pd.Series(LEVEL_BAD, index=delay_ms.index, dtype="float64")
    lvl[delay_ms <= thresholds["delay_max_ms"]] = LEVEL_MID
    lvl[delay_ms <= thresholds["delay_good_ms"]] = LEVEL_OK
    return lvl.where(delay_ms.notna())


def level_jitter(jitter_ms: pd.Series, thresholds: dict[str, Any]) -> pd.Series:
    """L_jitter: 1 jika ≤ jitter_max_ms, selain itu 3. NaN tetap NaN."""
    lvl = pd.Series(LEVEL_BAD, index=jitter_ms.index, dtype="float64")
    lvl[jitter_ms <= thresholds["jitter_max_ms"]] = LEVEL_OK
    return lvl.where(jitter_ms.notna())


def loss_good_pct(thresholds: dict[str, Any]) -> float:
    """Ambang loss baik; profil lama tanpa `loss_good_pct` memakai loss_max_pct (perilaku 2 level lama)."""
    return thresholds.get("loss_good_pct", thresholds["loss_max_pct"])


def level_loss(loss_pct: pd.Series, thresholds: dict[str, Any]) -> pd.Series:
    """L_loss: 1 jika ≤ loss_good_pct; 2 jika ≤ loss_max_pct; 3 jika lebih. NaN tetap NaN."""
    lvl = pd.Series(LEVEL_BAD, index=loss_pct.index, dtype="float64")
    lvl[loss_pct <= thresholds["loss_max_pct"]] = LEVEL_MID
    lvl[loss_pct <= loss_good_pct(thresholds)] = LEVEL_OK
    return lvl.where(loss_pct.notna())


def flow_levels(
    delay_ms: pd.Series, jitter_ms: pd.Series, loss_pct: pd.Series, thresholds: dict[str, Any]
) -> pd.DataFrame:
    """Tingkat per flow (sebelum klasterisasi): kolom delay, jitter, loss, dan total T = max."""
    out = pd.DataFrame(
        {
            "delay": level_delay(delay_ms, thresholds),
            "jitter": level_jitter(jitter_ms, thresholds),
            "loss": level_loss(loss_pct, thresholds),
        }
    )
    out["total"] = out.max(axis=1, skipna=False)
    return out


METRIC_NAMES: dict[str, str] = {"delay": "Delay", "jitter": "Jitter", "loss": "Packet Loss"}
CATEGORY_BAIK, CATEGORY_CUKUP = "QoS Baik", "QoS Cukup"
CATEGORY_BURUK, CATEGORY_ANOMALI = "QoS Buruk", "Anomali"
ACTIONS: dict[str, str] = {
    CATEGORY_BAIK: "-",
    CATEGORY_CUKUP: "Pemantauan",
    CATEGORY_BURUK: "Perbaikan Kapasitas/Konfigurasi",
    CATEGORY_ANOMALI: "Investigasi Kasus Tunggal",
}
NOISE_ID = -1


def join_metrics(names: list[str]) -> str:
    """Gabung nama metrik: "A", "A dan B", "A, B dan C"."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " dan " + names[-1]


def metrics_at(levels: dict[str, int], level: int) -> list[str]:
    """Nama metrik (urutan Delay, Jitter, Packet Loss) yang levelnya persis `level`."""
    return [METRIC_NAMES[m] for m in METRIC_NAMES if levels[m] == level]


def label_cluster(levels: dict[str, int], noise: bool = False) -> tuple[str, str]:
    """Label dan kategori tindakan dari tingkat per metrik {delay, jitter, loss} (Pers. 2.13–2.14).

    Klaster biasa: T = max level; T=1 → "QoS Baik", selain itu "QoS Cukup|Buruk - {metrik penyebab} {Meningkat|Melebihi Batas}".
    Noise: metrik berlevel 3 → "Anomali - {metrik} Ekstrem"; tanpa itu "Anomali".
    """
    if noise:
        bad = metrics_at(levels, LEVEL_BAD)
        label = f"{CATEGORY_ANOMALI} - {join_metrics(bad)} Ekstrem" if bad else CATEGORY_ANOMALI
        return label, ACTIONS[CATEGORY_ANOMALI]
    total = max(levels.values())
    if total == LEVEL_OK:
        return CATEGORY_BAIK, ACTIONS[CATEGORY_BAIK]
    category, note = (CATEGORY_CUKUP, "Meningkat") if total == LEVEL_MID else (CATEGORY_BURUK, "Melebihi Batas")
    return f"{category} - {join_metrics(metrics_at(levels, total))} {note}", ACTIONS[category]


def cluster_levels(stats: dict[str, float], thresholds: dict[str, Any]) -> dict[str, int]:
    """Tingkat per metrik dari statistik pelabelan klaster (mean_delay_ms, p999_jitter_ms, plr_pct)."""
    one = lambda v: pd.Series([v])  # noqa: E731
    return {
        "delay": int(level_delay(one(stats["mean_delay_ms"]), thresholds).iloc[0]),
        "jitter": int(level_jitter(one(stats["p999_jitter_ms"]), thresholds).iloc[0]),
        "loss": int(level_loss(one(stats["plr_pct"]), thresholds).iloc[0]),
    }


CAUSE_SEPARATOR = ";"


def cause_metrics(levels: dict[str, int], noise: bool = False) -> list[str]:
    """Kunci metrik penyebab (delay/jitter/loss, urutan tetap) menurut aturan label.

    Klaster biasa: metrik dengan level = T (kosong bila QoS Baik). Noise: metrik berlevel 3.
    """
    if noise:
        target = LEVEL_BAD
    else:
        target = max(levels.values())
        if target == LEVEL_OK:
            return []
    return [m for m in METRIC_NAMES if levels[m] == target]


def apply_labels(stats: dict[str, float], thresholds: dict[str, Any], noise: bool = False) -> dict[str, Any]:
    """Tingkat, label, kategori tindakan, dan metrik_penyebab ("delay;loss") dari statistik pelabelan + ambang."""
    levels = cluster_levels(stats, thresholds)
    label, action = label_cluster(levels, noise)
    return {
        "tingkat": {**levels, "klaster": max(levels.values())},
        "label": label,
        "kategori_tindakan": action,
        "metrik_penyebab": CAUSE_SEPARATOR.join(cause_metrics(levels, noise)),
    }
