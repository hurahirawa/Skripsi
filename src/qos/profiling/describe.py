"""Deskripsi posisi klaster terhadap ambang ITU. Murni deskriptif: BUKAN label kualitas dan tidak memengaruhi klasterisasi."""
from __future__ import annotations

from typing import Any

WITHIN, BETWEEN, OVER = 1, 2, 3   # di dalam batas baik / di antara batas baik dan maksimum / melewati maksimum
METRICS: tuple[str, ...] = ("delay", "jitter", "loss")
METRIC_NAMES: dict[str, str] = {"delay": "Delay", "jitter": "Jitter", "loss": "Packet Loss"}
UNITS: dict[str, str] = {"delay": "ms", "jitter": "ms", "loss": "%"}


def _limits(metric: str, th: dict[str, float]) -> tuple[float, float | None]:
    """(batas dalam, batas maksimum) metrik; jitter hanya punya satu batas."""
    if metric == "delay":
        return th["delay_good_ms"], th["delay_max_ms"]
    if metric == "jitter":
        return th["jitter_max_ms"], None
    return th["loss_good_pct"], th["loss_max_pct"]


def position(metric: str, value: float, th: dict[str, float]) -> int:
    """WITHIN bila ≤ batas dalam, BETWEEN bila ≤ batas maksimum, selain itu OVER (jitter: WITHIN atau OVER)."""
    good, limit = _limits(metric, th)
    if value <= good:
        return WITHIN
    if limit is not None and value <= limit:
        return BETWEEN
    return OVER


def band_text(metric: str, pos: int, th: dict[str, float]) -> str:
    """Teks rentang, mis. "≤150 ms", "150–400 ms", ">400 ms"."""
    good, limit = _limits(metric, th)
    unit = UNITS[metric]
    if pos == WITHIN:
        return f"≤{good:g} {unit}"
    if limit is None:
        return f">{good:g} {unit}"
    return f"{good:g}–{limit:g} {unit}" if pos == BETWEEN else f">{limit:g} {unit}"


def describe(delay_ms: float, jitter_ms: float, fai_pct: float, th: dict[str, float]) -> dict[str, Any]:
    """Posisi median klaster terhadap ambang ITU: per metrik, teks ringkas, dan metrik yang melewati batas maksimum.

    FAI dipakai sebagai proksi loss. Hasil deskriptif, bukan label kualitas.
    """
    values = {"delay": delay_ms, "jitter": jitter_ms, "loss": fai_pct}
    per = {}
    for m, v in values.items():
        pos = position(m, float(v), th)
        per[m] = {"nilai": float(v), "posisi": pos, "rentang": band_text(m, pos, th)}
    text = "; ".join(f"{METRIC_NAMES[m]} {per[m]['nilai']:,.1f} {UNITS[m]} ({per[m]['rentang']})" for m in METRICS)
    return {
        "metrik": per,
        "teks": f"Median: {text}",
        "ringkas": " · ".join(f"{METRIC_NAMES[m]} {per[m]['rentang']}" for m in METRICS),
        "melewati_ambang": [m for m in METRICS if per[m]["posisi"] == OVER],
    }
