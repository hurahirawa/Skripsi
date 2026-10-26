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


def _short(rentang: str) -> str:
    """Rentang tanpa satuan, mis. "≤150 ms" → "≤150"."""
    return rentang.removesuffix(" ms").removesuffix(" %")


def _count(v: float) -> str:
    """Bilangan bulat dengan titik ribuan (gaya Indonesia)."""
    return f"{round(v):,}".replace(",", ".")


def _seconds(v: float) -> str:
    """Detik: bulat bila ≥ 10, selain itu 1 desimal."""
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def protocol_text(karakter: dict[str, Any] | None) -> str:
    """Teks komposisi protokol, mis. "UDP 98% · TCP 2%"; kosong bila protokol tidak diketahui."""
    if not karakter or karakter.get("persen_udp") is None:
        return ""
    parts = [f"UDP {karakter['persen_udp']:.0f}%", f"TCP {karakter['persen_tcp']:.0f}%"]
    if (karakter.get("persen_lain") or 0) >= 0.5:  # abaikan sisa pembulatan
        parts.append(f"lain {karakter['persen_lain']:.0f}%")
    return " · ".join(parts)


def character_text(karakter: dict[str, Any] | None) -> str:
    """Median total paket dan durasi flow, mis. "32 paket · 90 dtk" (bagian yang tidak diketahui dilewati)."""
    if not karakter:
        return ""
    parts = []
    if karakter.get("paket_median") is not None:
        parts.append(f"{_count(karakter['paket_median'])} paket")
    if karakter.get("durasi_median_s") is not None:
        parts.append(f"{_seconds(karakter['durasi_median_s'])} dtk")
    return " · ".join(parts)


def describe(
    delay_ms: float, jitter_ms: float, fai_pct: float, th: dict[str, float],
    nama: str | None = None, karakter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Posisi median klaster terhadap ambang ITU: per metrik, teks ringkas, dan metrik yang melewati batas maksimum.

    FAI dipakai sebagai proksi loss. Dengan `karakter` (median paket/durasi, % protokol) `ringkas` berformat
    "Klaster 2 · 32 paket · 90 dtk · Delay >400 · Jitter >50 · Loss ≤1"; `itu` selalu berisi bagian ITU saja. Deskriptif, bukan label kualitas.
    """
    values = {"delay": delay_ms, "jitter": jitter_ms, "loss": fai_pct}
    per = {}
    for m, v in values.items():
        pos = position(m, float(v), th)
        per[m] = {"nilai": float(v), "posisi": pos, "rentang": band_text(m, pos, th)}
    text = "; ".join(f"{METRIC_NAMES[m]} {per[m]['nilai']:,.1f} {UNITS[m]} ({per[m]['rentang']})" for m in METRICS)
    itu = " · ".join(f"{METRIC_NAMES[m]} {per[m]['rentang']}" for m in METRICS)
    ringkas = itu
    proto = protocol_text(karakter)
    if karakter:
        short = " · ".join(f"{label} {_short(per[m]['rentang'])}" for m, label in (("delay", "Delay"), ("jitter", "Jitter"), ("loss", "Loss")))
        ringkas = " · ".join(p for p in (nama, character_text(karakter), short) if p)
    return {
        "metrik": per,
        "teks": f"Median: {text}" + (f"; protokol {proto}" if proto else ""),
        "itu": itu,
        "ringkas": ringkas,
        "protokol": proto,
        "melewati_ambang": [m for m in METRICS if per[m]["posisi"] == OVER],
    }
