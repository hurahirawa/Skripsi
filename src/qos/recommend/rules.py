"""Rekomendasi mitigasi rule-based (PRD FR-REC); aturan dibaca dari config/recommendations.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from qos.profiling.labeling import (
    CATEGORY_ANOMALI,
    CATEGORY_BURUK,
    CAUSE_SEPARATOR,
    METRIC_NAMES,
)

RULE_KEYS: tuple[str, ...] = ("delay", "jitter", "loss", "anomali")
METRIC_ORDER: tuple[str, ...] = tuple(METRIC_NAMES)  # delay, jitter, loss
NAME_TO_METRIC: dict[str, str] = {v: k for k, v in METRIC_NAMES.items()}
EXCEEDED_SUFFIX = " Melebihi Batas"


class RulesError(ValueError):
    """Berkas aturan rekomendasi tidak valid."""


@dataclass(frozen=True)
class Recommendation:
    """Rekomendasi untuk satu label: pemicu, kategori tindakan, dan daftar saran (tanpa duplikat)."""

    label: str
    category: str
    triggers: tuple[str, ...]
    suggestions: tuple[str, ...]


def load_rules(path: str | Path) -> dict[str, dict[str, Any]]:
    """Muat dan validasi aturan: tiap kunci di RULE_KEYS wajib punya pemicu, kategori, dan saran."""
    path = Path(path)
    if not path.is_file():
        raise RulesError(f"Berkas aturan rekomendasi tidak ditemukan: {path}")
    rules = (yaml.safe_load(path.read_text("utf-8")) or {}).get("rules", {})
    for key in RULE_KEYS:
        rule = rules.get(key)
        if not rule or not all(rule.get(f) for f in ("pemicu", "kategori", "saran")):
            raise RulesError(f"Aturan '{key}' tidak lengkap di {path} (butuh pemicu, kategori, saran)")
        if not isinstance(rule["saran"], list):
            raise RulesError(f"'saran' pada aturan '{key}' harus berupa daftar")
    return rules


def metrics_from_label(label: str) -> list[str]:
    """Kunci metrik penyebab (delay/jitter/loss) dari label "QoS Buruk - A, B dan C Melebihi Batas"."""
    body = label.split(" - ", 1)[1] if " - " in label else ""
    body = body.removesuffix(EXCEEDED_SUFFIX)
    names = [n.strip() for n in body.replace(" dan ", ", ").split(",") if n.strip()]
    found = {NAME_TO_METRIC[n] for n in names if n in NAME_TO_METRIC}
    return [m for m in METRIC_ORDER if m in found]


def causes_from_column(value: Any) -> list[str] | None:
    """Parse kolom `metrik_penyebab` ("delay;loss") menjadi daftar kunci; None bila kosong/tidak ada (CSV lama)."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return None
    keys = [k for k in str(value).split(CAUSE_SEPARATOR) if k in METRIC_ORDER]
    return [m for m in METRIC_ORDER if m in keys] or None


def recommend(label: str, rules: dict[str, dict[str, Any]], causes: Any = None) -> Recommendation | None:
    """Rekomendasi untuk `label`; None untuk QoS Baik, QoS Cukup, dan Data Belum Cukup.

    QoS Buruk: gabungan saran semua metrik penyebab (urutan delay, jitter, loss). Metrik dibaca dari
    `causes` (kolom metrik_penyebab) bila ada; parser teks label hanya fallback untuk data lama.
    Anomali: hanya saran aturan `anomali` (Lampiran 5).
    """
    if label.startswith(CATEGORY_ANOMALI):
        r = rules["anomali"]
        return Recommendation(label, r["kategori"], (r["pemicu"],), tuple(r["saran"]))
    if not label.startswith(CATEGORY_BURUK):
        return None
    metrics = causes_from_column(causes) or metrics_from_label(label)
    if not metrics:
        return None
    suggestions: list[str] = []
    for m in metrics:
        suggestions += [s for s in rules[m]["saran"] if s not in suggestions]
    return Recommendation(
        label, rules[metrics[0]]["kategori"], tuple(rules[m]["pemicu"] for m in metrics), tuple(suggestions)
    )


def session_recommendations(rows: pd.DataFrame, rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rekomendasi tingkat sesi dari baris jendela: label terakhir tiap flow → kelompok per label bermasalah.

    Kembalian terurut dari yang paling banyak flow-nya; tiap item: label, jumlah_flow, kategori, pemicu, saran.
    """
    if rows.empty:
        return []
    cols = ["label", "metrik_penyebab"] if "metrik_penyebab" in rows.columns else ["label"]
    last = rows.groupby("flow_id", sort=False)[cols].last()
    if "metrik_penyebab" not in last.columns:
        last["metrik_penyebab"] = None
    out = []
    for (label, causes), n in last.fillna({"metrik_penyebab": ""}).value_counts().items():
        rec = recommend(str(label), rules, causes)
        if rec is not None:
            out.append(
                {"label": rec.label, "jumlah_flow": int(n), "kategori": rec.category,
                 "pemicu": list(rec.triggers), "saran": list(rec.suggestions)}
            )
    return out


def cluster_recommendations(profile: Iterable[dict[str, Any]], rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Rekomendasi per klaster bermasalah pada profil model (QoS Buruk dan Anomali)."""
    out = []
    for c in profile:
        rec = recommend(c["label"], rules, c.get("metrik_penyebab"))
        if rec is not None:
            out.append(
                {"id_klaster": c["id_klaster"], "jumlah_anggota": c["jumlah_anggota"], "label": rec.label,
                 "kategori": rec.category, "pemicu": list(rec.triggers), "saran": list(rec.suggestions)}
            )
    return out
