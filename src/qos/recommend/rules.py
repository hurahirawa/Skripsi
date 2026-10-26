"""Saran berdasarkan karakteristik klaster (PRD FR-REC); aturan dibaca dari config/recommendations.yaml.

Pemicu: metrik klaster yang melewati ambang ITU (deskripsi, bukan label kualitas) dan titik "Di luar klaster".
Dimatikan lewat config `recommendations.enabled`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from qos.monitor.calls import OUTSIDE_KEY, STATUS_CLUSTERED, STATUS_OUTSIDE
from qos.profiling.describe import METRICS

RULE_KEYS: tuple[str, ...] = (*METRICS, "di_luar_klaster")
TITLE = "Saran berdasarkan karakteristik klaster"
NOISE_ID = -1


class RulesError(ValueError):
    """Berkas aturan rekomendasi tidak valid."""


@dataclass(frozen=True)
class Recommendation:
    """Saran untuk satu klaster atau kelompok "Di luar klaster": pemicu, jenis tindakan, dan saran (tanpa duplikat)."""

    key: str
    category: str
    triggers: tuple[str, ...]
    suggestions: tuple[str, ...]


def recommendations_enabled(cfg: dict[str, Any]) -> bool:
    """True bila config recommendations.enabled (default true)."""
    return bool(cfg.get("recommendations", {}).get("enabled", True))


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


def recommend_metrics(metrics: Iterable[str], rules: dict[str, dict[str, Any]], key: str) -> Recommendation | None:
    """Saran gabungan untuk metrik yang melewati ambang (urutan delay, jitter, loss); None bila tidak ada metrik."""
    ordered = [m for m in METRICS if m in set(metrics)]
    if not ordered:
        return None
    suggestions: list[str] = []
    for m in ordered:
        suggestions += [s for s in rules[m]["saran"] if s not in suggestions]
    return Recommendation(key, rules[ordered[0]]["kategori"], tuple(rules[m]["pemicu"] for m in ordered), tuple(suggestions))


def recommend_outside(rules: dict[str, dict[str, Any]], key: str = OUTSIDE_KEY) -> Recommendation:
    """Saran investigasi kasus tunggal untuk titik di luar semua klaster."""
    r = rules["di_luar_klaster"]
    return Recommendation(key, r["kategori"], (r["pemicu"],), tuple(r["saran"]))


def cluster_metric_map(profile: Iterable[dict[str, Any]]) -> dict[int, list[str]]:
    """Nomor klaster → metrik yang melewati ambang ITU (dari `metrik_melewati_ambang` di profil)."""
    return {int(c["id_klaster"]): list(c.get("metrik_melewati_ambang", [])) for c in profile}


def cluster_recommendations(profile: Iterable[dict[str, Any]], rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Saran per klaster pada profil model: klaster yang ada metrik melewati ambang; Noise latih → investigasi kasus tunggal."""
    out = []
    for c in profile:
        cid = int(c["id_klaster"])
        rec = recommend_outside(rules, "Noise") if cid == NOISE_ID else recommend_metrics(c.get("metrik_melewati_ambang", []), rules, f"Klaster {cid}")
        if rec is not None:
            out.append(
                {"id_klaster": cid, "nama": c.get("nama", rec.key), "jumlah_anggota": c["jumlah_anggota"], "kategori": rec.category,
                 "pemicu": list(rec.triggers), "saran": list(rec.suggestions), "deskripsi": c.get("deskripsi", {}).get("ringkas", "")}
            )
    return out


def _state_recommendation(
    status: Any, cluster: Any, metric_map: dict[int, list[str]], rules: dict[str, dict[str, Any]]
) -> Recommendation | None:
    if status == STATUS_OUTSIDE:
        return recommend_outside(rules)
    if status == STATUS_CLUSTERED and not pd.isna(cluster):
        return recommend_metrics(metric_map.get(int(cluster), []), rules, f"Klaster {int(cluster)}")
    return None


def session_recommendations(
    rows: pd.DataFrame, metric_map: dict[int, list[str]], rules: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Saran tingkat sesi dari baris jendela: status terakhir tiap flow dikelompokkan per klaster / Di luar klaster.

    Kembalian terurut dari yang paling banyak flow-nya; tiap item: kunci, jumlah_flow, kategori, pemicu, saran.
    """
    if rows.empty:
        return []
    last = rows.groupby("flow_id", sort=False)[["status", "klaster_terdekat"]].last()
    counts: dict[str, list] = {}
    for status, cluster in zip(last["status"], last["klaster_terdekat"]):
        rec = _state_recommendation(status, cluster, metric_map, rules)
        if rec is not None:
            counts.setdefault(rec.key, [rec, 0])[1] += 1
    items = sorted(counts.values(), key=lambda rn: -rn[1])
    return [
        {"kunci": rec.key, "jumlah_flow": n, "kategori": rec.category, "pemicu": list(rec.triggers), "saran": list(rec.suggestions)}
        for rec, n in items
    ]


def call_recommendations(
    calls: pd.DataFrame, metric_map: dict[int, list[str]], rules: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Saran per panggilan dari tabel panggilan: yang klaster akhirnya punya metrik melewati ambang, atau di luar klaster."""
    out = []
    for row in calls.itertuples():
        rec = _state_recommendation(row.status_akhir, row.klaster_akhir, metric_map, rules)
        if rec is not None:
            out.append(
                {"flow_id": row.flow_id, "kunci": rec.key, "kategori": rec.category, "pemicu": list(rec.triggers), "saran": list(rec.suggestions)}
            )
    return out
