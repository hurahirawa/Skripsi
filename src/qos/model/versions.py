"""Versioning artefak: daftar versi, aktivasi (current.txt), dan Terapkan Ulang Pelabelan (tanpa refit)."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from qos.config import resolve_path
from qos.model.train import CURRENT_FILE, _jsonable, allocate_version
from qos.profiling.labeling import NOISE_ID, apply_labels
from qos.profiling.profile import label_map

REQUIRED_FILES: tuple[str, ...] = (
    "hdbscan_model.joblib", "scaler.joblib", "pca.joblib", "cluster_profile.json", "train_meta.json",
)
COPY_FILES: tuple[str, ...] = ("hdbscan_model.joblib", "scaler.joblib", "pca.joblib", "grid_results.csv", "training_sample.csv")
THRESHOLD_KEYS: tuple[str, ...] = ("delay_good_ms", "delay_max_ms", "jitter_max_ms", "loss_max_pct")


class VersionError(RuntimeError):
    """Versi tidak ditemukan, tidak lengkap, atau operasi ditolak."""


def _root(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["paths"]["artifacts_dir"])


def active_version(cfg: dict[str, Any]) -> str | None:
    """Versi di current.txt, atau None."""
    current = _root(cfg) / CURRENT_FILE
    return current.read_text("utf-8").strip() if current.is_file() else None


def is_complete(folder: Path) -> bool:
    """True jika semua berkas wajib ada."""
    return all((folder / n).is_file() for n in REQUIRED_FILES)


def list_versions(cfg: dict[str, Any]) -> pd.DataFrame:
    """Tabel semua versi lengkap (terbaru dulu): skor, asal data, versi induk bila hasil relabel, dan penanda aktif."""
    root = _root(cfg)
    active = active_version(cfg)
    rows = []
    for folder in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True) if root.is_dir() else []:
        if not is_complete(folder):
            continue
        profile = json.loads((folder / "cluster_profile.json").read_text("utf-8"))
        meta = json.loads((folder / "train_meta.json").read_text("utf-8"))
        ev = profile["evaluation"]
        src = meta.get("sumber_data", {}).get("jenis", "csv" if meta.get("filter_voip") else "-")
        rows.append(
            {
                "versi": folder.name, "aktif": folder.name == active, "waktu": meta.get("timestamp"),
                "sumber": "relabel" if "relabeled_from" in meta else src,
                "induk": meta.get("relabeled_from"), "klaster": ev["n_clusters"],
                "silhouette": ev["silhouette"], "chi": ev["chi"], "noise_ratio": ev["noise_ratio"],
            }
        )
    return pd.DataFrame(rows, columns=["versi", "aktif", "waktu", "sumber", "induk", "klaster", "silhouette", "chi", "noise_ratio"])


def activate_version(cfg: dict[str, Any], version: str, monitoring_active: bool = False) -> Path:
    """Jadikan `version` aktif (tulis current.txt secara atomik). Ditolak bila versi tak lengkap atau monitoring aktif."""
    if monitoring_active:
        raise VersionError("Mengganti model aktif ditolak selama monitoring aktif. Hentikan sesi dulu.")
    folder = _root(cfg) / version
    if not folder.is_dir():
        raise VersionError(f"Versi {version} tidak ditemukan")
    if not is_complete(folder):
        raise VersionError(f"Versi {version} tidak lengkap; kurang: {[n for n in REQUIRED_FILES if not (folder / n).is_file()]}")
    current = _root(cfg) / CURRENT_FILE
    tmp = current.with_suffix(".tmp")
    tmp.write_text(version, encoding="utf-8")
    os.replace(tmp, current)
    return current


def validate_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Pastikan semua kunci ambang ada, ≥ 0, dan delay baik ≤ delay maks."""
    missing = [k for k in THRESHOLD_KEYS if k not in thresholds]
    if missing:
        raise VersionError(f"Ambang tidak lengkap, kurang: {missing}")
    out = {k: float(thresholds[k]) for k in THRESHOLD_KEYS}
    if any(v < 0 for v in out.values()):
        raise VersionError("Ambang tidak boleh negatif")
    if out["delay_good_ms"] > out["delay_max_ms"]:
        raise VersionError("delay_good_ms tidak boleh lebih besar dari delay_max_ms")
    return out


def relabel_profile(profile_doc: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """Hitung ulang tingkat, label, kategori tindakan, dan metrik_penyebab dari `statistik_pelabelan` tersimpan.

    Murni: tidak membaca data latih dan tidak menyentuh model (tanpa refit HDBSCAN).
    """
    th = validate_thresholds(thresholds)
    clusters = []
    for c in profile_doc["clusters"]:
        labeled = apply_labels(c["statistik_pelabelan"], th, noise=int(c["id_klaster"]) == NOISE_ID)
        clusters.append({**c, **labeled})
    return {**profile_doc, "thresholds": th, "clusters": clusters, "label_map": label_map(clusters)}


def label_changes(old_doc: dict[str, Any], new_doc: dict[str, Any]) -> pd.DataFrame:
    """Tabel klaster dengan label lama vs baru dan penanda `berubah`."""
    new = {c["id_klaster"]: c for c in new_doc["clusters"]}
    return pd.DataFrame(
        [
            {"klaster": c["id_klaster"], "label_lama": c["label"], "label_baru": new[c["id_klaster"]]["label"],
             "berubah": c["label"] != new[c["id_klaster"]]["label"]}
            for c in old_doc["clusters"]
        ]
    )


def relabel_version(cfg: dict[str, Any], version: str, thresholds: dict[str, float]) -> Path:
    """Terapkan Ulang Pelabelan: buat versi BARU dari `version` dengan ambang baru.

    Model, scaler, PCA, dan statistik tidak berubah (disalin); hanya label yang dihitung ulang.
    Versi lama dan current.txt tidak disentuh; aktivasi dilakukan terpisah.
    """
    root = _root(cfg)
    src = root / version
    if not is_complete(src):
        raise VersionError(f"Versi {version} tidak ditemukan atau tidak lengkap")
    old = json.loads((src / "cluster_profile.json").read_text("utf-8"))
    new_doc = relabel_profile(old, thresholds)
    meta = json.loads((src / "train_meta.json").read_text("utf-8"))

    new_version, out = allocate_version(root)
    for name in COPY_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, out / name)
    new_doc["model_version"] = new_version
    meta.update(
        {
            "model_version": new_version,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "relabeled_from": version,
            "thresholds_sebelum": old["thresholds"],
            "thresholds": new_doc["thresholds"],
        }
    )
    (out / "cluster_profile.json").write_text(json.dumps(_jsonable(new_doc), indent=2, ensure_ascii=False), "utf-8")
    (out / "train_meta.json").write_text(json.dumps(_jsonable(meta), indent=2, ensure_ascii=False), "utf-8")
    return out
