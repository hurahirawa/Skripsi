"""Versioning artefak: daftar versi, aktivasi (current.txt), dan Perbarui Deskripsi Klaster (tanpa refit)."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from qos.config import resolve_path
from qos.model.train import ARTIFACT_FILES, CURRENT_FILE, SCHEMA, _jsonable, allocate_version
from qos.profiling.profile import redescribe

LEGACY_REQUIRED: tuple[str, ...] = ("hdbscan_model.joblib", "scaler.joblib", "pca.joblib", "cluster_profile.json", "train_meta.json")
REQUIRED_FILES: tuple[str, ...] = tuple(n for n in ARTIFACT_FILES if n not in ("grid_results.csv", "training_sample.csv"))
COPY_FILES: tuple[str, ...] = tuple(n for n in ARTIFACT_FILES if n not in ("cluster_profile.json", "train_meta.json"))
THRESHOLD_KEYS: tuple[str, ...] = ("delay_good_ms", "delay_max_ms", "jitter_max_ms", "loss_good_pct", "loss_max_pct")
LEGACY_MESSAGE = (
    "Versi {v} dibuat dengan pelabelan QoS (Baik/Cukup/Buruk) dan tidak kompatibel dengan aplikasi klasterisasi murni. "
    "Latih versi baru: python scripts/train.py --activate"
)


class VersionError(RuntimeError):
    """Versi tidak ditemukan, tidak lengkap, atau operasi ditolak."""


def _root(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["paths"]["artifacts_dir"])


def active_version(cfg: dict[str, Any]) -> str | None:
    """Versi di current.txt, atau None."""
    current = _root(cfg) / CURRENT_FILE
    return current.read_text("utf-8").strip() if current.is_file() else None


def is_legacy(folder: Path) -> bool:
    """True bila artefak berasal dari pelabelan QoS lama (cluster_profile.json tanpa penanda `schema`)."""
    path = folder / "cluster_profile.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text("utf-8")).get("schema") != SCHEMA


def is_complete(folder: Path) -> bool:
    """True jika semua berkas wajib versi klasterisasi murni ada."""
    return all((folder / n).is_file() for n in REQUIRED_FILES) and not is_legacy(folder)


def list_versions(cfg: dict[str, Any]) -> pd.DataFrame:
    """Tabel semua versi (terbaru dulu): skor, asal data, penanda aktif, dan `kompatibel` (False untuk versi pelabelan QoS lama)."""
    root = _root(cfg)
    active = active_version(cfg)
    rows = []
    for folder in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True) if root.is_dir() else []:
        if not all((folder / n).is_file() for n in LEGACY_REQUIRED):
            continue
        profile = json.loads((folder / "cluster_profile.json").read_text("utf-8"))
        meta = json.loads((folder / "train_meta.json").read_text("utf-8"))
        ev = profile["evaluation"]
        src = meta.get("sumber_data", {}).get("jenis", "csv" if meta.get("filter_voip") else "-")
        rows.append(
            {
                "versi": folder.name, "aktif": folder.name == active, "waktu": meta.get("timestamp"),
                "sumber": "deskripsi diperbarui" if "deskripsi_dari" in meta else src,
                "induk": meta.get("deskripsi_dari"), "klaster": ev["n_clusters"],
                "silhouette": ev["silhouette"], "dbcv": ev.get("dbcv"), "chi": ev["chi"], "noise_ratio": ev["noise_ratio"],
                "kompatibel": is_complete(folder),
            }
        )
    return pd.DataFrame(
        rows, columns=["versi", "aktif", "waktu", "sumber", "induk", "klaster", "silhouette", "dbcv", "chi", "noise_ratio", "kompatibel"]
    )


def activate_version(cfg: dict[str, Any], version: str, monitoring_active: bool = False) -> Path:
    """Jadikan `version` aktif (tulis current.txt secara atomik).

    Ditolak bila versi tak ditemukan, berasal dari pelabelan QoS lama, tidak lengkap, atau monitoring aktif.
    """
    if monitoring_active:
        raise VersionError("Mengganti model aktif ditolak selama monitoring aktif. Hentikan sesi dulu.")
    folder = _root(cfg) / version
    if not folder.is_dir():
        raise VersionError(f"Versi {version} tidak ditemukan")
    if is_legacy(folder):
        raise VersionError(LEGACY_MESSAGE.format(v=version))
    if not is_complete(folder):
        raise VersionError(f"Versi {version} tidak lengkap; kurang: {[n for n in REQUIRED_FILES if not (folder / n).is_file()]}")
    current = _root(cfg) / CURRENT_FILE
    tmp = current.with_suffix(".tmp")
    tmp.write_text(version, encoding="utf-8")
    os.replace(tmp, current)
    return current


def validate_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Pastikan semua kunci ambang ada, ≥ 0, dan batas baik ≤ batas maksimum."""
    missing = [k for k in THRESHOLD_KEYS if k not in thresholds]
    if missing:
        raise VersionError(f"Ambang tidak lengkap, kurang: {missing}")
    out = {k: float(thresholds[k]) for k in THRESHOLD_KEYS}
    if any(v < 0 for v in out.values()):
        raise VersionError("Ambang tidak boleh negatif")
    if out["delay_good_ms"] > out["delay_max_ms"]:
        raise VersionError("delay_good_ms tidak boleh lebih besar dari delay_max_ms")
    if out["loss_good_pct"] > out["loss_max_pct"]:
        raise VersionError("loss_good_pct tidak boleh lebih besar dari loss_max_pct")
    return out


def redescribe_profile(profile_doc: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """Hitung ulang deskripsi ITU dari `statistik` tersimpan. Murni: tanpa data latih, tanpa menyentuh model."""
    th = validate_thresholds(thresholds)
    return {**profile_doc, "thresholds": th, "clusters": redescribe(profile_doc["clusters"], th)}


def description_changes(old_doc: dict[str, Any], new_doc: dict[str, Any]) -> pd.DataFrame:
    """Tabel klaster dengan deskripsi ringkas lama vs baru dan penanda `berubah`."""
    new = {c["id_klaster"]: c for c in new_doc["clusters"]}
    return pd.DataFrame(
        [
            {
                "klaster": c["nama"], "deskripsi_lama": c["deskripsi"]["ringkas"],
                "deskripsi_baru": new[c["id_klaster"]]["deskripsi"]["ringkas"],
                "berubah": c["deskripsi"]["ringkas"] != new[c["id_klaster"]]["deskripsi"]["ringkas"],
            }
            for c in old_doc["clusters"]
        ]
    )


def refresh_descriptions(cfg: dict[str, Any], version: str, thresholds: dict[str, float]) -> Path:
    """Perbarui Deskripsi Klaster: buat versi BARU dari `version` dengan ambang ITU baru.

    Model, scaler, PCA, centroid, dan statistik tidak berubah (disalin); hanya deskripsi yang dihitung ulang.
    Versi lama dan current.txt tidak disentuh; aktivasi dilakukan terpisah.
    """
    root = _root(cfg)
    src = root / version
    if is_legacy(src):
        raise VersionError(LEGACY_MESSAGE.format(v=version))
    if not is_complete(src):
        raise VersionError(f"Versi {version} tidak ditemukan atau tidak lengkap")
    old = json.loads((src / "cluster_profile.json").read_text("utf-8"))
    new_doc = redescribe_profile(old, thresholds)
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
            "deskripsi_dari": version,
            "thresholds_sebelum": old["thresholds"],
            "thresholds": new_doc["thresholds"],
        }
    )
    (out / "cluster_profile.json").write_text(json.dumps(_jsonable(new_doc), indent=2, ensure_ascii=False), "utf-8")
    (out / "train_meta.json").write_text(json.dumps(_jsonable(meta), indent=2, ensure_ascii=False), "utf-8")
    return out
