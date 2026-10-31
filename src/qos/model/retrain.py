"""Latih ulang dari unggahan CSV (PRD TR-5, TR-6, UC-06): mode Tambah atau Ganti, perbandingan, dan pemetaan klaster.

Latih ulang SELALU membuat versi baru dan tidak pernah menimpa artefak lama maupun current.txt;
mengaktifkan versi baru adalah langkah terpisah (qos.model.versions.activate_version).
Fungsi yang sama dipakai UI (halaman Pengaturan) dan CLI (scripts/train.py --upload).
"""
from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qos.config import resolve_path
from qos.features.preprocess import build_features
from qos.model.dataset import DatasetError, Prepared, prepare_dataset
from qos.model.train import save_artifacts, train
from qos.model.versions import active_version

logger = logging.getLogger(__name__)

US_PER_MS = 1000.0
COMPARE_ROWS: tuple[tuple[str, str], ...] = (
    ("Jumlah data latih", "n_data"), ("Jumlah klaster", "n_clusters"), ("Noise Ratio", "noise_ratio"),
    ("Silhouette", "silhouette"), ("Davies-Bouldin", "davies_bouldin"), ("Calinski-Harabasz", "chi"),
    ("DBCV", "dbcv"), ("ARI bootstrap", "ari_mean"),
)


class RetrainError(RuntimeError):
    """Latih ulang ditolak atau gagal."""


class InsufficientDataError(RetrainError):
    """Jumlah baris latih di bawah minimum atau tidak ada baris baru yang lolos."""


class MonitoringActiveError(RetrainError):
    """Latih ulang ditolak selama monitoring aktif (PRD TR-5)."""


def apply_overrides(cfg: dict[str, Any], params: dict[str, Any] | None) -> dict[str, Any]:
    """Salinan config dengan parameter dari `params` (kunci: min_cluster_size, min_samples, grid_enabled, thresholds,
    selection (aturan seleksi), limit_percentile, limit_rule); config asli tidak diubah."""
    out = copy.deepcopy(cfg)
    p = params or {}
    if "min_cluster_size" in p:
        out["hdbscan"]["min_cluster_size"] = int(p["min_cluster_size"])
    if "min_samples" in p:
        out["hdbscan"]["min_samples"] = int(p["min_samples"])
    if "grid_enabled" in p:
        out["hdbscan"]["grid_search"]["enabled"] = bool(p["grid_enabled"])
    if "thresholds" in p:
        out["thresholds"].update(p["thresholds"])
    if "selection" in p:
        out["hdbscan"]["selection"].update(p["selection"])
    for key in ("limit_percentile", "limit_rule"):
        if key in p:
            out["assignment"][key] = p[key]
    return out


def preview(
    cfg: dict[str, Any], uploads: list[tuple[str | Path, str]], mode: str, parent_version: str | None = None
) -> Prepared:
    """Pratinjau unggahan (format, baris per tahap, lama + baru = total) tanpa melatih. Versi induk default = versi aktif."""
    try:
        return prepare_dataset(cfg, uploads, mode, parent_version or active_version(cfg))
    except DatasetError as exc:
        raise RetrainError(str(exc)) from exc


def load_version_documents(cfg: dict[str, Any], version: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(cluster_profile.json, train_meta.json) sebuah versi."""
    folder = resolve_path(cfg["paths"]["artifacts_dir"]) / version
    return (
        json.loads((folder / "cluster_profile.json").read_text("utf-8")),
        json.loads((folder / "train_meta.json").read_text("utf-8")),
    )


def comparison_table(old_profile: dict, old_meta: dict, new_eval: dict, new_n: int, old_version: str, new_version: str) -> pd.DataFrame:
    """Tabel metrik model lama vs baru (kolom: metrik, <versi lama>, <versi baru>)."""
    old_eval = {**old_profile["evaluation"], "n_data": old_meta.get("jumlah_baris_latih")}
    new = {**new_eval, "n_data": new_n}
    rows = []
    for label, key in COMPARE_ROWS:
        rows.append({"metrik": label, f"aktif ({old_version})": old_eval.get(key), f"baru ({new_version})": new.get(key)})
    return pd.DataFrame(rows)


def map_old_to_new(old_profile: dict[str, Any], result: dict[str, Any], cfg: dict[str, Any]) -> pd.DataFrame:
    """Pemetaan klaster lama → klaster baru terdekat.

    Centroid lama (satuan asli: delay_ms, jitter_ms, fai_pct) dikonversi ke ruang fitur model baru
    (µs → kolom fitur → log1p → scaler baru) lalu dicari centroid baru terdekat (Euclidean).
    """
    feat = cfg["features"]
    columns, log_columns = feat["cluster_columns"], feat["log_transform"]
    originals = {feat["delay_column"]: ("delay_ms", US_PER_MS), feat["jitter_column"]: ("jitter_ms", US_PER_MS), "fai_pct": ("fai_pct", 1.0)}
    unsupported = [c for c in columns if c not in originals]
    if unsupported:
        raise RetrainError(f"Pemetaan klaster butuh fitur delay/jitter/fai; fitur {unsupported} tidak didukung.")
    olds = [c for c in old_profile["clusters"] if c.get("centroid_asli") and int(c["id_klaster"]) > 0]
    new_cents = {int(k): np.asarray(v) for k, v in result["assignment"]["hdbscan"]["centroid"].items()}
    new_names = {int(c["id_klaster"]): c for c in result["profile"]}
    ids = sorted(new_cents)
    rows = []
    for c in olds:
        raw = pd.DataFrame([{col: c["centroid_asli"][originals[col][0]] * originals[col][1] for col in columns}])
        x = result["scaler"].transform(build_features(raw, columns, log_columns))[0]
        dist = {k: float(np.linalg.norm(x - new_cents[k])) for k in ids}
        nearest = min(dist, key=lambda k: (dist[k], k))
        rows.append(
            {
                "klaster_lama": c["nama"], "anggota_lama": c["jumlah_anggota"], "klaster_baru": new_names[nearest]["nama"],
                "jarak": round(dist[nearest], 4), "anggota_baru": new_names[nearest]["jumlah_anggota"],
                "deskripsi_baru": new_names[nearest]["deskripsi"]["ringkas"],
            }
        )
    return pd.DataFrame(rows, columns=["klaster_lama", "anggota_lama", "klaster_baru", "jarak", "anggota_baru", "deskripsi_baru"])


def retrain_upload(
    cfg: dict[str, Any],
    uploads: list[tuple[str | Path, str]],
    mode: str,
    params: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
    monitoring_active: bool = False,
    parent_version: str | None = None,
) -> dict[str, Any]:
    """Latih model baru dari unggahan CSV dan simpan sebagai versi baru (TIDAK diaktifkan).

    mode "tambah": data_latih.csv versi induk (default: aktif) + data baru; "ganti": hanya data baru.
    Kembalian: version, path, evaluation, params, profile, data (Prepared), peringatan, perbandingan, pemetaan.
    """
    if monitoring_active:
        raise MonitoringActiveError("Latih ulang ditolak selama monitoring aktif. Hentikan sesi (Stop) dulu.")
    c = apply_overrides(cfg, params)
    parent = parent_version or active_version(c)
    prep = preview(c, uploads, mode, parent)
    if not prep.ok:
        err = InsufficientDataError if any("minimum" in p or "baris baru" in p for p in prep.problems) else RetrainError
        raise err(" | ".join(prep.problems))
    source = {"jenis": "unggah", "mode": mode, "berkas": [(r.name, r.format) for r in prep.files], "sumber": ", ".join(prep.sources)}
    result = train(prep.frame, c, progress=progress, source=source, data_meta=prep.info())
    out = save_artifacts(result, c, activate=False)  # tidak pernah mengaktifkan / menimpa current.txt
    logger.info("Latih ulang (%s, %d baris) selesai: versi %s", mode, prep.n_total, out.name)

    comparison = mapping = None
    if parent:
        old_profile, old_meta = load_version_documents(c, parent)
        comparison = comparison_table(old_profile, old_meta, result["evaluation"], result["n_train_rows"], parent, out.name)
        try:
            mapping = map_old_to_new(old_profile, result, c)
        except RetrainError as exc:
            result["peringatan"].append(str(exc))
    return {
        "version": out.name, "path": out, "evaluation": result["evaluation"], "params": result["params"],
        "profile": result["profile"], "data": prep, "peringatan": result["peringatan"], "parent": parent,
        "perbandingan": comparison, "pemetaan": mapping,
    }
