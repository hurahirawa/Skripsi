"""Latih ulang model (PRD TR-5, TR-6, UC-06): sumber CSV CIC/Darknet atau log jendela monitoring.

Latih ulang SELALU membuat versi baru dan tidak pernah menimpa artefak lama maupun current.txt;
mengaktifkan versi baru adalah langkah terpisah (qos.model.versions.activate_version).
"""
from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qos.config import resolve_path
from qos.features.preprocess import session_mask
from qos.model.train import COUNT_COLUMNS, load_training_frame, save_artifacts, train
from qos.monitor.calls import INSUFFICIENT, SIGNALING

logger = logging.getLogger(__name__)

SOURCES: tuple[str, ...] = ("csv", "log")
LOG_GLOB = "jendela_*.csv"
LOG_COLUMNS: tuple[str, ...] = ("id_sesi", "flow_id", "delay_ms", "jitter_ms", "fwd", "bwd")
OPTIONAL_LOG_COLUMNS: tuple[str, ...] = ("status", "label", "di_luar_klaster", "feature_mode", "diimputasi")  # tidak ada di CSV lama → "tidak diketahui"
FEATURE_MODES: tuple[str, ...] = ("window", "cumulative")
BOOL_MAP = {True: True, False: False, "True": True, "False": False, "true": True, "false": False}
US_PER_MS = 1000.0


class RetrainError(RuntimeError):
    """Latih ulang ditolak atau gagal."""


class InsufficientDataError(RetrainError):
    """Jumlah baris latih di bawah minimum."""


class MonitoringActiveError(RetrainError):
    """Latih ulang ditolak selama monitoring aktif (PRD TR-5)."""


def list_log_files(cfg: dict[str, Any]) -> list[Path]:
    """CSV jendela hasil monitoring di <reports_dir>/riwayat."""
    folder = resolve_path(cfg["paths"]["reports_dir"]) / "riwayat"
    return sorted(folder.glob(LOG_GLOB)) if folder.is_dir() else []


def load_log_frame(cfg: dict[str, Any], files: list[Path] | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Ubah log jendela monitoring menjadi tabel latih (kolom mentah iat_mean_us, iat_std_us, fwd, bwd).

    Urutan: buang "Data Belum Cukup" (wajib), mode fitur selain `feature_mode` (CSV lama tanpa kolom = tidak
    diketahui → dibuang), baris diimputasi (opsi), di luar klaster (opsi), satu baris per flow bila per_flow=last,
    lalu filter sesi minimum pelatihan. Jumlah minimum baris (`min_rows`) dicek SEBELUM pelatihan.
    Kembalian: (tabel latih, hitungan tiap tahap).
    """
    opt = cfg["retrain"]["log_source"]
    if not opt["exclude_insufficient"]:
        raise RetrainError(
            "retrain.log_source.exclude_insufficient harus true: baris 'Data Belum Cukup' tidak punya fitur valid."
        )
    if opt["per_flow"] not in ("last", "all"):
        raise RetrainError(f"retrain.log_source.per_flow tidak dikenal: {opt['per_flow']!r} (last | all)")
    files = files if files is not None else list_log_files(cfg)
    if not files:
        raise InsufficientDataError("Tidak ada log jendela monitoring (reports/riwayat/jendela_*.csv).")

    mode = opt["feature_mode"]
    if mode is not None and mode not in FEATURE_MODES:
        raise RetrainError(f"retrain.log_source.feature_mode tidak dikenal: {mode!r} ({list(FEATURE_MODES)} atau null)")
    wanted = (*LOG_COLUMNS, *OPTIONAL_LOG_COLUMNS)
    raw = pd.concat([pd.read_csv(f, usecols=lambda c: c in wanted) for f in files], ignore_index=True)
    missing = [c for c in LOG_COLUMNS if c not in raw.columns]
    if "status" not in raw.columns and "label" not in raw.columns:
        missing.append("status")
    if missing:
        raise RetrainError(f"Log jendela tidak punya kolom {missing}")
    if "status" not in raw.columns:  # log lama (pelabelan QoS): kolom `label` memuat status Data Belum Cukup / Sinyal SIP
        raw["status"] = raw["label"]
    for col in OPTIONAL_LOG_COLUMNS:  # CSV lama: kolom tidak ada → nilai tidak diketahui (NaN)
        if col not in raw.columns:
            raw[col] = np.nan
    counts = {"baris_log": len(raw)}
    df = raw[~raw["status"].isin([INSUFFICIENT, SIGNALING])]  # keduanya tanpa fitur valid
    counts["setelah_buang_data_belum_cukup"] = len(df)
    if mode is not None:
        unknown = df["feature_mode"].isna()
        other = ~unknown & (df["feature_mode"] != mode)
        counts["dikecualikan_mode_tidak_diketahui"] = int(unknown.sum())
        counts["dikecualikan_mode_lain"] = int(other.sum())
        df = df[~unknown & ~other]
    imputed = df["diimputasi"].map(BOOL_MAP)
    if opt["exclude_imputed"]:
        counts["dikecualikan_diimputasi"] = int((imputed == True).sum())  # noqa: E712
        counts["dikecualikan_imputasi_tidak_diketahui"] = int(imputed.isna().sum())
        df = df[imputed == False]  # noqa: E712 — hanya baris yang pasti tidak diimputasi
    counts["setelah_mode_dan_imputasi"] = len(df)
    if opt["exclude_outside_cluster"]:
        outside = df["di_luar_klaster"].map(BOOL_MAP) if "di_luar_klaster" in df else pd.Series(False, index=df.index)
        df = df[outside != True]  # noqa: E712 — log lama tanpa kolom ini: tidak ada yang dibuang
    counts["setelah_buang_di_luar_klaster"] = len(df)
    if opt["per_flow"] == "last":
        df = df.groupby(["id_sesi", "flow_id"], sort=False).tail(1)
    counts["setelah_per_flow"] = len(df)
    ms = cfg["data"]["min_session"]
    df = df.rename(columns={"fwd": COUNT_COLUMNS[0], "bwd": COUNT_COLUMNS[1]})
    df = df[session_mask(df, ms["fwd"], ms["bwd"])]
    counts["setelah_filter_sesi"] = len(df)

    if len(df) < opt["min_rows"]:
        raise InsufficientDataError(
            f"Data log tidak cukup: {len(df)} baris < min_rows ({opt['min_rows']}) "
            f"(baris log {counts['baris_log']} -> tanpa 'Data Belum Cukup' {counts['setelah_buang_data_belum_cukup']}"
            f" -> mode/imputasi {counts['setelah_mode_dan_imputasi']} (mode tidak diketahui "
            f"{counts.get('dikecualikan_mode_tidak_diketahui', 0)}, mode lain {counts.get('dikecualikan_mode_lain', 0)}, "
            f"diimputasi {counts.get('dikecualikan_diimputasi', 0)})"
            f" -> tanpa di luar klaster {counts['setelah_buang_di_luar_klaster']} -> per flow {counts['setelah_per_flow']}"
            f" -> filter sesi {counts['setelah_filter_sesi']})."
        )
    feat = cfg["features"]
    table = pd.DataFrame(
        {
            feat["delay_column"]: df["delay_ms"].to_numpy() * US_PER_MS,
            feat["jitter_column"]: df["jitter_ms"].to_numpy() * US_PER_MS,
            COUNT_COLUMNS[0]: df[COUNT_COLUMNS[0]].to_numpy(),
            COUNT_COLUMNS[1]: df[COUNT_COLUMNS[1]].to_numpy(),
        }
    ).reset_index(drop=True)
    return table, counts


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


def retrain(
    cfg: dict[str, Any],
    source: str,
    params: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
    monitoring_active: bool = False,
    csv_path: str | Path | None = None,
    log_files: list[Path] | None = None,
) -> dict[str, Any]:
    """Latih model baru dan simpan sebagai versi baru (TIDAK diaktifkan).

    Kembalian: version, path, evaluation, params, profile, counts, source.
    """
    if monitoring_active:
        raise MonitoringActiveError("Latih ulang ditolak selama monitoring aktif. Hentikan sesi (Stop) dulu.")
    if source not in SOURCES:
        raise RetrainError(f"Sumber tidak dikenal: {source!r} (pilihan: {list(SOURCES)})")
    c = apply_overrides(cfg, params)
    if source == "csv":
        path = resolve_path(csv_path or c["data"]["raw_dir"])
        flows, counts = load_training_frame(path, c)
        info = {"jenis": "csv", "lokasi": str(path)}
    else:
        flows, counts = load_log_frame(c, log_files)
        files = log_files if log_files is not None else list_log_files(c)
        info = {"jenis": "log", "berkas": [str(f) for f in files], "opsi": c["retrain"]["log_source"]}
    result = train(flows, c, counts=counts, progress=progress, source={**info, "jumlah_baris": counts})
    out = save_artifacts(result, c, activate=False)  # tidak pernah mengaktifkan / menimpa current.txt
    logger.info("Latih ulang (%s) selesai: versi %s", source, out.name)
    return {
        "version": out.name, "path": out, "evaluation": result["evaluation"], "params": result["params"],
        "profile": result["profile"], "counts": counts, "source": info,
    }
