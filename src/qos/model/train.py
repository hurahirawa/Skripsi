"""Pelatihan HDBSCAN: grid search, evaluasi, profil, dan penyimpanan artefak versi (PRD FR-TR)."""
from __future__ import annotations

import itertools
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

import hdbscan
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from qos.config import resolve_path
from qos.data.loader import filter_voip, load_flows
from qos.features.extract import derive_metrics
from qos.features.preprocess import apply_session_filters, build_features, clean_features, fit_scaler
from qos.model.evaluate import evaluate_clustering, silhouette_label
from qos.profiling.profile import build_cluster_profile, label_map

logger = logging.getLogger(__name__)

VERSION_FORMAT = "%Y%m%d_%H%M%S"
CURRENT_FILE = "current.txt"
COUNT_COLUMNS: tuple[str, ...] = ("fwd_packets_count", "bwd_packets_count")
ARTIFACT_FILES: tuple[str, ...] = (
    "hdbscan_model.joblib",
    "scaler.joblib",
    "pca.joblib",
    "cluster_profile.json",
    "train_meta.json",
    "grid_results.csv",
    "training_sample.csv",
)


def fit_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int, method: str) -> hdbscan.HDBSCAN:
    """Fit HDBSCAN (paket hdbscan, euclidean, prediction_data=True) — PRD TR-1."""
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=method,
        prediction_data=True,
    )
    return model.fit(X)


TIE_BREAK_RULE = (
    "Silhouette tertinggi; seri → CHI tertinggi; seri → min_cluster_size terkecil; "
    "seri → min_samples terkecil (kandidat identik menghasilkan klaster yang sama)"
)
TIE_DECIMALS = 9  # selisih di bawah 1e-9 dianggap seri


def rank_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Kandidat valid terurut menurut TIE_BREAK_RULE (terbaik di baris pertama)."""
    valid = results[results["valid"]].copy()
    valid["_sil"] = valid["silhouette"].round(TIE_DECIMALS)
    valid["_chi"] = valid["chi"].round(TIE_DECIMALS)
    ranked = valid.sort_values(
        ["_sil", "_chi", "min_cluster_size", "min_samples"], ascending=[False, False, True, True], kind="stable"
    )
    return ranked.drop(columns=["_sil", "_chi"])


def tied_with_best(results: pd.DataFrame) -> list[tuple[int, int]]:
    """Pasangan (min_cluster_size, min_samples) yang seri penuh (Silhouette dan CHI) dengan kandidat terbaik."""
    ranked = rank_candidates(results)
    if ranked.empty:
        return []
    best = ranked.iloc[0]
    same = ranked[
        (ranked["silhouette"].round(TIE_DECIMALS) == round(best["silhouette"], TIE_DECIMALS))
        & (ranked["chi"].round(TIE_DECIMALS) == round(best["chi"], TIE_DECIMALS))
    ]
    return [(int(a), int(b)) for a, b in zip(same["min_cluster_size"], same["min_samples"])]


def pick_best(results: pd.DataFrame) -> pd.Series | None:
    """Kandidat terbaik menurut TIE_BREAK_RULE; None jika tak ada kandidat valid."""
    ranked = rank_candidates(results)
    return None if ranked.empty else ranked.iloc[0]


def grid_search(
    X: np.ndarray, cfg: dict[str, Any], progress: Callable[[int, int], None] | None = None
) -> pd.DataFrame:
    """Grid min_cluster_size × min_samples (PRD TR-2); kolom `valid` = n_klaster ≥ 2 dan noise ≤ batas.

    `progress(selesai, total)` dipanggil setelah tiap kandidat (untuk progress bar UI).
    """
    hcfg, rs = cfg["hdbscan"], cfg["data"]["random_state"]
    grid = hcfg["grid_search"]
    rows = []
    combos = list(itertools.product(grid["min_cluster_size"], grid["min_samples"]))
    for i, (mcs, ms) in enumerate(combos, start=1):
        labels = fit_hdbscan(X, mcs, ms, hcfg["cluster_selection_method"]).labels_
        ev = evaluate_clustering(X, labels, grid["silhouette_sample_size"], rs)
        valid = ev["n_clusters"] >= 2 and ev["noise_ratio"] <= grid["max_noise_ratio"]
        rows.append({"min_cluster_size": mcs, "min_samples": ms, **ev, "valid": bool(valid)})
        if progress is not None:
            progress(i, len(combos))
    return pd.DataFrame(rows)


def _jsonable(obj: Any) -> Any:
    """Ubah numpy/NaN menjadi tipe JSON valid (NaN → null)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def load_training_frame(source: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Muat CSV → filter VoIP → exclude traffic type + sesi minimum; kembalikan juga hitungan tiap tahap."""
    data_cfg = cfg["data"]
    raw = load_flows(source, data_cfg)
    voip = filter_voip(raw, data_cfg["voip_filter"])
    final = apply_session_filters(voip, data_cfg)
    return final, {"total_flow": len(raw), "voip": len(voip), "setelah_filter_sesi": len(final)}


def train(
    flows: pd.DataFrame,
    cfg: dict[str, Any],
    use_grid: bool | None = None,
    counts: dict[str, int] | None = None,
    progress: Callable[[int, int], None] | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Latih model dari flow yang sudah difilter; `source` (asal data) dicatat di train_meta.json. Mengembalikan semua objek hasil pelatihan (belum disimpan)."""
    data_cfg, feat, hcfg, ev_cfg = cfg["data"], cfg["features"], cfg["hdbscan"], cfg["evaluation"]
    rs = data_cfg["random_state"]
    use_grid = hcfg["grid_search"]["enabled"] if use_grid is None else use_grid
    columns, log_columns = feat["cluster_columns"], feat["log_transform"]
    # cleaning pada kolom mentah; fitur turunan (delay_ms, jitter_ms, fai_pct) dihitung sesudahnya
    base = list(dict.fromkeys([feat["delay_column"], feat["jitter_column"], *COUNT_COLUMNS]))

    clean, medians = clean_features(flows, base, data_cfg["max_train_rows"], rs)
    clean = derive_metrics(clean, feat["delay_column"], feat["jitter_column"], feat["fai_clip_negative"])
    features = build_features(clean, columns, log_columns)
    scaler = fit_scaler(features)
    X = scaler.transform(features)

    grid = None
    fallback = False
    tied: list[tuple[int, int]] = []
    if use_grid:
        grid = grid_search(X, cfg, progress)
        best = pick_best(grid)
        tied = tied_with_best(grid)
        if best is None:
            fallback = True
            logger.warning("Tidak ada kandidat grid yang valid; memakai parameter tetap dari config")
            mcs, ms = hcfg["min_cluster_size"], hcfg["min_samples"]
        else:
            mcs, ms = int(best["min_cluster_size"]), int(best["min_samples"])
    else:
        mcs, ms = hcfg["min_cluster_size"], hcfg["min_samples"]

    model = fit_hdbscan(X, mcs, ms, hcfg["cluster_selection_method"])
    labels = model.labels_
    evaluation = evaluate_clustering(X, labels, hcfg["grid_search"]["silhouette_sample_size"], rs)
    evaluation["silhouette_label"] = silhouette_label(evaluation["silhouette"], ev_cfg["silhouette_bands"])
    profile = build_cluster_profile(clean, labels, cfg)

    pca = PCA(n_components=2, random_state=rs).fit(X)
    pcs = pca.transform(X)
    sample = clean.assign(cluster_id=labels, membership_prob=model.probabilities_, pc1=pcs[:, 0], pc2=pcs[:, 1])
    if len(sample) > ev_cfg["pca_sample_size"]:
        sample = sample.sample(n=ev_cfg["pca_sample_size"], random_state=rs)

    return {
        "model": model, "scaler": scaler, "frame": clean, "pca": pca, "grid": grid, "profile": profile,
        "evaluation": evaluation, "sample": sample.reset_index(drop=True), "medians": medians,
        "params": {"min_cluster_size": mcs, "min_samples": ms, "grid_search": bool(use_grid),
                   "grid_fallback": fallback, "tie_break_rule": TIE_BREAK_RULE,
                   "kandidat_seri_penuh": tied},
        "n_train_rows": int(len(clean)), "counts": counts or {}, "labels": labels, "source": source or {},
    }


def allocate_version(root: Path) -> tuple[str, Path]:
    """Buat folder versi baru artifacts/<YYYYMMDD_HHMMSS>/; tidak pernah memakai folder yang sudah ada."""
    version = datetime.now().strftime(VERSION_FORMAT)
    while (root / version).exists():  # nama versi berdetak per detik; tunggu detik berikutnya
        time.sleep(0.2)
        version = datetime.now().strftime(VERSION_FORMAT)
    out = root / version
    out.mkdir(parents=True, exist_ok=False)
    return version, out


def save_artifacts(result: dict[str, Any], cfg: dict[str, Any], activate: bool | None = None) -> Path:
    """Simpan artefak ke artifacts/<YYYYMMDD_HHMMSS>/ (PRD §10.1) dan perbarui current.txt bila perlu.

    `activate=None`: aktifkan hanya jika belum ada versi aktif.
    """
    root = resolve_path(cfg["paths"]["artifacts_dir"])
    version, out = allocate_version(root)

    joblib.dump(result["model"], out / "hdbscan_model.joblib")
    joblib.dump(result["scaler"], out / "scaler.joblib")
    joblib.dump(result["pca"], out / "pca.joblib")
    profile_doc = {
        "model_version": version,
        "thresholds": cfg["thresholds"],
        "evaluation": result["evaluation"],
        "clusters": result["profile"],
        "label_map": label_map(result["profile"]),
    }
    meta = {
        "model_version": version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "parameters": result["params"],
        "cluster_selection_method": cfg["hdbscan"]["cluster_selection_method"],
        "cluster_columns": cfg["features"]["cluster_columns"],
        "log_transform": cfg["features"]["log_transform"],
        "kolom_dasar_imputasi": list(result["medians"]),
        "thresholds": cfg["thresholds"],
        "median_imputasi": result["medians"],
        "jumlah_baris_latih": result["n_train_rows"],
        "filter_voip": {**cfg["data"]["voip_filter"], "jumlah_flow": result["counts"]},
        "filter_sesi": {"min_session": cfg["data"]["min_session"],
                        "exclude_traffic_types": cfg["data"]["exclude_traffic_types"]},
        "fai_clip_negative": cfg["features"]["fai_clip_negative"],
        "random_state": cfg["data"]["random_state"],
        "sumber_data": result.get("source", {}),
    }
    (out / "cluster_profile.json").write_text(json.dumps(_jsonable(profile_doc), indent=2, ensure_ascii=False), "utf-8")
    (out / "train_meta.json").write_text(json.dumps(_jsonable(meta), indent=2, ensure_ascii=False), "utf-8")
    grid = result["grid"]
    (grid if grid is not None else pd.DataFrame()).to_csv(out / "grid_results.csv", index=False)
    result["sample"].to_csv(out / "training_sample.csv", index=False)

    current = root / CURRENT_FILE
    if activate or (activate is None and not current.is_file()):
        current.write_text(version, encoding="utf-8")
    logger.info("Artefak disimpan di %s (aktif=%s)", out, current.is_file() and current.read_text("utf-8").strip() == version)
    return out
