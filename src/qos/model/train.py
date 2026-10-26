"""Pelatihan klasterisasi murni: HDBSCAN (aturan seleksi final), pembanding K-Means, centroid, batas jarak, profil, artefak (PRD FR-TR)."""
from __future__ import annotations

import itertools
import json
import logging
import math
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import hdbscan
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from qos.analysis import clustering as cl
from qos.analysis.final import centroids_original_units
from qos.config import resolve_path
from qos.data.loader import filter_voip, load_flows
from qos.features.extract import derive_metrics
from qos.features.preprocess import apply_session_filters, build_features, clean_features, fit_scaler
from qos.model.assignment import RULE_TEXT, assign, centroids_of, distance_limits
from qos.model.evaluate import evaluate_clustering, silhouette_label
from qos.profiling.profile import build_cluster_profile

logger = logging.getLogger(__name__)

VERSION_FORMAT = "%Y%m%d_%H%M%S"
CURRENT_FILE = "current.txt"
SCHEMA = "klaster_murni"       # penanda artefak; versi lama (pelabelan QoS) tidak punya penanda ini
SCHEMA_VERSION = 2
COUNT_COLUMNS: tuple[str, ...] = ("fwd_packets_count", "bwd_packets_count")
ARTIFACT_FILES: tuple[str, ...] = (
    "hdbscan_model.joblib",
    "kmeans_n.joblib",
    "kmeans_opt.joblib",
    "scaler.joblib",
    "pca.joblib",
    "assignment.json",
    "cluster_profile.json",
    "train_meta.json",
    "grid_results.csv",
    "training_sample.csv",
)
SELECTION_RULE = (
    "Kandidat valid = noise <= batas, klaster >= minimum, ARI bootstrap rata-rata >= minimum; "
    "pilih DBCV tertinggi; seri -> Silhouette tertinggi; seri -> min_cluster_size lalu min_samples terkecil"
)


class TrainError(RuntimeError):
    """Pelatihan tidak menghasilkan klaster yang dapat dipakai."""


def fit_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int, method: str, prediction_data: bool = True) -> hdbscan.HDBSCAN:
    """Fit HDBSCAN (paket hdbscan, euclidean) — PRD TR-1."""
    return hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size), min_samples=int(min_samples), metric="euclidean",
        cluster_selection_method=method, prediction_data=prediction_data,
    ).fit(X)


def select_final(grid: pd.DataFrame, sel: dict[str, Any]) -> tuple[pd.Series | None, pd.DataFrame]:
    """Aturan seleksi final (SELECTION_RULE): (kandidat terbaik atau None, kandidat valid terurut)."""
    ok = grid[
        (grid["noise_ratio"] <= sel["max_noise_ratio"]) & (grid["n_clusters"] >= sel["min_clusters"])
        & (grid["ari_mean"] >= sel["min_ari_mean"]) & grid["dbcv"].notna()
    ].copy()
    ok["_dbcv"], ok["_sil"] = ok["dbcv"].round(9), ok["silhouette"].round(9)
    ok = ok.sort_values(["_dbcv", "_sil", "min_cluster_size", "min_samples"], ascending=[False, False, True, True], kind="stable")
    ok = ok.drop(columns=["_dbcv", "_sil"])
    return (None if ok.empty else ok.iloc[0]), ok


def grid_search(X: np.ndarray, cfg: dict[str, Any], progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """Grid min_cluster_size × min_samples dengan metrik dan ARI bootstrap tiap kandidat; kolom `valid` menurut aturan seleksi.

    `progress(selesai, total)` dipanggil setelah tiap kandidat (untuk progress bar UI).
    """
    h, rs, ev_cfg = cfg["hdbscan"], cfg["data"]["random_state"], cfg["evaluation"]
    grid, st, sel, method = h["grid_search"], h["stability"], h["selection"], h["cluster_selection_method"]
    combos = list(itertools.product(grid["min_cluster_size"], grid["min_samples"]))
    rows = []
    for i, (mcs, ms) in enumerate(combos, start=1):
        labels = fit_hdbscan(X, mcs, ms, method, prediction_data=False).labels_
        ev = evaluate_clustering(X, labels, grid["silhouette_sample_size"], rs, ev_cfg["dbcv_max_points"])
        mean, sd = cl.stability_ari(
            lambda Xs, a=mcs, b=ms: fit_hdbscan(Xs, a, b, method, prediction_data=False).labels_,
            X, labels, st["n_bootstrap"], st["fraction"], rs,
        )
        rows.append({"min_cluster_size": mcs, "min_samples": ms, **ev, "ari_mean": mean, "ari_sd": sd})
        if progress is not None:
            progress(i, len(combos))
    out = pd.DataFrame(rows)
    _, ok = select_final(out, sel)
    out["valid"] = out.index.isin(ok.index)
    return out


def _jsonable(obj: Any) -> Any:
    """Ubah numpy/NaN menjadi tipe JSON valid (NaN → null)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
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


def _kmeans_block(X: np.ndarray, k: int, cfg: dict[str, Any], title: str) -> dict[str, Any]:
    """Fit K-Means k tertentu: model, label ber-nomor 1..k, centroid, metrik, ARI bootstrap, waktu fit."""
    km, ev_cfg, rs = cfg["kmeans"], cfg["evaluation"], cfg["data"]["random_state"]
    st = cfg["hdbscan"]["stability"]
    t0 = time.perf_counter()
    model = cl.kmeans_fit(X, k, km["n_init"], rs)
    fit_time = time.perf_counter() - t0
    labels = cl.apply_mapping(model.labels_, cl.renumber_by_size(model.labels_))
    ari = cl.stability_ari(lambda Xs: cl.kmeans_fit(Xs, k, km["n_init"], rs).labels_, X, labels, st["n_bootstrap"], st["fraction"], rs)
    ev = evaluate_clustering(X, labels, cfg["hdbscan"]["grid_search"]["silhouette_sample_size"], rs, ev_cfg["dbcv_max_points"])
    return {
        "nama": title, "k": k, "model": model, "labels": labels, "centroids": centroids_of(X, labels),
        "metrik": {**ev, "ari_mean": ari[0], "ari_sd": ari[1], "waktu_fit_s": fit_time},
    }


def train(
    flows: pd.DataFrame,
    cfg: dict[str, Any],
    use_grid: bool | None = None,
    counts: dict[str, int] | None = None,
    progress: Callable[[int, int], None] | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Latih model dari flow yang sudah difilter; `source` (asal data) dicatat di train_meta.json.

    Mengembalikan semua objek hasil pelatihan (belum disimpan). Klaster bernomor 1..k (urut ukuran menurun), noise -1.
    """
    data_cfg, feat, hcfg, ev_cfg = cfg["data"], cfg["features"], cfg["hdbscan"], cfg["evaluation"]
    rs = data_cfg["random_state"]
    use_grid = hcfg["grid_search"]["enabled"] if use_grid is None else use_grid
    columns, log_columns = feat["cluster_columns"], feat["log_transform"]
    base = list(dict.fromkeys([feat["delay_column"], feat["jitter_column"], *COUNT_COLUMNS]))

    clean, medians = clean_features(flows, base, data_cfg["max_train_rows"], rs)
    clean = derive_metrics(clean, feat["delay_column"], feat["jitter_column"], feat["fai_clip_negative"])
    features = build_features(clean, columns, log_columns)
    scaler = fit_scaler(features)
    X = scaler.transform(features)
    method = hcfg["cluster_selection_method"]

    grid, fallback = None, False
    mcs, ms = hcfg["min_cluster_size"], hcfg["min_samples"]
    if use_grid:
        grid = grid_search(X, cfg, progress)
        best, _ = select_final(grid, hcfg["selection"])
        if best is None:
            fallback = True
            logger.warning("Tidak ada kandidat grid yang memenuhi aturan seleksi; memakai parameter tetap dari config")
        else:
            mcs, ms = int(best["min_cluster_size"]), int(best["min_samples"])

    model = fit_hdbscan(X, mcs, ms, method)
    raw_labels = model.labels_
    mapping = cl.renumber_by_size(raw_labels)
    labels = cl.apply_mapping(raw_labels, mapping)
    n_h = int(len(set(labels.tolist()) - {-1}))
    if n_h < 1:
        raise TrainError(f"HDBSCAN (min_cluster_size={mcs}, min_samples={ms}) tidak menemukan klaster; semua titik noise.")

    acfg = cfg["assignment"]
    cents = centroids_of(X, labels)
    limits, global_limit = distance_limits(X, labels, cents, acfg["limit_percentile"], acfg["limit_rule"])
    asg = assign(X, cents, limits)
    sil_n = hcfg["grid_search"]["silhouette_sample_size"]
    st = hcfg["stability"]
    ari = cl.stability_ari(
        lambda Xs: fit_hdbscan(Xs, mcs, ms, method, prediction_data=False).labels_, X, labels, st["n_bootstrap"], st["fraction"], rs
    )
    evaluation = evaluate_clustering(X, labels, sil_n, rs, ev_cfg["dbcv_max_points"])
    evaluation.update(ari_mean=ari[0], ari_sd=ari[1], silhouette_label=silhouette_label(evaluation["silhouette"], ev_cfg["silhouette_bands"]))
    assigned = evaluate_clustering(X, cl.assign_noise_to_nearest(X, labels), sil_n, rs, ev_cfg["dbcv_max_points"])

    # pembanding K-Means: k = jumlah klaster HDBSCAN dan k optimal
    km = cfg["kmeans"]
    ks = list(range(km["k_min"], min(km["k_max"], len(X) - 1) + 1))
    sweep_rows = []
    for k in ks:
        m = cl.kmeans_fit(X, k, km["n_init"], rs)
        ev = evaluate_clustering(X, m.labels_, sil_n, rs, ev_cfg["dbcv_max_points"])
        sweep_rows.append({"k": k, "inertia": float(m.inertia_), **ev, "calinski_harabasz": ev["chi"]})
    sweep = pd.DataFrame(sweep_rows)
    k_opt, k_reason = cl.select_kmeans_k(sweep, km["rank_metrics"])
    kmeans_n = _kmeans_block(X, n_h, cfg, f"K-Means k = {n_h} (= klaster HDBSCAN)")
    if k_opt == n_h:
        kmeans_opt = {**kmeans_n, "nama": f"K-Means k = {k_opt} (optimal = klaster HDBSCAN)"}
    else:
        kmeans_opt = _kmeans_block(X, k_opt, cfg, f"K-Means k = {k_opt} (optimal)")

    centroids_raw = {
        int(r["klaster"]): {k: float(v) for k, v in r.items() if k != "klaster"}
        for r in centroids_original_units(cents, scaler, columns, log_columns).to_dict("records")
    }
    own = {}
    for c, centre in cents.items():
        d = np.linalg.norm(X[labels == c] - centre, axis=1)
        own[c] = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "maks": float(d.max())}
    profile = build_cluster_profile(clean, labels, cfg, centroids_raw, limits, own)

    pca = PCA(n_components=2, random_state=rs).fit(X)
    pcs = pca.transform(X)
    sample = clean.assign(
        cluster_id=labels, jarak_terdekat=asg.nearest_dist, di_luar_klaster=asg.outside,
        klaster_kmeans_n=kmeans_n["labels"], klaster_kmeans_opt=kmeans_opt["labels"], pc1=pcs[:, 0], pc2=pcs[:, 1],
    )
    if len(sample) > ev_cfg["pca_sample_size"]:
        sample = sample.sample(n=ev_cfg["pca_sample_size"], random_state=rs)

    def cent_json(c: dict[int, np.ndarray]) -> dict[str, list[float]]:
        return {str(k): v.tolist() for k, v in c.items()}

    assignment = {
        "aturan": acfg["limit_rule"], "aturan_teks": RULE_TEXT[acfg["limit_rule"]], "persentil": acfg["limit_percentile"],
        "batas_global": global_limit, "urutan_fitur": list(columns),
        "hdbscan": {"centroid": cent_json(cents), "batas": {str(k): v for k, v in limits.items()}},
        "kmeans_n": {"k": kmeans_n["k"], "centroid": cent_json(kmeans_n["centroids"])},
        "kmeans_opt": {"k": kmeans_opt["k"], "centroid": cent_json(kmeans_opt["centroids"])},
    }
    comparison = {
        "hdbscan": {"nama": "HDBSCAN (tanpa noise)", **evaluation},
        "hdbscan_assigned": {"nama": "HDBSCAN (noise di-assign ke centroid terdekat)", **assigned, "ari_mean": ari[0], "ari_sd": ari[1]},
        "kmeans_n": {"nama": kmeans_n["nama"], "k": kmeans_n["k"], **kmeans_n["metrik"]},
        "kmeans_opt": {"nama": kmeans_opt["nama"], "k": kmeans_opt["k"], **kmeans_opt["metrik"]},
    }

    return {
        "model": model, "scaler": scaler, "frame": clean, "pca": pca, "grid": grid, "profile": profile,
        "evaluation": evaluation, "comparison": comparison, "assignment": assignment,
        "kmeans_models": {"n": kmeans_n["model"], "opt": kmeans_opt["model"]},
        "kmeans_sweep": sweep.drop(columns=["calinski_harabasz"]), "k_reason": k_reason,
        "sample": sample.reset_index(drop=True), "medians": medians,
        "params": {"min_cluster_size": mcs, "min_samples": ms, "grid_search": bool(use_grid),
                   "grid_fallback": fallback, "aturan_seleksi": SELECTION_RULE, "k_kmeans_optimal": k_opt},
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
    joblib.dump(result["kmeans_models"]["n"], out / "kmeans_n.joblib")
    joblib.dump(result["kmeans_models"]["opt"], out / "kmeans_opt.joblib")
    joblib.dump(result["scaler"], out / "scaler.joblib")
    joblib.dump(result["pca"], out / "pca.joblib")
    profile_doc = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "model_version": version,
        "thresholds": cfg["thresholds"],
        "evaluation": result["evaluation"],
        "comparison": result["comparison"],
        "kmeans_sweep": result["kmeans_sweep"].to_dict("records"),
        "k_optimal_alasan": result["k_reason"],
        "clusters": result["profile"],
    }
    meta = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "model_version": version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "parameters": result["params"],
        "cluster_selection_method": cfg["hdbscan"]["cluster_selection_method"],
        "aturan_seleksi": cfg["hdbscan"]["selection"],
        "stabilitas": cfg["hdbscan"]["stability"],
        "kmeans": cfg["kmeans"],
        "penugasan": cfg["assignment"],
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
    (out / "assignment.json").write_text(json.dumps(_jsonable(result["assignment"]), indent=2, ensure_ascii=False), "utf-8")
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
