"""Bagian G analisis klasterisasi: aturan pemilihan HDBSCAN final (ARI bootstrap semua kandidat), perbandingan, profil, batas jarak.

Menambahkan bagian G di akhir reports/experiments/cluster_analysis.md (A-F tidak diubah; menjalankan ulang hanya mengganti bagian G).
Tidak mengubah aplikasi, model aktif, config default, atau artefak. Aturan dari config/analysis_config.yaml (analysis.final.selection).

Pemakaian: python scripts/cluster_analysis_selection.py [--config config/analysis_config.yaml] [--main-config config/default_config.yaml]
"""
from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qos.analysis import clustering as cl  # noqa: E402
from qos.analysis import plots  # noqa: E402
from qos.analysis.data import prepare  # noqa: E402
from qos.analysis.final import (  # noqa: E402
    centroids_original_units,
    distance_rules,
    recommend_rule,
    select_final_hdbscan,
)
from qos.analysis.report import md_table  # noqa: E402
from qos.analysis.run import RunResult, _build_config, analyze  # noqa: E402
from qos.config import load_config, resolve_path, setup_logging  # noqa: E402
from qos.data.loader import filter_voip, load_flows  # noqa: E402
from cluster_analysis import REPORT_NAME, load_analysis_config  # noqa: E402
from cluster_analysis_final import hdbscan_config, metric_row, profile_table, timing_md  # noqa: E402


def grid_with_ari(res: RunResult, acfg: dict) -> pd.DataFrame:
    """Tambah ari_mean/ari_sd (bootstrap subsampel) untuk SETIAP kombinasi grid."""
    method, st, X = acfg["hdbscan"]["cluster_selection_method"], acfg["stability"], res.prep.X
    ari = []
    for r in res.grid.itertuples():
        mcs, ms = int(r.min_cluster_size), int(r.min_samples)
        full = cl.hdbscan_fit(X, mcs, ms, method).labels_
        ari.append(cl.stability_ari(lambda Xs: cl.hdbscan_fit(Xs, mcs, ms, method).labels_, X, full, st["n_bootstrap"], st["fraction"], acfg["random_state"]))
    return res.grid.assign(ari_mean=[a[0] for a in ari], ari_sd=[a[1] for a in ari])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--main-config", default="config/default_config.yaml")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    main_cfg = load_config(resolve_path(args.main_config), use_user=False)
    setup_logging(main_cfg)
    acfg = load_analysis_config(resolve_path(args.config))
    fin, sel = acfg["final"], acfg["final"]["selection"]
    out_dir = resolve_path(acfg["output_dir"])
    plot_dir, dpi = out_dir / "cluster_analysis", acfg["plots"]["dpi"]

    voip = filter_voip(load_flows(resolve_path(main_cfg["data"]["raw_dir"]), main_cfg["data"]), main_cfg["data"]["voip_filter"])
    set_cfg = next(s for s in acfg["sets"] if s["name"] == fin["set_name"])
    prep = prepare(voip, set_cfg, acfg["features"][fin["variant"]], acfg["cleaning"], acfg["fai_clip_negative"])
    facfg = copy.deepcopy(acfg)
    facfg["hdbscan"].update(min_cluster_size=fin["hdbscan"]["min_cluster_size"], min_samples=fin["hdbscan"]["min_samples"])
    name = f"final_g_{set_cfg['name']}__{fin['variant']}"
    res = analyze(prep, name, set_cfg["title"], fin["variant"], facfg)
    n = len(prep.X)

    grid = grid_with_ari(res, facfg)
    best, valid, reason = select_final_hdbscan(grid, sel)
    mcs, ms = int(best["min_cluster_size"]), int(best["min_samples"])
    h = hdbscan_config(res, mcs, ms, "HDBSCAN final", facfg)
    n_h = int(h.metrics["n_clusters"])

    km_opt = res.configs["kmeans_opt"]
    km_n = res.configs["kmeans_fair"]
    if n_h != res.k_opt:
        k_ = n_h
        fit_t, km = cl.median_time(lambda: cl.kmeans_fit(prep.X, k_, acfg["kmeans"]["n_init"], acfg["random_state"]), acfg["timing"]["repeats"])
        km_n = _build_config(f"K-Means k = klaster HDBSCAN final", f"k={k_}, n_init={acfg['kmeans']['n_init']}", prep, km.labels_, km, fit_t,
                             lambda Xs: cl.kmeans_fit(Xs, k_, acfg["kmeans"]["n_init"], acfg["random_state"]).labels_, acfg, False)
    same_k = n_h == res.k_opt

    # grafik: PCA dan jarak memakai RunResult dengan konfigurasi final
    res.configs["hdbscan"], res.configs["kmeans_fair"], res.hdb_best = h, km_n, best
    plots.pca_scatter(res, plot_dir / f"{name}_pca.png", dpi, acfg["plots"]["max_scatter_points"], acfg["random_state"])
    plots.own_distance_plot(res, plot_dir / f"{name}_jarak_centroid.png", dpi)

    all_tbl = grid.assign(n_clusters=grid["n_clusters"].astype(int), valid=grid.index.isin(valid.index)).rename(
        columns={"silhouette": "Silhouette", "davies_bouldin": "DBI", "dbcv": "DBCV", "ari_mean": "ARI mean", "ari_sd": "ARI sd"})
    cols = ["min_cluster_size", "min_samples", "n_clusters", "noise_ratio", "Silhouette", "DBI", "DBCV", "ARI mean", "ARI sd"]
    valid_tbl = valid.assign(n_clusters=valid["n_clusters"].astype(int)).rename(
        columns={"silhouette": "Silhouette", "davies_bouldin": "DBI", "dbcv": "DBCV", "ari_mean": "ARI mean", "ari_sd": "ARI sd"})[cols]

    rows = [metric_row("HDBSCAN final (i) tanpa noise", h), metric_row("HDBSCAN final (ii) noise → centroid terdekat", h, True),
            metric_row(km_opt.name + (" (= k HDBSCAN final)" if same_k else ""), km_opt)]
    if not same_k:
        rows.append(metric_row(km_n.name, km_n))

    rules = distance_rules(prep.X, h.labels, fin["distance_limit_percentile"])
    rec = recommend_rule(rules)
    cent = centroids_original_units(h.centroids, prep.scaler, prep.columns, prep.log_columns)
    thresholds = main_cfg["thresholds"]
    rel = lambda s: f"cluster_analysis/{name}_{s}.png"  # noqa: E731

    L = [
        sel["report_marker"], "",
        f"# Bagian G — Aturan pemilihan HDBSCAN final ({n} flow, {set_cfg['title']})", "",
        "Dihasilkan oleh `scripts/cluster_analysis_selection.py` (aturan di `analysis.final.selection`). Masih analisis; aplikasi dan model aktif tidak diubah.", "",
        "## G.1 Aturan dan kandidat",
        f"Kandidat valid = noise ≤ {sel['max_noise_ratio']:.0%}, klaster ≥ {sel['min_clusters']}, ARI bootstrap mean ≥ {sel['min_ari_mean']:g} "
        f"({acfg['stability']['n_bootstrap']}× subsampel {acfg['stability']['fraction']:.0%}). Pilih DBCV tertinggi; seri → Silhouette. "
        f"Dari {len(grid)} kombinasi grid, **{len(valid)}** valid.", "",
        "### Kandidat valid (urut DBCV)", md_table(valid_tbl), "",
        f"**Terpilih:** {reason}", "",
        "### Seluruh grid dengan ARI bootstrap",
        md_table(all_tbl[[*cols, "valid"]].assign(valid=lambda d: d["valid"].map({True: "ya", False: "-"}))), "",
        "## G.2 Perbandingan: HDBSCAN final vs K-Means",
        md_table(pd.DataFrame(rows)), "",
        ("Catatan: k optimal K-Means sama dengan jumlah klaster HDBSCAN final, sehingga K-Means k = klaster HDBSCAN identik dengan k optimal. " if same_k else "")
        + "Versi (i) HDBSCAN menghitung metrik tanpa titik noise; versi (ii) memasukkan semua titik (noise → centroid terdekat), sepadan dengan K-Means.", "",
        timing_md(res, {}), f"![PCA]({rel('pca')})", "",
        "## G.3 Profil klaster HDBSCAN final (posisi ITU = deskripsi median terhadap ambang config, bukan label)",
        f"Ambang: delay ≤{thresholds['delay_good_ms']:g}/≤{thresholds['delay_max_ms']:g} ms, jitter ≤{thresholds['jitter_max_ms']:g} ms, "
        f"loss ≤{thresholds['loss_good_pct']:g}/≤{thresholds['loss_max_pct']:g} % (FAI sebagai proksi loss).", "",
        f"#### {h.name} ({h.description})", profile_table(h, thresholds), "",
        "Centroid (satuan asli):", md_table(cent, digits=3), "",
        "Jarak ke centroid sendiri:", md_table(h.own_distance, digits=3), "", f"![jarak]({rel('jarak_centroid')})", "",
        "## G.4 Batas \"jauh dari semua klaster\": tiga aturan",
        "Titik ditandai jauh bila jaraknya ke SETIAP centroid melebihi batas klaster tersebut.", "",
        md_table(rules.drop(columns=["noise total", "non-noise total", "batas per klaster"]), digits=1), "",
        "Batas per klaster (ruang ter-skala):", "",
        *[f"- {a}: {b}" for a, b in zip(rules["aturan"], rules["batas per klaster"])], "",
        f"**Rekomendasi: {rules.iloc[rec]['aturan']}** — noise ditandai {rules.iloc[rec]['noise ditandai jauh (%)']:.1f}% "
        f"({rules.iloc[rec]['noise ditandai']}/{rules.iloc[rec]['noise total']}), non-noise ditandai {rules.iloc[rec]['non-noise ditandai jauh (%)']:.1f}% "
        f"({rules.iloc[rec]['non-noise ditandai']}/{rules.iloc[rec]['non-noise total']}). Kriteria: salah tandai paling sedikit (noise yang lolos + non-noise yang ditandai jauh), seri → noise ditandai lebih banyak; lihat trade-off antar aturan pada tabel.", "",
    ]
    rep = out_dir / REPORT_NAME
    old = rep.read_text(encoding="utf-8")
    rep.write_text(old.split(sel["report_marker"])[0].rstrip() + "\n\n" + "\n".join(L), encoding="utf-8")
    print("Laporan:", rep, "| terpilih:", mcs, ms, "klaster", n_h, "| valid:", len(valid), "| aturan:", rules.iloc[rec]["aturan"])


if __name__ == "__main__":
    main()
