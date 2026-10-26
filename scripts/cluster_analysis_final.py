"""Bagian F analisis klasterisasi: himpunan keputusan (fwd >= 5, bwd >= 5, tanpa Tor), HDBSCAN utama, K-Means pembanding.

Menambahkan bagian F ke reports/experiments/cluster_analysis.md (A-E tidak dihapus; menjalankan ulang hanya mengganti bagian F).
Tidak mengubah aplikasi, model aktif, config default, atau artefak. Parameter dari config/analysis_config.yaml (blok analysis.final).

Pemakaian: python scripts/cluster_analysis_final.py [--config config/analysis_config.yaml] [--main-config config/default_config.yaml]
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
    add_itu_column,
    centroids_original_units,
    distance_limits,
    far_from_all,
)
from qos.analysis.report import METRIC_LABELS, md_table  # noqa: E402
from qos.analysis.run import ConfigResult, RunResult, _build_config, analyze  # noqa: E402
from qos.config import load_config, resolve_path, setup_logging  # noqa: E402
from qos.data.loader import filter_voip, load_flows  # noqa: E402
from cluster_analysis import REPORT_NAME, load_analysis_config  # noqa: E402

PROFILE_COLS = ["Klaster", "n", "delay mean ms", "delay median ms", "jitter mean ms", "jitter median ms", "FAI mean %",
                "FAI median %", "paket median", "% jitter=0", "% bwd=0", "traffic type", "protokol", "posisi ITU (median)"]


def hdbscan_config(res: RunResult, mcs: int, ms: int, name: str, acfg: dict) -> ConfigResult:
    """Bangun ConfigResult HDBSCAN untuk pasangan parameter tertentu pada data `res.prep`."""
    method = acfg["hdbscan"]["cluster_selection_method"]
    fit_t, model = cl.median_time(lambda: cl.hdbscan_fit(res.prep.X, mcs, ms, method), acfg["timing"]["repeats"])
    return _build_config(name, f"min_cluster_size={mcs}, min_samples={ms}, {method}", res.prep, model.labels_, model, fit_t,
                         lambda Xs: cl.hdbscan_fit(Xs, mcs, ms, method).labels_, acfg, True)


def metric_row(label: str, c: ConfigResult, assigned: bool = False) -> dict:
    """Satu baris tabel perbandingan: metrik, ARI bootstrap, waktu fit."""
    m = c.metrics_assigned if assigned else c.metrics
    return {"konfigurasi": label, "parameter": c.description,
            **{METRIC_LABELS[k]: (int(m[k]) if k == "n_clusters" else m[k]) for k in METRIC_LABELS},
            "ARI bootstrap (mean ± sd)": f"{c.stability[0]:.3f} ± {c.stability[1]:.3f}", "waktu fit (dtk)": c.fit_time}


def pick_row(label: str, best: pd.Series) -> dict:
    """Satu baris tabel pilihan grid."""
    return {"kriteria": label, "min_cluster_size": int(best["min_cluster_size"]), "min_samples": int(best["min_samples"]),
            "klaster": int(best["n_clusters"]), "noise": best["noise_ratio"], "Silhouette": best["silhouette"],
            "DBI": best["davies_bouldin"], "CHI": best["calinski_harabasz"], "DBCV": best["dbcv"]}


def profile_table(c: ConfigResult, thresholds: dict) -> str:
    """Profil klaster + kolom posisi ITU sebagai tabel Markdown."""
    prof = add_itu_column(c.profile, thresholds)
    prof.columns = PROFILE_COLS
    return md_table(prof, digits=2)


def timing_md(res: RunResult, extra: dict) -> str:
    """Waktu fit dan prediksi per titik."""
    rows = [{"konfigurasi": c.name, "waktu fit (dtk)": c.fit_time, "metode prediksi": m, "prediksi per titik (µs)": t * 1e6}
            for c in [*res.configs.values(), *extra.values()] for m, t in c.predict_time_point.items()]
    return "### Waktu\n" + md_table(pd.DataFrame(rows)) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--main-config", default="config/default_config.yaml")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    main_cfg = load_config(resolve_path(args.main_config), use_user=False)
    setup_logging(main_cfg)
    acfg = load_analysis_config(resolve_path(args.config))
    fin = acfg["final"]
    thresholds = main_cfg["thresholds"]
    out_dir = resolve_path(acfg["output_dir"])
    plot_dir = out_dir / "cluster_analysis"
    dpi = acfg["plots"]["dpi"]

    voip = filter_voip(load_flows(resolve_path(main_cfg["data"]["raw_dir"]), main_cfg["data"]), main_cfg["data"]["voip_filter"])
    set_cfg = next(s for s in acfg["sets"] if s["name"] == fin["set_name"])
    prep = prepare(voip, set_cfg, acfg["features"][fin["variant"]], acfg["cleaning"], acfg["fai_clip_negative"])
    n = len(prep.frame)
    print(f"{set_cfg['title']}: {n} flow")

    facfg = copy.deepcopy(acfg)
    facfg["hdbscan"].update(min_cluster_size=fin["hdbscan"]["min_cluster_size"], min_samples=fin["hdbscan"]["min_samples"])
    name = f"final_{set_cfg['name']}__{fin['variant']}"
    res = analyze(prep, name, set_cfg["title"], fin["variant"], facfg)   # HDBSCAN utama = DBCV tertinggi (aturan config)
    plots.kmeans_curves(res, plot_dir / f"{name}_kmeans.png", dpi)
    plots.hdbscan_heatmaps(res, plot_dir / f"{name}_hdbscan_heatmap.png", dpi)
    plots.pca_scatter(res, plot_dir / f"{name}_pca.png", dpi, acfg["plots"]["max_scatter_points"], acfg["random_state"])
    plots.own_distance_plot(res, plot_dir / f"{name}_jarak_centroid.png", dpi)

    sel = acfg["selection"]
    lo, hi = fin["candidate_clusters"]
    best_dbcv = res.hdb_best
    best_sil, _ = cl.select_hdbscan(res.grid, sel["hdbscan_min_clusters"], sel["hdbscan_max_noise_ratio"], "silhouette")
    best_rng, _ = cl.select_hdbscan(res.grid, lo, sel["hdbscan_max_noise_ratio"], sel["hdbscan_primary_metric"], max_clusters=hi)
    picks = pd.DataFrame([pick_row("DBCV tertinggi (terpilih)", best_dbcv), pick_row("Silhouette tertinggi", best_sil),
                          pick_row(f"DBCV tertinggi dengan {lo}–{hi} klaster", best_rng)])
    key = lambda b: (int(b["min_cluster_size"]), int(b["min_samples"]))  # noqa: E731
    extra: dict[str, ConfigResult] = {}
    if key(best_sil) != key(best_dbcv):
        extra["Silhouette tertinggi"] = hdbscan_config(res, *key(best_sil), "HDBSCAN (Silhouette tertinggi)", facfg)
    rng_label = f"kandidat {lo}–{hi} klaster"
    if key(best_rng) not in (key(best_dbcv), key(best_sil)):
        extra[rng_label] = hdbscan_config(res, *key(best_rng), f"HDBSCAN ({rng_label})", facfg)
    h = res.configs["hdbscan"]
    grid = res.grid.assign(n_clusters=res.grid["n_clusters"].astype(int)).rename(
        columns={"silhouette": "Silhouette", "davies_bouldin": "DBI", "calinski_harabasz": "CHI", "dbcv": "DBCV"})

    rows = [metric_row("HDBSCAN terpilih (i) tanpa noise", h), metric_row("HDBSCAN terpilih (ii) noise → centroid terdekat", h, True)]
    for c in extra.values():
        rows.append(metric_row(f"{c.name} (i)", c))
    rows += [metric_row(res.configs["kmeans_opt"].name, res.configs["kmeans_opt"]),
             metric_row(res.configs["kmeans_fair"].name, res.configs["kmeans_fair"])]

    # penugasan Euclidean: batas per klaster dari HDBSCAN terpilih
    pct = fin["distance_limit_percentile"]
    limits = distance_limits(prep.X, h.labels, pct)
    noise = h.labels == cl.NOISE
    far = far_from_all(prep.X[noise], h.centroids, limits) if noise.any() else np.array([], dtype=bool)
    nonnoise_far = far_from_all(prep.X[~noise], h.centroids, limits)
    lim_df = h.own_distance.assign(**{f"batas p{pct}": [limits[int(c)] for c in h.own_distance["klaster"]]})
    cons = pd.DataFrame([{"konfigurasi": c.name, "titik dinilai": c.consistency["n_dinilai"], "noise": c.consistency["n_noise"],
                          "cocok dengan label asli (%)": c.consistency["cocok_persen"]} for c in [*res.configs.values(), *extra.values()]])
    cent = centroids_original_units(h.centroids, prep.scaler, prep.columns, prep.log_columns)
    ids = sorted(h.centroids)
    cent_sc = pd.DataFrame(np.vstack([h.centroids[i] for i in ids]), columns=[f"{c} (ter-skala)" for c in prep.columns])
    cent_sc.insert(0, "klaster", ids)
    ratio = np.array([])
    if noise.any():
        ratio = (cl.assign_nearest(prep.X[noise], h.centroids)[1] / np.array([limits[i] for i in ids])[None, :]).min(axis=1)

    rel = lambda s: f"cluster_analysis/{name}_{s}.png"  # noqa: E731
    r = prep.report
    if key(best_rng) != key(best_dbcv):
        delta = (f"Δ Silhouette = {best_rng['silhouette'] - best_dbcv['silhouette']:+.4f}, Δ DBI = {best_rng['davies_bouldin'] - best_dbcv['davies_bouldin']:+.4f}, "
                 f"Δ DBCV = {best_rng['dbcv'] - best_dbcv['dbcv']:+.4f}, Δ noise = {best_rng['noise_ratio'] - best_dbcv['noise_ratio']:+.4f}, "
                 f"klaster {int(best_rng['n_clusters'])} vs {int(best_dbcv['n_clusters'])}.")
    else:
        delta = "sama dengan DBCV tertinggi."
    L = [
        fin["report_marker"], "",
        f"# Bagian F — Himpunan keputusan: {set_cfg['title']} ({n} flow)", "",
        "Keputusan: data utama = fwd ≥ 5, bwd ≥ 5, tanpa Tor; HDBSCAN algoritma utama, K-Means pembanding. Masih analisis; aplikasi dan model aktif tidak diubah. "
        "Bagian A–E di atas tidak diubah. Dihasilkan oleh `scripts/cluster_analysis_final.py` (blok `analysis.final` di `config/analysis_config.yaml`). "
        f"Grid HDBSCAN lebih rapat daripada bagian B (min_cluster_size {fin['hdbscan']['min_cluster_size']}, min_samples {fin['hdbscan']['min_samples']}) karena datanya kecil.", "",
        "## F.0 Ringkasan data", md_table(pd.DataFrame(r["tahap"], columns=["tahap", "jumlah flow"])), "",
        f"- Jitter = 0: {r['persen_jitter_nol']:.2f}% · bwd = 0: {r['persen_bwd_nol']:.2f}% · traffic type: {r['per_traffic_type']} · protokol: {r['per_protokol']}", "",
        "## F.B Penentuan jumlah klaster",
        "### K-Means (k = 2..10)", md_table(res.sweep[["k", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz"]].rename(
            columns={"silhouette": "Silhouette", "davies_bouldin": "DBI", "calinski_harabasz": "CHI"})),
        "", f"![kurva K-Means]({rel('kmeans')})", "", f"**k optimal K-Means = {res.k_opt}.** {res.k_reason}", "",
        "### HDBSCAN: grid lengkap (metrik pada titik non-noise)",
        md_table(grid[["min_cluster_size", "min_samples", "n_clusters", "noise_ratio", "Silhouette", "DBI", "CHI", "DBCV"]]),
        "", f"![heatmap HDBSCAN]({rel('hdbscan_heatmap')})", "",
        "### Pilihan terbaik menurut kriteria berbeda (batas: ≥ 2 klaster, noise ≤ batas config)", md_table(picks), "",
        f"Kandidat {lo}–{hi} klaster dibanding DBCV tertinggi: {delta}", "",
        "## F.C Perbandingan", md_table(pd.DataFrame(rows)), "",
        "Waktu fit = median 3 pengulangan. ARI bootstrap = ARI label subsampel 80% (50×) vs label data penuh.", "",
        timing_md(res, extra), f"![PCA]({rel('pca')})", "",
        "### Profil per klaster (satuan asli; posisi ITU = deskripsi median klaster terhadap ambang config, bukan label)",
        f"Ambang: delay ≤{thresholds['delay_good_ms']:g}/≤{thresholds['delay_max_ms']:g} ms, jitter ≤{thresholds['jitter_max_ms']:g} ms, "
        f"loss ≤{thresholds['loss_good_pct']:g}/≤{thresholds['loss_max_pct']:g} % (FAI dipakai sebagai proksi loss).", "",
    ]
    for label, c in [("HDBSCAN terpilih", h), *extra.items(), ("K-Means k optimal", res.configs["kmeans_opt"]),
                     ("K-Means k = klaster HDBSCAN", res.configs["kmeans_fair"])]:
        L += [f"#### {label}: {c.name} ({c.description})", profile_table(c, thresholds), ""]
    L += [
        "## F.D Penugasan Euclidean (HDBSCAN terpilih)",
        "### Centroid (satuan asli; delay/jitter dalam ms)", md_table(cent, digits=3),
        "", "Centroid di ruang ter-skala:", md_table(cent_sc, digits=3),
        "", "### Konsistensi: penugasan ulang ke centroid terdekat vs label asli", md_table(cons, digits=2),
        "", f"### Jarak ke centroid sendiri dan batas usulan (p{pct})", md_table(lim_df, digits=3),
        "", f"![jarak]({rel('jarak_centroid')})", "",
        f"**Usulan batas \"jauh dari semua klaster\":** jarak ke SETIAP centroid > batas p{pct} klaster tersebut.", "",
        f"- Titik noise melewati batas pada semua klaster: **{int(far.sum())} dari {int(noise.sum())} ({(far.mean() * 100 if noise.any() else 0):.1f}%)**.",
        f"- Pembanding: titik non-noise yang melewati batas semua klaster: {int(nonnoise_far.sum())} dari {int((~noise).sum())} ({nonnoise_far.mean() * 100:.1f}%).",
    ]
    if noise.any():
        L.append(f"- Rasio jarak noise terhadap batas klaster terdekat (min d/batas): median {np.median(ratio):.2f}, "
                 f"p95 {np.percentile(ratio, 95):.2f}, maks {ratio.max():.2f} (>1 = jauh dari semua klaster).")
    L.append("")

    rep = out_dir / REPORT_NAME
    old = rep.read_text(encoding="utf-8")
    g_marker = fin["selection"]["report_marker"]
    g_part = "\n\n" + g_marker + old.split(g_marker, 1)[1] if g_marker in old else ""   # pertahankan bagian G
    text = old.split(fin["report_marker"])[0].rstrip() + "\n\n"
    rep.write_text(text + "\n".join(L) + g_part, encoding="utf-8")
    print(f"Laporan: {rep}")


if __name__ == "__main__":
    main()
