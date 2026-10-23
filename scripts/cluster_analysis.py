"""Analisis klasterisasi murni (Tahap 1): K-Means vs HDBSCAN pada flow VOIP Darknet.CSV.

Keluaran: reports/experiments/cluster_analysis.md dan grafik di reports/experiments/cluster_analysis/.
Tidak mengubah aplikasi, model aktif, config default, atau artefak. Parameter dari config/analysis_config.yaml.

Pemakaian: python scripts/cluster_analysis.py [--config config/analysis_config.yaml] [--main-config config/default_config.yaml]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qos.analysis import plots  # noqa: E402
from qos.analysis.data import prepare  # noqa: E402
from qos.analysis.report import (  # noqa: E402
    comparison_rows,
    conclusion,
    dict_table,
    fixture_assignment,
    md_table,
    summary_row,
    timing_rows,
)
from qos.analysis.run import RunResult, analyze  # noqa: E402
from qos.config import load_config, resolve_path, setup_logging  # noqa: E402
from qos.data.loader import filter_voip, load_flows  # noqa: E402

REPORT_NAME = "cluster_analysis.md"


def load_analysis_config(path: Path) -> dict:
    """Muat blok `analysis:` dari YAML."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["analysis"]


def section_data(res: RunResult, img: str) -> list[str]:
    """A. Data dan cleaning (himpunan utama)."""
    r = res.prep.report
    stages = pd.DataFrame(r["tahap"], columns=["tahap", "jumlah flow"])
    stages["dibuang"] = (-stages["jumlah flow"].diff()).fillna(0).astype(int)
    iqr = pd.DataFrame(r["ekstrem_iqr"]).T.reset_index().rename(columns={"index": "fitur"})
    iqr["jumlah"] = iqr["jumlah"].astype(int)
    anomalies = pd.DataFrame(
        {"kolom": list(r["nan_per_kolom"]), "NaN": list(r["nan_per_kolom"].values()),
         "inf": list(r["inf_per_kolom"].values()), "negatif": list(r["negatif_per_kolom"].values())}
    )
    return [
        "## A. Data dan cleaning",
        f"Himpunan utama: **{res.set_title}**; fitur: `{res.prep.columns}`, log1p pada `{res.prep.log_columns}`, lalu StandardScaler.",
        "", "### Jumlah flow per tahap", md_table(stages),
        "", f"Duplikat (baris identik pada semua kolom yang dimuat): **{r.get('duplikat', 0)}**; baris bernilai negatif dibuang: "
        f"**{r.get('baris_negatif_dibuang', 0)}**; baris diimputasi median: **{r['baris_diimputasi']}** (median: {r['median_imputasi']}).",
        "", "### NaN / inf / negatif (sebelum cleaning, pada himpunan ini)", md_table(anomalies),
        "", "### Nilai ekstrem (metode IQR, pada data bersih, nilai mentah)", md_table(iqr),
        "", f"- Jitter = 0 (iat_std = 0): **{r['persen_jitter_nol']:.2f}%** flow",
        f"- bwd = 0: **{r['persen_bwd_nol']:.2f}%** flow",
        "", "### Jumlah per traffic type", dict_table(r["per_traffic_type"], "traffic type", "flow"),
        "", "### Jumlah per protokol", dict_table(r["per_protokol"], "protokol", "flow"),
        "", "### Korelasi antar-fitur (setelah transformasi)",
        "Pearson:", md_table(r["korelasi_pearson"].round(4), index=True),
        "", "Spearman:", md_table(r["korelasi_spearman"].round(4), index=True),
        "", f"### Histogram fitur sebelum dan sesudah transformasi\n\n![histogram]({img})", "",
    ]


def section_k(res: RunResult, curves: str, heat: str) -> list[str]:
    """B. Penentuan jumlah klaster optimal."""
    sweep = res.sweep.rename(columns={"inertia": "inertia", "silhouette": "Silhouette", "davies_bouldin": "DBI", "calinski_harabasz": "CHI"})
    grid = res.grid.assign(n_clusters=res.grid["n_clusters"].astype(int)).rename(columns={"silhouette": "Silhouette", "davies_bouldin": "DBI", "calinski_harabasz": "CHI", "dbcv": "DBCV"})
    return [
        "## B. Penentuan jumlah klaster optimal",
        "### K-Means (k = 2..10, n_init = 10, random_state 42)",
        md_table(sweep[["k", "inertia", "Silhouette", "DBI", "CHI"]]),
        "", f"![kurva K-Means]({curves})",
        "", f"**k optimal K-Means = {res.k_opt}.** {res.k_reason}",
        "", "### HDBSCAN (grid min_cluster_size × min_samples; metrik pada titik non-noise, DBCV via `hdbscan.validity.validity_index`)",
        md_table(grid[["min_cluster_size", "min_samples", "n_clusters", "noise_ratio", "Silhouette", "DBI", "CHI", "DBCV"]]),
        "", f"![heatmap HDBSCAN]({heat})",
        "", f"**Parameter optimal HDBSCAN:** {res.hdb_reason}",
        "", "Aturan pemilihan (dari config): K-Means = rata-rata peringkat Silhouette dan DBI (CHI hanya pendukung karena cenderung naik terhadap k); "
        "HDBSCAN = DBCV tertinggi dengan batas jumlah klaster dan noise. Catatan: jumlah klaster HDBSCAN yang besar pada min_samples tinggi "
        "menunjukkan banyak mikro-klaster; lihat tabel grid bila dosen menginginkan jumlah klaster kecil.", "",
    ]


def section_compare(res: RunResult, pca_img: str) -> list[str]:
    """C. Perbandingan K-Means vs HDBSCAN."""
    lines = [
        "## C. Perbandingan K-Means vs HDBSCAN",
        "Konfigurasi: (1) HDBSCAN optimal, (2) K-Means k optimal, (3) K-Means dengan k = jumlah klaster HDBSCAN optimal (adil). "
        "Untuk HDBSCAN dihitung dua versi: (i) metrik tanpa titik noise, (ii) noise di-assign ke centroid terdekat (semua titik).",
        "", md_table(comparison_rows(res)),
    ]
    if res.configs["kmeans_opt"].description == res.configs["kmeans_fair"].description:
        lines.append("\nCatatan: k optimal K-Means sama dengan jumlah klaster HDBSCAN, sehingga konfigurasi (2) dan (3) identik.")
    lines += [
        "", "### Stabilitas dan waktu",
        "Stabilitas = ARI antara label data penuh dan label hasil fit pada subsampel 80% (50×), dihitung pada titik subsampel.",
        "", md_table(timing_rows(res)),
        "", f"### PCA 2D\n\n![PCA]({pca_img})", "",
    ]
    for key in ("hdbscan", "kmeans_opt", "kmeans_fair"):
        c = res.configs[key]
        prof = c.profile.copy()
        prof.columns = ["Klaster", "n", "delay mean ms", "delay median ms", "jitter mean ms", "jitter median ms", "FAI mean %",
                        "FAI median %", "paket median", "% jitter=0", "% bwd=0", "traffic type", "protokol"]
        lines += [f"### Profil per klaster: {c.name} ({c.description})", md_table(prof, digits=2), ""]
    lines += ["### Kesimpulan perbandingan", *conclusion(res), ""]
    return lines


def section_centroid(res: RunResult, fixture: Path, dist_img: str) -> list[str]:
    """D. Penugasan data baru dengan jarak Euclidean ke centroid."""
    lines = ["## D. Penugasan dengan jarak Euclidean ke centroid", "Centroid = rata-rata anggota klaster di ruang fitur ter-skala (noise HDBSCAN tidak punya centroid).", ""]
    cons = pd.DataFrame(
        [{"konfigurasi": c.name, "titik dinilai": c.consistency["n_dinilai"], "noise": c.consistency["n_noise"],
          "cocok dengan label asli (%)": c.consistency["cocok_persen"]} for c in res.configs.values()]
    )
    lines += ["### Konsistensi: penugasan ulang ke centroid terdekat vs label asli algoritma", md_table(cons, digits=2), ""]
    for key in ("hdbscan", "kmeans_opt"):
        c = res.configs[key]
        lines += [f"### Jarak ke centroid sendiri: {c.name}", md_table(c.own_distance, digits=3), ""]
    lines += [f"![jarak]({dist_img})", "", f"### Contoh: 10 flow `{fixture.name}` (ruang fitur himpunan utama)",
              md_table(fixture_assignment(fixture, res), digits=3), ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--main-config", default="config/default_config.yaml")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    main_cfg = load_config(resolve_path(args.main_config), use_user=False)
    setup_logging(main_cfg)
    acfg = load_analysis_config(resolve_path(args.config))
    out_dir = resolve_path(acfg["output_dir"])
    plot_dir = out_dir / "cluster_analysis"
    dpi = acfg["plots"]["dpi"]

    print("Memuat data…")
    voip = filter_voip(load_flows(resolve_path(main_cfg["data"]["raw_dir"]), main_cfg["data"]), main_cfg["data"]["voip_filter"])
    print(f"  flow VOIP (kategori VOIP, termasuk Tor): {len(voip)}")

    results: list[RunResult] = []
    for s in acfg["sets"]:
        for variant in s["feature_variants"]:
            name = f"{s['name']}__{variant}"
            t0 = time.time()
            prep = prepare(voip, s, acfg["features"][variant], acfg["cleaning"], acfg["fai_clip_negative"])
            res = analyze(prep, name, s["title"], variant, acfg)
            res.primary = bool(s.get("primary")) and variant == "base"
            results.append(res)
            plots.kmeans_curves(res, plot_dir / f"{name}_kmeans.png", dpi)
            plots.hdbscan_heatmaps(res, plot_dir / f"{name}_hdbscan_heatmap.png", dpi)
            plots.pca_scatter(res, plot_dir / f"{name}_pca.png", dpi, acfg["plots"]["max_scatter_points"], acfg["random_state"])
            plots.own_distance_plot(res, plot_dir / f"{name}_jarak_centroid.png", dpi)
            print(f"  {name}: {len(prep.frame)} flow, k={res.k_opt}, HDBSCAN klaster={int(res.hdb_best['n_clusters'])} ({time.time() - t0:.0f} dtk)")

    main_res = next(r for r in results if r.primary)
    raw_cols = main_res.prep.frame
    plots.hist_before_after(raw_cols, main_res.prep.features, plot_dir / "histogram_fitur.png", dpi)

    rel = lambda n: f"cluster_analysis/{n}"  # noqa: E731
    name = main_res.run_name
    lines = [
        "# Analisis klasterisasi murni: K-Means vs HDBSCAN (Tahap 1)",
        "", "Dihasilkan oleh `scripts/cluster_analysis.py` (`config/analysis_config.yaml`, random_state 42). "
        "Tidak mengubah aplikasi, model aktif, config default, atau artefak. Nomor klaster dimulai dari 1 (urut ukuran menurun); noise HDBSCAN ditulis \"Noise\".",
        "", f"Data: {len(voip):,} flow berkategori VOIP di Darknet.CSV (termasuk Tor). Himpunan utama: {main_res.set_title}.", "",
        *section_data(main_res, rel("histogram_fitur.png")),
        *section_k(main_res, rel(f"{name}_kmeans.png"), rel(f"{name}_hdbscan_heatmap.png")),
        *section_compare(main_res, rel(f"{name}_pca.png")),
        *section_centroid(main_res, resolve_path(acfg["assignment"]["fixture"]), rel(f"{name}_jarak_centroid.png")),
    ]

    rows = [summary_row(r, a) for r in results for a in ("kmeans", "hdbscan")]
    lines += [
        "## E. Sensitivitas data dan fitur",
        "Setiap baris mengulang B dan C secara ringkas. Metrik K-Means pada k optimal (semua titik); metrik HDBSCAN pada titik non-noise.",
        "", md_table(pd.DataFrame(rows)), "",
    ]
    for r in results:
        lines += [f"### {r.set_title} · fitur `{r.variant}`", f"- k optimal K-Means: {r.k_reason}", f"- HDBSCAN: {r.hdb_reason}",
                  f"- Grafik: `{r.run_name}_kmeans.png`, `{r.run_name}_hdbscan_heatmap.png`, `{r.run_name}_pca.png`", ""]
    lines += ["## Rekomendasi untuk Tahap 2", "Lihat ringkasan di percakapan; bagian ini diisi manual setelah laporan ditinjau.", ""]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_NAME).write_text("\n".join(lines), encoding="utf-8")
    print(f"Laporan: {out_dir / REPORT_NAME}")


if __name__ == "__main__":
    main()
