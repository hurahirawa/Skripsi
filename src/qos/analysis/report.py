"""Penyusun laporan Markdown analisis klasterisasi."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qos.analysis import clustering as cl
from qos.analysis.data import add_derived, transform_features
from qos.analysis.run import ConfigResult, RunResult

METRIC_LABELS: dict[str, str] = {
    "n_clusters": "Jumlah klaster", "noise_ratio": "Noise ratio", "silhouette": "Silhouette (↑)",
    "davies_bouldin": "Davies-Bouldin (↓)", "calinski_harabasz": "Calinski-Harabasz (↑)", "dbcv": "DBCV (↑)",
}


def fmt(v: Any, digits: int = 4) -> str:
    """Teks sel: '-' untuk NaN, ribuan untuk bilangan besar."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{digits}f}" if abs(v) < 1000 else f"{v:,.1f}"
    return str(v)


def md_table(df: pd.DataFrame, index: bool = False, digits: int = 4) -> str:
    """DataFrame → tabel Markdown."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v, digits) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def dict_table(d: dict, key_name: str, value_name: str) -> str:
    """dict sederhana → tabel dua kolom."""
    return md_table(pd.DataFrame({key_name: list(d), value_name: list(d.values())}))


def comparison_rows(res: RunResult) -> pd.DataFrame:
    """Tabel perbandingan utama: HDBSCAN (i) tanpa noise, HDBSCAN (ii) noise di-assign, K-Means k optimal, K-Means k adil."""
    h, ko, kf = res.configs["hdbscan"], res.configs["kmeans_opt"], res.configs["kmeans_fair"]
    rows = [
        ("HDBSCAN (i) tanpa noise", h.description, h.metrics, h),
        ("HDBSCAN (ii) noise → centroid terdekat", h.description, h.metrics_assigned, h),
        (f"{ko.name}", ko.description, ko.metrics, ko),
        (f"{kf.name}", kf.description, kf.metrics, kf),
    ]
    out = []
    for name, desc, m, c in rows:
        out.append({"konfigurasi": name, "parameter": desc, **{METRIC_LABELS[k]: (int(m[k]) if k == "n_clusters" else m[k]) for k in METRIC_LABELS},
                    "ARI stabilitas (mean ± std)": f"{c.stability[0]:.3f} ± {c.stability[1]:.3f}"})
    return pd.DataFrame(out)


def timing_rows(res: RunResult) -> pd.DataFrame:
    """Waktu fit dan prediksi per titik tiap konfigurasi."""
    rows = []
    for c in res.configs.values():
        for method, t in c.predict_time_point.items():
            rows.append({"konfigurasi": c.name, "waktu fit (detik)": c.fit_time, "metode prediksi": method, "prediksi per titik (µs)": t * 1e6})
    return pd.DataFrame(rows)


def summary_row(res: RunResult, algo: str) -> dict[str, Any]:
    """Satu baris tabel sensitivitas (E) untuk algoritma 'hdbscan' atau 'kmeans'."""
    c = res.configs["hdbscan" if algo == "hdbscan" else "kmeans_opt"]
    is_h = algo == "hdbscan"
    return {
        "himpunan": res.set_title, "fitur": res.variant, "jumlah flow": len(res.prep.frame),
        "% jitter = 0": round(res.prep.report["persen_jitter_nol"], 2), "algoritma": "HDBSCAN" if is_h else "K-Means",
        "k optimal (K-Means)": "-" if is_h else res.k_opt,
        "parameter HDBSCAN": f"mcs={int(res.hdb_best['min_cluster_size'])}, ms={int(res.hdb_best['min_samples'])}" if is_h else "-",
        "klaster": int(c.metrics["n_clusters"]), "Silhouette": c.metrics["silhouette"], "DBI": c.metrics["davies_bouldin"],
        "CHI": c.metrics["calinski_harabasz"], "DBCV": c.metrics["dbcv"], "noise": c.metrics["noise_ratio"],
        "ARI stabilitas": f"{c.stability[0]:.3f} ± {c.stability[1]:.3f}",
    }


def fixture_assignment(csv_path: Path, res: RunResult, top: int = 3) -> pd.DataFrame:
    """Jarak 10 flow fixture ke centroid HDBSCAN dan K-Means (ruang fitur ter-skala milik run `res`)."""
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame(
        {
            "iat_mean_us": raw["Flow IAT Mean"], "iat_std_us": raw["Flow IAT Std"],
            "fwd_packets_count": raw["Total Fwd Packets"], "bwd_packets_count": raw["Total Bwd Packets"],
        }
    )
    df = add_derived(df, True)
    prep = res.prep
    X = prep.scaler.transform(transform_features(df, prep.columns, prep.log_columns))
    out = pd.DataFrame({"flow": range(1, len(df) + 1), "delay_ms": df["delay_ms"], "jitter_ms": df["jitter_ms"], "FAI %": df["fai_pct"]})
    for key, title in (("hdbscan", "HDBSCAN"), ("kmeans_opt", "K-Means")):
        cfg = res.configs[key]
        nearest, dist = cl.assign_nearest(X, cfg.centroids)
        ids = sorted(cfg.centroids)
        out[f"{title}: terdekat"] = [f"Klaster {n}" for n in nearest]
        out[f"{title}: jarak"] = dist.min(axis=1)
        order = np.argsort(dist, axis=1)[:, :top]
        out[f"{title}: {top} terdekat"] = [", ".join(f"K{ids[j]} ({dist[i, j]:.2f})" for j in order[i]) for i in range(len(df))]
    return out


def conclusion(res: RunResult) -> list[str]:
    """Kesimpulan algoritma yang lebih sesuai, berdasarkan skor perbandingan run utama."""
    h, kf, ko = res.configs["hdbscan"], res.configs["kmeans_fair"], res.configs["kmeans_opt"]
    tests = [
        ("Silhouette", h.metrics["silhouette"], kf.metrics["silhouette"], True),
        ("Davies-Bouldin", h.metrics["davies_bouldin"], kf.metrics["davies_bouldin"], False),
        ("DBCV", h.metrics["dbcv"], kf.metrics["dbcv"], True),
        ("stabilitas ARI", h.stability[0], kf.stability[0], True),
    ]
    wins_h = [n for n, a, b, hi in tests if (a > b if hi else a < b)]
    wins_k = [n for n, *_ in tests if n not in wins_h]
    prof = h.profile
    noise_pct = h.metrics["noise_ratio"] * 100
    lines = [
        f"- Pada jumlah klaster yang sama (k = {int(h.metrics['n_clusters'])}), HDBSCAN unggul di: {', '.join(wins_h) or '-'}; "
        f"K-Means unggul di: {', '.join(wins_k) or '-'}.",
        f"- HDBSCAN menandai {noise_pct:.1f}% flow sebagai noise (outlier) dan memakai kerapatan sehingga tidak memaksa setiap titik masuk "
        f"klaster; K-Means selalu menugaskan semua titik (noise 0%) dan sangat cepat untuk prediksi, tetapi titik ekstrem ikut menarik centroid.",
        f"- K-Means pada k optimal ({ko.name}, {ko.description}) mencapai Silhouette {ko.metrics['silhouette']:.4f}, namun DBCV "
        f"{ko.metrics['dbcv']:.4f} (DBCV memeriksa kerapatan internal dan keterpisahan berbasis kerapatan; nilai negatif berarti klaster tidak padat/terpisah menurut ukuran itu).",
    ]
    dom = prof[prof["klaster"] != "Noise"]
    short = dom[(dom["jitter_nol_pct"] > 50) | (dom["bwd_nol_pct"] > 50)]
    lines.append(
        f"- Profil HDBSCAN: {len(short)} dari {len(dom)} klaster didominasi aliran dengan jitter = 0 atau bwd = 0 (aliran satu arah/sangat pendek), "
        f"mencakup {int(short['n'].sum()):,} dari {int(dom['n'].sum()):,} flow berklaster. Lihat tabel profil untuk melihat klaster yang berisi panggilan sungguhan."
    )
    if len(wins_h) == len(wins_k):
        verdict = "metrik tidak memberi pemenang tunggal (imbang): K-Means unggul pada metrik berbasis jarak (Silhouette, stabilitas), HDBSCAN pada DBI dan DBCV"
    else:
        verdict = f"{'HDBSCAN' if len(wins_h) > len(wins_k) else 'K-Means'} unggul pada lebih banyak metrik"
    lines.append(f"- **Kesimpulan:** {verdict}. Pertimbangan di luar metrik (noise/outlier eksplisit, jumlah klaster, stabilitas, kecocokan dengan penugasan centroid) dibahas pada rekomendasi di percakapan.")
    km = ko.profile
    kd = km[km["klaster"] != "Noise"]
    kshort = kd[(kd["jitter_nol_pct"] > 50) | (kd["bwd_nol_pct"] > 50)]
    lines.append(
        f"- Profil K-Means (k optimal): {len(kshort)} dari {len(kd)} klaster didominasi aliran jitter = 0 atau bwd = 0, mencakup {int(kshort['n'].sum()):,} dari {int(kd['n'].sum()):,} flow. "
        "Artinya pada himpunan tanpa filter sesi, kedua algoritma terutama memisahkan aliran pendek/satu arah dari aliran panjang, bukan tingkat kualitas panggilan."
    )
    return lines
