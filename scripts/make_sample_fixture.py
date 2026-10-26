"""Buat fixture 10 flow nyata (dari flow final) dan data antara untuk Lampiran 2–5 naskah.

Keluaran:
  tests/fixtures/sample_real_10.csv       10 flow mentah + nomor klaster model aktif
  reports/lampiran/lampiran_data.md       FAI, log1p, Z-score, penugasan klaster, dan profil

Pemilihan: 1 flow dari tiap klaster (termasuk noise), sisa slot dibagi ke klaster terbesar;
pengacakan memakai data.random_state. Nomor klaster berasal dari model aktif (artifacts/current.txt).

Pemakaian: python scripts/make_sample_fixture.py --config config/default_config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from qos.config import PROJECT_ROOT, load_config, resolve_path, setup_logging
from qos.data.loader import filter_voip, load_csv
from qos.features.extract import derive_metrics
from qos.features.preprocess import apply_session_filters, build_features
from qos.profiling.profile import build_cluster_profile

ID_COLUMNS = {"Flow ID": "flow_id", "Src IP": "src_ip", "Dst IP": "dst_ip"}
FIXTURE_COLUMNS = [
    "Flow ID", "Src IP", "Dst IP", "Protocol", "Flow Duration", "Total Fwd Packets", "Total Bwd Packets",
    "Flow IAT Mean", "Flow IAT Std", "cluster",
]


def read_ids(path: Path) -> pd.DataFrame:
    """Baca kolom identitas flow apa adanya (urutan baris sama dengan load_csv)."""
    ids = pd.read_csv(
        path, usecols=lambda c: c.strip() in ID_COLUMNS, dtype=str, encoding="utf-8-sig", encoding_errors="replace"
    )
    return ids.rename(columns=lambda c: ID_COLUMNS[c.strip()])


def select_indices(clusters: np.ndarray, n_total: int, seed: int) -> list[int]:
    """Pilih n_total posisi: 1 per klaster, sisanya dibagi ke klaster terbesar (acak, reproducible)."""
    rng = np.random.RandomState(seed)
    ids, sizes = np.unique(clusters, return_counts=True)
    if n_total < len(ids):
        raise ValueError(f"n_total ({n_total}) < jumlah klaster termasuk noise ({len(ids)})")
    quota = {int(i): 1 for i in ids}
    for i in ids[np.argsort(-sizes, kind="stable")][: n_total - len(ids)]:
        quota[int(i)] += 1
    chosen: list[int] = []
    for cid in sorted(quota):
        pool = np.flatnonzero(clusters == cid)
        chosen += rng.choice(pool, size=min(quota[cid], len(pool)), replace=False).tolist()
    return sorted(chosen)


def md(df: pd.DataFrame, floatfmt: str = "{:,.4f}") -> str:
    """Render DataFrame (tanpa indeks) sebagai tabel Markdown."""
    def fmt(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            return "-" if pd.isna(v) else (f"{v:,.0f}" if float(v).is_integer() else floatfmt.format(v))
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep, *("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False))])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None, help="berkas Darknet CSV (default: satu-satunya CSV di data.raw_dir)")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--fixture", default=str(PROJECT_ROOT / "tests" / "fixtures" / "sample_real_10.csv"))
    parser.add_argument("--out", default=None, help="folder keluaran lampiran (default: <reports_dir>/lampiran)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    th, feat = cfg["thresholds"], cfg["features"]
    seed = cfg["data"]["random_state"]
    root = resolve_path(cfg["paths"]["artifacts_dir"])
    version = (root / "current.txt").read_text("utf-8").strip()
    art = root / version
    meta = json.loads((art / "train_meta.json").read_text("utf-8"))
    model_profile = json.loads((art / "cluster_profile.json").read_text("utf-8"))
    scaler = joblib.load(art / "scaler.joblib")
    sample = pd.read_csv(art / "training_sample.csv")

    src = Path(args.data) if args.data else next(resolve_path(cfg["data"]["raw_dir"]).glob("*.csv"))
    raw = load_csv(src, cfg["data"])
    raw = pd.concat([raw, read_ids(src)], axis=1)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = apply_session_filters(filter_voip(raw, cfg["data"]["voip_filter"]), cfg["data"])
    if len(final) != len(sample) or not np.allclose(final["iat_mean_us"], sample["iat_mean_us"]):
        raise SystemExit("Flow final tidak selaras dengan training_sample.csv model aktif; latih ulang atau cek config")
    final = final.assign(cluster=sample["cluster_id"].to_numpy(), jarak_terdekat=sample["jarak_terdekat"].to_numpy())

    pick = final.iloc[select_indices(final["cluster"].to_numpy(), args.n, seed)].copy()
    pick.insert(0, "sample", [f"s{i}" for i in range(1, len(pick) + 1)])

    fixture = pick.rename(
        columns={
            "flow_id": "Flow ID", "src_ip": "Src IP", "dst_ip": "Dst IP", "protocol": "Protocol",
            "duration_us": "Flow Duration", "fwd_packets_count": "Total Fwd Packets",
            "bwd_packets_count": "Total Bwd Packets", "iat_mean_us": "Flow IAT Mean", "iat_std_us": "Flow IAT Std",
        }
    )[FIXTURE_COLUMNS]
    fixture["Protocol"] = fixture["Protocol"].astype(int)
    fixture_path = Path(args.fixture)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(fixture_path, index=False)

    # data antara: tepat seperti alur pelatihan/inferensi
    d = derive_metrics(pick, feat["delay_column"], feat["jitter_column"], feat["fai_clip_negative"])
    raw_fai = (d["fwd_packets_count"] - d["bwd_packets_count"]) / d["fwd_packets_count"] * 100
    logged = build_features(d, feat["cluster_columns"], feat["log_transform"])
    z = pd.DataFrame(scaler.transform(logged), columns=[f"z_{c}" for c in logged.columns], index=d.index)
    labels = {int(c["id_klaster"]): c["deskripsi"]["ringkas"] for c in model_profile["clusters"]}

    t_raw = d[["sample", "flow_id", "src_ip", "dst_ip", "protocol", "duration_us", "fwd_packets_count",
               "bwd_packets_count", "iat_mean_us", "iat_std_us"]]
    t_derived = pd.DataFrame(
        {"sample": d["sample"], "delay_ms": d["delay_ms"], "jitter_ms": d["jitter_ms"],
         "fai_mentah_%": raw_fai, "fai_clip_%": d["fai_pct"]}
    )
    t_feat = pd.concat(
        [d[["sample"]], logged.rename(columns=lambda c: f"{c}" + (" (log1p)" if c in feat["log_transform"] else "")), z],
        axis=1,
    )
    t_assign = pd.DataFrame(
        {"sample": d["sample"], "cluster": d["cluster"], "jarak_terdekat": d["jarak_terdekat"],
         "deskripsi_klaster": [labels[int(c)] for c in d["cluster"]]}
    )
    scaler_tbl = pd.DataFrame(
        {"fitur": logged.columns, "mean": scaler.mean_, "scale (std)": scaler.scale_,
         "median imputasi (kolom mentah)": [meta["median_imputasi"].get(c, np.nan) for c in logged.columns]}
    )
    sample_profile = build_cluster_profile(d, d["cluster"].to_numpy(), cfg)
    t_prof = pd.DataFrame(
        [
            {
                "klaster": c["id_klaster"], "n": c["jumlah_anggota"],
                "mean delay": c["statistik"]["delay_ms"]["mean"], "std delay": c["statistik"]["delay_ms"]["std"] or np.nan,
                "P99,9 jitter": c["statistik_tambahan"]["p999_jitter_ms"], "PLR %": c["statistik_tambahan"]["plr_pct"],
                "Q1 delay": c["boxplot"]["delay_ms"]["q1"], "Q3 delay": c["boxplot"]["delay_ms"]["q3"],
                "pagar bawah": c["boxplot"]["delay_ms"]["lower_fence"], "pagar atas": c["boxplot"]["delay_ms"]["upper_fence"],
                "whisker": f"{c['boxplot']['delay_ms']['whisker_low']:.2f} / {c['boxplot']['delay_ms']['whisker_high']:.2f}",
                "deskripsi (10 sampel)": c["deskripsi"]["ringkas"], "deskripsi (model, 247 flow)": labels.get(c["id_klaster"], "-"),
            }
            for c in sample_profile
        ]
    )
    ev = model_profile["evaluation"]
    t_model = pd.DataFrame(
        [
            {"klaster": c["id_klaster"], "n": c["jumlah_anggota"], "mean delay": c["statistik_tambahan"]["mean_delay_ms"],
             "P99,9 jitter": c["statistik_tambahan"]["p999_jitter_ms"], "PLR %": c["statistik_tambahan"]["plr_pct"],
             "deskripsi": c["deskripsi"]["ringkas"]}
            for c in model_profile["clusters"]
        ]
    )

    out_dir = resolve_path(args.out) if args.out else resolve_path(cfg["paths"]["reports_dir"]) / "lampiran"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data antara untuk Lampiran 2–5",
        "",
        f"Sumber: `{src.name}` · model: `{version}` · random_state = {seed} · fixture: `{fixture_path.name}`",
        f"Fitur klaster: {feat['cluster_columns']} · log1p: {feat['log_transform']} · FAI clip negatif: {feat['fai_clip_negative']}",
        f"Ambang ITU: {th}",
        "",
        "Rumus: delay_ms = Flow IAT Mean / 1000; jitter_ms = Flow IAT Std / 1000; FAI = (fwd − bwd) / fwd × 100 "
        "(dijadikan 0 bila negatif); fitur Z = (x' − mean) / std dengan x' = log1p(x) untuk IAT Mean dan IAT Std.",
        "",
        "## Tabel 1 — Data mentah (Lampiran 2)", "", md(t_raw), "",
        "## Tabel 2 — Satuan asli dan FAI", "", md(t_derived), "",
        "## Tabel 3 — Fitur klaster: log1p dan Z-score (Lampiran 3)", "", md(t_feat), "",
        "Parameter StandardScaler (dari model aktif):", "", md(scaler_tbl), "",
        "## Tabel 4 — Penugasan klaster (Lampiran 4)", "",
        "`cluster` dan `jarak_terdekat` dari model aktif yang dilatih pada 247 flow (bernomor mulai 1; -1 = noise latih).", "", md(t_assign), "",
        "## Tabel 5 — Profil dari 10 sampel (Lampiran 5)", "",
        "Dihitung dari 10 flow ini saja (seperti Lampiran 5 lama); deskripsi model memakai statistik seluruh klaster.", "",
        md(t_prof), "",
        "## Tabel 6 — Profil model aktif (247 flow)", "",
        f"Silhouette = {ev['silhouette']:.4f} ({ev['silhouette_label']}) · CHI = {ev['chi']:,.2f} · "
        f"Noise Ratio = {ev['noise_ratio']:.2%} · Klaster = {ev['n_clusters']}", "",
        md(t_model), "",
    ]
    md_path = out_dir / "lampiran_data.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Fixture: {fixture_path}\nLampiran: {md_path}")
    print(fixture.to_string(index=False))


if __name__ == "__main__":
    main()
