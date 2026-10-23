"""Perbandingan varian fitur klaster (config experiments.feature_variants); tidak mengubah current.txt.

Tiap varian dilatih penuh (grid search) dan artefaknya disimpan dengan activate=False.
Profil dan label tetap dihitung dari metrik satuan asli (ms, %), bukan dari nilai log.

Pemakaian: python scripts/compare_variants.py --config config/default_config.yaml [--data path] [--out folder]
"""
from __future__ import annotations

import argparse
import copy
import sys

import pandas as pd

from qos.config import load_config, resolve_path, setup_logging
from qos.model.train import load_training_frame, save_artifacts, train


def _fmt(v: object) -> str:
    """Format sel tabel Markdown."""
    if isinstance(v, bool):
        return "ya" if v else "tidak"
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer() and abs(v) > 1000):
        return f"{int(v):,}"
    if isinstance(v, float):
        return "-" if pd.isna(v) else f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    return str(v)


def md(df: pd.DataFrame) -> str:
    """Render DataFrame (tanpa indeks) sebagai tabel Markdown."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def profile_table(profile: list[dict]) -> pd.DataFrame:
    """Tabel profil per klaster dari statistik satuan asli."""
    rows = []
    for c in profile:
        s = c["statistik_pelabelan"]
        rows.append(
            {"klaster": c["id_klaster"], "n": c["jumlah_anggota"], "mean delay (ms)": s["mean_delay_ms"],
             "P99,9 jitter (ms)": s["p999_jitter_ms"], "PLR (%)": s["plr_pct"], "label": c["label"]}
        )
    return pd.DataFrame(rows)


def fai_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    """Diagnostik FAI: flow |fwd − bwd| = 1, FAI nol, dan FAI minimum yang bukan 0."""
    diff = (frame["fwd_packets_count"] - frame["bwd_packets_count"]).abs()
    fai = frame["fai_pct"]
    pos = fai[fai > 0]
    one = frame[diff == 1]
    return {
        "n": len(frame),
        "selisih_abs_1": int((diff == 1).sum()),
        "selisih_abs_1_fwd_lebih_besar": int((one["fwd_packets_count"] > one["bwd_packets_count"]).sum()),
        "fai_nol": int((fai == 0).sum()),
        "fai_positif": int(len(pos)),
        "fai_min_bukan_nol": float(pos.min()) if len(pos) else float("nan"),
        "fai_median_positif": float(pos.median()) if len(pos) else float("nan"),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    exp = cfg["experiments"]
    source = resolve_path(args.data or cfg["data"]["raw_dir"])
    out_dir = resolve_path(args.out) if args.out else resolve_path(cfg["paths"]["reports_dir"]) / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    flows, counts = load_training_frame(source, cfg)
    th = cfg["thresholds"]
    lines = [
        "# Perbandingan varian fitur klaster (M3)",
        "",
        f"Flow latih: {len(flows):,} · ambang ITU tidak diubah: {th}",
        "Profil dan label dari metrik satuan asli (ms, %); varian hanya mengubah ruang fitur HDBSCAN.",
        "",
    ]
    diag = None
    for name, spec in exp["feature_variants"].items():
        vcfg = copy.deepcopy(cfg)
        vcfg["features"].update(spec)
        result = train(flows, vcfg, counts=counts)
        out = save_artifacts(result, vcfg, activate=False)
        ev, params = result["evaluation"], result["params"]
        grid = result["grid"]
        top = grid[grid["valid"]].sort_values(["silhouette", "chi"], ascending=False, kind="stable").head(exp["top_n"])
        diag = fai_diagnostics(result["frame"])
        lines += [
            f"## Varian {name}",
            "",
            f"Fitur: `{spec['cluster_columns']}` · log1p: `{spec['log_transform'] or 'tidak ada'}` · artefak: `{out.name}`",
            f"Kandidat valid: {int(grid['valid'].sum())} dari {len(grid)}",
            "",
            f"### {exp['top_n']} kandidat valid teratas",
            "",
            md(top[["min_cluster_size", "min_samples", "silhouette", "chi", "noise_ratio", "n_clusters"]]),
            "",
            f"**Terpilih:** min_cluster_size = {params['min_cluster_size']}, min_samples = {params['min_samples']}"
            + (" (FALLBACK: tidak ada kandidat valid)" if params["grid_fallback"] else ""),
            "",
            f"Silhouette = {ev['silhouette']:.4f} ({ev['silhouette_label']}) · CHI = {ev['chi']:,.2f} · "
            f"Noise Ratio = {ev['noise_ratio']:.2%} · Klaster = {ev['n_clusters']}",
            "",
            "### Profil per klaster",
            "",
            md(profile_table(result["profile"])),
            "",
        ]
    lines += [
        "## Diagnostik FAI (flow latih, FAI ter-clip)",
        "",
        md(pd.DataFrame([diag])),
        "",
        "`selisih_abs_1` = flow dengan |fwd − bwd| = 1; `fai_min_bukan_nol` = FAI positif terkecil (%).",
        "",
    ]
    text = "\n".join(lines)
    path = out_dir / "compare_features.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Laporan: {path}")


if __name__ == "__main__":
    main()
