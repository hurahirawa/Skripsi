"""Laporan konfigurasi final (M1): jumlah flow, ringkasan fitur, dan distribusi posisi terhadap ambang ITU per flow.

Posisi dihitung per flow SEBELUM klasterisasi (delay = iat_mean, jitter = iat_std, loss = FAI), hanya sebagai
gambaran sebaran (deskriptif, bukan label kualitas); deskripsi per klaster ada di profil model.

Pemakaian: python scripts/report_final.py --config config/default_config.yaml [--data path] [--out folder]
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from qos.config import load_config, resolve_path, setup_logging
from qos.data.loader import filter_voip, load_flows
from qos.features.preprocess import apply_session_filters, compute_fai
from qos.profiling.describe import position

PERCENTILES = (0.5, 0.95)


def _fmt(v: float) -> str:
    """Format angka: bilangan bulat dengan pemisah ribuan, selain itu 2 desimal."""
    if pd.isna(v):
        return "-"
    return f"{int(v):,}" if float(v).is_integer() else f"{v:,.2f}"


def _md(df: pd.DataFrame, first: str = "") -> str:
    """Render DataFrame sebagai tabel Markdown."""
    head = f"| {first} | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|---|" + "---:|" * len(df.columns)
    rows = [f"| {i} | " + " | ".join(_fmt(v) for v in r) + " |" for i, r in df.iterrows()]
    return "\n".join([head, sep, *rows])


def summary(df: pd.DataFrame, columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Median dan p95 per kolom (inf dijadikan NaN, NaN diabaikan)."""
    rows = {}
    for name, s in columns.items():
        s = s.replace([np.inf, -np.inf], np.nan)
        rows[name] = {"n": int(s.notna().sum()), "nan/inf": int(s.isna().sum()),
                      "median": s.median(), "p95": s.quantile(0.95)}
    return pd.DataFrame.from_dict(rows, orient="index")


def distribution(levels: pd.DataFrame) -> pd.DataFrame:
    """Jumlah dan persen flow per tingkat (1/2/3) untuk tiap metrik dan total."""
    out = {}
    n = len(levels)
    for col in levels.columns:
        counts = levels[col].value_counts()
        row = {f"L{lv}": int(counts.get(lv, 0)) for lv in (1, 2, 3)}
        row.update({f"L{lv} %": row[f"L{lv}"] / n * 100 if n else float("nan") for lv in (1, 2, 3)})
        row["NaN"] = int(levels[col].isna().sum())
        out[col] = row
    return pd.DataFrame.from_dict(out, orient="index")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    data_cfg, th = cfg["data"], cfg["thresholds"]
    source = resolve_path(args.data or data_cfg["raw_dir"])
    out_dir = resolve_path(args.out) if args.out else resolve_path(cfg["paths"]["reports_dir"]) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_flows(source, data_cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        voip = filter_voip(df, data_cfg["voip_filter"])
    final = apply_session_filters(voip, data_cfg)

    fai = compute_fai(final["fwd_packets_count"], final["bwd_packets_count"], cfg["features"]["fai_clip_negative"])
    delay_ms = final["iat_mean_us"] / 1000
    jitter_ms = final["iat_std_us"] / 1000
    levels = pd.DataFrame(
        {
            m: s.map(lambda v, m=m: position(m, v, th) if pd.notna(v) else np.nan)
            for m, s in (("delay", delay_ms), ("jitter", jitter_ms), ("loss", fai))
        }
    )
    levels["total"] = levels.max(axis=1, skipna=False)

    warn_min = data_cfg["voip_filter"]["min_flows_warning"]
    ms, ex = data_cfg["min_session"], data_cfg["exclude_traffic_types"]
    stages = pd.DataFrame(
        {"jumlah": [len(df), len(voip), len(final)]},
        index=["Total flow", "Lolos filter VoIP", f"Final (exclude {ex}, fwd≥{ms['fwd']}, bwd≥{ms['bwd']})"],
    )
    proto = final["protocol"].value_counts().sort_index().to_frame("jumlah")
    proto.index = [f"protokol {int(p)}" for p in proto.index]
    types = final["traffic_type"].value_counts().to_frame("jumlah") if "traffic_type" in final else None

    lines = [
        "# Laporan konfigurasi final (M1)",
        "",
        f"Sumber: `{source}` · proksi delay: Flow IAT Mean · kolom klaster: {cfg['features']['cluster_columns']}",
        f"Ambang: {th} · FAI clip negatif: {cfg['features']['fai_clip_negative']}",
        "",
        "## Jumlah flow",
        "",
        _md(stages),
        "",
        *([f"> PERINGATAN: flow final ({len(final):,}) di bawah min_flows_warning ({warn_min})", ""]
          if len(final) < warn_min else [f"Flow final ({len(final):,}) ≥ min_flows_warning ({warn_min}).", ""]),
        "Per protokol:",
        "",
        _md(proto),
        "",
        *(["Per traffic type:", "", _md(types), ""] if types is not None else []),
        "## Median dan p95 fitur (flow final)",
        "",
        _md(summary(final, {
            "iat_mean_us": final["iat_mean_us"],
            "iat_std_us": final["iat_std_us"],
            "fai_pct": fai,
            "delay_ms (iat_mean/1000)": delay_ms,
            "jitter_ms (iat_std/1000)": jitter_ms,
        })),
        "",
        "## Distribusi posisi terhadap ambang ITU per flow (sebelum klasterisasi)",
        "",
        "L1 = di dalam batas baik, L2 = di antara batas baik dan maksimum, L3 = melewati batas maksimum; total = max ketiga metrik (deskriptif).",
        "Per flow: jitter memakai iat_std dan loss memakai FAI ter-clip.",
        "",
        _md(distribution(levels)),
        "",
    ]
    for name, sub in (final.groupby("traffic_type") if "traffic_type" in final else []):
        lines += [f"### {name} (n = {len(sub):,})", "", _md(distribution(levels.loc[sub.index])), ""]
    text = "\n".join(lines)
    path = out_dir / "final_config_m1.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Laporan: {path}")


if __name__ == "__main__":
    main()
