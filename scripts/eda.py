"""Laporan EDA M1: jumlah flow, rincian kandidat VoIP, dan distribusi 4 fitur.

Pemakaian: python scripts/eda.py --config config/default_config.yaml [--data path] [--out folder]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qos.config import load_config, resolve_path, setup_logging
from qos.data.eda import (
    corrupt_counts,
    describe_by_group,
    describe_features,
    packet_medians,
    voip_breakdown,
    voip_by_group,
)
from qos.data.loader import filter_voip, load_flows

PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
HIST_BINS = 60


def _pct(part: int, whole: int) -> str:
    """Format persentase aman terhadap pembagi nol."""
    return f"{part / whole * 100:.2f}%" if whole else "-"


def _md_table(df: pd.DataFrame) -> str:
    """Render DataFrame sebagai tabel Markdown."""
    head = "| fitur | " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|---|" + "---:|" * len(df.columns)
    fmt = lambda v: f"{int(v):,}" if float(v).is_integer() else f"{v:,.2f}"  # noqa: E731
    rows = ["| " + str(idx) + " | " + " | ".join(fmt(v) for v in row) + " |" for idx, row in df.iterrows()]
    return "\n".join([head, sep, *rows])


def plot_histograms(df: pd.DataFrame, columns: list[str], path: Path, group_col: str = "traffic_type") -> None:
    """Simpan histogram log10(x+1) tiap fitur, satu garis per kelompok."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    groups = sorted(df[group_col].dropna().unique()) if group_col in df.columns else [None]
    for ax, col in zip(axes.ravel(), columns):
        for g in groups:
            sub = df if g is None else df[df[group_col] == g]
            values = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
            values = values[values >= 0]
            ax.hist(np.log10(values + 1), bins=HIST_BINS, histtype="step", label=f"{g} ({len(values):,})")
        ax.set_title(col)
        ax.set_xlabel("log10(x + 1)")
        ax.set_ylabel("jumlah flow")
        ax.legend(fontsize=7)
    fig.suptitle("Distribusi fitur flow VoIP per kelompok")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None, help="berkas CSV atau folder (default: data.raw_dir)")
    parser.add_argument("--out", default=None, help="folder keluaran (default: <reports_dir>/eda)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    voip_cfg = cfg["data"]["voip_filter"]
    suspect_ports = cfg["eda"]["suspect_ports"]
    columns = cfg["features"]["cluster_columns"]
    source = resolve_path(args.data or cfg["data"]["raw_dir"])
    out_dir = resolve_path(args.out) if args.out else resolve_path(cfg["paths"]["reports_dir"]) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_flows(source, cfg["data"])
    has_groups = "traffic_type" in df.columns
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        voip = filter_voip(df, voip_cfg)
    hist_path = out_dir / "eda_m1_hist.png"
    plot_histograms(voip, columns, hist_path)
    udp_total = int((df["protocol"] == voip_cfg["protocol"]).sum())
    b = voip_breakdown(df, voip_cfg, suspect_ports)
    all_cols = (*columns, "protocol", "src_port", "dst_port")

    lines = [
        "# Laporan EDA M1",
        "",
        f"Sumber: `{source}` · mode filter: `{voip_cfg['mode']}`"
        + (f" (kategori == {voip_cfg['label_value']})" if voip_cfg["mode"] == "label" else ""),
        "",
        "## Jumlah flow",
        "",
        "| Tahap | Jumlah | Proporsi |",
        "|---|---:|---:|",
        f"| Total flow | {len(df):,} | 100% |",
        f"| UDP (protocol == {voip_cfg['protocol']}) | {udp_total:,} | {_pct(udp_total, len(df))} |",
        f"| **Lolos filter VoIP** | **{len(voip):,}** | {_pct(len(voip), len(df))} |",
        "",
        *(f"> PERINGATAN: {w.message}" for w in caught),
        "",
    ]
    if has_groups:
        lines += ["## Flow VoIP per traffic type", "", _md_table(voip_by_group(df, voip)), ""]
        proto = voip.groupby("traffic_type")["protocol"].value_counts().unstack(fill_value=0)
        lines += ["Protokol flow VoIP per traffic type:", "", _md_table(proto), ""]
    lines += [
        "## Nilai rusak per kolom",
        "",
        "Seluruh flow (nan = kosong/tak terbaca):",
        "",
        _md_table(corrupt_counts(df)),
        "",
        "Flow VoIP:",
        "",
        _md_table(corrupt_counts(voip)),
        "",
        "## Median paket per arah (flow VoIP)",
        "",
        _md_table(packet_medians(voip)) if has_groups else "(tidak ada kolom traffic_type)",
        "",
        "## Distribusi 4 fitur (flow VoIP)",
        "",
    ]
    clean = voip.replace([np.inf, -np.inf], np.nan)
    complete = int(clean[columns].notna().all(axis=1).sum())
    lines += [f"Baris dengan 4 fitur lengkap: {complete:,} dari {len(voip):,}", ""]
    lines += ["### Semua kelompok", "", _md_table(describe_features(clean, columns, PERCENTILES)), ""]
    if has_groups:
        for name, desc in describe_by_group(voip, columns, PERCENTILES).items():
            lines += [f"### {name}", "", _md_table(desc), ""]
    lines += [f"![Histogram fitur]({hist_path.name})", ""]
    if voip_cfg["mode"] == "port":
        rtp_clean = b["rtp_only"] - b["rtp_only_suspect"]
        lines += [
            "## Rincian filter port",
            "",
            f"Port SIP: {voip_cfg['sip_ports']} · rentang RTP: {voip_cfg['rtp_port_ranges']}",
            "",
            f"SIP {b['sip']:,} (curiga {b['sip_suspect']:,}) · hanya-RTP {b['rtp_only']:,} "
            f"(curiga {b['rtp_only_suspect']:,}, bersih {rtp_clean:,})",
            "",
        ]
    else:
        lines += [
            "Pembanding filter port (tidak dipakai): "
            f"UDP + port SIP/RTP meloloskan {b['voip']:,} flow dari {b['total']:,}.",
            "",
        ]
    report_path = out_dir / "eda_m1.md"
    text = "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Laporan: {report_path}")


if __name__ == "__main__":
    main()
