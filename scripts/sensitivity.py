"""Analisis sensitivitas filter sesi minimum (fwd ≥ N dan bwd ≥ N) pada flow VoIP; hanya laporan.

Pemakaian: python scripts/sensitivity.py --config config/default_config.yaml [--data path] [--out folder]
"""
from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from qos.config import load_config, resolve_path, setup_logging
from qos.data.eda import sensitivity
from qos.data.loader import filter_voip, load_flows


def _md(df: pd.DataFrame) -> str:
    """Render DataFrame sebagai tabel Markdown (indeks = N)."""
    fmt = lambda v: f"{int(v):,}" if float(v).is_integer() else f"{v:,.2f}"  # noqa: E731
    head = "| N | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|---|" + "---:|" * len(df.columns)
    rows = [f"| {i} | " + " | ".join(fmt(v) for v in r) + " |" for i, r in df.iterrows()]
    return "\n".join([head, sep, *rows])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    eda = cfg["eda"]
    clip = cfg["features"]["fai_clip_negative"]
    source = resolve_path(args.data or cfg["data"]["raw_dir"])
    out_dir = resolve_path(args.out) if args.out else resolve_path(cfg["paths"]["reports_dir"]) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_flows(source, cfg["data"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        voip = filter_voip(df, cfg["data"]["voip_filter"])
    res = sensitivity(
        voip, eda["sensitivity_n"], eda["sensitivity_exclude"],
        eda["duration_cap_s"] * 1e6, eda["duration_cap_tol_ms"] * 1e3, clip,
    )
    wo = "+".join(eda["sensitivity_exclude"])
    lines = [
        "# Sensitivitas filter sesi minimum (M1)",
        "",
        f"Sumber: `{source}` · flow VoIP awal: {len(voip):,} · filter: fwd ≥ N dan bwd ≥ N · "
        f"FAI clip negatif: {clip} · batas durasi: {eda['duration_cap_s']} s "
        f"(toleransi {eda['duration_cap_tol_ms']} ms)",
        "",
        "Hanya laporan; tidak ada nilai N yang ditetapkan.",
        "",
        f"## Jumlah flow tersisa per traffic type (`total_tanpa` = total tanpa {wo})",
        "",
        _md(res["jumlah"]),
        "",
        "## Jumlah flow tersisa per protokol",
        "",
        _md(res["protokol"]),
        "",
        "## Ringkasan fitur — semua traffic type",
        "",
        _md(res["statistik_semua"]),
        "",
        f"## Ringkasan fitur — tanpa {wo}",
        "",
        _md(res["statistik_tanpa"]),
        "",
        "Keterangan: `iat_nol_pct` = % iat_std = 0; `durasi_batas_pct` = % durasi menempel di batas; "
        "duration/iat dalam µs, FAI dalam %.",
        "",
    ]
    text = "\n".join(lines)
    path = out_dir / "sensitivity_m1.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Laporan: {path}")


if __name__ == "__main__":
    main()
