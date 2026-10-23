"""Ekspor laporan sesi: CSV jendela dan PDF ringkasan (PRD UC-08)."""
from __future__ import annotations

import io
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402

from qos.config import resolve_path  # noqa: E402
from qos.monitor.engine import CARD_KEYS, INSUFFICIENT, SIGNALING, card_category  # noqa: E402
from qos.recommend.rules import call_recommendations, session_recommendations  # noqa: E402
from qos.viz.plots import LABEL_COLORS  # noqa: E402

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # DejaVu bawaan matplotlib (Unicode)
FONT_NAME = "DejaVu"
MAX_FLOW_ROWS = 40  # baris tabel flow di PDF
CATEGORIES: tuple[str, ...] = (*CARD_KEYS, INSUFFICIENT)


def per_flow_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan per flow: jumlah jendela, label terakhir, rata-rata delay/jitter/FAI."""
    if rows.empty:
        return pd.DataFrame(columns=["flow_id", "jendela", "label_terakhir", "delay_ms", "jitter_ms", "fai_pct"])
    g = rows.groupby("flow_id", sort=False)
    out = pd.DataFrame(
        {
            "jendela": g.size(),
            "label_terakhir": g["label"].last(),
            "delay_ms": g["delay_ms"].mean(),
            "jitter_ms": g["jitter_ms"].mean(),
            "fai_pct": g["fai_pct"].mean(),
        }
    )
    return out.reset_index()


def category_counts(flows: pd.DataFrame) -> dict[str, int]:
    """Jumlah flow per kategori berdasarkan label terakhir (termasuk Data Belum Cukup)."""
    counts = {k: 0 for k in CATEGORIES}
    for label in flows["label_terakhir"]:
        cat = card_category(str(label))
        if cat in counts:  # "Sinyal SIP" tidak dihitung
            counts[cat] += 1
    return counts


def distribution_chart(counts: dict[str, int]) -> bytes:
    """Bar chart jumlah flow per kategori (warna sesuai PRD) sebagai PNG."""
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=150)
    ax.bar(list(counts), list(counts.values()), color=[LABEL_COLORS[k] for k in counts])
    for i, v in enumerate(counts.values()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("jumlah flow")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _fmt(v: Any) -> str:
    """Teks sel: '-' untuk kosong, 2 desimal untuk float."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:,.2f}" if isinstance(v, float) else str(v)


def build_session_pdf(
    summary: dict[str, Any],
    rows: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
    model_profile: dict[str, Any] | None = None,
    calls: pd.DataFrame | None = None,
) -> bytes:
    """PDF ringkasan sesi: info sesi dan model, distribusi label, rekomendasi, tabel flow, profil klaster."""
    n_signaling = rows.loc[rows["label"] == SIGNALING, "flow_id"].nunique() if "label" in rows else 0
    flows = per_flow_table(rows[rows["label"] != SIGNALING] if "label" in rows else rows)  # sinyal SIP bukan aliran dinilai
    counts = category_counts(flows)
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font(FONT_NAME, "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font(FONT_NAME, "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_page()

    def heading(text: str) -> None:
        pdf.ln(3)
        pdf.set_font(FONT_NAME, "B", 12)
        pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(FONT_NAME, "", 9)

    def line(text: str, bold: bool = False) -> None:
        pdf.set_font(FONT_NAME, "B" if bold else "", 9)
        pdf.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(FONT_NAME, "", 9)

    def table(header: list[str], body: list[list[str]], widths: list[float]) -> None:
        with pdf.table(col_widths=widths, text_align="LEFT", line_height=5) as t:
            row = t.row()
            for c in header:
                row.cell(c)
            for r in body:
                row = t.row()
                for c in r:
                    row.cell(c)

    pdf.set_font(FONT_NAME, "B", 16)
    pdf.cell(0, 10, "Laporan Sesi Pemantauan QoS VoIP", new_x="LMARGIN", new_y="NEXT")
    line(f"Dibuat {datetime.now():%Y-%m-%d %H:%M:%S}")

    heading("Informasi sesi")
    info = [
        ["ID sesi", _fmt(summary.get("id_sesi_pemantauan"))],
        ["Waktu mulai / selesai", f"{_fmt(summary.get('waktu_mulai'))} / {_fmt(summary.get('waktu_selesai'))}"],
        ["Status keseluruhan", _fmt(summary.get("status_keseluruhan"))],
        ["Total aliran", _fmt(summary.get("total_aliran"))],
        ["Jumlah baris jendela", str(len(rows))],
    ]
    if model_profile:
        ev = model_profile["evaluation"]
        info += [
            ["Model", str(model_profile.get("model_version", "-"))],
            ["Silhouette / CHI / Noise Ratio",
             f"{ev['silhouette']:.4f} ({ev['silhouette_label']}) / {ev['chi']:,.2f} / {ev['noise_ratio']:.2%}"],
            ["Jumlah klaster", str(ev["n_clusters"])],
        ]
    table(["Item", "Nilai"], info, [55, 135])

    heading("Distribusi label (label terakhir tiap aliran)")
    pdf.image(io.BytesIO(distribution_chart(counts)), w=170)
    if counts[INSUFFICIENT]:
        line(f"{counts[INSUFFICIENT]} aliran berstatus 'Data Belum Cukup' dan tidak dihitung sebagai Anomali maupun status Kritis.")

    if n_signaling:
        line(f"{n_signaling} flow sinyal SIP dicatat di CSV jendela tetapi tidak dinilai dan tidak dihitung.")

    heading("Rekomendasi mitigasi")
    recs = session_recommendations(rows, rules)
    if not recs:
        line("Tidak ada aliran QoS Buruk atau Anomali pada sesi ini.")
    for r in recs:
        line(f"{r['label']} — {r['jumlah_flow']} aliran ({r['kategori']})", bold=True)
        for s in r["saran"]:
            line(f"•  {s}")
        pdf.ln(1)

    if calls is not None and len(calls):
        heading(f"Panggilan selesai (maks {MAX_FLOW_ROWS} dari {len(calls)})")
        if calls["catatan"].fillna("").ne("").any():
            line("Catatan: sesi memakai mode window; label_akhir berasal dari jendela terakhir, bukan kumulatif.")
        table(
            ["Panggilan", "Durasi s", "Jendela", "Label akhir", "Label terburuk", "Delay ms", "FAI %"],
            [
                [r.flow_id, _fmt(float(r.durasi_s)), str(r.jumlah_jendela), r.label_akhir, r.label_terburuk,
                 _fmt(r.delay_ms), _fmt(r.fai_pct)]
                for r in calls.head(MAX_FLOW_ROWS).itertuples()
            ],
            [44, 14, 14, 42, 42, 18, 16],
        )
        call_recs = call_recommendations(calls, rules)
        heading("Rekomendasi per panggilan bermasalah")
        if not call_recs:
            line("Tidak ada panggilan bermasalah.")
        for r in call_recs:
            line(f"{r['flow_id']} — {r['label']} ({r['kategori']})", bold=True)
            for sg in r["saran"]:
                line(f"•  {sg}")
            pdf.ln(1)

    heading(f"Aliran (maks {MAX_FLOW_ROWS} dari {len(flows)})")
    table(
        ["Flow", "Jendela", "Label terakhir", "Delay ms", "Jitter ms", "FAI %"],
        [
            [r.flow_id, str(r.jendela), r.label_terakhir, _fmt(r.delay_ms), _fmt(r.jitter_ms), _fmt(r.fai_pct)]
            for r in flows.head(MAX_FLOW_ROWS).itertuples()
        ],
        [50, 18, 52, 22, 24, 24],
    )

    if model_profile:
        heading("Profil klaster model")
        table(
            ["Klaster", "n", "Mean delay ms", "P99,9 jitter ms", "PLR %", "Label"],
            [
                [str(c["id_klaster"]), str(c["jumlah_anggota"]), _fmt(c["statistik_pelabelan"]["mean_delay_ms"]),
                 _fmt(c["statistik_pelabelan"]["p999_jitter_ms"]), _fmt(c["statistik_pelabelan"]["plr_pct"]), c["label"]]
                for c in model_profile["clusters"]
            ],
            [16, 12, 26, 28, 18, 90],
        )
    return bytes(pdf.output())


def export_session(
    summary: dict[str, Any], cfg: dict[str, Any], rules: dict[str, dict[str, Any]], model_profile: dict[str, Any] | None = None
) -> dict[str, Path]:
    """Tulis laporan sesi (salinan CSV jendela + PDF) ke <reports_dir>/exports dan kembalikan path-nya."""
    window_csv = Path(summary["lokasi_laporan"])
    if not window_csv.is_file():
        raise FileNotFoundError(f"CSV jendela tidak ditemukan: {window_csv}")
    rows = pd.read_csv(window_csv)
    out_dir = resolve_path(cfg["paths"]["reports_dir"]) / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = str(summary["id_sesi_pemantauan"])
    csv_path, pdf_path = out_dir / f"laporan_{sid}.csv", out_dir / f"laporan_{sid}.pdf"
    shutil.copyfile(window_csv, csv_path)
    calls_csv = window_csv.with_name(window_csv.name.replace("jendela_", "panggilan_", 1))
    calls = pd.read_csv(calls_csv) if calls_csv.is_file() else None
    pdf_path.write_bytes(build_session_pdf(summary, rows, rules, model_profile, calls))
    out = {"csv": csv_path, "pdf": pdf_path}
    if calls is not None:
        out["calls_csv"] = out_dir / f"laporan_panggilan_{sid}.csv"
        shutil.copyfile(calls_csv, out["calls_csv"])
    return out
