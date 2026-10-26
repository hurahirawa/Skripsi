"""Ekspor laporan sesi: CSV jendela/panggilan dan PDF ringkasan (PRD UC-08)."""
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
from qos.monitor.calls import (  # noqa: E402
    INSUFFICIENT,
    OUTSIDE_KEY,
    SIGNALING,
    STATUS_CLUSTERED,
    STATUS_OUTSIDE,
    call_state_text,
    cluster_key,
    parse_percentages,
)
from qos.recommend.rules import (  # noqa: E402
    TITLE as RECOMMENDATION_TITLE,
    call_recommendations,
    cluster_metric_map,
    recommendations_enabled,
    session_recommendations,
)
from qos.viz.plots import state_color  # noqa: E402

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # DejaVu bawaan matplotlib (Unicode)
FONT_NAME = "DejaVu"
MAX_FLOW_ROWS = 40  # baris tabel flow di PDF


def _state_text(status: Any, cluster: Any) -> str:
    """Teks status terakhir: "Klaster N", "Di luar klaster (terdekat: Klaster N)", atau status lain."""
    return call_state_text(str(status), cluster if not pd.isna(cluster) else 0) if status in (STATUS_CLUSTERED, STATUS_OUTSIDE) else str(status)


def per_flow_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan per flow: jumlah jendela, klaster/status terakhir, jarak terakhir, rata-rata delay/jitter/FAI."""
    cols = ["flow_id", "jendela", "status_terakhir", "klaster_terakhir", "jarak_terakhir", "delay_ms", "jitter_ms", "fai_pct"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    g = rows.groupby("flow_id", sort=False)
    last_status, last_cluster = g["status"].last(), g["klaster_terdekat"].last()
    out = pd.DataFrame(
        {
            "jendela": g.size(),
            "status_terakhir": last_status,
            "klaster_terakhir": [_state_text(s, c) for s, c in zip(last_status, last_cluster)],
            "jarak_terakhir": g["jarak_terdekat"].last(),
            "delay_ms": g["delay_ms"].mean(),
            "jitter_ms": g["jitter_ms"].mean(),
            "fai_pct": g["fai_pct"].mean(),
        }
    )
    return out.reset_index()[cols]


def state_counts(rows: pd.DataFrame, cluster_ids: list[int]) -> dict[str, int]:
    """Jumlah flow per klaster, Di luar klaster, dan Data Belum Cukup menurut status terakhir (Sinyal SIP tidak dihitung)."""
    counts = {cluster_key(c): 0 for c in cluster_ids}
    counts[OUTSIDE_KEY] = 0
    counts[INSUFFICIENT] = 0
    if rows.empty:
        return counts
    last = rows[rows["status"] != SIGNALING].groupby("flow_id", sort=False)[["status", "klaster_terdekat"]].last()
    for status, cluster in zip(last["status"], last["klaster_terdekat"]):
        if status == STATUS_CLUSTERED:
            key = cluster_key(cluster)
            counts[key] = counts.get(key, 0) + 1
        elif status == STATUS_OUTSIDE:
            counts[OUTSIDE_KEY] += 1
        elif status == INSUFFICIENT:
            counts[INSUFFICIENT] += 1
    return counts


def distribution_chart(counts: dict[str, int]) -> bytes:
    """Bar chart jumlah flow per klaster/status (warna kategorikal, tanpa makna baik/buruk) sebagai PNG."""
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=150)
    ax.bar(list(counts), list(counts.values()), color=[state_color(k) for k in counts])
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
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return "-"
    return f"{v:,.2f}" if isinstance(v, float) else str(v)


def build_session_pdf(
    summary: dict[str, Any],
    rows: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
    model_profile: dict[str, Any] | None = None,
    calls: pd.DataFrame | None = None,
    show_recommendations: bool = True,
) -> bytes:
    """PDF ringkasan sesi: info sesi dan model, distribusi klaster, saran, tabel panggilan dan flow, profil klaster."""
    has_status = "status" in rows
    n_signaling = rows.loc[rows["status"] == SIGNALING, "flow_id"].nunique() if has_status else 0
    flows = per_flow_table(rows[rows["status"] != SIGNALING] if has_status else rows)  # sinyal SIP bukan aliran dinilai
    clusters = model_profile["clusters"] if model_profile else []
    cluster_ids = sorted(int(c["id_klaster"]) for c in clusters if int(c["id_klaster"]) != -1)
    counts = state_counts(rows if has_status else rows.iloc[0:0], cluster_ids)
    metric_map = cluster_metric_map(clusters)
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
    pdf.cell(0, 10, "Laporan Sesi Pemantauan VoIP (Klasterisasi)", new_x="LMARGIN", new_y="NEXT")
    line(f"Dibuat {datetime.now():%Y-%m-%d %H:%M:%S}")

    heading("Informasi sesi")
    info = [
        ["ID sesi", _fmt(summary.get("id_sesi_pemantauan"))],
        ["Waktu mulai / selesai", f"{_fmt(summary.get('waktu_mulai'))} / {_fmt(summary.get('waktu_selesai'))}"],
        ["Status sesi", _fmt(summary.get("status_sesi"))],
        ["Total aliran", _fmt(summary.get("total_aliran"))],
        ["Klaster mayoritas (panggilan)", _fmt(summary.get("klaster_mayoritas"))],
        ["% panggilan di luar klaster", _fmt(float(summary["persen_di_luar_klaster"])) if "persen_di_luar_klaster" in summary and not pd.isna(summary["persen_di_luar_klaster"]) else "-"],
        ["Jumlah baris jendela", str(len(rows))],
    ]
    if model_profile:
        ev = model_profile["evaluation"]
        info += [
            ["Model", str(model_profile.get("model_version", "-"))],
            ["Silhouette / DBCV / Noise Ratio",
             f"{ev['silhouette']:.4f} ({ev['silhouette_label']}) / {_fmt(ev.get('dbcv'))} / {ev['noise_ratio']:.2%}"],
            ["Jumlah klaster", str(ev["n_clusters"])],
        ]
    table(["Item", "Nilai"], info, [65, 125])

    heading("Distribusi klaster (klaster terdekat terakhir tiap aliran)")
    pdf.image(io.BytesIO(distribution_chart(counts)), w=170)
    if counts.get(INSUFFICIENT):
        line(f"{counts[INSUFFICIENT]} aliran berstatus 'Data Belum Cukup' dan tidak ditugaskan ke klaster.")
    if n_signaling:
        line(f"{n_signaling} flow sinyal SIP dicatat di CSV jendela tetapi tidak dinilai dan tidak dihitung.")

    if show_recommendations:
        heading(RECOMMENDATION_TITLE)
        recs = session_recommendations(rows, metric_map, rules) if has_status else []
        if not recs:
            line("Tidak ada klaster dengan metrik yang melewati ambang ITU dan tidak ada aliran di luar klaster pada sesi ini.")
        for r in recs:
            line(f"{r['kunci']} — {r['jumlah_flow']} aliran ({r['kategori']}): {'; '.join(r['pemicu'])}", bold=True)
            for s in r["saran"]:
                line(f"•  {s}")
            pdf.ln(1)

    if calls is not None and len(calls):
        heading(f"Panggilan selesai (maks {MAX_FLOW_ROWS} dari {len(calls)})")
        if calls["catatan"].fillna("").ne("").any():
            line("Catatan: sesi memakai mode window; klaster akhir berasal dari jendela terakhir, bukan kumulatif.")
        table(
            ["Panggilan", "Durasi s", "Jendela", "Klaster akhir", "Jarak", "% di luar", "Delay ms", "FAI %"],
            [
                [r.flow_id, _fmt(float(r.durasi_s)), str(r.jumlah_jendela), _state_text(r.status_akhir, r.klaster_akhir),
                 _fmt(float(r.jarak_akhir)) if not pd.isna(r.jarak_akhir) else "-", _fmt(float(r.persen_di_luar_klaster)),
                 _fmt(r.delay_ms), _fmt(r.fai_pct)]
                for r in calls.head(MAX_FLOW_ROWS).itertuples()
            ],
            [44, 14, 14, 44, 14, 16, 22, 16],
        )
        if show_recommendations:
            call_recs = call_recommendations(calls, metric_map, rules)
            heading("Saran per panggilan")
            if not call_recs:
                line("Tidak ada panggilan yang memerlukan saran.")
            for r in call_recs:
                line(f"{r['flow_id']} — {r['kunci']} ({r['kategori']})", bold=True)
                for sg in r["saran"]:
                    line(f"•  {sg}")
                pdf.ln(1)

    heading(f"Aliran (maks {MAX_FLOW_ROWS} dari {len(flows)})")
    table(
        ["Flow", "Jendela", "Klaster terakhir", "Jarak", "Delay ms", "Jitter ms", "FAI %"],
        [
            [r.flow_id, str(r.jendela), r.klaster_terakhir, _fmt(r.jarak_terakhir), _fmt(r.delay_ms), _fmt(r.jitter_ms), _fmt(r.fai_pct)]
            for r in flows.head(MAX_FLOW_ROWS).itertuples()
        ],
        [46, 14, 46, 14, 22, 24, 24],
    )

    if model_profile:
        heading("Profil klaster model (posisi terhadap ambang ITU = deskripsi, bukan label kualitas)")
        table(
            ["Klaster", "n", "Median delay ms", "Median jitter ms", "Median FAI %", "Batas jarak", "Deskripsi"],
            [
                [c["nama"], str(c["jumlah_anggota"]), _fmt(c["statistik"]["delay_ms"]["median"]), _fmt(c["statistik"]["jitter_ms"]["median"]),
                 _fmt(c["statistik"]["fai_pct"]["median"]), _fmt(c.get("batas_jarak")), c["deskripsi"]["ringkas"]]
                for c in clusters
            ],
            [18, 10, 24, 24, 22, 20, 72],
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
    pdf_path.write_bytes(build_session_pdf(summary, rows, rules, model_profile, calls, recommendations_enabled(cfg)))
    out = {"csv": csv_path, "pdf": pdf_path}
    if calls is not None:
        out["calls_csv"] = out_dir / f"laporan_panggilan_{sid}.csv"
        shutil.copyfile(calls_csv, out["calls_csv"])
    return out
