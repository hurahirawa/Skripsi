"""P4 Saran berdasarkan karakteristik klaster & Ekspor Laporan: saran per klaster dan per sesi, riwayat, unduh CSV dan PDF."""
import pandas as pd
import streamlit as st

import ui
from qos.monitor.calls import parse_counts
from qos.recommend.rules import (
    TITLE as RECOMMENDATION_TITLE,
    call_recommendations,
    cluster_metric_map,
    cluster_recommendations,
    recommendations_enabled,
    session_recommendations,
)
from qos.report.export import export_session

st.set_page_config(page_title="Saran & Ekspor", page_icon="📝", layout="wide")
cfg = ui.get_cfg()
art = ui.require_artifacts(cfg)
ui.render_sidebar(cfg, art)
rules = ui.load_rules_cached(cfg)
enabled = recommendations_enabled(cfg)
metric_map = cluster_metric_map(art.profile["clusters"])

st.title("Ekspor Laporan dan Saran Klaster")


def show_recommendation(rec: dict, title: str) -> None:
    """Kotak saran: investigasi kasus tunggal (di luar klaster) memakai kotak merah, selain itu kuning; tanpa makna kualitas."""
    box = st.error if rec["kategori"] == "Investigasi Kasus Tunggal" else st.warning
    items = "\n".join(f"- {s}" for s in rec["saran"])
    box(f"**{title}**  \nTindakan: {rec['kategori']} · pemicu: {', '.join(rec['pemicu'])}\n\n{items}")


if enabled:
    st.subheader(f"{RECOMMENDATION_TITLE} (model aktif)")
    st.caption("Saran dipicu oleh metrik median klaster yang melewati ambang ITU; ini deskripsi karakteristik, bukan label kualitas.")
    clusters = cluster_recommendations(art.profile["clusters"], rules)
    if not clusters:
        st.success("Tidak ada klaster dengan metrik yang melewati ambang ITU.")
    for rec in clusters:
        show_recommendation(rec, f"{rec['nama']} ({rec['jumlah_anggota']} aliran) — {rec['deskripsi']}")
else:
    st.info("Saran dimatikan (`recommendations.enabled: false` di config).")

reports = ui.resolve_path(cfg["paths"]["reports_dir"]) / "riwayat"
summaries = sorted(reports.glob("riwayat_pemantauan_*.csv"), reverse=True) if reports.is_dir() else []
st.subheader("Riwayat sesi")
if not summaries:
    st.info("Belum ada riwayat. Jalankan sesi pemantauan lalu tekan Stop.")
    st.stop()

history = pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True)
history = history.sort_values(["waktu_mulai", "waktu_selesai"], ascending=False, ignore_index=True)  # terbaru di atas
st.dataframe(history, width="stretch", hide_index=True)

st.subheader("Laporan sesi")
chosen = st.selectbox("Sesi (terbaru dipilih otomatis)", history["id_sesi_pemantauan"], index=0)
summary = history[history["id_sesi_pemantauan"] == chosen].iloc[0].to_dict()
csv_path = ui.Path(summary["lokasi_laporan"])
if not csv_path.is_file():
    st.warning(
        f"Sesi **{chosen}** tidak punya data jendela ({csv_path.name}): tidak ada paket yang menjadi flow. "
        "Periksa interface (harus aktif dan terhubung, mis. Wi-Fi), filter BPF, dan izin Npcap/Administrator."
    )
    st.stop()

rows = pd.read_csv(csv_path)
status = summary.get("status_sesi")
if status == "Dihentikan tidak normal":
    st.warning(f"Sesi **{chosen}** dihentikan tidak normal (aplikasi ditutup atau capture terputus tanpa Stop); skor model tidak tercatat.")
elif status is None or pd.isna(status):
    st.warning(f"Sesi **{chosen}** dicatat dengan format lama (pelabelan QoS) dan tidak punya ringkasan klaster.")
else:
    per_cluster = parse_counts(summary.get("panggilan_per_klaster"))
    majority = summary.get("klaster_mayoritas")
    st.info(
        f"Status sesi: **{status}** · panggilan per klaster: "
        f"{', '.join(f'Klaster {k}: {v}' for k, v in per_cluster.items()) or '-'} · "
        f"di luar klaster: {summary.get('persen_di_luar_klaster', 0):.0f}% · "
        f"klaster mayoritas: {'-' if pd.isna(majority) else f'Klaster {int(majority)}'}"
    )
if enabled and "status" in rows:
    recs = session_recommendations(rows, metric_map, rules)
    if not recs:
        st.success("Tidak ada klaster dengan metrik melewati ambang ITU dan tidak ada aliran di luar klaster pada sesi ini.")
    for rec in recs:
        show_recommendation(rec, f"{rec['kunci']} — {rec['jumlah_flow']} aliran")

calls_path = csv_path.with_name(csv_path.name.replace("jendela_", "panggilan_", 1))
calls = pd.read_csv(calls_path) if calls_path.is_file() else pd.DataFrame()
st.subheader("Panggilan per sesi")
if calls.empty:
    st.caption("Belum ada panggilan selesai yang tercatat pada sesi ini.")
elif "status_akhir" not in calls:
    st.warning("Berkas panggilan memakai format lama (pelabelan QoS) dan tidak ditampilkan.")
else:
    ui.show_calls_table(calls, newest_first=False)
    if "catatan" in calls and calls["catatan"].fillna("").ne("").any():
        st.warning("Sesi berjalan pada mode window: klaster akhir berasal dari jendela terakhir; ringkasan yang sah hanya pada mode cumulative.")
    if enabled:
        call_recs = call_recommendations(calls, metric_map, rules)
        for rec in call_recs:
            show_recommendation(rec, f"Panggilan {rec['flow_id']} — {rec['kunci']}")
        if not call_recs:
            st.success("Tidak ada panggilan yang memerlukan saran pada sesi ini.")
    st.download_button("Unduh CSV panggilan", calls_path.read_bytes(), file_name=calls_path.name, mime="text/csv")

c1, c2 = st.columns(2)
c1.download_button("Unduh CSV jendela", csv_path.read_bytes(), file_name=csv_path.name, mime="text/csv")
if c2.button("Buat PDF ringkasan"):
    paths = export_session(summary, cfg, rules, art.profile)
    st.session_state["pdf_ready"] = (chosen, paths["pdf"].name, paths["pdf"].read_bytes())
ready = st.session_state.get("pdf_ready")
if ready and ready[0] == chosen:
    c2.download_button("Unduh PDF ringkasan", ready[2], file_name=ready[1], mime="application/pdf")
st.dataframe(rows.head(200), width="stretch", hide_index=True)
