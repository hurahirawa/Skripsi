"""P4 Rekomendasi Mitigasi & Ekspor Laporan: alert per klaster dan per sesi, riwayat, unduh CSV dan PDF."""
import pandas as pd
import streamlit as st

import ui
from qos.monitor.engine import card_category
from qos.recommend.rules import cluster_recommendations, session_recommendations
from qos.report.export import export_session
from qos.viz.plots import cluster_name

st.set_page_config(page_title="Rekomendasi & Ekspor", page_icon="📝", layout="wide")
cfg = ui.get_cfg()
art = ui.require_artifacts(cfg)
ui.render_sidebar(cfg, art)
rules = ui.load_rules_cached(cfg)

st.title("Rekomendasi Mitigasi & Ekspor Laporan")


def show_recommendation(rec: dict, title: str) -> None:
    """Alert berwarna menurut kategori label, dengan saran di bawahnya."""
    box = st.error if card_category(rec["label"]) == "Anomali" else st.warning
    items = "\n".join(f"- {s}" for s in rec["saran"])
    box(f"**{title}** — {rec['label']}  \nTindakan: {rec['kategori']} · pemicu: {', '.join(rec['pemicu'])}\n\n{items}")


st.subheader("Klaster bermasalah (model aktif)")
clusters = cluster_recommendations(art.profile["clusters"], rules)
if not clusters:
    st.success("Tidak ada klaster bermasalah.")
by_id = {c["id_klaster"]: c for c in art.profile["clusters"]}
for rec in clusters:
    p = by_id[rec["id_klaster"]]["statistik_pelabelan"]
    show_recommendation(rec, f"{cluster_name(rec['id_klaster'])} ({rec['jumlah_anggota']} aliran)")
    st.caption(f"mean delay {p['mean_delay_ms']:.2f} ms · P99,9 jitter {p['p999_jitter_ms']:.2f} ms · PLR {p['plr_pct']:.2f}%")

reports = ui.resolve_path(cfg["paths"]["reports_dir"]) / "riwayat"
summaries = sorted(reports.glob("riwayat_pemantauan_*.csv"), reverse=True) if reports.is_dir() else []
st.subheader("Riwayat sesi")
if not summaries:
    st.info("Belum ada riwayat. Jalankan sesi pemantauan lalu tekan Stop.")
    st.stop()

history = pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True)
st.dataframe(history, width="stretch", hide_index=True)

st.subheader("Laporan sesi")
chosen = st.selectbox("Sesi", history["id_sesi_pemantauan"])
summary = history[history["id_sesi_pemantauan"] == chosen].iloc[0].to_dict()
csv_path = ui.Path(summary["lokasi_laporan"])
if not csv_path.is_file():
    st.warning(f"Berkas jendela tidak ditemukan: {csv_path}")
    st.stop()

rows = pd.read_csv(csv_path)
if summary["status_keseluruhan"] == "Kritis":
    st.error(f"Sesi **{chosen}** berstatus **Kritis**: proporsi aliran QoS Buruk + Anomali melebihi {cfg['monitoring']['kritis_ratio']:.0%}.")
else:
    st.info(f"Status sesi: **{summary['status_keseluruhan']}**")
recs = session_recommendations(rows, rules)
if not recs:
    st.success("Tidak ada aliran QoS Buruk atau Anomali pada sesi ini.")
for rec in recs:
    show_recommendation(rec, f"{rec['jumlah_flow']} aliran")

c1, c2 = st.columns(2)
c1.download_button("Unduh CSV jendela", csv_path.read_bytes(), file_name=csv_path.name, mime="text/csv")
if c2.button("Buat PDF ringkasan"):
    paths = export_session(summary, cfg, rules, art.profile)
    st.session_state["pdf_ready"] = (chosen, paths["pdf"].name, paths["pdf"].read_bytes())
ready = st.session_state.get("pdf_ready")
if ready and ready[0] == chosen:
    c2.download_button("Unduh PDF ringkasan", ready[2], file_name=ready[1], mime="application/pdf")
st.dataframe(rows.head(200), width="stretch", hide_index=True)
