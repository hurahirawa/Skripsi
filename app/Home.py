"""Beranda: ringkasan aplikasi dan status model aktif. Jalankan: streamlit run app/Home.py"""
import ui  # noqa: F401  (menyiapkan sys.path ke src/)
import streamlit as st

st.set_page_config(page_title="VoIP QoS · HDBSCAN", page_icon="📞", layout="wide")

cfg = ui.get_cfg()
art = ui.require_artifacts(cfg)
ui.render_sidebar(cfg, art)

st.title("Pemetaan Karakteristik VoIP untuk Identifikasi QoS")
st.caption("HDBSCAN · ITU-T G.114 dan Y.1541 · pemantauan lokal tanpa database")

ev = art.profile["evaluation"]
c1, c2, c3, c4 = st.columns([2, 1.6, 1, 1])
c1.metric("Model aktif", art.version)
c2.metric("Silhouette", f"{ev['silhouette']:.4f}", ev["silhouette_label"], delta_color="off")
c3.metric("Klaster", ev["n_clusters"])
c4.metric("Noise Ratio", f"{ev['noise_ratio']:.2%}")

st.markdown(
    """
**Alur kerja**
1. Pilih mode **Replay PCAP** di sidebar, pilih atau unggah berkas PCAP, lalu tekan **Start**.
2. Buka **Dashboard Overview** untuk melihat label tiap aliran secara berkala.
3. **Analisis & Evaluasi Klaster** menampilkan skor, profil, dan box plot model.
4. **Pengaturan Parameter** mengatur parameter HDBSCAN dan ambang ITU.
5. **Rekomendasi & Ekspor** menampilkan klaster bermasalah dan riwayat sesi.
"""
)
st.info(
    "Mode **Replay PCAP** selalu tersedia. Mode **Live** butuh Npcap (Windows) dan, untuk backend pyshark, "
    "Wireshark/TShark; status prasyarat ditampilkan di sidebar saat Live dipilih."
)
