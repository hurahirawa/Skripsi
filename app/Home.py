"""Beranda: ringkasan aplikasi dan status model aktif. Jalankan: streamlit run app/Home.py"""
import ui  # noqa: F401  (menyiapkan sys.path ke src/)
import streamlit as st

st.set_page_config(page_title="VoIP · Klasterisasi HDBSCAN", page_icon="📞", layout="wide")

cfg = ui.get_cfg()
art = ui.require_artifacts(cfg)
ui.render_sidebar(cfg, art)

st.title("Pemetaan Karakteristik Trafik VoIP dengan Klasterisasi")
st.caption("HDBSCAN (pembanding K-Means) · penugasan jarak Euclidean ke centroid · ambang ITU hanya sebagai deskripsi · tanpa database")

ev = art.profile["evaluation"]
c1, c2, c3, c4, c5 = st.columns([2, 1.6, 1.2, 1, 1])
c1.metric("Model aktif", art.version)
c2.metric("Silhouette", f"{ev['silhouette']:.4f}", ev["silhouette_label"], delta_color="off")
c3.metric("DBCV", "-" if ev.get("dbcv") is None else f"{ev['dbcv']:.4f}")
c4.metric("Klaster", ev["n_clusters"])
c5.metric("Noise Ratio", f"{ev['noise_ratio']:.2%}")

st.markdown(
    """
**Alur kerja**
1. Pilih mode **Replay PCAP** di sidebar, pilih atau unggah berkas PCAP, lalu tekan **Start**.
2. Buka **Dashboard Overview** untuk melihat klaster terdekat tiap panggilan secara berkala.
3. **Analisis & Evaluasi Klaster** menampilkan perbandingan HDBSCAN dan K-Means, profil, box plot, centroid, dan batas jarak.
4. **Pengaturan Parameter** mengatur HDBSCAN, aturan seleksi, aturan batas jarak, dan ambang ITU (deskripsi saja).
5. **Saran & Ekspor** menampilkan saran berdasarkan karakteristik klaster dan riwayat sesi.
"""
)
st.info(
    "Mode **Replay PCAP** selalu tersedia. Mode **Live** butuh Npcap (Windows) dan, untuk backend pyshark, "
    "Wireshark/TShark; status prasyarat ditampilkan di sidebar saat Live dipilih."
)
