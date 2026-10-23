"""P2 Analisis & Evaluasi Klaster: skor, tabel profil, box plot 3 metrik, dan sampel data."""
import ui  # noqa: F401
import pandas as pd
import streamlit as st

from qos.viz.plots import BOX_METRICS, boxplot_fig, cluster_name

st.set_page_config(page_title="Analisis & Evaluasi Klaster", page_icon="🔬", layout="wide")
cfg = ui.get_cfg()
art = ui.require_artifacts(cfg)
ui.render_sidebar(cfg, art)
profile = art.profile["clusters"]
th = art.profile["thresholds"]
ev = art.profile["evaluation"]

st.title("Analisis & Evaluasi Klaster")
st.info(
    "Halaman ini untuk **evaluasi model** (skor, profil klaster, box plot), bukan hasil sesi pemantauan. "
    "Hasil sesi dan panggilan ada di halaman Rekomendasi & Ekspor."
)
try:
    st.page_link("pages/4_Rekomendasi_Ekspor.py", label="Buka Rekomendasi & Ekspor (hasil sesi)", icon="📝")
except st.errors.StreamlitAPIException:  # halaman dijalankan di luar navigasi multipage (mis. uji AppTest)
    pass
c1, c2, c3 = st.columns(3)
c1.metric("Silhouette", f"{ev['silhouette']:.4f}", ev["silhouette_label"], delta_color="off")
c2.metric("Calinski-Harabasz", f"{ev['chi']:,.2f}")
c3.metric("Noise Ratio", f"{ev['noise_ratio']:.2%}")
st.caption(f"Model {art.version} · {ev['n_clusters']} klaster · fitur {art.meta['cluster_columns']} · log1p {art.meta['log_transform']}")


def fmt(v: float | None) -> str:
    """Angka 2 desimal; None (mis. std klaster beranggota 1) ditampilkan '-'."""
    return "-" if v is None or pd.isna(v) else f"{v:,.2f}"


rows = []
for c in profile:
    row = {"ID Klaster": c["id_klaster"], "Jumlah Anggota": c["jumlah_anggota"]}
    for metric, (_, title, _, _) in BOX_METRICS.items():
        s = c["statistik"][metric]
        row.update({f"{title} mean": fmt(s["mean"]), f"{title} median": fmt(s["median"]), f"{title} std": fmt(s["std"])})
    p, lv = c["statistik_pelabelan"], c["tingkat"]
    row.update(
        {
            "Mean delay (ms)": fmt(p["mean_delay_ms"]), "P99,9 jitter (ms)": fmt(p["p999_jitter_ms"]), "PLR (%)": fmt(p["plr_pct"]),
            "Level delay": lv["delay"], "Level jitter": lv["jitter"], "Level loss": lv["loss"],
            "Label": c["label"], "Kategori Tindakan": c["kategori_tindakan"],
        }
    )
    rows.append(row)
st.subheader("Profil klaster")
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.subheader("Box plot per metrik")
log = st.toggle("Skala log", value=False)
cols = st.columns(3)
for col, metric in zip(cols, BOX_METRICS):
    col.plotly_chart(boxplot_fig(profile, metric, th, log), width="stretch", key=f"box_{metric}")

st.subheader("Sampel data latih")
sample = pd.read_csv(ui.resolve_path(cfg["paths"]["artifacts_dir"]) / art.version / "training_sample.csv")
choice = st.multiselect("Filter klaster", sorted(sample["cluster_id"].unique()), format_func=cluster_name)
view = sample[sample["cluster_id"].isin(choice)] if choice else sample
show = [c for c in ("flow_id", "delay_ms", "jitter_ms", "fwd_packets_count", "bwd_packets_count", "fai_pct", "cluster_id", "membership_prob") if c in view]
st.dataframe(view[show].head(500), width="stretch", hide_index=True)
