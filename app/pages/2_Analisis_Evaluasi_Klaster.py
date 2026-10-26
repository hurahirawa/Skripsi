"""P2 Analisis & Evaluasi Klaster: skor, perbandingan HDBSCAN vs K-Means, Elbow/Silhouette/DBI, grid, profil, box plot, centroid."""
import ui  # noqa: F401
import pandas as pd
import streamlit as st

from qos.viz.plots import BOX_METRICS, boxplot_fig, cluster_name, grid_heatmaps_fig, kmeans_curves_fig, style_by_state

st.set_page_config(page_title="Analisis & Evaluasi Klaster", page_icon="🔬", layout="wide")
cfg = ui.get_cfg()
art = ui.require_artifacts(cfg)
ui.render_sidebar(cfg, art)
profile = art.profile["clusters"]
th = art.profile["thresholds"]
ev = art.profile["evaluation"]
comparison = art.profile["comparison"]
folder = ui.resolve_path(cfg["paths"]["artifacts_dir"]) / art.version

st.title("Analisis & Evaluasi Klaster")
st.info(
    "Halaman ini untuk **evaluasi model** (skor, perbandingan algoritma, profil klaster, box plot), bukan hasil sesi pemantauan. "
    "Hasil sesi dan panggilan ada di halaman Rekomendasi & Ekspor."
)
try:
    st.page_link("pages/4_Rekomendasi_Ekspor.py", label="Buka Rekomendasi & Ekspor (hasil sesi)", icon="📝")
except st.errors.StreamlitAPIException:  # halaman dijalankan di luar navigasi multipage (mis. uji AppTest)
    pass


def fmt(v: float | None, digits: int = 2) -> str:
    """Angka berdesimal; None/NaN (mis. std klaster beranggota 1) ditampilkan '-'."""
    return "-" if v is None or pd.isna(v) else f"{v:,.{digits}f}"


st.subheader("Evaluasi HDBSCAN (titik non-noise)")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Silhouette", fmt(ev["silhouette"], 4), ev["silhouette_label"], delta_color="off")
c2.metric("Davies-Bouldin", fmt(ev["davies_bouldin"], 4))
c3.metric("Calinski-Harabasz", fmt(ev["chi"]))
c4.metric("DBCV", fmt(ev.get("dbcv"), 4))
c5.metric("Noise Ratio", f"{ev['noise_ratio']:.2%}")
c6.metric("ARI bootstrap", f"{fmt(ev.get('ari_mean'), 3)} ± {fmt(ev.get('ari_sd'), 3)}")
p = art.meta["parameters"]
st.caption(
    f"Model {art.version} · {ev['n_clusters']} klaster · HDBSCAN min_cluster_size={p['min_cluster_size']}, min_samples={p['min_samples']} · "
    f"fitur {art.meta['cluster_columns']} · log1p {art.meta['log_transform']} · "
    f"batas jarak: {art.assignment['aturan_teks']} (p{art.assignment['persentil']:g}, batas global {art.assignment['batas_global']:.3f})"
)

st.subheader("Perbandingan HDBSCAN vs K-Means")
rows = []
for key in ("hdbscan", "hdbscan_assigned", "kmeans_n", "kmeans_opt"):
    m = comparison[key]
    rows.append(
        {
            "Konfigurasi": m["nama"], "Klaster": m["n_clusters"], "Noise": f"{m['noise_ratio']:.2%}", "Silhouette": fmt(m["silhouette"], 4),
            "DBI": fmt(m["davies_bouldin"], 4), "CHI": fmt(m["chi"]), "DBCV": fmt(m["dbcv"], 4),
            "ARI bootstrap": f"{fmt(m.get('ari_mean'), 3)} ± {fmt(m.get('ari_sd'), 3)}",
            "Waktu fit (dtk)": fmt(m.get("waktu_fit_s"), 4),
        }
    )
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.caption(
    "HDBSCAN dihitung dua versi: tanpa titik noise, dan dengan noise di-assign ke centroid terdekat (sepadan dengan K-Means yang menugaskan semua titik)."
)

st.subheader("K-Means: Elbow, Silhouette, dan DBI per k")
sweep = pd.DataFrame(art.profile["kmeans_sweep"])
st.plotly_chart(kmeans_curves_fig(sweep, comparison["kmeans_n"]["k"], comparison["kmeans_opt"]["k"]), width="stretch", key="kmeans_curves")
st.caption(art.profile.get("k_optimal_alasan", ""))

st.subheader("Grid HDBSCAN (min_cluster_size × min_samples)")
grid_path = folder / "grid_results.csv"
grid = pd.read_csv(grid_path) if grid_path.is_file() and grid_path.stat().st_size > 2 else pd.DataFrame()
if grid.empty:
    st.caption("Model ini dilatih tanpa grid search (parameter tetap).")
else:
    st.plotly_chart(grid_heatmaps_fig(grid, (int(p["min_cluster_size"]), int(p["min_samples"]))), width="stretch", key="grid_heatmap")
    sel = art.meta["aturan_seleksi"]
    st.caption(
        f"Kandidat valid: noise ≤ {sel['max_noise_ratio']:.0%}, klaster ≥ {sel['min_clusters']}, ARI mean ≥ {sel['min_ari_mean']:g}; "
        "dipilih DBCV tertinggi (seri → Silhouette). Kotak merah = terpilih."
    )
    with st.expander(f"Tabel grid lengkap ({len(grid)} kandidat, {int(grid['valid'].sum())} valid)"):
        st.dataframe(grid, width="stretch", hide_index=True)

st.subheader("Profil klaster")
st.caption("Kolom *Deskripsi ITU* = posisi median klaster terhadap ambang ITU dari config; deskriptif, bukan label kualitas.")
table = []
for c in profile:
    row = {"Klaster": c["nama"], "Jumlah Anggota": c["jumlah_anggota"]}
    for metric, (title, _) in BOX_METRICS.items():
        s = c["statistik"][metric]
        row.update({f"{title} mean": fmt(s["mean"]), f"{title} median": fmt(s["median"]), f"{title} std": fmt(s["std"])})
    row.update({"Deskripsi ITU": c["deskripsi"]["ringkas"], "Metrik melewati ambang": ", ".join(c["metrik_melewati_ambang"]) or "-"})
    table.append(row)
st.dataframe(style_by_state(pd.DataFrame(table), "Klaster"), width="stretch", hide_index=True)

st.subheader("Box plot per klaster (garis putus = ambang ITU, hanya referensi)")
log = st.toggle("Skala log", value=False)
cols = st.columns(3)
for col, metric in zip(cols, BOX_METRICS):
    col.plotly_chart(boxplot_fig(profile, metric, th, log), width="stretch", key=f"box_{metric}")

st.subheader("Centroid dan batas jarak")
centroid_rows = []
for c in profile:
    if c["id_klaster"] == -1:
        continue
    raw = c["centroid_asli"] or {}
    own = c["jarak_sendiri"] or {}
    centroid_rows.append(
        {
            "Klaster": c["nama"], "Delay centroid (ms)": fmt(raw.get("delay_ms")), "Jitter centroid (ms)": fmt(raw.get("jitter_ms")),
            "FAI centroid (%)": fmt(raw.get("fai_pct")), "Jarak median": fmt(own.get("median"), 3), "Jarak p95": fmt(own.get("p95"), 3),
            "Jarak maks": fmt(own.get("maks"), 3), "Batas di luar klaster": fmt(c["batas_jarak"], 3),
        }
    )
st.dataframe(style_by_state(pd.DataFrame(centroid_rows), "Klaster"), width="stretch", hide_index=True)
st.caption(
    "Titik ditandai **Di luar klaster** bila jaraknya ke SETIAP centroid melebihi batas klaster tersebut. "
    f"Aturan: {art.assignment['aturan_teks']}; batas global p{art.assignment['persentil']:g} = {art.assignment['batas_global']:.3f}."
)

st.subheader("Sampel data latih")
sample = pd.read_csv(folder / "training_sample.csv")
choice = st.multiselect("Filter klaster", sorted(sample["cluster_id"].unique()), format_func=cluster_name)
view = sample[sample["cluster_id"].isin(choice)] if choice else sample
show = [c for c in ("flow_id", "delay_ms", "jitter_ms", "fwd_packets_count", "bwd_packets_count", "fai_pct", "cluster_id", "jarak_terdekat",
                    "di_luar_klaster", "klaster_kmeans_n") if c in view]
st.dataframe(view[show].head(500), width="stretch", hide_index=True)
