"""P1 Dashboard Overview: status saat ini, kartu per klaster, panggilan aktif, PCA scatter dengan centroid, tabel jendela."""
import ui  # noqa: F401
import pandas as pd
import streamlit as st

from qos.recommend.rules import TITLE as RECOMMENDATION_TITLE
from qos.recommend.rules import cluster_metric_map, recommendations_enabled, session_recommendations
from qos.viz.plots import pca_scatter, style_by_state

st.set_page_config(page_title="Dashboard Overview", page_icon="📊", layout="wide")
cfg = ui.get_cfg()
art = ui.require_artifacts(cfg)
ui.render_sidebar(cfg, art)
rules = ui.load_rules_cached(cfg)

st.title("Dashboard Overview")
if st.session_state.get("session") is None and not len(st.session_state.get("live_rows", [])):
    st.info("Belum ada sesi. Pilih PCAP di sidebar lalu tekan **Start**.")

background = pd.read_csv(
    ui.resolve_path(cfg["paths"]["artifacts_dir"]) / art.version / "training_sample.csv", usecols=["pc1", "pc2", "cluster_id"]
)
ids = art.cluster_ids
pcs = art.pca.transform([art.centroids[i] for i in ids])
centroid_pc = {i: (float(x), float(y)) for i, (x, y) in zip(ids, pcs)}
metric_map = cluster_metric_map(art.profile["clusters"])
TABLE_COLUMNS = ["waktu_jendela", "flow_id", "delay_ms", "jitter_ms", "fwd", "bwd", "fai_pct", "status", "klaster_terdekat", "jarak_terdekat"]


def description_table() -> pd.DataFrame:
    """Tabel deskripsi klaster model (posisi median terhadap ambang ITU; deskriptif, bukan label)."""
    return pd.DataFrame(
        [{"Klaster": c["nama"], "Anggota latih": c["jumlah_anggota"], "Deskripsi": c["deskripsi"]["ringkas"]}
         for c in art.profile["clusters"]]
    )


@st.fragment(run_every=cfg["monitoring"]["refresh_sec"])
def live_view() -> None:
    """Auto-refresh: kosongkan queue lalu gambar ulang banner, kartu, panggilan aktif, scatter, dan tabel."""
    rows = ui.drain_session()
    calls = ui.drain_calls()
    session = st.session_state.get("session")
    counts = session.card_counts() if session is not None else {}
    ui.render_connection_banner(session, art)
    ui.render_capture_status(cfg, session)
    ui.render_cluster_cards(counts, art)
    if counts.get("Data Belum Cukup"):
        st.caption(f"{counts['Data Belum Cukup']} aliran berstatus *Data Belum Cukup* (belum ditugaskan ke klaster).")
    if counts.get("Sinyal SIP"):
        st.caption(f"{counts['Sinyal SIP']} flow *Sinyal SIP* dicatat tetapi tidak dinilai dan tidak dihitung sebagai panggilan.")
    ui.render_active_calls(session, art)

    left, right = st.columns(2)
    left.subheader("Sebaran PCA dan centroid")
    left.plotly_chart(pca_scatter(background, rows, centroid_pc), width="stretch", key="pca")
    right.subheader("Deskripsi klaster")
    right.caption("Posisi median klaster terhadap ambang ITU. Deskripsi saja, bukan label kualitas.")
    desc = description_table()
    right.dataframe(style_by_state(desc, "Klaster"), width="stretch", hide_index=True)

    recs = session_recommendations(rows, metric_map, rules) if len(rows) and recommendations_enabled(cfg) else []
    if recs:
        with st.expander(f"{RECOMMENDATION_TITLE} ({len(recs)} kelompok)", expanded=False):
            for r in recs:
                st.markdown(f"**{r['kunci']}** — {r['jumlah_flow']} aliran · {r['kategori']}")
                for s in r["saran"]:
                    st.markdown(f"- {s}")

    st.subheader("Panggilan Selesai")
    if len(calls):
        ui.show_calls_table(calls)  # terbaru di atas
    else:
        st.caption(
            f"Belum ada panggilan selesai (panggilan selesai setelah idle {cfg['monitoring']['call_end_idle_sec']} detik atau saat Stop)."
        )

    st.subheader("Jendela terbaru")
    if len(rows):
        st.dataframe(rows[TABLE_COLUMNS].tail(50).iloc[::-1], width="stretch", hide_index=True)
    else:
        st.caption("Belum ada data jendela.")


live_view()
