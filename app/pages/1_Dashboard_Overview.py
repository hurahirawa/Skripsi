"""P1 Dashboard Overview: kartu ringkasan, PCA scatter live, bar chart vs ambang ITU, tabel jendela terbaru."""
import ui  # noqa: F401
import pandas as pd
import streamlit as st

from qos.recommend.rules import session_recommendations
from qos.viz.plots import labeling_bars, pca_scatter

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
TABLE_COLUMNS = ["waktu_jendela", "flow_id", "delay_ms", "jitter_ms", "fwd", "bwd", "fai_pct", "cluster_id", "label"]


@st.fragment(run_every=cfg["monitoring"]["refresh_sec"])
def live_view() -> None:
    """Auto-refresh: kosongkan queue lalu gambar ulang kartu, scatter, bar chart, dan tabel."""
    rows = ui.drain_session()
    calls = ui.drain_calls()
    session = st.session_state.get("session")
    counts = session.card_counts() if session is not None else {}
    ui.render_connection_banner(session)
    ui.render_capture_status(cfg, session)
    ui.render_active_calls(session)
    cols = st.columns(5)
    for col, key in zip(cols, ["Total Aliran Trafik", "QoS Baik", "QoS Cukup", "QoS Buruk", "Anomali"]):
        col.metric(key, counts.get(key, 0))
    if counts.get("Data Belum Cukup"):
        st.caption(f"{counts['Data Belum Cukup']} aliran berstatus *Data Belum Cukup* (tidak dihitung sebagai Anomali).")

    if counts.get("Sinyal SIP"):
        st.caption(f"{counts['Sinyal SIP']} flow *Sinyal SIP* dicatat tetapi tidak dinilai dan tidak dihitung sebagai panggilan.")

    left, right = st.columns(2)
    left.subheader("Sebaran PCA")
    left.plotly_chart(pca_scatter(background, art.labels, rows), width="stretch", key="pca")
    right.subheader("Statistik pelabelan vs ambang ITU")
    right.plotly_chart(labeling_bars(art.profile["clusters"], art.profile["thresholds"]), width="stretch", key="bars")

    recs = session_recommendations(rows, rules) if len(rows) else []
    if recs:
        with st.expander(f"Rekomendasi mitigasi ({len(recs)} kelompok label bermasalah)", expanded=False):
            for r in recs:
                st.markdown(f"**{r['label']}** — {r['jumlah_flow']} aliran · {r['kategori']}")
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
