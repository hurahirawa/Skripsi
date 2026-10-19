"""P3 Pengaturan Parameter & Ambang Batas: konfigurasi, latih ulang, Terapkan Ulang Pelabelan, dan versi model."""
import json

import pandas as pd
import streamlit as st
import yaml

import ui
from qos.model.retrain import InsufficientDataError, RetrainError, load_log_frame, retrain
from qos.model.versions import VersionError, activate_version, label_changes, list_versions, relabel_version

st.set_page_config(page_title="Pengaturan Parameter", page_icon="⚙️", layout="wide")
cfg = ui.get_cfg()
art = ui.require_artifacts(cfg)
ui.render_sidebar(cfg, art)

st.title("Pengaturan Parameter & Ambang Batas")
h, t = cfg["hdbscan"], cfg["thresholds"]
monitoring = ui.monitoring_active()

st.subheader("HDBSCAN")
c1, c2, c3 = st.columns(3)
mcs = c1.slider("min_cluster_size", 2, 50, int(h["min_cluster_size"]))
ms = c2.slider("min_samples", 1, 20, int(h["min_samples"]))
grid = c3.toggle("Grid search otomatis", value=bool(h["grid_search"]["enabled"]))

st.subheader("Ambang ITU")
default_th = art.profile["thresholds"]  # ambang saat model dilatih = default skripsi
st.warning(
    "Perubahan ambang ITU di sini hanya untuk **eksplorasi**. Hasil skripsi memakai ambang default "
    "(ITU-T G.114 untuk delay; ITU-T Y.1541 Kelas 0/1 untuk jitter dan packet loss): "
    f"delay {default_th['delay_good_ms']:g}/{default_th['delay_max_ms']:g} ms, "
    f"jitter {default_th['jitter_max_ms']:g} ms, loss {default_th['loss_max_pct']:g}%."
)
d1, d2, d3, d4 = st.columns(4)
delay_good = d1.number_input("Delay baik (ms)", min_value=0.0, value=float(t["delay_good_ms"]))
delay_max = d2.number_input("Delay maks (ms)", min_value=0.0, value=float(t["delay_max_ms"]))
jitter_max = d3.number_input("Jitter maks (ms)", min_value=0.0, value=float(t["jitter_max_ms"]))
loss_max = d4.number_input("Loss maks (%)", min_value=0.0, value=float(t["loss_max_pct"]), format="%.3f")
thresholds = {
    "delay_good_ms": delay_good, "delay_max_ms": delay_max, "jitter_max_ms": jitter_max, "loss_max_pct": loss_max,
}

errors = []
if delay_good > delay_max:
    errors.append("Delay baik tidak boleh lebih besar dari delay maks.")
for e in errors:
    st.error(e)

overrides = {"hdbscan": {"min_cluster_size": mcs, "min_samples": ms, "grid_search": {"enabled": grid}}, "thresholds": thresholds}
b1, b2 = st.columns(2)
if b1.button("Save Configuration", type="primary", disabled=bool(errors)):
    path = ui.user_config_path()
    path.write_text(yaml.safe_dump(overrides, sort_keys=False, allow_unicode=True), encoding="utf-8")
    ui.reload_cfg()
    st.success(f"Konfigurasi disimpan di {path}")
    st.rerun()
if b2.button("Reset to Default"):
    ui.user_config_path().unlink(missing_ok=True)
    ui.reload_cfg()
    st.success("Konfigurasi dikembalikan ke default.")
    st.rerun()


def activate_button(version: str, key: str) -> None:
    """Tombol "Aktifkan model ini": menulis current.txt (ditolak selama monitoring aktif)."""
    help_text = "Dinonaktifkan selama monitoring aktif" if monitoring else None
    if st.button("Aktifkan model ini", key=key, disabled=monitoring, help=help_text):
        try:
            activate_version(cfg, version, monitoring_active=monitoring)
        except VersionError as exc:
            st.error(str(exc))
            return
        for k in ("retrain_result", "relabel_result"):
            st.session_state.pop(k, None)
        st.success(f"Model aktif sekarang: {version}")
        st.rerun()


# ---------------------------------------------------------------- Terapkan Ulang Pelabelan
st.divider()
st.subheader("Terapkan Ulang Pelabelan")
st.caption(
    f"Menghitung ulang label dari statistik tersimpan model aktif ({art.version}) dengan ambang di atas, "
    "**tanpa melatih ulang HDBSCAN**. Hasilnya versi baru; model aktif tidak berubah sampai Anda mengaktifkannya."
)
if st.button("Terapkan Ulang Pelabelan", disabled=bool(errors)):
    try:
        out = relabel_version(cfg, art.version, thresholds)
        new_doc = json.loads((out / "cluster_profile.json").read_text("utf-8"))
        st.session_state["relabel_result"] = {
            "version": out.name, "changes": label_changes(art.profile, new_doc), "thresholds": new_doc["thresholds"],
        }
    except VersionError as exc:
        st.error(str(exc))
res = st.session_state.get("relabel_result")
if res:
    st.success(f"Versi baru dibuat: {res['version']} (ambang {res['thresholds']})")
    st.dataframe(res["changes"], width="stretch", hide_index=True)
    if not res["changes"]["berubah"].any():
        st.info("Tidak ada label yang berubah dengan ambang ini.")
    activate_button(res["version"], "act_relabel")

# ------------------------------------------------------------------------- Latih ulang
st.divider()
st.subheader("Latih Ulang Model")
source_label = st.radio("Sumber data latih", ["CSV CIC/Darknet (data.raw_dir)", "Log monitoring (reports/riwayat)"], horizontal=True, key="retrain_source")
source = "csv" if source_label.startswith("CSV") else "log"
log_ok = True
if source == "log":
    opt = cfg["retrain"]["log_source"]
    st.caption(
        f"Log dipakai dengan pengecualian: Data Belum Cukup (wajib), Anomali ({'ya' if opt['exclude_anomaly'] else 'tidak'}); "
        f"{'satu baris per flow' if opt['per_flow'] == 'last' else 'semua jendela'}; minimum {opt['min_rows']} baris."
    )
    try:
        _, counts = load_log_frame(cfg)
        st.success(f"Data log mencukupi: {counts['setelah_filter_sesi']} baris (dari {counts['baris_log']} baris log).")
    except InsufficientDataError as exc:
        log_ok = False
        st.warning(str(exc))
    except RetrainError as exc:
        log_ok = False
        st.error(str(exc))
if monitoring:
    st.warning("Monitoring sedang berjalan: pelatihan ulang dinonaktifkan.")

if st.button("Latih Ulang Model", disabled=monitoring or bool(errors) or not log_ok):
    bar = st.progress(0.0, text="Melatih model…")
    params = {"min_cluster_size": mcs, "min_samples": ms, "grid_enabled": grid, "thresholds": thresholds}

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Grid search {done}/{total}")

    try:
        result = retrain(cfg, source, params, on_progress, monitoring_active=monitoring)
        bar.progress(1.0, text="Selesai")
        st.session_state["retrain_result"] = result
    except RetrainError as exc:
        bar.empty()
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 — kegagalan pelatihan ditampilkan, bukan dilempar ke UI
        bar.empty()
        st.error(f"Pelatihan gagal: {type(exc).__name__}: {exc}")
res = st.session_state.get("retrain_result")
if res:
    ev = res["evaluation"]
    st.success(
        f"Versi baru: {res['version']} (belum aktif) · parameter {res['params']['min_cluster_size']}/{res['params']['min_samples']}"
    )
    m1, m2, m3, m4 = st.columns(4)
    sil = "-" if pd.isna(ev["silhouette"]) else f"{ev['silhouette']:.4f}"
    m1.metric("Silhouette", sil, ev["silhouette_label"], delta_color="off")
    m2.metric("CHI", "-" if pd.isna(ev["chi"]) else f"{ev['chi']:,.2f}")
    m3.metric("Noise Ratio", f"{ev['noise_ratio']:.2%}")
    m4.metric("Klaster", ev["n_clusters"])
    st.dataframe(
        pd.DataFrame([{"klaster": c["id_klaster"], "n": c["jumlah_anggota"], "label": c["label"]} for c in res["profile"]]),
        width="stretch", hide_index=True,
    )
    activate_button(res["version"], "act_retrain")

# ------------------------------------------------------------------------- Versi model
st.divider()
st.subheader("Versi model")
versions = list_versions(cfg)
st.dataframe(versions.fillna({"induk": "-"}), width="stretch", hide_index=True)
inactive = versions[~versions["aktif"]]["versi"].tolist()
if inactive:
    pick = st.selectbox("Aktifkan versi lain", inactive)
    activate_button(pick, "act_pick")
st.caption(f"Ambang pada model aktif ({art.version}): {art.profile['thresholds']}")
