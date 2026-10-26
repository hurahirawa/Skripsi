"""P3 Pengaturan: parameter HDBSCAN, aturan seleksi, aturan batas jarak, ambang ITU, Perbarui Deskripsi Klaster, latih ulang, versi."""
import json

import pandas as pd
import streamlit as st
import yaml

import ui
from qos.model.assignment import RULE_TEXT, RULES
from qos.model.retrain import InsufficientDataError, RetrainError, load_log_frame, retrain
from qos.model.versions import VersionError, activate_version, description_changes, list_versions, refresh_descriptions

st.set_page_config(page_title="Pengaturan Parameter", page_icon="⚙️", layout="wide")
cfg = ui.get_cfg()
art = ui.require_artifacts(cfg)
ui.render_sidebar(cfg, art)

st.title("Pengaturan Parameter")
h, t, a = cfg["hdbscan"], cfg["thresholds"], cfg["assignment"]
monitoring = ui.monitoring_active()

st.subheader("HDBSCAN")
c1, c2, c3 = st.columns(3)
mcs = c1.slider("min_cluster_size", 2, 50, int(h["min_cluster_size"]))
ms = c2.slider("min_samples", 1, 20, int(h["min_samples"]))
grid = c3.toggle("Grid search otomatis", value=bool(h["grid_search"]["enabled"]))
st.caption("Parameter tetap dipakai bila grid search mati atau tidak ada kandidat yang memenuhi aturan seleksi.")

st.subheader("Aturan seleksi HDBSCAN (grid search)")
st.caption("Kandidat valid memenuhi semua syarat; dipilih DBCV tertinggi (seri → Silhouette).")
s1, s2, s3 = st.columns(3)
max_noise = s1.number_input("Noise maksimum", min_value=0.0, max_value=1.0, value=float(h["selection"]["max_noise_ratio"]), step=0.05, format="%.2f")
min_clusters = s2.number_input("Klaster minimum", min_value=1, max_value=20, value=int(h["selection"]["min_clusters"]), step=1)
min_ari = s3.number_input("ARI bootstrap minimum", min_value=0.0, max_value=1.0, value=float(h["selection"]["min_ari_mean"]), step=0.05, format="%.2f")

st.subheader("Aturan batas \"Di luar klaster\"")
st.caption("Titik ditandai di luar klaster bila jaraknya ke SETIAP centroid melebihi batas klaster tersebut.")
r1, r2 = st.columns(2)
limit_rule = r1.selectbox("Aturan batas", list(RULES), index=list(RULES).index(a["limit_rule"]), format_func=lambda k: f"{k} — {RULE_TEXT[k]}")
limit_pct = r2.slider("Persentil jarak", 50, 100, int(a["limit_percentile"]))

st.subheader("Ambang ITU")
default_th = art.profile["thresholds"]  # ambang saat model dilatih
st.info(
    "Ambang ITU **tidak memengaruhi klasterisasi atau penugasan**. Ambang hanya dipakai untuk deskripsi klaster, garis referensi grafik, "
    "dan saran. Default: ITU-T G.114 (delay), ITU-T Y.1541 Kelas 0/1 (jitter), ITU-T G.1010 dan panduan Cisco VoIP (packet loss): "
    f"delay {default_th['delay_good_ms']:g}/{default_th['delay_max_ms']:g} ms, jitter {default_th['jitter_max_ms']:g} ms, "
    f"loss {default_th['loss_good_pct']:g}/{default_th['loss_max_pct']:g}%."
)
d1, d2, d3, d4, d5 = st.columns(5)
delay_good = d1.number_input("Delay baik (ms)", min_value=0.0, value=float(t["delay_good_ms"]))
delay_max = d2.number_input("Delay maks (ms)", min_value=0.0, value=float(t["delay_max_ms"]))
jitter_max = d3.number_input("Jitter maks (ms)", min_value=0.0, value=float(t["jitter_max_ms"]))
loss_good = d4.number_input("Loss baik (%)", min_value=0.0, value=float(t["loss_good_pct"]), format="%.3f")
loss_max = d5.number_input("Loss maks (%)", min_value=0.0, value=float(t["loss_max_pct"]), format="%.3f")
thresholds = {
    "delay_good_ms": delay_good, "delay_max_ms": delay_max, "jitter_max_ms": jitter_max,
    "loss_good_pct": loss_good, "loss_max_pct": loss_max,
}

errors = []
if delay_good > delay_max:
    errors.append("Delay baik tidak boleh lebih besar dari delay maks.")
if loss_good > loss_max:
    errors.append("Loss baik tidak boleh lebih besar dari loss maks.")
for e in errors:
    st.error(e)

selection = {"max_noise_ratio": max_noise, "min_clusters": int(min_clusters), "min_ari_mean": min_ari}
overrides = {
    "hdbscan": {"min_cluster_size": mcs, "min_samples": ms, "grid_search": {"enabled": grid}, "selection": selection},
    "assignment": {"limit_rule": limit_rule, "limit_percentile": limit_pct},
    "thresholds": thresholds,
}
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
    """Tombol "Aktifkan model ini": menulis current.txt (ditolak selama monitoring aktif atau untuk versi pelabelan QoS lama)."""
    help_text = "Dinonaktifkan selama monitoring aktif" if monitoring else None
    if st.button("Aktifkan model ini", key=key, disabled=monitoring, help=help_text):
        try:
            activate_version(cfg, version, monitoring_active=monitoring)
        except VersionError as exc:
            st.error(str(exc))
            return
        for k in ("retrain_result", "describe_result"):
            st.session_state.pop(k, None)
        st.success(f"Model aktif sekarang: {version}")
        st.rerun()


# ------------------------------------------------------------------ Perbarui Deskripsi Klaster
st.divider()
st.subheader("Perbarui Deskripsi Klaster")
st.caption(
    f"Menghitung ulang deskripsi posisi klaster terhadap ambang ITU dari statistik tersimpan model aktif ({art.version}), "
    "**tanpa melatih ulang** (model, centroid, dan batas jarak tidak berubah). Hasilnya versi baru; model aktif tidak berubah "
    "sampai Anda mengaktifkannya."
)
if st.button("Perbarui Deskripsi Klaster", disabled=bool(errors)):
    try:
        out = refresh_descriptions(cfg, art.version, thresholds)
        new_doc = json.loads((out / "cluster_profile.json").read_text("utf-8"))
        st.session_state["describe_result"] = {
            "version": out.name, "changes": description_changes(art.profile, new_doc), "thresholds": new_doc["thresholds"],
        }
    except VersionError as exc:
        st.error(str(exc))
res = st.session_state.get("describe_result")
if res:
    st.success(f"Versi baru dibuat: {res['version']} (ambang {res['thresholds']})")
    st.dataframe(res["changes"], width="stretch", hide_index=True)
    if not res["changes"]["berubah"].any():
        st.info("Tidak ada deskripsi yang berubah dengan ambang ini.")
    activate_button(res["version"], "act_describe")

# ------------------------------------------------------------------------- Latih ulang
st.divider()
st.subheader("Latih Ulang Model")
source_label = st.radio("Sumber data latih", ["CSV CIC/Darknet (data.raw_dir)", "Log monitoring (reports/riwayat)"], horizontal=True, key="retrain_source")
source = "csv" if source_label.startswith("CSV") else "log"
log_ok = True
if source == "log":
    opt = cfg["retrain"]["log_source"]
    st.caption(
        f"Log dipakai dengan pengecualian: Data Belum Cukup (wajib), di luar klaster ({'ya' if opt['exclude_outside_cluster'] else 'tidak'}); "
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

confirm_retrain = st.checkbox("Saya mengerti ini membuat versi model baru", key="confirm_retrain", disabled=monitoring)
if st.button("Latih Ulang Model", disabled=monitoring or bool(errors) or not log_ok or not confirm_retrain):
    bar = st.progress(0.0, text="Melatih model…")
    params = {
        "min_cluster_size": mcs, "min_samples": ms, "grid_enabled": grid, "thresholds": thresholds, "selection": selection,
        "limit_rule": limit_rule, "limit_percentile": limit_pct,
    }

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
    m1, m2, m3, m4, m5 = st.columns(5)
    sil = "-" if pd.isna(ev["silhouette"]) else f"{ev['silhouette']:.4f}"
    m1.metric("Silhouette", sil, ev["silhouette_label"], delta_color="off")
    m2.metric("DBCV", "-" if pd.isna(ev["dbcv"]) else f"{ev['dbcv']:.4f}")
    m3.metric("CHI", "-" if pd.isna(ev["chi"]) else f"{ev['chi']:,.2f}")
    m4.metric("Noise Ratio", f"{ev['noise_ratio']:.2%}")
    m5.metric("Klaster", ev["n_clusters"])
    st.dataframe(
        pd.DataFrame([{"klaster": c["nama"], "n": c["jumlah_anggota"], "deskripsi": c["deskripsi"]["ringkas"]} for c in res["profile"]]),
        width="stretch", hide_index=True,
    )
    activate_button(res["version"], "act_retrain")

# ------------------------------------------------------------------------- Versi model
st.divider()
st.subheader("Versi model")
versions = list_versions(cfg)
st.dataframe(versions.fillna({"induk": "-"}), width="stretch", hide_index=True)
if not versions["kompatibel"].all():
    st.caption("Versi dengan *kompatibel* = False dibuat dengan pelabelan QoS lama dan ditolak saat diaktifkan.")
inactive = versions[~versions["aktif"]]["versi"].tolist()
if inactive:
    pick = st.selectbox("Aktifkan versi lain", inactive)
    activate_button(pick, "act_pick")
st.caption(f"Ambang pada model aktif ({art.version}): {art.profile['thresholds']}")
