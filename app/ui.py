"""Helper bersama halaman Streamlit: konfigurasi, model aktif, sidebar, dan kontrol sesi pemantauan."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import copy  # noqa: E402
import html  # noqa: E402
import logging  # noqa: E402
from datetime import datetime  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from qos.capture import sniffer  # noqa: E402
from qos.capture.sniffer import LIVE_BACKENDS, CaptureError, iter_pcap  # noqa: E402
from qos.config import DEFAULT_CONFIG_PATH, USER_CONFIG_NAME, load_config, resolve_path, setup_logging  # noqa: E402
from qos.monitor.calls import call_state_text, call_title, call_toast_message  # noqa: E402
from qos.monitor.engine import (  # noqa: E402
    Artifacts,
    InferenceEngine,
    ModelNotFoundError,
    MonitoringSession,
    connection_status,
    load_artifacts,
)
from qos.monitor.recover import recover_orphan_sessions  # noqa: E402
from qos.recommend.rules import load_rules  # noqa: E402
from qos.viz.plots import NEUTRAL_COLOR, OUTSIDE_COLOR, cluster_color, distance_bars, state_color, style_by_state  # noqa: E402

logger = logging.getLogger(__name__)
CONFIG_ENV = "QOS_CONFIG"          # path config alternatif (dipakai uji)
MAX_LIVE_ROWS = 20000              # batas baris jendela yang disimpan di memori UI
MODE_REPLAY, MODE_LIVE = "Replay PCAP", "Live"
PCAP_SUFFIXES = (".pcap", ".pcapng")


def config_dir() -> Path:
    """Folder config aktif (tempat user_config.yaml ditulis)."""
    return Path(os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH).parent


def user_config_path() -> Path:
    """Lokasi config/user_config.yaml."""
    return config_dir() / USER_CONFIG_NAME


def get_cfg() -> dict:
    """Konfigurasi aktif di st.session_state (default + user_config.yaml)."""
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = load_config(os.environ.get(CONFIG_ENV) or None)
        setup_logging(st.session_state["cfg"])
    return st.session_state["cfg"]


def reload_cfg() -> dict:
    """Muat ulang konfigurasi dari berkas (setelah Save / Reset)."""
    st.session_state.pop("cfg", None)
    return get_cfg()


@st.cache_resource(show_spinner="Memuat model aktif…")
def _cached_artifacts(artifacts_dir: str, version: str | None) -> Artifacts:
    return load_artifacts({"paths": {"artifacts_dir": artifacts_dir}}, version)


def active_version(cfg: dict) -> str | None:
    """Versi di artifacts/current.txt, atau None jika belum ada."""
    current = resolve_path(cfg["paths"]["artifacts_dir"]) / "current.txt"
    return current.read_text("utf-8").strip() if current.is_file() else None


def require_artifacts(cfg: dict) -> Artifacts:
    """Muat model aktif atau tampilkan error jelas lalu hentikan halaman (MON-8)."""
    version = active_version(cfg)
    if version is None:
        st.error("Tidak ada model aktif. Latih model dulu: `python scripts/train.py --config config/default_config.yaml --activate`")
        st.stop()
    try:
        return _cached_artifacts(cfg["paths"]["artifacts_dir"], version)
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.stop()


@st.cache_data(show_spinner=False)
def _cached_rules(path: str, mtime: float) -> dict:
    return load_rules(path)


def load_rules_cached(cfg: dict) -> dict:
    """Aturan rekomendasi dari config/recommendations.yaml (di-cache; dimuat ulang jika berkas berubah)."""
    path = resolve_path(cfg["paths"]["recommendations_file"])
    return _cached_rules(str(path), path.stat().st_mtime)


def monitoring_active() -> bool:
    """True jika ada sesi pemantauan yang sedang berjalan."""
    session = st.session_state.get("session")
    return bool(session is not None and session.running)


def drain_session() -> pd.DataFrame:
    """Pindahkan hasil di queue ke st.session_state['live_rows'] dan kembalikan seluruh baris."""
    rows = st.session_state.get("live_rows", pd.DataFrame())
    session = st.session_state.get("session")
    if session is not None:
        batches = session.drain()
        if batches:
            rows = pd.concat([rows, *batches], ignore_index=True).tail(MAX_LIVE_ROWS)
            st.session_state["live_rows"] = rows
    return rows


def drain_calls() -> pd.DataFrame:
    """Pindahkan panggilan selesai ke st.session_state['live_calls'], tampilkan toast tiap panggilan, kembalikan semuanya."""
    calls = st.session_state.get("live_calls", pd.DataFrame())
    session = st.session_state.get("session")
    if session is not None:
        batches = session.drain_calls()
        if batches:
            new = pd.concat(batches, ignore_index=True)
            st.toast(call_toast_message(new), icon="📞")  # satu toast per siklus refresh
            calls = pd.concat([calls, new], ignore_index=True)
            st.session_state["live_calls"] = calls
    return calls


COMPACT_CALL_COLUMNS = ["waktu_selesai", "pasangan", "durasi_s", "klaster", "jarak_akhir", "persen_di_luar_klaster"]
CALL_COLUMN_CONFIG = {
    "waktu_selesai": st.column_config.TextColumn("Selesai", width=145),
    "pasangan": st.column_config.TextColumn("Pasangan alamat", width=165, help="flow_id lengkap ada di Detail"),
    "durasi_s": st.column_config.NumberColumn("Durasi (s)", format="%.1f", width=70),
    "klaster": st.column_config.TextColumn("Klaster akhir", width=240),
    "jarak_akhir": st.column_config.NumberColumn("Jarak", format="%.3f", width=70),
    "persen_di_luar_klaster": st.column_config.NumberColumn("% jendela di luar", format="%.0f", width=110),
}


def show_calls_table(calls: pd.DataFrame, newest_first: bool = True) -> None:
    """Tabel panggilan ringkas berwarna menurut klaster akhir; kolom lengkap di expander "Detail"."""
    shown = calls.iloc[::-1] if newest_first else calls
    compact = shown.assign(
        pasangan=shown["flow_id"].map(call_title),
        klaster=[call_state_text(str(s), c if not pd.isna(c) else 0) for s, c in zip(shown["status_akhir"], shown["klaster_akhir"])],
    )[COMPACT_CALL_COLUMNS]
    st.dataframe(style_by_state(compact, "klaster"), column_config=CALL_COLUMN_CONFIG, width="stretch", hide_index=True)
    with st.expander("Detail"):
        st.dataframe(shown, width="stretch", hide_index=True)


CARDS_PER_ROW = 3
SPARK_BLOCK = "■"


def _foreground(color: str) -> str:
    """Warna teks (gelap/putih) yang terbaca di atas latar `color` (luminans relatif)."""
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#111827" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#FFFFFF"


def _badge(text: str, color: str) -> str:
    """Badge HTML berwarna."""
    return (
        f'<span style="background:{color};color:{_foreground(color)};padding:2px 10px;border-radius:12px;'
        f'font-weight:600">{html.escape(text)}</span>'
    )


def banner_color(status: dict) -> str:
    """Warna banner: warna klaster, abu-abu tua untuk di luar klaster, abu-abu netral selain itu."""
    if status["di_luar"]:
        return OUTSIDE_COLOR
    return cluster_color(status["klaster"]) if status["klaster"] is not None else NEUTRAL_COLOR


def banner_text(status: dict) -> str:
    """Judul banner: "Status Saat Ini: paling dekat ke Klaster N (jarak X)", varian di luar klaster, atau status tunggu."""
    if status["klaster"] is None:
        return f"Status Saat Ini: {status['teks']}"
    return f"Status Saat Ini: {status['teks']} (jarak {status['jarak']:.2f})"


def render_connection_banner(session, art: Artifacts) -> None:
    """Banner "Status Saat Ini: paling dekat ke Klaster N (jarak X)" + deskripsi singkat klaster; murni deskriptif."""
    status = connection_status(session.active_calls() if session is not None else [])
    color = banner_color(status)
    desc = ""
    if status["klaster"] is not None:
        desc = art.descriptions.get(status["klaster"], "")
        proto = art.profile_protocol(status["klaster"])
        if proto:
            desc = f"{desc} · {proto}"
        if status["di_luar"]:
            desc = f"Jauh dari semua klaster. Klaster terdekat: {desc}" if desc else "Jauh dari semua klaster."
        if status["n_aktif"] > 1:
            desc += f" · {status['n_kelompok']} dari {status['n_aktif']} panggilan aktif"
    st.markdown(
        f'<div style="background:{color};color:{_foreground(color)};padding:1rem 1.4rem;border-radius:.7rem;margin-bottom:.6rem">'
        f'<div style="font-size:2.1rem;font-weight:700;line-height:1.2">{html.escape(banner_text(status))}</div>'
        f'<div style="font-size:.9rem;opacity:.9">{html.escape(desc)}</div></div>',
        unsafe_allow_html=True,
    )


def render_capture_status(cfg: dict, session) -> None:
    """Status capture kecil (paket, flow aktif, paket terakhir) + peringatan jika berjalan tapi belum ada paket."""
    if session is None:
        return
    s = session.capture_stats()
    last = s["paket_terakhir"]
    text = f"Capture: **{s['paket']:,}** paket · **{s['flow_aktif']}** flow aktif · paket terakhir: " + (
        f"{last:%H:%M:%S} ({s['detik_sejak_paket']:.0f} dtk lalu)" if last else "belum ada"
    )
    st.caption(text)
    elapsed = (datetime.now() - session.started_at).total_seconds()
    if session.running and not s["paket"] and elapsed > 2 * cfg["monitoring"]["window_sec"]:
        st.warning(
            "Belum ada paket tertangkap. Periksa interface (harus aktif dan terhubung), filter BPF, dan izin Npcap/Administrator."
        )


def render_cluster_cards(counts: dict[str, int], art: Artifacts) -> None:
    """Kartu jumlah flow: Total, tiap klaster (palet kategorikal), dan Di luar klaster (abu-abu)."""
    keys = ["Total Aliran Trafik", *[f"Klaster {c}" for c in art.cluster_ids], "Di luar klaster"]
    for col, key in zip(st.columns(len(keys)), keys):
        color = NEUTRAL_COLOR if key == "Total Aliran Trafik" else state_color(key)
        col.markdown(f'<div style="height:5px;border-radius:3px;background:{color}"></div>', unsafe_allow_html=True)
        col.metric(key, counts.get(key, 0))
        if key.startswith("Klaster "):
            col.caption(art.characters.get(int(key.split()[1]), ""))


def render_active_calls(session, art: Artifacts) -> None:
    """Panel "Panggilan Aktif": klaster terdekat, jarak, bar jarak ke semua centroid, dan sparkline klaster per jendela."""
    st.subheader("Panggilan Aktif")
    calls = session.active_calls() if session is not None else []
    if not calls:
        st.caption("Tidak ada panggilan aktif.")
        return
    for start in range(0, len(calls), CARDS_PER_ROW):
        for col, call in zip(st.columns(CARDS_PER_ROW), calls[start:start + CARDS_PER_ROW]):
            with col.container(border=True):
                st.markdown(f"**{call_title(call['flow_id'])}**")
                text = call_state_text(call["status"], call["klaster"]) if call["klaster"] is not None else call["status"]
                st.markdown(_badge(text, state_color(call["kunci"])), unsafe_allow_html=True)
                if call["klaster"] is not None:
                    limit = art.limits.get(call["klaster"])
                    st.caption(f"Jarak ke centroid terdekat {call['jarak']:.3f}" + (f" (batas {limit:.3f})" if limit is not None else ""))
                    st.plotly_chart(
                        distance_bars(call["jarak_ke_klaster"], art.limits, call["klaster"], height=24 * len(call["jarak_ke_klaster"]) + 50),
                        width="stretch", key=f"dist_{call['flow_id']}", config={"displayModeBar": False},
                    )
                st.caption(
                    f"Durasi {call['durasi_s']:.0f} dtk · delay {call['delay_ms']:.1f} ms · "
                    f"jitter {call['jitter_ms']:.1f} ms · FAI {call['fai_pct']:.2f}%"
                )
                spark = "".join(
                    f'<span style="color:{state_color(c)}" title="{html.escape(c)}">{SPARK_BLOCK}</span>' for c in call["riwayat"]
                )
                st.markdown(f'<div style="font-size:1.1rem;letter-spacing:2px">{spark}</div>', unsafe_allow_html=True)


def list_sample_pcaps(cfg: dict) -> list[Path]:
    """Daftar PCAP di data/sample_pcap."""
    folder = ROOT / "data" / "sample_pcap"
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in PCAP_SUFFIXES) if folder.is_dir() else []


@st.cache_data(ttl=30, show_spinner=False)
def list_interfaces(tshark_path: str | None) -> list[str]:
    """Daftar interface (tshark -D bila ada, selain itu psutil); di-cache 30 detik."""
    return sniffer.list_interfaces({"monitoring": {"tshark_path": tshark_path}})


def _start(
    cfg: dict, art: Artifacts, mode: str, interface: str | None, pcap: Path | None, speed: float, backend: str
) -> None:
    """Buat dan jalankan MonitoringSession; error prasyarat ditampilkan di sidebar (MON-8)."""
    c = copy.deepcopy(cfg)
    c["monitoring"]["replay_speed"] = speed
    c["monitoring"]["capture_backend"] = backend
    engine = InferenceEngine(art, c)
    if mode == MODE_LIVE:
        report = sniffer.check_live_environment(c, backend)
        if not report.ok:
            for problem in report.problems:
                st.sidebar.error(problem)
            return
        if not interface:
            st.sidebar.error("Pilih network interface dulu.")
            return
        source = lambda stop: sniffer.open_source(c, backend, stop=stop, interface=interface)  # noqa: E731
        name = f"{backend}_{interface}"
    else:
        if pcap is None:
            st.sidebar.error("Pilih atau unggah berkas PCAP dulu.")
            return
        source = lambda stop: iter_pcap(pcap, speed, stop)  # noqa: E731
        name = f"replay_{pcap.stem}"
    try:
        session = MonitoringSession(c, engine, source, name)
    except OSError as exc:
        st.sidebar.error(f"Gagal membuat sesi: {exc}")
        return
    st.session_state.update(session=session, live_rows=pd.DataFrame(), live_calls=pd.DataFrame(), last_summary=None)
    session.start()


def _recover_sessions(cfg: dict) -> None:
    """Beri ringkasan abnormal pada sesi terputus (CSV jendela tanpa ringkasan), kecuali sesi yang sedang berjalan."""
    session = st.session_state.get("session")
    try:
        recover_orphan_sessions(cfg, [session.session_id] if getattr(session, "session_id", None) else [])
    except OSError as exc:
        logger.warning("Pemulihan sesi terputus gagal: %s", exc)


def render_sidebar(cfg: dict, art: Artifacts) -> None:
    """Sidebar: versi model, mode, sumber (interface / PCAP), Start/Stop Monitoring."""
    sb = st.sidebar
    _recover_sessions(cfg)
    sb.caption(f"Model aktif: **{art.version}**")
    mode = sb.radio("Mode capture", [MODE_REPLAY, MODE_LIVE], key="mode", disabled=monitoring_active())
    interface = pcap = None
    speed = float(cfg["monitoring"]["replay_speed"])
    default_backend = cfg["monitoring"]["capture_backend"]
    backend = default_backend if default_backend in LIVE_BACKENDS else LIVE_BACKENDS[0]
    if mode == MODE_LIVE:
        backend = sb.selectbox(
            "Backend capture", list(LIVE_BACKENDS), index=LIVE_BACKENDS.index(backend), key="backend", disabled=monitoring_active()
        )
        names = list_interfaces(cfg["monitoring"].get("tshark_path")) or [""]
        status = sniffer.interface_status()
        interface = sb.selectbox(
            "Network interface", names, index=sniffer.default_interface_index(names, status), key="interface",
            format_func=lambda n: f"{n} (tidak aktif)" if status and not status.get(n, True) else n,
            disabled=monitoring_active(),
        )
        report = sniffer.check_live_environment(cfg, backend)
        with sb.expander("Status prasyarat live capture", expanded=not report.ok):
            for problem in report.problems:
                st.error(problem)
            for note in report.notes:
                st.caption(note)
    else:
        samples = list_sample_pcaps(cfg)
        choice = sb.selectbox("PCAP contoh (data/sample_pcap)", ["—"] + [p.name for p in samples], key="pcap_choice")
        upload = sb.file_uploader("…atau unggah PCAP", type=[s.lstrip(".") for s in PCAP_SUFFIXES], key="pcap_upload")
        speed = sb.number_input("Kecepatan replay (×, 0 = secepat mungkin)", min_value=0.0, value=speed, step=1.0, key="speed")
        if upload is not None:
            target = resolve_path(cfg["paths"]["reports_dir"]) / "uploads"
            target.mkdir(parents=True, exist_ok=True)
            pcap = target / Path(upload.name).name
            pcap.write_bytes(upload.getvalue())
        elif choice != "—":
            pcap = next(p for p in samples if p.name == choice)
    running = monitoring_active()
    col1, col2 = sb.columns(2)
    if col1.button("Start", type="primary", disabled=running, width="stretch"):
        _start(cfg, art, mode, interface, pcap, speed, backend)
        st.rerun()
    if col2.button("Stop", disabled=st.session_state.get("session") is None, width="stretch"):
        path = st.session_state["session"].stop()
        st.session_state["last_summary"] = path
        st.session_state["session"] = None
        st.rerun()
    _session_status(float(cfg["monitoring"]["refresh_sec"]))


def _session_status(refresh_sec: float) -> None:
    """Status sesi di sidebar; fragment yang menyegarkan diri agar "Replay selesai" muncul tanpa interaksi."""

    @st.fragment(run_every=refresh_sec)
    def render() -> None:
        with st.sidebar:
            session = st.session_state.get("session")
            if session is not None:
                if session.running:
                    st.success("Monitoring berjalan")
                else:
                    st.success("Replay selesai — tekan Stop untuk menyimpan ringkasan")
                if session.error:
                    st.error(f"Sesi berhenti karena error: {session.error}")
            if st.session_state.get("last_summary"):
                st.info(f"Ringkasan sesi: `{st.session_state['last_summary']}`")

    render()
