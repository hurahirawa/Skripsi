"""Helper bersama halaman Streamlit: konfigurasi, model aktif, sidebar, dan kontrol sesi pemantauan."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import copy  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from qos.capture import sniffer  # noqa: E402
from qos.capture.sniffer import LIVE_BACKENDS, CaptureError, iter_pcap  # noqa: E402
from qos.recommend.rules import load_rules  # noqa: E402
from qos.config import DEFAULT_CONFIG_PATH, USER_CONFIG_NAME, load_config, resolve_path, setup_logging  # noqa: E402
from qos.monitor.engine import (  # noqa: E402
    Artifacts,
    InferenceEngine,
    ModelNotFoundError,
    MonitoringSession,
    load_artifacts,
)

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
    st.session_state.update(session=session, live_rows=pd.DataFrame(), last_summary=None)
    session.start()


def render_sidebar(cfg: dict, art: Artifacts) -> None:
    """Sidebar: versi model, mode, sumber (interface / PCAP), Start/Stop Monitoring."""
    sb = st.sidebar
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
        interface = sb.selectbox(
            "Network interface", list_interfaces(cfg["monitoring"].get("tshark_path")) or [""], key="interface",
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
