"""Uji live capture (M6): prasyarat, konversi paket, adapter scapy/pyshark (dengan sniffer palsu), detak idle."""
import copy
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from qos.capture import sniffer
from qos.capture.flow_table import PacketRecord, WindowAggregator
from qos.capture.sniffer import (
    CaptureError,
    check_live_environment,
    explain_error,
    find_tshark,
    iter_live_pyshark,
    iter_live_scapy,
    npcap_installed,
    open_source,
    packet_from_pyshark,
    parse_tshark_interfaces,
)
from qos.config import load_config

CFG = load_config(use_user=False)


@pytest.fixture(autouse=True)
def no_startup_delay(monkeypatch):
    monkeypatch.setattr(sniffer, "STARTUP_CHECK_SEC", 0.0)


# ---- prasyarat ----------------------------------------------------------------------------------
def test_find_tshark_config_path_dan_which():
    assert find_tshark("C:/x/tshark.exe", is_file=lambda p: True) == "C:/x/tshark.exe"
    assert find_tshark("C:/x/tshark.exe", is_file=lambda p: False) is None  # path eksplisit tidak ada: tidak mencari lagi
    assert find_tshark(None, which=lambda n: "/usr/bin/tshark") == "/usr/bin/tshark"
    assert find_tshark(None, which=lambda n: None, platform="linux") is None


def test_find_tshark_lokasi_windows(monkeypatch):
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    found = find_tshark(None, which=lambda n: None, is_file=lambda p: p.endswith("tshark.exe"), platform="win32")
    assert found.endswith("tshark.exe") and "Wireshark" in found
    assert find_tshark(None, which=lambda n: None, is_file=lambda p: False, platform="win32") is None


def test_npcap_installed(monkeypatch):
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    assert npcap_installed("win32", is_file=lambda p: "Npcap" in p)
    assert npcap_installed("win32", is_file=lambda p: p.endswith("wpcap.dll") and "Npcap" not in p)
    assert not npcap_installed("win32", is_file=lambda p: False)
    assert npcap_installed("linux", find_lib=lambda n: "libpcap.so.0.8")
    assert not npcap_installed("linux", find_lib=lambda n: None)


def test_parse_tshark_interfaces():
    text = (
        "1. \\Device\\NPF_{ABC} (Wi-Fi)\n2. \\Device\\NPF_{DEF} (Ethernet 2)\n3. \\Device\\NPF_Loopback (Adapter for loopback traffic capture)\n"
        "4. eth0\n\nbukan interface\n"
    )
    assert parse_tshark_interfaces(text) == ["Wi-Fi", "Ethernet 2", "Adapter for loopback traffic capture", "eth0"]
    assert parse_tshark_interfaces("") == []


def test_check_env_pyshark_tanpa_tshark(monkeypatch):
    monkeypatch.setattr(sniffer, "find_tshark", lambda p=None: None)
    monkeypatch.setattr(sniffer, "npcap_installed", lambda: True)
    monkeypatch.setattr(sniffer, "npcap_admin_only", lambda: False)
    r = check_live_environment(CFG, "pyshark")
    assert not r.ok and any("tshark tidak ditemukan" in p and "scapy" in p for p in r.problems)
    assert check_live_environment(CFG, "scapy").ok  # scapy tidak butuh tshark


def test_check_env_tanpa_npcap_dan_admin_only(monkeypatch):
    monkeypatch.setattr(sniffer, "find_tshark", lambda p=None: "tshark")
    monkeypatch.setattr(sniffer, "npcap_installed", lambda: False)
    monkeypatch.setattr(sniffer, "is_admin", lambda: False)
    r = check_live_environment(CFG, "scapy")
    assert not r.ok and any("Npcap" in p or "libpcap" in p for p in r.problems)
    monkeypatch.setattr(sniffer, "npcap_installed", lambda: True)
    monkeypatch.setattr(sniffer, "npcap_admin_only", lambda: True)
    monkeypatch.setattr(sniffer.sys, "platform", "win32")
    r = check_live_environment(CFG, "scapy")
    assert not r.ok and any("Administrator only" in p for p in r.problems)
    monkeypatch.setattr(sniffer, "is_admin", lambda: True)
    assert check_live_environment(CFG, "scapy").ok


def test_check_env_backend_tak_dikenal():
    assert not check_live_environment(CFG, "replay").ok


def test_explain_error():
    assert "Administrator" in explain_error(PermissionError("denied"))
    assert "Npcap" in explain_error(RuntimeError("Sniffing and sending packets is not available: No libpcap provider"))
    assert "tidak ditemukan" in explain_error(ValueError("Unknown interface"), "eth9")
    assert "Capture gagal" in explain_error(RuntimeError("aneh"))


# ---- konversi paket pyshark ---------------------------------------------------------------------
class _TS:
    def timestamp(self):
        return 1234.5


def fake_pyshark_packet(**layers):
    return SimpleNamespace(sniff_time=_TS(), length="120", **layers)


def test_packet_from_pyshark_udp_ipv4():
    pkt = fake_pyshark_packet(
        ip=SimpleNamespace(src="1.1.1.1", dst="2.2.2.2", proto="17"), udp=SimpleNamespace(srcport="5000", dstport="16384")
    )
    assert packet_from_pyshark(pkt) == PacketRecord(1234.5, "1.1.1.1", "2.2.2.2", 5000, 16384, 17, 120)


def test_packet_from_pyshark_ipv6_dan_tanpa_port_dan_non_ip():
    v6 = fake_pyshark_packet(ipv6=SimpleNamespace(src="::1", dst="::2", nxt="6"), tcp=SimpleNamespace(srcport="1", dstport="2"))
    assert packet_from_pyshark(v6).protocol == 6
    icmp = fake_pyshark_packet(ip=SimpleNamespace(src="1.1.1.1", dst="2.2.2.2", proto="1"))
    rec = packet_from_pyshark(icmp)
    assert (rec.src_port, rec.dst_port, rec.protocol) == (0, 0, 1)
    assert packet_from_pyshark(fake_pyshark_packet()) is None  # mis. ARP


# ---- adapter scapy ------------------------------------------------------------------------------
def scapy_udp(ts=1.0):
    from scapy.layers.inet import IP, UDP
    from scapy.layers.l2 import Ether

    pkt = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=1000, dport=2000) / b"x"
    pkt.time = ts
    return pkt


class FakeSniffer:
    instances: list = []
    start_error: Exception | None = None
    exception_after_start: Exception | None = None

    def __init__(self, iface, filter, prn, store):
        self.iface, self.filter, self.prn, self.store = iface, filter, prn, store
        self.exception, self.stopped = None, False
        FakeSniffer.instances.append(self)

    def start(self):
        if FakeSniffer.start_error:
            raise FakeSniffer.start_error
        self.exception = FakeSniffer.exception_after_start

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake_sniffer():
    FakeSniffer.instances, FakeSniffer.start_error, FakeSniffer.exception_after_start = [], None, None
    return FakeSniffer


def test_scapy_live_paket_idle_dan_stop(fake_sniffer):
    stop = threading.Event()
    gen = iter_live_scapy("Wi-Fi", "udp", stop, 0.01, sniffer_factory=fake_sniffer)
    assert next(gen) is None  # idle → detak
    sn = fake_sniffer.instances[0]
    assert (sn.iface, sn.filter, sn.store) == ("Wi-Fi", "udp", False)
    sn.prn(scapy_udp(5.0))
    rec = next(gen)
    assert isinstance(rec, PacketRecord) and (rec.src_port, rec.dst_port, rec.protocol, rec.ts) == (1000, 2000, 17, 5.0)
    stop.set()
    assert list(gen) == []
    assert sn.stopped


def test_scapy_live_non_ip_diabaikan(fake_sniffer):
    from scapy.layers.l2 import ARP, Ether

    stop = threading.Event()
    gen = iter_live_scapy("x", None, stop, 0.01, sniffer_factory=fake_sniffer)
    next(gen)
    fake_sniffer.instances[0].prn(Ether() / ARP())
    assert next(gen) is None
    stop.set()


def test_scapy_live_izin_ditolak(fake_sniffer):
    fake_sniffer.start_error = PermissionError("Access is denied")
    with pytest.raises(CaptureError, match="Administrator"):
        next(iter_live_scapy("Wi-Fi", "udp", threading.Event(), 0.01, sniffer_factory=fake_sniffer))


def test_scapy_live_error_thread_setelah_start(fake_sniffer):
    fake_sniffer.exception_after_start = RuntimeError("No libpcap provider available !")
    with pytest.raises(CaptureError, match="Npcap"):
        next(iter_live_scapy("Wi-Fi", "udp", threading.Event(), 0.01, sniffer_factory=fake_sniffer))


# ---- adapter pyshark ----------------------------------------------------------------------------
class FakeCapture:
    created: list = []

    def __init__(self, **kwargs):
        self.kwargs, self.closed, self.inbox = kwargs, threading.Event(), queue.Queue()
        FakeCapture.created.append(self)

    def sniff_continuously(self):
        while not self.closed.is_set():
            try:
                yield self.inbox.get(timeout=0.01)
            except queue.Empty:
                continue

    def close(self):
        self.closed.set()


def test_pyshark_live_paket_idle_dan_stop():
    FakeCapture.created = []
    stop = threading.Event()
    gen = iter_live_pyshark("Wi-Fi", "udp", stop, 0.02, tshark_path="C:/ts/tshark.exe", capture_factory=FakeCapture)
    assert next(gen) is None
    cap = FakeCapture.created[0]
    assert cap.kwargs == {"interface": "Wi-Fi", "bpf_filter": "udp", "tshark_path": "C:/ts/tshark.exe"}
    cap.inbox.put(fake_pyshark_packet(ip=SimpleNamespace(src="1.1.1.1", dst="2.2.2.2", proto="17"),
                                      udp=SimpleNamespace(srcport="1", dstport="2")))
    cap.inbox.put(fake_pyshark_packet())  # non-IP diabaikan
    rec = None
    while rec is None:
        rec = next(gen)
    assert rec.src_ip == "1.1.1.1"
    stop.set()
    assert list(gen) == []
    assert cap.closed.is_set()


def test_pyshark_live_error_pembuatan_capture():
    def boom(**kwargs):
        raise PermissionError("permission denied")

    with pytest.raises(CaptureError, match="Administrator"):
        next(iter_live_pyshark("Wi-Fi", "udp", threading.Event(), 0.01, capture_factory=boom))


def test_pyshark_live_tshark_mati_mendadak():
    class Dies(FakeCapture):
        def sniff_continuously(self):
            return iter(())  # thread selesai tanpa paket

    gen = iter_live_pyshark("Wi-Fi", "udp", threading.Event(), 0.01, capture_factory=Dies)
    with pytest.raises(CaptureError, match="tshark berhenti"):
        for _ in range(100):
            next(gen)
            time.sleep(0.01)


# ---- open_source --------------------------------------------------------------------------------
def test_open_source_live_tanpa_tshark_gagal_jelas(monkeypatch):
    monkeypatch.setattr(sniffer, "find_tshark", lambda p=None: None)
    with pytest.raises(CaptureError, match="tshark tidak ditemukan"):
        open_source(CFG, "pyshark", interface="Wi-Fi")


def test_open_source_live_validasi_dan_dispatch(monkeypatch):
    ok = sniffer.EnvReport("scapy")
    monkeypatch.setattr(sniffer, "check_live_environment", lambda cfg, backend: ok)
    with pytest.raises(CaptureError, match="interface"):
        open_source(CFG, "scapy", interface=None)
    monkeypatch.setattr(sniffer, "iter_live_scapy", lambda *a, **k: ("scapy", a))
    monkeypatch.setattr(sniffer, "iter_live_pyshark", lambda *a, **k: ("pyshark", a))
    monkeypatch.setattr(sniffer, "find_tshark", lambda p=None: "tshark")
    assert open_source(CFG, "scapy", interface="Wi-Fi")[0] == "scapy"
    assert open_source(CFG, "pyshark", interface="Wi-Fi")[0] == "pyshark"
    with pytest.raises(CaptureError, match="tidak dikenal"):
        open_source(CFG, "ngawur", interface="x")


# ---- detak idle pada agregator & sesi -----------------------------------------------------------
def pkt(ts, src=("10.0.0.1", 5000), dst=("10.0.0.2", 6000)):
    return PacketRecord(ts, src[0], dst[0], src[1], dst[1], 17, 100)


def test_advance_menutup_jendela_tanpa_paket_baru():
    a = WindowAggregator(5.0, "window", 1, 120.0)
    assert a.advance(100.0) == []  # belum ada jendela
    for i in range(3):
        a.add(pkt(100.0 + i * 0.1))
    assert a.advance(102.0) == []  # jendela belum selesai
    (w,) = a.advance(106.0)
    assert w.n_packets == 3
    assert a.advance(107.0) == []  # tidak ada paket baru → tidak ada baris ganda
    a.add(pkt(113.0))
    (w2,) = a.flush()
    assert w2.window_start == 110.0 and w2.n_packets == 1


def _session_env(tmp_path):
    import numpy as np
    import pandas as pd

    from qos.model.train import save_artifacts, train

    cfg = copy.deepcopy(CFG)
    cfg["paths"].update({"artifacts_dir": str(tmp_path / "a"), "reports_dir": str(tmp_path / "r")})
    cfg["hdbscan"]["grid_search"].update({"min_cluster_size": [5], "min_samples": [2]})
    rng = np.random.default_rng(0)
    parts = []
    for mean_us, std_us, pk in ((1000.0, 500.0, 100), (10_000.0, 8_000.0, 300)):
        fwd = rng.normal(pk, pk * 0.02, 80).round()
        parts.append(pd.DataFrame({
            "iat_mean_us": rng.normal(mean_us, mean_us * 0.03, 80), "iat_std_us": rng.normal(std_us, std_us * 0.03, 80),
            "fwd_packets_count": fwd, "bwd_packets_count": (fwd * rng.normal(0.9, 0.02, 80)).round(),
        }))
    save_artifacts(train(pd.concat(parts, ignore_index=True), cfg), cfg)
    return cfg


def test_sesi_live_menutup_jendela_saat_idle_dan_menutup_sumber(tmp_path):
    from qos.monitor.engine import InferenceEngine, MonitoringSession, load_artifacts

    cfg = _session_env(tmp_path)
    cfg["monitoring"].update({"window_sec": 5, "feature_mode": "window", "min_packets_per_window": 1})
    closed = threading.Event()

    def source(stop):
        try:
            base = time.time() - 60  # timestamp lampau → jam dinding sudah melewati batas jendela
            for i in range(30):
                yield pkt(base + i * 0.05, ("10.0.0.1", 5000), ("10.0.0.2", 6000))
                yield pkt(base + i * 0.05 + 0.01, ("10.0.0.2", 6000), ("10.0.0.1", 5000))
            while not stop.is_set():  # lalu idle: hanya detak
                yield None
                time.sleep(0.01)
        finally:
            closed.set()

    s = MonitoringSession(cfg, InferenceEngine(load_artifacts(cfg), cfg), source, "live uji")
    s.start()
    df = s.results.get(timeout=10)  # muncul tanpa paket baru → berkat detak idle
    assert len(df) == 1 and df["fwd"].iloc[0] == 30 and df["bwd"].iloc[0] == 30
    assert s.running
    s.stop()
    assert closed.is_set() and not s.running  # sumber live ditutup saat Stop


def test_sesi_melaporkan_error_sumber_live(tmp_path):
    from qos.monitor.engine import InferenceEngine, MonitoringSession, load_artifacts

    cfg = _session_env(tmp_path)

    def source(stop):
        raise CaptureError("Izin capture ditolak")
        yield

    s = MonitoringSession(cfg, InferenceEngine(load_artifacts(cfg), cfg), source, "gagal")
    s.start()
    s.wait(10)
    s.stop()
    assert s.error and "Izin capture ditolak" in s.error
