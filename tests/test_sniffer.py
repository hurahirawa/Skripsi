"""Uji sumber paket replay PCAP."""
import threading

import pytest

from qos.capture.demo import build_packets, write_demo_pcap
from qos.capture.sniffer import CaptureError, iter_pcap, open_source, packet_from_scapy
from qos.config import load_config


@pytest.fixture(scope="module")
def pcap(tmp_path_factory):
    return write_demo_pcap(tmp_path_factory.mktemp("pcap") / "demo.pcap", duration_sec=6.0)


def test_iter_pcap_tanpa_jeda(pcap):
    pkts = list(iter_pcap(pcap, speed=0))
    assert len(pkts) == len(build_packets(6.0))
    assert all(p.protocol == 17 for p in pkts)
    assert [p.ts for p in pkts] == sorted(p.ts for p in pkts)
    assert {p.dst_port for p in pkts} >= {16384, 5060}


def test_packet_from_scapy_non_ip_dilewati():
    from scapy.layers.l2 import ARP, Ether

    pkt = Ether() / ARP()
    pkt.time = 1.0
    assert packet_from_scapy(pkt) is None


def test_packet_from_scapy_raw_ip_tanpa_ethernet():
    from scapy.layers.inet import IP, UDP
    from scapy.packet import Raw

    raw = Raw(bytes(IP(src="1.1.1.1", dst="2.2.2.2") / UDP(sport=5, dport=6) / b"x"))
    raw.time = 7.0
    rec = packet_from_scapy(raw)
    assert (rec.src_ip, rec.dst_port, rec.protocol, rec.ts) == ("1.1.1.1", 6, 17, 7.0)


def test_timing_replay_proporsional_speed(pcap):
    now = [0.0]
    sleeps = []

    def sleep(sec):
        sleeps.append(sec)
        now[0] += sec

    list(iter_pcap(pcap, speed=2.0, sleep=sleep, clock=lambda: now[0]))
    ts = [p.ts for p in iter_pcap(pcap, speed=0)]
    assert now[0] == pytest.approx((ts[-1] - ts[0]) / 2.0, abs=0.01)  # waktu dinding = durasi PCAP / speed
    assert max(sleeps) <= 0.2 + 1e-9


def test_stop_menghentikan_iterasi(pcap):
    stop = threading.Event()
    seen = 0
    for _ in iter_pcap(pcap, speed=0, stop=stop):
        seen += 1
        if seen == 5:
            stop.set()
    assert seen == 5


def test_berkas_tidak_ada_dan_backend():
    with pytest.raises(CaptureError):
        list(iter_pcap("tidak_ada.pcap", speed=0))
    cfg = load_config(use_user=False)
    with pytest.raises(CaptureError):
        open_source(cfg, "replay", None)
    with pytest.raises(CaptureError):  # live tanpa interface / prasyarat
        open_source(cfg, "scapy", interface=None)
    with pytest.raises(CaptureError):
        open_source(cfg, "ngawur")
