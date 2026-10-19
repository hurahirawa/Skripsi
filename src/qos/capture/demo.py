"""Generator PCAP sintetis mirip VoIP (SIP + RTP) untuk demo dan uji replay. BUKAN data nyata."""
from __future__ import annotations

from pathlib import Path

import numpy as np

RTP_INTERVAL_SEC = 0.02  # G.711, 20 ms per paket
RTP_PAYLOAD = b"\x80\x00" + bytes(158)
# (ip_a, port_a, ip_b, port_b, jitter_sec, loss_prob_b_to_a)
DEFAULT_CALLS: tuple[tuple[str, int, str, int, float, float], ...] = (
    ("10.0.0.1", 16384, "10.0.0.2", 16384, 0.0005, 0.0),
    ("10.0.0.3", 16500, "10.0.0.4", 16500, 0.006, 0.03),
    ("10.0.0.5", 17000, "10.0.0.6", 17000, 0.015, 0.10),
)


def build_packets(duration_sec: float = 30.0, seed: int = 42, t0: float = 1_700_000_000.0) -> list:
    """Bangun daftar paket scapy (urut waktu): beberapa panggilan RTP dua arah + sinyal SIP jarang."""
    from scapy.layers.inet import IP, UDP
    from scapy.layers.l2 import Ether

    rng = np.random.default_rng(seed)
    packets = []

    def add(ts: float, src: str, sport: int, dst: str, dport: int, payload: bytes) -> None:
        pkt = Ether() / IP(src=src, dst=dst) / UDP(sport=sport, dport=dport) / payload
        pkt.time = ts
        packets.append(pkt)

    for ip_a, pa, ip_b, pb, jitter, loss in DEFAULT_CALLS:
        n = int(duration_sec / RTP_INTERVAL_SEC)
        for i in range(n):
            base = t0 + i * RTP_INTERVAL_SEC
            add(base + abs(rng.normal(0, jitter)), ip_a, pa, ip_b, pb, RTP_PAYLOAD)
            if rng.random() >= loss:
                add(base + abs(rng.normal(0, jitter)) + 0.001, ip_b, pb, ip_a, pa, RTP_PAYLOAD)
    for k in range(int(duration_sec // 5)):  # SIP: satu pasang paket per 5 detik
        add(t0 + k * 5 + 0.5, "10.0.0.1", 5060, "10.0.0.2", 5060, b"INVITE sip:test SIP/2.0\r\n")
        add(t0 + k * 5 + 0.55, "10.0.0.2", 5060, "10.0.0.1", 5060, b"SIP/2.0 200 OK\r\n")
    packets.sort(key=lambda p: float(p.time))
    return packets


def write_demo_pcap(path: str | Path, duration_sec: float = 30.0, seed: int = 42) -> Path:
    """Tulis PCAP sintetis ke `path`."""
    from scapy.utils import wrpcap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(path), build_packets(duration_sec, seed))
    return path
