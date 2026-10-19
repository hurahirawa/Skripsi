"""Sumber paket: replay PCAP, live pyshark (default, butuh tshark), dan live scapy (fallback, butuh Npcap/libpcap).

Sumber live mengeluarkan `None` sebagai detak saat tidak ada paket, agar jendela tetap tertutup tepat waktu.
"""
from __future__ import annotations

import asyncio
import ctypes.util
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qos.capture.flow_table import PacketRecord

logger = logging.getLogger(__name__)

SLEEP_STEP_SEC = 0.2  # tidur dipecah agar sinyal stop cepat ditanggapi
BACKENDS: tuple[str, ...] = ("pyshark", "scapy", "replay")
LIVE_BACKENDS: tuple[str, ...] = ("pyshark", "scapy")
TSHARK_TIMEOUT_SEC = 10
STARTUP_CHECK_SEC = 0.5  # jeda setelah sniffer start untuk menangkap error awal (izin, interface)
MAX_QUEUE = 200_000      # paket yang menunggu diproses; sisanya dibuang dan dicatat


class CaptureError(RuntimeError):
    """Kegagalan akuisisi paket (berkas hilang, izin ditolak, tshark/Npcap tidak ada, dsb.)."""


# ---------------------------------------------------------------- lingkungan
@dataclass
class EnvReport:
    """Hasil pemeriksaan lingkungan capture: `problems` menghalangi, `notes` hanya informasi."""

    backend: str
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True jika tidak ada masalah penghalang."""
        return not self.problems


def find_tshark(
    configured: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    is_file: Callable[[str], bool] = os.path.isfile,
    platform: str = sys.platform,
) -> str | None:
    """Cari tshark: path dari config, PATH, lalu lokasi instalasi umum Windows."""
    if configured:
        return configured if is_file(configured) else None
    found = which("tshark")
    if found:
        return found
    if platform == "win32":
        for env in ("ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(env)
            if base and is_file(candidate := os.path.join(base, "Wireshark", "tshark.exe")):
                return candidate
    return None


def npcap_installed(
    platform: str = sys.platform,
    is_file: Callable[[str], bool] = os.path.isfile,
    find_lib: Callable[[str], Any] = ctypes.util.find_library,
) -> bool:
    """Windows: Npcap/WinPcap (wpcap.dll) terpasang; sistem lain: libpcap tersedia."""
    if platform == "win32":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        return any(is_file(os.path.join(root, "System32", *parts)) for parts in (("Npcap", "wpcap.dll"), ("wpcap.dll",)))
    return bool(find_lib("pcap"))


def npcap_admin_only() -> bool:
    """True jika Npcap dikonfigurasi 'Administrator only' (Windows)."""
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services\npcap\Parameters") as key:
            return bool(winreg.QueryValueEx(key, "AdminOnly")[0])
    except OSError:
        return False


def is_admin() -> bool:
    """True jika proses berjalan sebagai Administrator (Windows) atau root (POSIX)."""
    try:
        if sys.platform == "win32":
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except (AttributeError, OSError):
        return False


def check_live_environment(cfg: dict[str, Any], backend: str) -> EnvReport:
    """Periksa prasyarat live capture untuk `backend`; pesan berbahasa Indonesia dengan langkah perbaikan."""
    report = EnvReport(backend)
    if backend not in LIVE_BACKENDS:
        report.problems.append(f"Backend live tidak dikenal: {backend!r} (pilihan: {list(LIVE_BACKENDS)})")
        return report
    windows = sys.platform == "win32"
    if backend == "pyshark":
        tshark = find_tshark(cfg["monitoring"].get("tshark_path"))
        if tshark is None:
            report.problems.append(
                "tshark tidak ditemukan. Pasang Wireshark (sertakan komponen TShark) atau isi monitoring.tshark_path; "
                "alternatifnya atur monitoring.capture_backend: scapy."
            )
        else:
            report.notes.append(f"tshark: {tshark}")
    if not npcap_installed():
        report.problems.append(
            "Npcap belum terpasang (wpcap.dll tidak ditemukan). Pasang Npcap dari https://npcap.com."
            if windows
            else "libpcap tidak ditemukan. Pasang paket libpcap (mis. apt install libpcap0.8)."
        )
    else:
        report.notes.append("Npcap/libpcap: terpasang")
    admin = is_admin()
    if windows and npcap_admin_only() and not admin:
        report.problems.append("Npcap dikonfigurasi 'Administrator only'. Jalankan terminal/aplikasi sebagai Administrator.")
    elif not admin:
        report.notes.append(
            "Proses tidak berjalan sebagai Administrator/root; jika capture ditolak, jalankan sebagai Administrator/root."
        )
    return report


def parse_tshark_interfaces(text: str) -> list[str]:
    """Parse keluaran `tshark -D` menjadi nama interface (nama ramah di dalam kurung bila ada)."""
    names = []
    for line in text.splitlines():
        line = line.strip()
        head = line.split(" ", 1)[0]
        if not line or not head.endswith(".") or not head[:-1].isdigit():
            continue
        body = line[len(head):].strip()
        if body.endswith(")") and "(" in body:
            names.append(body[body.rindex("(") + 1 : -1])
        else:
            names.append(body.split(" ", 1)[0])
    return names


def list_interfaces(cfg: dict[str, Any]) -> list[str]:
    """Daftar interface: `tshark -D` bila tshark ada (MON-2), selain itu psutil."""
    tshark = find_tshark(cfg["monitoring"].get("tshark_path"))
    if tshark:
        try:
            out = subprocess.run([tshark, "-D"], capture_output=True, text=True, timeout=TSHARK_TIMEOUT_SEC, check=True)
            names = parse_tshark_interfaces(out.stdout)
            if names:
                return names
        except (OSError, subprocess.SubprocessError):
            logger.warning("tshark -D gagal; memakai psutil")
    try:
        import psutil

        return sorted(psutil.net_if_addrs())
    except Exception:  # noqa: BLE001 — psutil tak tersedia / dibatasi OS
        return []


def explain_error(exc: BaseException, interface: str | None = None) -> str:
    """Ubah error capture mentah menjadi pesan yang bisa ditindaklanjuti."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if isinstance(exc, PermissionError) or any(k in low for k in ("permission", "denied", "not permitted", "access is")):
        return f"Izin capture ditolak ({text}). Jalankan sebagai Administrator (Windows) atau root / CAP_NET_RAW (Linux)."
    if any(k in low for k in ("no libpcap", "wpcap", "winpcap", "npcap", "pcap provider")):
        return f"Npcap/libpcap tidak dapat dimuat ({text}). Pasang atau perbaiki Npcap (https://npcap.com)."
    if any(k in low for k in ("no such device", "not found", "unknown interface", "does not exist", "invalid interface")):
        return f"Interface {interface!r} tidak ditemukan ({text}). Pilih interface dari daftar."
    return f"Capture gagal ({text})."


# --------------------------------------------------------------- konversi paket
def _parse_raw_ip(pkt: Any) -> Any:
    """Parse payload mentah sebagai IPv4/IPv6 berdasarkan nibble versi; kembalikan paket asli bila bukan IP."""
    from scapy.layers.inet import IP
    from scapy.layers.inet6 import IPv6

    data = bytes(pkt)
    version = data[0] >> 4 if data else 0
    if version not in (4, 6):
        return pkt
    parsed = (IP if version == 4 else IPv6)(data)
    parsed.time = pkt.time
    return parsed


def packet_from_scapy(pkt: Any) -> PacketRecord | None:
    """Ubah paket scapy menjadi PacketRecord; non-IP dilewati (None). Port 0 untuk protokol tanpa port."""
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.packet import Raw

    if not (pkt.haslayer(IP) or pkt.haslayer(IPv6)) and pkt.haslayer(Raw):
        pkt = _parse_raw_ip(pkt)  # linktype raw-IP (mis. 101/228) tidak dikenali scapy
    if pkt.haslayer(IP):
        ip = pkt[IP]
        src, dst, proto = ip.src, ip.dst, int(ip.proto)
    elif pkt.haslayer(IPv6):
        ip = pkt[IPv6]
        src, dst, proto = ip.src, ip.dst, int(ip.nh)
    else:
        return None
    sport = dport = 0
    for layer in (TCP, UDP):
        if pkt.haslayer(layer):
            sport, dport = int(pkt[layer].sport), int(pkt[layer].dport)
            break
    return PacketRecord(float(pkt.time), src, dst, sport, dport, proto, len(pkt))


def _layer(pkt: Any, name: str) -> Any | None:
    """Layer pyshark bernama `name`, atau None."""
    try:
        return getattr(pkt, name)
    except AttributeError:
        return None


def packet_from_pyshark(pkt: Any) -> PacketRecord | None:
    """Ubah paket pyshark menjadi PacketRecord; non-IP dilewati (None)."""
    ip = _layer(pkt, "ip")
    if ip is not None:
        src, dst, proto = str(ip.src), str(ip.dst), int(ip.proto)
    else:
        ip6 = _layer(pkt, "ipv6")
        if ip6 is None:
            return None
        src, dst, proto = str(ip6.src), str(ip6.dst), int(ip6.nxt)
    sport = dport = 0
    for name in ("tcp", "udp"):
        layer = _layer(pkt, name)
        if layer is not None:
            sport, dport = int(layer.srcport), int(layer.dstport)
            break
    length = int(getattr(pkt, "length", 0) or 0)
    return PacketRecord(pkt.sniff_time.timestamp(), src, dst, sport, dport, proto, length)


# ------------------------------------------------------------------ replay
def iter_pcap(
    path: str | Path,
    speed: float | None = 1.0,
    stop: threading.Event | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[PacketRecord]:
    """Baca .pcap/.pcapng dan keluarkan paket dengan timing asli dibagi `speed`.

    speed None atau ≤ 0: tanpa jeda (secepat mungkin). `stop` yang di-set menghentikan iterasi.
    """
    # impor layer dulu: PcapReader mendaftarkan tipe link-layer dari layer yang sudah dimuat
    import scapy.layers.inet  # noqa: F401
    import scapy.layers.inet6  # noqa: F401
    import scapy.layers.l2  # noqa: F401
    from scapy.utils import PcapReader

    path = Path(path)
    if not path.is_file():
        raise CaptureError(f"Berkas PCAP tidak ditemukan: {path}")
    realtime = speed is not None and speed > 0
    t0_pcap: float | None = None
    t0_wall = clock()
    try:
        reader = PcapReader(str(path))
    except Exception as exc:  # scapy melempar beragam error untuk berkas rusak
        raise CaptureError(f"Gagal membuka PCAP {path.name}: {exc}") from exc
    with reader:
        for raw in reader:
            if stop is not None and stop.is_set():
                return
            rec = packet_from_scapy(raw)
            if rec is None:
                continue
            if realtime:
                t0_pcap = rec.ts if t0_pcap is None else t0_pcap
                due = t0_wall + (rec.ts - t0_pcap) / speed
                while (wait := due - clock()) > 0:
                    if stop is not None and stop.is_set():
                        return
                    sleep(min(wait, SLEEP_STEP_SEC))
            yield rec


# -------------------------------------------------------------------- live
def _drain_queue(
    q: queue.Queue, stop: threading.Event, poll_sec: float, failure: Callable[[], BaseException | None]
) -> Iterator[PacketRecord | None]:
    """Generator bersama: ambil dari queue; keluarkan None saat idle; naikkan CaptureError jika sumber gagal."""
    while not stop.is_set():
        try:
            yield q.get(timeout=poll_sec)
        except queue.Empty:
            exc = failure()
            if exc is not None:
                raise CaptureError(explain_error(exc)) from exc
            yield None


def iter_live_scapy(
    interface: str,
    bpf: str | None,
    stop: threading.Event,
    poll_sec: float,
    sniffer_factory: Callable[..., Any] | None = None,
) -> Iterator[PacketRecord | None]:
    """Live capture scapy AsyncSniffer (fallback). Mengeluarkan None tiap `poll_sec` saat tidak ada paket."""
    if sniffer_factory is None:
        import scapy.layers.inet  # noqa: F401
        import scapy.layers.inet6  # noqa: F401
        import scapy.layers.l2  # noqa: F401
        from scapy.sendrecv import AsyncSniffer as sniffer_factory  # noqa: N813

    q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
    dropped = [0]

    def on_packet(p: Any) -> None:
        rec = packet_from_scapy(p)
        if rec is not None:
            try:
                q.put_nowait(rec)
            except queue.Full:
                dropped[0] += 1

    sniffer = None
    try:
        sniffer = sniffer_factory(iface=interface, filter=bpf or None, prn=on_packet, store=False)
        sniffer.start()
        time.sleep(STARTUP_CHECK_SEC)
        if getattr(sniffer, "exception", None) is not None:
            raise sniffer.exception
    except Exception as exc:  # noqa: BLE001 — scapy melempar beragam tipe error
        raise CaptureError(explain_error(exc, interface)) from exc
    try:
        yield from _drain_queue(q, stop, poll_sec, lambda: getattr(sniffer, "exception", None))
    finally:
        try:
            sniffer.stop()
        except Exception:  # noqa: BLE001 — sniffer mungkin sudah berhenti
            logger.debug("sniffer.stop() gagal", exc_info=True)
        if dropped[0]:
            logger.warning("%d paket dibuang karena antrean penuh", dropped[0])


def iter_live_pyshark(
    interface: str,
    bpf: str | None,
    stop: threading.Event,
    poll_sec: float,
    tshark_path: str | None = None,
    capture_factory: Callable[..., Any] | None = None,
) -> Iterator[PacketRecord | None]:
    """Live capture pyshark LiveCapture (default; butuh tshark), di thread sendiri dengan event loop baru."""
    if capture_factory is None:
        import pyshark

        capture_factory = pyshark.LiveCapture

    q: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
    errors: list[BaseException] = []
    holder: dict[str, Any] = {}

    def worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            kwargs: dict[str, Any] = {"interface": interface, "bpf_filter": bpf or None}
            if tshark_path:
                kwargs["tshark_path"] = tshark_path
            cap = holder["cap"] = capture_factory(**kwargs)
            for pkt in cap.sniff_continuously():
                rec = packet_from_pyshark(pkt)
                if rec is not None:
                    try:
                        q.put_nowait(rec)
                    except queue.Full:
                        pass
        except BaseException as exc:  # noqa: BLE001 — diteruskan ke thread konsumen
            errors.append(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=worker, name="pyshark-capture", daemon=True)
    thread.start()
    time.sleep(STARTUP_CHECK_SEC)
    if errors:
        raise CaptureError(explain_error(errors[0], interface)) from errors[0]

    def failure() -> BaseException | None:
        if errors:
            return errors[0]
        return None if thread.is_alive() else RuntimeError("proses tshark berhenti tiba-tiba")

    try:
        yield from _drain_queue(q, stop, poll_sec, failure)
    finally:
        cap = holder.get("cap")
        if cap is not None:
            try:
                cap.close()
            except Exception:  # noqa: BLE001 — penutupan dari thread lain bisa gagal; thread daemon
                logger.debug("pyshark close gagal", exc_info=True)


def open_source(
    cfg: dict[str, Any],
    backend: str,
    pcap_path: str | Path | None = None,
    stop: threading.Event | None = None,
    interface: str | None = None,
) -> Iterator[PacketRecord | None]:
    """Buka sumber paket. Prasyarat live diperiksa SEKARANG (bukan saat iterasi) dan gagal dengan pesan jelas."""
    mon = cfg["monitoring"]
    if backend == "replay":
        if pcap_path is None:
            raise CaptureError("Mode replay butuh path berkas PCAP")
        return iter_pcap(pcap_path, mon["replay_speed"], stop)
    if backend not in LIVE_BACKENDS:
        raise CaptureError(f"Backend tidak dikenal: {backend!r} (pilihan: {list(BACKENDS)})")
    report = check_live_environment(cfg, backend)
    if not report.ok:
        raise CaptureError(" ".join(report.problems))
    if not interface:
        raise CaptureError("Pilih network interface untuk live capture")
    stop = stop or threading.Event()
    poll = float(mon["live_poll_sec"])
    if backend == "scapy":
        return iter_live_scapy(interface, mon["bpf_filter"], stop, poll)
    return iter_live_pyshark(interface, mon["bpf_filter"], stop, poll, find_tshark(mon.get("tshark_path")))
