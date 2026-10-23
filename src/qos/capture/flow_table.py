"""Flow table 5-tuple dua arah dengan jendela waktu (PRD MON-3, MON-4).

Jendela digerakkan oleh timestamp paket (bukan jam dinding), sehingga hasil replay sama pada kecepatan berapa pun.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

FEATURE_MODES: tuple[str, ...] = ("window", "cumulative")
US_PER_SEC = 1_000_000.0


@dataclass(frozen=True)
class PacketRecord:
    """Satu paket IP: timestamp (detik), 5-tuple, dan panjang."""

    ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    length: int = 0


@dataclass(frozen=True)
class WindowFlow:
    """Fitur satu flow pada satu jendela."""

    flow_id: str
    window_start: float
    window_end: float
    protocol: int
    fwd: int
    bwd: int
    n_packets: int
    duration_us: float
    iat_mean_us: float
    iat_std_us: float


@dataclass(frozen=True)
class FlowEnd:
    """Flow yang dianggap selesai (panggilan berakhir): waktu paket pertama dan terakhir (detik)."""

    flow_id: str
    first_ts: float
    last_ts: float


class RunningStats:
    """Mean dan simpangan baku sampel (ddof=1) berjalan (Welford)."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x: float) -> None:
        """Tambah satu nilai."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def get_mean(self) -> float:
        """Mean; NaN jika belum ada nilai."""
        return self.mean if self.n else math.nan

    def get_std(self) -> float:
        """Std sampel (ddof=1); NaN jika nilai < 2."""
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else math.nan


def canonical_key(p: PacketRecord) -> tuple[tuple[str, int], tuple[str, int], int]:
    """Kunci 5-tuple kanonik: kedua arah paket menghasilkan kunci yang sama."""
    a, b = (p.src_ip, p.src_port), (p.dst_ip, p.dst_port)
    lo, hi = sorted((a, b))
    return lo, hi, p.protocol


def make_flow_id(key: tuple[tuple[str, int], tuple[str, int], int]) -> str:
    """ID flow terbaca: `ipA:portA-ipB:portB-protokol`."""
    (ip1, p1), (ip2, p2), proto = key
    return f"{ip1}:{p1}-{ip2}:{p2}-{proto}"


def flow_ports(flow_id: str) -> tuple[int, int]:
    """Kedua port dari flow_id `ipA:portA-ipB:portB-proto`; (0, 0) jika format tidak dikenal."""
    parts = flow_id.split("-")
    try:
        return int(parts[0].rsplit(":", 1)[1]), int(parts[1].rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 0, 0


@dataclass
class _Flow:
    """Keadaan satu flow: kumulatif sejak awal dan per jendela."""

    fwd_endpoint: tuple[str, int]
    protocol: int
    first_ts: float
    last_ts: float
    cum_fwd: int = 0
    cum_bwd: int = 0
    cum_iat: RunningStats = field(default_factory=RunningStats)
    win_first: float | None = None
    win_last: float | None = None
    win_fwd: int = 0
    win_bwd: int = 0
    win_iat: RunningStats = field(default_factory=RunningStats)

    def reset_window(self) -> None:
        """Kosongkan statistik jendela."""
        self.win_first = self.win_last = None
        self.win_fwd = self.win_bwd = 0
        self.win_iat = RunningStats()


class WindowAggregator:
    """Agregasi paket → fitur per flow per jendela.

    Arah forward = pengirim paket pertama flow (seperti CICFlowMeter). IAT = selisih waktu antar paket
    berurutan dalam flow (dua arah); Std memakai ddof=1.
    feature_mode `window`: paket dalam jendela; `cumulative`: sejak awal flow.
    Flow dengan paket < min_packets dilewati. Flow dianggap selesai (dibuang dari tabel dan dicatat di `pop_ended`)
    saat idle > min(idle_timeout_sec, call_end_idle_sec), atau saat `end_all` (Stop / akhir sumber).
    """

    def __init__(
        self,
        window_sec: float,
        feature_mode: str,
        min_packets: int,
        idle_timeout_sec: float,
        call_end_idle_sec: float | None = None,
    ) -> None:
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"feature_mode tidak dikenal: {feature_mode!r} (pilihan: {list(FEATURE_MODES)})")
        if window_sec <= 0:
            raise ValueError("window_sec harus > 0")
        self.window_sec, self.feature_mode = float(window_sec), feature_mode
        self.min_packets, self.idle_timeout_sec = min_packets, idle_timeout_sec
        self.end_idle_sec = idle_timeout_sec if call_end_idle_sec is None else min(idle_timeout_sec, call_end_idle_sec)
        self._flows: dict[tuple, _Flow] = {}
        self._ended: list[FlowEnd] = []
        self._start: float | None = None
        self.skipped_min_packets = 0

    @property
    def active_flows(self) -> int:
        """Jumlah flow di flow table."""
        return len(self._flows)

    def pop_ended(self) -> list[FlowEnd]:
        """Ambil (dan kosongkan) daftar flow yang selesai sejak pemanggilan terakhir."""
        out, self._ended = self._ended, []
        return out

    def end_all(self) -> list[FlowEnd]:
        """Akhiri semua flow yang tersisa (Stop / akhir sumber); kembalikan semua flow selesai yang tertunda."""
        for key, f in self._flows.items():
            self._ended.append(FlowEnd(make_flow_id(key), f.first_ts, f.last_ts))
        self._flows.clear()
        return self.pop_ended()

    def add(self, p: PacketRecord) -> list[WindowFlow]:
        """Masukkan satu paket; kembalikan fitur jendela yang selesai karena paket ini."""
        if self._start is None:
            self._start = p.ts
        out: list[WindowFlow] = []
        out += self.advance(p.ts)
        self._update(p)
        return out

    def advance(self, now: float) -> list[WindowFlow]:
        """Majukan jam ke `now`: tutup jendela yang sudah lewat (dipakai juga sebagai detak saat live idle)."""
        if self._start is None or now < self._start + self.window_sec:
            return []
        out = self._close_window(now)
        # lompati jendela kosong
        self._start += math.floor((now - self._start) / self.window_sec) * self.window_sec
        return out

    def flush(self) -> list[WindowFlow]:
        """Tutup jendela yang sedang berjalan (akhir replay / Stop)."""
        if self._start is None:
            return []
        out = self._close_window()
        self._start += self.window_sec
        return out

    def _update(self, p: PacketRecord) -> None:
        key = canonical_key(p)
        flow = self._flows.get(key)
        if flow is None:
            flow = _Flow((p.src_ip, p.src_port), p.protocol, p.ts, p.ts)
            self._flows[key] = flow
        elif p.ts >= flow.last_ts:
            flow.cum_iat.add((p.ts - flow.last_ts) * US_PER_SEC)
            if flow.win_last is not None:
                flow.win_iat.add((p.ts - flow.win_last) * US_PER_SEC)
        forward = (p.src_ip, p.src_port) == flow.fwd_endpoint
        if forward:
            flow.cum_fwd += 1
            flow.win_fwd += 1
        else:
            flow.cum_bwd += 1
            flow.win_bwd += 1
        if flow.win_first is None:
            flow.win_first = p.ts
        flow.win_last = max(p.ts, flow.win_last or p.ts)
        flow.last_ts = max(p.ts, flow.last_ts)

    def _close_window(self, now: float | None = None) -> list[WindowFlow]:
        assert self._start is not None
        start, end = self._start, self._start + self.window_sec
        ref = end if now is None else max(end, now)  # jeda panjang: idle diukur sampai `now`, bukan akhir jendela
        out: list[WindowFlow] = []
        for key, f in list(self._flows.items()):
            if f.win_first is not None:
                wf = self._to_window_flow(key, f, start, end)
                if wf.n_packets >= self.min_packets:
                    out.append(wf)
                else:
                    self.skipped_min_packets += 1
            f.reset_window()
            if ref - f.last_ts > self.end_idle_sec:
                self._ended.append(FlowEnd(make_flow_id(key), f.first_ts, f.last_ts))
                del self._flows[key]
        return out

    def _to_window_flow(self, key: tuple, f: _Flow, start: float, end: float) -> WindowFlow:
        if self.feature_mode == "window":
            fwd, bwd = f.win_fwd, f.win_bwd
            duration = (f.win_last - f.win_first) * US_PER_SEC
            iat = f.win_iat
        else:
            fwd, bwd = f.cum_fwd, f.cum_bwd
            duration = (f.last_ts - f.first_ts) * US_PER_SEC
            iat = f.cum_iat
        return WindowFlow(
            flow_id=make_flow_id(key), window_start=start, window_end=end, protocol=f.protocol,
            fwd=fwd, bwd=bwd, n_packets=fwd + bwd, duration_us=duration,
            iat_mean_us=iat.get_mean(), iat_std_us=iat.get_std(),
        )
