"""Uji flow table 5-tuple dua arah dan jendela waktu."""
import math

import pytest

from qos.capture.flow_table import (
    PacketRecord,
    RunningStats,
    WindowAggregator,
    canonical_key,
    make_flow_id,
)

A, B = ("10.0.0.1", 5000), ("10.0.0.2", 6000)


def pkt(ts, src=A, dst=B, proto=17):
    return PacketRecord(ts, src[0], dst[0], src[1], dst[1], proto, 100)


def agg(mode="window", window=5.0, min_packets=1, idle=120.0):
    return WindowAggregator(window, mode, min_packets, idle)


def test_running_stats_ddof1():
    s = RunningStats()
    assert math.isnan(s.get_mean()) and math.isnan(s.get_std())
    s.add(2.0)
    assert s.get_mean() == 2.0 and math.isnan(s.get_std())  # satu nilai: std tidak terdefinisi
    for x in (4.0, 6.0):
        s.add(x)
    assert s.get_mean() == pytest.approx(4.0)
    assert s.get_std() == pytest.approx(2.0)  # std sampel [2,4,6] = 2


def test_kunci_kanonik_dua_arah():
    assert canonical_key(pkt(0, A, B)) == canonical_key(pkt(0, B, A))
    assert canonical_key(pkt(0, A, B, 17)) != canonical_key(pkt(0, A, B, 6))
    assert make_flow_id(canonical_key(pkt(0))) == "10.0.0.1:5000-10.0.0.2:6000-17"


def test_window_fwd_bwd_dan_iat():
    a = agg()
    for ts, (s, d) in [(0.0, (A, B)), (0.1, (B, A)), (0.2, (A, B)), (0.3, (B, A))]:
        assert a.add(pkt(ts, s, d)) == []
    (w,) = a.flush()
    assert (w.fwd, w.bwd, w.n_packets) == (2, 2, 4)  # forward = pengirim paket pertama
    assert w.duration_us == pytest.approx(300_000)
    assert w.iat_mean_us == pytest.approx(100_000)
    assert w.iat_std_us == pytest.approx(0.0, abs=1e-6)
    assert w.window_end - w.window_start == 5.0 and w.protocol == 17


def test_arah_forward_dari_paket_pertama_walau_urutan_terbalik():
    a = agg()
    for ts in (0.0, 0.1, 0.2):
        a.add(pkt(ts, B, A))  # B pengirim pertama
    a.add(pkt(0.3, A, B))
    (w,) = a.flush()
    assert (w.fwd, w.bwd) == (3, 1)


def test_jendela_tertutup_saat_paket_melewati_batas():
    a = agg(window=5.0)
    a.add(pkt(0.0))
    a.add(pkt(1.0))
    out = a.add(pkt(5.0))  # memulai jendela kedua
    assert len(out) == 1 and out[0].n_packets == 2 and out[0].window_start == 0.0
    (w2,) = a.flush()
    assert w2.n_packets == 1 and w2.window_start == 5.0


def test_lompat_jendela_kosong():
    a = agg(window=5.0)
    a.add(pkt(0.0))
    out = a.add(pkt(23.0))
    assert len(out) == 1  # jendela kosong tidak menghasilkan baris
    (w,) = a.flush()
    assert w.window_start == 20.0 and w.window_end == 25.0


def test_min_packets_dilewati_dan_dihitung():
    a = agg(min_packets=3)
    a.add(pkt(0.0))
    a.add(pkt(0.1))
    assert a.flush() == [] and a.skipped_min_packets == 1


def test_std_nan_bila_iat_kurang_dari_dua():
    a = agg()
    a.add(pkt(0.0))
    a.add(pkt(0.2))  # hanya satu IAT
    (w,) = a.flush()
    assert w.iat_mean_us == pytest.approx(200_000) and math.isnan(w.iat_std_us)


def test_mode_window_iat_tidak_lintas_jendela():
    a = agg("window")
    a.add(pkt(0.0))
    a.add(pkt(0.1))
    a.add(pkt(0.2))
    a.add(pkt(5.5))
    a.add(pkt(5.6))
    a.add(pkt(5.7))
    (w2,) = a.flush()
    assert w2.iat_mean_us == pytest.approx(100_000)  # selisih 0.2→5.5 tidak ikut
    assert w2.n_packets == 3


def test_mode_cumulative_menghitung_sejak_awal_flow():
    a = agg("cumulative")
    for ts in (0.0, 0.1, 0.2):
        a.add(pkt(ts))
    (w1,) = a.add(pkt(5.5)) or a.flush()  # jendela pertama
    assert w1.n_packets in (3, 4)
    a.add(pkt(5.6))
    (w2,) = a.flush()
    assert w2.n_packets >= 5 and w2.duration_us == pytest.approx(5_600_000)  # sejak paket pertama


def test_cumulative_min_packets_memakai_total_kumulatif():
    a = agg("cumulative", min_packets=4)
    a.add(pkt(0.0))
    a.add(pkt(0.1))
    first = a.add(pkt(5.5))
    assert first == []  # kumulatif baru 2 paket
    a.add(pkt(5.6))
    (w,) = a.flush()
    assert w.n_packets == 4


def test_flow_idle_dibuang():
    a = agg(idle=10.0)
    a.add(pkt(0.0))
    a.add(pkt(5.0, ("10.9.9.9", 1), ("10.9.9.8", 2)))  # jendela 1 ditutup, flow pertama masih baru
    assert a.active_flows == 2
    a.add(pkt(20.0, ("10.9.9.9", 1), ("10.9.9.8", 2)))
    a.add(pkt(26.0, ("10.9.9.9", 1), ("10.9.9.8", 2)))
    assert a.active_flows == 1  # flow pertama idle > 10 detik


def test_validasi_argumen():
    with pytest.raises(ValueError):
        WindowAggregator(5, "bukan_mode", 1, 10)
    with pytest.raises(ValueError):
        WindowAggregator(0, "window", 1, 10)


def test_flush_tanpa_paket():
    assert agg().flush() == []
