"""Uji profil klaster, deskripsi ITU, dan Lampiran 5 (PRD §13)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qos.config import load_config
from qos.features.extract import derive_metrics, us_to_ms
from qos.model.evaluate import noise_ratio
from qos.profiling.describe import BETWEEN, OVER, WITHIN, describe, position
from qos.profiling.profile import (
    basic_stats,
    boxplot_stats,
    build_cluster_profile,
    extra_stats,
    redescribe,
    percentile,
    plr_pct,
)

CFG = load_config(use_user=False)
FIXTURE = Path(__file__).parent / "fixtures" / "lampiran_l21.csv"


@pytest.fixture(scope="module")
def l21():
    """Fixture Lampiran 2.1: delay dari flow_duration_us (bukan iat_mean) sesuai naskah."""
    raw = pd.read_csv(FIXTURE)
    df = raw.rename(
        columns={
            "total_fwd_packets": "fwd_packets_count",
            "total_backward_packets": "bwd_packets_count",
        }
    )
    return derive_metrics(df, "flow_duration_us", "flow_iat_std_us", clip_negative=False)


@pytest.fixture(scope="module")
def profile(l21):
    return build_cluster_profile(l21, l21["cluster"], CFG)


def by_id(profile, cid):
    return next(c for c in profile if c["id_klaster"] == cid)


# ---- Lampiran 5 ----------------------------------------------------------------------------------
def test_l5_mean_dan_std_delay(profile):
    for cid, mean, std in ((0, 109.23, 18.71), (1, 485.26, 49.70), (-1, 1183.45, 241.17)):
        st = by_id(profile, cid)["statistik"]["delay_ms"]
        assert st["mean"] == pytest.approx(mean, abs=0.005)
        assert st["std"] == pytest.approx(std, abs=0.005)


def test_l5_p999_jitter_dan_plr(profile):
    for cid, p999, plr in ((0, 0.92, 0.99), (1, 4.12, 22.92), (-1, 24.09, 47.11)):
        s = by_id(profile, cid)["statistik_tambahan"]
        assert s["p999_jitter_ms"] == pytest.approx(p999, abs=0.005)
        assert s["plr_pct"] == pytest.approx(plr, abs=0.005)


def test_l5_q1_q3_delay(profile):
    for cid, q1, q3 in ((0, 95.40, 120.54), (1, 467.69, 502.83), (-1, 1050.18, 1285.00)):
        b = by_id(profile, cid)["boxplot"]["delay_ms"]
        assert b["q1"] == pytest.approx(q1, abs=0.006)
        assert b["q3"] == pytest.approx(q3, abs=0.006)


def test_l5_deskripsi_itu_deskriptif(profile):
    # posisi median klaster terhadap ambang ITU: deskripsi, bukan label kualitas
    k0, k1, noise = by_id(profile, 0), by_id(profile, 1), by_id(profile, -1)
    assert k0["deskripsi"]["metrik"]["delay"]["posisi"] == WITHIN and "delay" not in k0["metrik_melewati_ambang"]
    assert k1["deskripsi"]["metrik"]["delay"]["posisi"] == OVER and "delay" in k1["metrik_melewati_ambang"]
    assert "delay" in noise["metrik_melewati_ambang"] and noise["nama"] == "Noise"
    assert k0["nama"] == "Klaster 0"
    assert "label" not in k0 and "kategori_tindakan" not in k0


def test_l5_fai(l21):
    fai = l21.set_index("sample")["fai_pct"]
    assert fai["s1"] == pytest.approx(1.33, abs=0.005)
    assert fai["s2"] == pytest.approx(20.45, abs=0.005)
    assert fai["s10"] == pytest.approx(52.00, abs=0.005)


def test_l5_pagar_dan_whisker_k0(profile):
    b = by_id(profile, 0)["boxplot"]["delay_ms"]
    assert b["lower_fence"] == pytest.approx(57.69, abs=0.005)
    assert b["upper_fence"] == pytest.approx(158.25, abs=0.005)
    assert b["whisker_low"] == pytest.approx(89.40, abs=0.005)
    assert b["whisker_high"] == pytest.approx(135.20, abs=0.005)
    assert b["outliers"] == []


def test_l5_noise_ratio(l21):
    assert noise_ratio(l21["cluster"].to_numpy()) == pytest.approx(0.30)


def test_l5_jumlah_anggota(profile):
    assert [c["jumlah_anggota"] for c in profile] == [5, 2, 3]  # urutan 0, 1, lalu Noise (-1) di akhir


# ---- Deskripsi posisi terhadap ambang ITU ----------------------------------------------------------
TH = CFG["thresholds"]


def test_position_batas_inklusif():
    assert position("delay", 150.0, TH) == WITHIN and position("delay", 150.1, TH) == BETWEEN
    assert position("delay", 400.0, TH) == BETWEEN and position("delay", 400.1, TH) == OVER
    assert position("jitter", 50.0, TH) == WITHIN and position("jitter", 50.1, TH) == OVER
    assert position("loss", 1.0, TH) == WITHIN and position("loss", 3.0, TH) == BETWEEN and position("loss", 3.01, TH) == OVER


def test_describe_teks_dan_metrik_melewati_ambang():
    d = describe(200.0, 10.0, 5.0, TH)
    assert d["ringkas"] == "Delay 150–400 ms · Jitter ≤50 ms · Packet Loss >3 %"
    assert d["melewati_ambang"] == ["loss"]
    assert "200,0" not in d["teks"] and "200.0 ms (150–400 ms)" in d["teks"]
    assert describe(1.0, 1.0, 0.0, TH)["melewati_ambang"] == []
    assert describe(500.0, 60.0, 4.0, TH)["melewati_ambang"] == ["delay", "jitter", "loss"]


def test_redescribe_memakai_ambang_baru_dari_statistik_tersimpan(profile):
    new_th = {**TH, "delay_good_ms": 50.0, "delay_max_ms": 100.0}
    out = redescribe(profile, new_th)
    k0 = next(c for c in out if c["id_klaster"] == 0)
    assert k0["deskripsi"]["metrik"]["delay"]["posisi"] == OVER  # median ~109 ms > 100 ms
    assert k0["jumlah_anggota"] == 5 and k0["statistik"] == by_id(profile, 0)["statistik"]  # statistik tak berubah


# ---- Fungsi dasar --------------------------------------------------------------------------------
def test_us_to_ms():
    assert us_to_ms(pd.Series([1500.0])).iloc[0] == 1.5


def test_derive_metrics_buang_fwd_nol_dan_clip():
    df = pd.DataFrame(
        {
            "iat_mean_us": [2000.0, 3000.0, 4000.0],
            "iat_std_us": [500.0, 600.0, 700.0],
            "fwd_packets_count": [10, 0, 5],
            "bwd_packets_count": [12, 3, 5],
        }
    )
    out = derive_metrics(df, "iat_mean_us", "iat_std_us", clip_negative=True)
    assert len(out) == 2
    assert out["delay_ms"].tolist() == [2.0, 4.0]
    assert out["jitter_ms"].tolist() == [0.5, 0.7]
    assert out["fai_pct"].tolist() == [0.0, 0.0]  # -20 di-clip ke 0
    raw = derive_metrics(df, "iat_mean_us", "iat_std_us", clip_negative=False)
    assert raw["fai_pct"].iloc[0] == pytest.approx(-20.0)


def test_basic_stats_ddof1_dan_satu_anggota():
    s = basic_stats(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert s["median"] == 2.5 and s["min"] == 1.0 and s["max"] == 4.0
    assert basic_stats(pd.Series([5.0]))["std"] is None


def test_boxplot_pencilan_dan_batas():
    v = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    b = boxplot_stats(v, max_outliers=500)
    assert b["outliers"] == [100.0]
    assert b["whisker_high"] == 5.0
    assert b["q1"] == pytest.approx(2.25) and b["q3"] == pytest.approx(4.75)
    many = pd.Series([0.0] * 50 + list(range(1000, 1010)) + [5000.0])
    top = boxplot_stats(many, max_outliers=3)["outliers"]
    assert len(top) == 3 and 5000.0 in top and 1009 in top


def test_percentile_linear():
    assert percentile(pd.Series([0.0, 10.0]), 50) == 5.0
    assert percentile(pd.Series([1.0, 2.0, 3.0]), 99.9) == pytest.approx(2.998)


def test_plr_pct():
    assert plr_pct(pd.Series([100, 100]), pd.Series([90, 120])) == pytest.approx(5.0)  # bwd>fwd tidak mengurangi
    with pytest.raises(ValueError):
        plr_pct(pd.Series([0]), pd.Series([0]))


def test_extra_stats_dari_config():
    df = pd.DataFrame(
        {
            "delay_ms": [10.0, 20.0],
            "jitter_ms": [1.0, 3.0],
            "fwd_packets_count": [100, 100],
            "bwd_packets_count": [100, 90],
        }
    )
    s = extra_stats(df, CFG["profiling"]["jitter_percentile"])
    assert s["mean_delay_ms"] == 15.0 and s["plr_pct"] == pytest.approx(5.0)
    assert s["p999_jitter_ms"] == pytest.approx(2.998)


def test_noise_ratio():
    assert noise_ratio(np.array([-1, 0, 0, 1])) == 0.25
    with pytest.raises(ValueError):
        noise_ratio(np.array([]))
