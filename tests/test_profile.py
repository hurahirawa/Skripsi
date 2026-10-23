"""Uji profil klaster, pelabelan, dan Lampiran 5 (PRD §13)."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qos.config import load_config
from qos.features.extract import derive_metrics, us_to_ms
from qos.model.evaluate import noise_ratio
from qos.profiling.labeling import cluster_levels, join_metrics, label_cluster
from qos.profiling.profile import (
    basic_stats,
    boxplot_stats,
    build_cluster_profile,
    label_map,
    labeling_stats,
    percentile,
    plr_pct,
)

CFG = load_config(use_user=False)
FIXTURE = Path(__file__).parent / "fixtures" / "lampiran_l21.csv"
BAIK = "QoS Baik"
BURUK_DELAY_LOSS = "QoS Buruk - Delay dan Packet Loss Melebihi Batas"
ANOMALI = "Anomali - Delay dan Packet Loss Ekstrem"


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
        s = by_id(profile, cid)["statistik_pelabelan"]
        assert s["p999_jitter_ms"] == pytest.approx(p999, abs=0.005)
        assert s["plr_pct"] == pytest.approx(plr, abs=0.005)


def test_l5_q1_q3_delay(profile):
    for cid, q1, q3 in ((0, 95.40, 120.54), (1, 467.69, 502.83), (-1, 1050.18, 1285.00)):
        b = by_id(profile, cid)["boxplot"]["delay_ms"]
        assert b["q1"] == pytest.approx(q1, abs=0.006)
        assert b["q3"] == pytest.approx(q3, abs=0.006)


def test_l5_label(profile):
    # K0: PLR 0,99% ≤ loss_good_pct 1% → QoS Baik (ambang loss 3 level, S1)
    assert by_id(profile, 0)["label"] == BAIK
    assert by_id(profile, 1)["label"] == BURUK_DELAY_LOSS
    assert by_id(profile, -1)["label"] == ANOMALI
    assert label_map(profile) == {"-1": ANOMALI, "0": BAIK, "1": BURUK_DELAY_LOSS}
    assert by_id(profile, 0)["kategori_tindakan"] == "-"
    assert by_id(profile, -1)["kategori_tindakan"] == "Investigasi Kasus Tunggal"
    assert by_id(profile, 0)["tingkat"] == {"delay": 1, "jitter": 1, "loss": 1, "klaster": 1}


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
    assert [c["jumlah_anggota"] for c in profile] == [3, 5, 2]  # urutan -1, 0, 1


# ---- Tabel 2.3: semua kombinasi level -------------------------------------------------------------
def _expected(levels, noise):
    names = {"delay": "Delay", "jitter": "Jitter", "loss": "Packet Loss"}

    def joined(items):
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " dan " + items[-1]

    if noise:
        bad = [names[m] for m in names if levels[m] == 3]
        return f"Anomali - {joined(bad)} Ekstrem" if bad else "Anomali"
    t = max(levels.values())
    if t == 1:
        return "QoS Baik"
    cause = [names[m] for m in names if levels[m] == t]
    prefix, note = ("QoS Cukup - ", " Meningkat") if t == 2 else ("QoS Buruk - ", " Melebihi Batas")
    return prefix + joined(cause) + note


@pytest.mark.parametrize("noise", [False, True])
def test_semua_kombinasi_label(noise):
    for d, j, l in itertools.product((1, 2, 3), (1, 3), (1, 3)):
        levels = {"delay": d, "jitter": j, "loss": l}
        assert label_cluster(levels, noise)[0] == _expected(levels, noise), levels


def test_contoh_label_prd():
    assert label_cluster({"delay": 2, "jitter": 1, "loss": 1})[0] == "QoS Cukup - Delay Meningkat"
    assert label_cluster({"delay": 1, "jitter": 1, "loss": 1}) == ("QoS Baik", "-")
    assert label_cluster({"delay": 2, "jitter": 1, "loss": 1})[1] == "Pemantauan"
    assert label_cluster({"delay": 3, "jitter": 3, "loss": 3})[0] == (
        "QoS Buruk - Delay, Jitter dan Packet Loss Melebihi Batas"
    )
    assert label_cluster({"delay": 2, "jitter": 1, "loss": 1}, noise=True)[0] == "Anomali"


def test_join_metrics():
    assert join_metrics(["A"]) == "A"
    assert join_metrics(["A", "B"]) == "A dan B"
    assert join_metrics(["A", "B", "C"]) == "A, B dan C"


def test_cluster_levels_batas():
    th = CFG["thresholds"]
    assert cluster_levels({"mean_delay_ms": 150.0, "p999_jitter_ms": 50.0, "plr_pct": 1.0}, th) == {
        "delay": 1, "jitter": 1, "loss": 1,
    }
    assert cluster_levels({"mean_delay_ms": 400.0, "p999_jitter_ms": 50.1, "plr_pct": 1.01}, th) == {
        "delay": 2, "jitter": 3, "loss": 2,
    }
    assert cluster_levels({"mean_delay_ms": 150.0, "p999_jitter_ms": 50.0, "plr_pct": 3.01}, th)["loss"] == 3


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


def test_labeling_stats_dari_config():
    df = pd.DataFrame(
        {
            "delay_ms": [10.0, 20.0],
            "jitter_ms": [1.0, 3.0],
            "fwd_packets_count": [100, 100],
            "bwd_packets_count": [100, 90],
        }
    )
    s = labeling_stats(df, CFG["profiling"]["jitter_percentile"])
    assert s["mean_delay_ms"] == 15.0 and s["plr_pct"] == pytest.approx(5.0)
    assert s["p999_jitter_ms"] == pytest.approx(2.998)


def test_noise_ratio():
    assert noise_ratio(np.array([-1, 0, 0, 1])) == 0.25
    with pytest.raises(ValueError):
        noise_ratio(np.array([]))
