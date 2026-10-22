"""Uji fungsi grafik Plotly (tanpa Streamlit)."""
import numpy as np
import pandas as pd
import pytest

from qos.config import load_config
from qos.viz.plots import LABEL_COLORS, boxplot_fig, cluster_name, label_color, labeling_bars, pca_scatter

TH = load_config(use_user=False)["thresholds"]


def _box(q1, med, q3, lo, hi, out=()):
    return {"q1": q1, "median": med, "q3": q3, "iqr": q3 - q1, "lower_fence": 0, "upper_fence": 1,
            "whisker_low": lo, "whisker_high": hi, "outliers": list(out)}


def _cluster(cid, label, mean, p999, plr):
    return {
        "id_klaster": cid, "jumlah_anggota": 5, "label": label,
        "statistik_pelabelan": {"mean_delay_ms": mean, "p999_jitter_ms": p999, "plr_pct": plr},
        "boxplot": {m: _box(1, 2, 3, 0.5, 3.5, [9.0]) for m in ("delay_ms", "jitter_ms", "fai_pct")},
    }


PROFILE = [
    _cluster(-1, "Anomali - Delay Ekstrem", 900.0, 100.0, 5.0),
    _cluster(0, "QoS Baik", 10.0, 5.0, 0.0),
    _cluster(1, "QoS Buruk - Packet Loss Melebihi Batas", 20.0, 6.0, 1.0),
]
LABELS = {int(c["id_klaster"]): c["label"] for c in PROFILE}


def test_warna_label_sesuai_prd():
    assert LABEL_COLORS["QoS Baik"] == "#2563EB"
    assert LABEL_COLORS["QoS Cukup"] == "#EAB308"
    assert LABEL_COLORS["QoS Buruk"] == "#F97316"
    assert LABEL_COLORS["Anomali"] == "#DC2626"
    assert label_color("QoS Buruk - Jitter Melebihi Batas") == "#F97316"
    assert label_color("Anomali") == "#DC2626"


def test_cluster_name():
    assert cluster_name(-1) == "Noise" and cluster_name(3) == "K3"


def test_pca_scatter_latar_pudar_dan_live_berwarna():
    bg = pd.DataFrame({"pc1": [0, 1, 2], "pc2": [0, 1, 2], "cluster_id": [-1, 0, 1]})
    live = pd.DataFrame(
        {"pc1": [0.5, np.nan], "pc2": [0.5, np.nan], "label": ["QoS Baik", "Data Belum Cukup"], "flow_id": ["a", "b"],
         "delay_ms": [1.0, 2.0], "jitter_ms": [1.0, 2.0], "fai_pct": [0.0, 0.0], "membership_prob": [0.9, np.nan]}
    )
    fig = pca_scatter(bg, LABELS, live)
    latih = [t for t in fig.data if t.name.startswith("Latih")]
    liv = [t for t in fig.data if t.name.startswith("Live")]
    assert len(latih) == 3 and all(t.marker.opacity < 0.5 for t in latih)
    assert len(liv) == 1 and liv[0].marker.color == "#2563EB"  # baris tanpa PC tidak digambar
    assert len(liv[0].x) == 1


def test_pca_scatter_tanpa_live():
    bg = pd.DataFrame({"pc1": [0.0], "pc2": [0.0], "cluster_id": [0]})
    assert all(t.name.startswith("Latih") for t in pca_scatter(bg, LABELS, None).data)


def test_labeling_bars_tiga_panel_dan_garis_ambang():
    fig = labeling_bars(PROFILE, TH)
    assert len(fig.data) == 3
    assert list(fig.data[0].x) == ["Noise", "K0", "K1"]
    assert list(fig.data[0].marker.color) == ["#DC2626", "#2563EB", "#F97316"]
    ys = sorted(s["y0"] for s in fig.layout.shapes)
    assert ys == sorted([TH["delay_good_ms"], TH["delay_max_ms"], TH["jitter_max_ms"], TH["loss_good_pct"], TH["loss_max_pct"]])


@pytest.mark.parametrize(
    "metric,expected",
    [("delay_ms", [TH["delay_good_ms"], TH["delay_max_ms"]]), ("jitter_ms", [TH["jitter_max_ms"]]), ("fai_pct", [TH["loss_good_pct"], TH["loss_max_pct"]])],
)
def test_boxplot_dari_statistik_tersimpan(metric, expected):
    fig = boxplot_fig(PROFILE, metric, TH)
    box = fig.data[0]
    assert box.type == "box"
    assert list(box.q1) == [1, 1, 1] and list(box.median) == [2, 2, 2] and list(box.q3) == [3, 3, 3]
    assert list(box.lowerfence) == [0.5] * 3 and list(box.upperfence) == [3.5] * 3
    assert sorted(s["y0"] for s in fig.layout.shapes) == sorted(expected)
    marker = fig.data[1]
    assert "pencilan 1" in marker.text[0]
    assert fig.layout.yaxis.type is None


def test_boxplot_penanda_dan_skala_log():
    fig = boxplot_fig(PROFILE, "delay_ms", TH, log=True)
    assert list(fig.data[1].y) == [900.0, 10.0, 20.0]  # mean delay
    assert fig.layout.yaxis.type == "log"
    assert list(boxplot_fig(PROFILE, "jitter_ms", TH).data[1].y) == [100.0, 5.0, 6.0]  # P99,9 jitter
    assert list(boxplot_fig(PROFILE, "fai_pct", TH).data[1].y) == [5.0, 0.0, 1.0]  # PLR


def test_boxplot_skala_log_memakai_rentang_nilai_positif():
    prof = [_cluster(0, "QoS Baik", 10.0, 5.0, 0.0), _cluster(1, "QoS Baik", 1000.0, 5.0, 0.0)]
    for c in prof:
        c["boxplot"]["fai_pct"] = _box(0.0, 0.0, 0.0, 0.0, 0.0)  # semua nol (mis. FAI klaster tanpa loss)
    fig = boxplot_fig(prof, "delay_ms", TH, log=True)
    lo, hi = fig.layout.yaxis.range
    assert fig.layout.yaxis.type == "log" and lo < 0.5 and 3 < hi < 4  # terbatas, tidak meledak ke 10^153
    zeros = boxplot_fig(prof, "fai_pct", TH, log=True)  # hanya ambang 0,1 yang positif
    assert zeros.layout.yaxis.type == "log" and zeros.layout.yaxis.range is not None


def test_plot_profil_lama_tanpa_loss_good():
    old = {k: v for k, v in TH.items() if k != "loss_good_pct"}
    fig = labeling_bars(PROFILE, old)
    ys = sorted(s["y0"] for s in fig.layout.shapes)
    assert ys == sorted([old["delay_good_ms"], old["delay_max_ms"], old["jitter_max_ms"], old["loss_max_pct"]])
    assert boxplot_fig(PROFILE, "fai_pct", old).layout.shapes[0]["y0"] == old["loss_max_pct"]
