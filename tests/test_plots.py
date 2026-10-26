"""Uji fungsi grafik Plotly (tanpa Streamlit)."""
import numpy as np
import pandas as pd
import pytest

from qos.config import load_config
from qos.profiling.describe import describe
from qos.viz.plots import (
    CLUSTER_PALETTE,
    NEUTRAL_COLOR,
    OUTSIDE_COLOR,
    boxplot_fig,
    cluster_color,
    cluster_name,
    distance_bars,
    grid_heatmaps_fig,
    kmeans_curves_fig,
    median_bars,
    pca_scatter,
    reference_lines,
    state_color,
)

TH = load_config(use_user=False)["thresholds"]


def _box(q1, med, q3, lo, hi, out=()):
    return {"q1": q1, "median": med, "q3": q3, "iqr": q3 - q1, "lower_fence": 0, "upper_fence": 1,
            "whisker_low": lo, "whisker_high": hi, "outliers": list(out)}


def _cluster(cid, delay, jitter, fai):
    stats = {m: {"mean": v, "median": v, "std": 1.0, "min": v, "max": v} for m, v in (("delay_ms", delay), ("jitter_ms", jitter), ("fai_pct", fai))}
    return {
        "id_klaster": cid, "nama": cluster_name(cid), "jumlah_anggota": 5, "statistik": stats, "deskripsi": describe(delay, jitter, fai, TH),
        "boxplot": {m: _box(1, 2, 3, 0.5, 3.5, [9.0]) for m in ("delay_ms", "jitter_ms", "fai_pct")},
    }


PROFILE = [_cluster(-1, 900.0, 100.0, 5.0), _cluster(1, 10.0, 5.0, 0.0), _cluster(2, 20.0, 6.0, 1.0)]


def test_palet_kategorikal_tanpa_makna_baik_buruk():
    assert len(CLUSTER_PALETTE) >= 5 and len(set(CLUSTER_PALETTE)) == len(CLUSTER_PALETTE)  # 5 klaster tetap berbeda warna
    assert cluster_color(1) == CLUSTER_PALETTE[0] and cluster_color(5) == CLUSTER_PALETTE[4]
    assert cluster_color(len(CLUSTER_PALETTE) + 1) == CLUSTER_PALETTE[0]  # berulang bila klaster banyak
    assert cluster_color(-1) == OUTSIDE_COLOR and OUTSIDE_COLOR not in CLUSTER_PALETTE and NEUTRAL_COLOR not in CLUSTER_PALETTE
    assert state_color("Klaster 2") == CLUSTER_PALETTE[1]
    assert state_color("Di luar klaster (terdekat: Klaster 2)") == OUTSIDE_COLOR and state_color("Di luar klaster") == OUTSIDE_COLOR
    assert state_color("Data Belum Cukup") == NEUTRAL_COLOR and state_color("Sinyal SIP") == NEUTRAL_COLOR


def test_cluster_name_dimulai_dari_1():
    assert cluster_name(-1) == "Noise" and cluster_name(1) == "Klaster 1"


def test_pca_scatter_latar_pudar_live_berwarna_dan_centroid_besar():
    bg = pd.DataFrame({"pc1": [0, 1, 2], "pc2": [0, 1, 2], "cluster_id": [-1, 1, 2]})
    live = pd.DataFrame(
        {"pc1": [0.5, 0.7, 5.0, np.nan], "pc2": [0.5, 0.6, 5.0, np.nan],
         "status": ["Terklaster", "Terklaster", "Di luar klaster", "Data Belum Cukup"], "klaster_terdekat": [1, 1, 2, pd.NA],
         "flow_id": list("abcd"), "jarak_terdekat": [0.1, 0.2, 9.0, np.nan], "delay_ms": [1.0] * 4, "jitter_ms": [1.0] * 4, "fai_pct": [0.0] * 4}
    )
    fig = pca_scatter(bg, live, {1: (0.0, 0.0), 2: (1.0, 1.0)})
    latih = [t for t in fig.data if t.name.startswith("Latih")]
    liv = {t.name: t for t in fig.data if t.name.startswith("Live")}
    assert len(latih) == 3 and all(t.marker.opacity < 0.5 for t in latih)
    assert set(liv) == {"Live · Klaster 1", "Live · Di luar klaster"}  # baris tanpa PC tidak digambar
    assert liv["Live · Klaster 1"].marker.color == CLUSTER_PALETTE[0] and len(liv["Live · Klaster 1"].x) == 2
    assert liv["Live · Di luar klaster"].marker.color == OUTSIDE_COLOR and liv["Live · Di luar klaster"].marker.symbol == "x"
    cent = next(t for t in fig.data if t.name == "Centroid")
    assert cent.marker.size >= 18 and list(cent.text) == ["K1", "K2"] and list(cent.marker.color) == list(CLUSTER_PALETTE[:2])


def test_pca_scatter_tanpa_live():
    bg = pd.DataFrame({"pc1": [0.0], "pc2": [0.0], "cluster_id": [1]})
    assert all(t.name.startswith("Latih") for t in pca_scatter(bg, None).data)


def test_median_bars_tiga_panel_dan_garis_ambang():
    fig = median_bars(PROFILE, TH)
    assert len(fig.data) == 3
    assert list(fig.data[0].x) == ["Noise", "Klaster 1", "Klaster 2"]
    assert list(fig.data[0].marker.color) == [OUTSIDE_COLOR, CLUSTER_PALETTE[0], CLUSTER_PALETTE[1]]
    ys = sorted(s["y0"] for s in fig.layout.shapes)
    assert ys == sorted([TH["delay_good_ms"], TH["delay_max_ms"], TH["jitter_max_ms"], TH["loss_good_pct"], TH["loss_max_pct"]])


@pytest.mark.parametrize(
    "metric,expected",
    [("delay_ms", [TH["delay_good_ms"], TH["delay_max_ms"]]), ("jitter_ms", [TH["jitter_max_ms"]]), ("fai_pct", [TH["loss_good_pct"], TH["loss_max_pct"]])],
)
def test_boxplot_per_klaster_dari_statistik_tersimpan(metric, expected):
    fig = boxplot_fig(PROFILE, metric, TH)
    boxes = [t for t in fig.data if t.type == "box"]
    assert len(boxes) == 3 and [b.x[0] for b in boxes] == ["Noise", "Klaster 1", "Klaster 2"]
    assert all(list(b.q1) == [1] and list(b.median) == [2] and list(b.q3) == [3] for b in boxes)
    assert all(list(b.lowerfence) == [0.5] and list(b.upperfence) == [3.5] for b in boxes)
    assert [b.marker.color for b in boxes] == [OUTSIDE_COLOR, CLUSTER_PALETTE[0], CLUSTER_PALETTE[1]]
    assert sorted(s["y0"] for s in fig.layout.shapes) == sorted(expected) == sorted(reference_lines(metric, TH))
    assert fig.layout.yaxis.type is None


def test_boxplot_penanda_mean_dan_skala_log():
    fig = boxplot_fig(PROFILE, "delay_ms", TH, log=True)
    mean = next(t for t in fig.data if t.name == "Mean")
    assert list(mean.y) == [900.0, 10.0, 20.0]
    assert fig.layout.yaxis.type == "log"


def test_boxplot_skala_log_memakai_rentang_nilai_positif():
    prof = [_cluster(1, 10.0, 5.0, 0.0), _cluster(2, 1000.0, 5.0, 0.0)]
    for c in prof:
        c["boxplot"]["fai_pct"] = _box(0.0, 0.0, 0.0, 0.0, 0.0)  # semua nol (mis. FAI klaster tanpa loss)
    lo, hi = boxplot_fig(prof, "delay_ms", TH, log=True).layout.yaxis.range
    assert lo < 0.5 and 3 < hi < 4  # terbatas, tidak meledak ke 10^153
    zeros = boxplot_fig(prof, "fai_pct", TH, log=True)
    assert zeros.layout.yaxis.type == "log" and zeros.layout.yaxis.range is not None


def test_distance_bars_satu_bar_per_klaster_dan_batas():
    fig = distance_bars({1: 0.2, 2: 1.5, 3: 9.0}, {1: 0.5, 2: 0.6, 3: 0.4}, nearest=1)
    bar = fig.data[0]
    assert list(bar.y) == ["Klaster 1", "Klaster 2", "Klaster 3"] and list(bar.x) == [0.2, 1.5, 9.0]
    assert list(bar.marker.line.width) == [2, 0, 0]  # klaster terdekat disorot
    assert list(fig.data[1].x) == [0.5, 0.6, 0.4]  # tanda batas
    assert len(distance_bars({1: 0.2}).data) == 1


def test_kmeans_curves_dan_heatmap_grid():
    sweep = pd.DataFrame({"k": [2, 3, 4], "inertia": [9.0, 5.0, 4.0], "silhouette": [0.5, 0.7, 0.6], "davies_bouldin": [0.9, 0.5, 0.6],
                          "chi": [10.0, 30.0, 25.0]})
    fig = kmeans_curves_fig(sweep, k_n=4, k_opt=3)
    assert len(fig.data) == 4 and len(fig.layout.shapes) == 8  # dua garis k pada tiap panel
    grid = pd.DataFrame({"min_cluster_size": [3, 3, 5, 5], "min_samples": [1, 2, 1, 2], "n_clusters": [4, 3, 3, 2],
                         "noise_ratio": [0.1] * 4, "silhouette": [0.8] * 4, "dbcv": [0.7] * 4, "ari_mean": [0.9, 0.95, 1.0, 0.8],
                         "davies_bouldin": [0.3] * 4})
    hm = grid_heatmaps_fig(grid, best=(3, 2))
    assert len(hm.data) == 6 and any(s["line"]["color"] == "#DC2626" for s in hm.layout.shapes)  # kotak merah = terpilih
    assert hm.data[4].z[1][0] == 1.0 and hm.data[4].z[1][1] == 0.8 and list(hm.data[0].x) == ["1", "2"]
