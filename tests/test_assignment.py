"""Uji penugasan Euclidean ke centroid dan tiga aturan batas "di luar klaster"."""
import numpy as np
import pytest

from qos.model.assignment import (
    NOISE_ID,
    assign,
    centroids_of,
    distance_limits,
    distance_matrix,
    format_distances,
    parse_distances,
)


@pytest.fixture
def two_clusters():
    """Klaster 1 padat di (0,0), klaster 2 longgar di (10,0), dua titik noise."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (60, 2)), rng.normal([10, 0], 1.0, (60, 2)), [[5.0, 5.0], [20.0, 20.0]]])
    labels = np.r_[np.ones(60), np.full(60, 2), NOISE_ID, NOISE_ID].astype(int)
    return X, labels


def test_centroid_rata_rata_anggota_noise_tanpa_centroid(two_clusters):
    X, labels = two_clusters
    c = centroids_of(X, labels)
    assert sorted(c) == [1, 2]  # nomor dari 1; noise tidak punya centroid
    assert c[1] == pytest.approx(X[labels == 1].mean(axis=0))


def test_assign_terdekat_dan_jarak_euclidean():
    cents = {1: np.array([0.0, 0.0]), 2: np.array([10.0, 0.0])}
    a = assign(np.array([[1.0, 0.0], [8.0, 0.0], [3.0, 4.0]]), cents)
    assert a.nearest.tolist() == [1, 2, 1]
    assert a.nearest_dist.tolist() == pytest.approx([1.0, 2.0, 5.0])
    assert a.dist.shape == (3, 2) and a.ids.tolist() == [1, 2]
    assert not a.outside.any()  # tanpa batas: tidak pernah di luar klaster
    assert distance_matrix(np.array([[3.0, 4.0]]), cents)[1][0, 0] == pytest.approx(5.0)


def test_aturan_batas_tiga_varian(two_clusters):
    X, labels = two_clusters
    cents = centroids_of(X, labels)
    per, glob = distance_limits(X, labels, cents, 95, "per_cluster")
    own = {c: np.linalg.norm(X[labels == c] - cents[c], axis=1) for c in cents}
    assert per[1] == pytest.approx(np.percentile(own[1], 95)) and per[2] == pytest.approx(np.percentile(own[2], 95))
    assert per[1] < per[2]
    assert glob == pytest.approx(np.percentile(np.concatenate(list(own.values())), 95))
    g, _ = distance_limits(X, labels, cents, 95, "global")
    assert g == {1: pytest.approx(glob), 2: pytest.approx(glob)}
    m, _ = distance_limits(X, labels, cents, 95, "max_cluster_global")
    assert m[1] == pytest.approx(max(per[1], glob)) and m[2] == pytest.approx(max(per[2], glob))
    assert m[1] >= per[1] and m[2] >= per[2]  # aturan (c) tidak pernah lebih ketat dari (a)


def test_aturan_batas_tak_dikenal_ditolak(two_clusters):
    X, labels = two_clusters
    with pytest.raises(ValueError, match="tidak dikenal"):
        distance_limits(X, labels, centroids_of(X, labels), 95, "acak")


def test_di_luar_klaster_jika_melebihi_batas_semua_centroid(two_clusters):
    X, labels = two_clusters
    cents = centroids_of(X, labels)
    limits, _ = distance_limits(X, labels, cents, 95, "max_cluster_global")
    probe = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0], [20.0, 20.0]])
    a = assign(probe, cents, limits)
    assert a.outside.tolist() == [False, False, True, True]
    assert a.nearest.tolist()[:2] == [1, 2]  # klaster terdekat tetap dilaporkan walau di luar klaster
    # dekat salah satu centroid cukup untuk tidak "jauh dari semua klaster"
    edge = np.array([[cents[2][0] + limits[2] * 0.9, cents[2][1]]])
    assert not assign(edge, cents, limits).outside[0]


def test_aturan_c_menandai_lebih_sedikit_non_noise(two_clusters):
    X, labels = two_clusters
    cents = centroids_of(X, labels)
    flagged = {}
    for rule in ("per_cluster", "max_cluster_global"):
        lim, _ = distance_limits(X, labels, cents, 95, rule)
        flagged[rule] = assign(X[labels != NOISE_ID], cents, lim).outside.sum()
    assert flagged["max_cluster_global"] <= flagged["per_cluster"]


def test_format_dan_parse_jarak():
    text = format_distances(np.array([1, 2, 10]), np.array([0.1234, 1.4, 12.0]))
    assert text == "1:0.123;2:1.400;10:12.000"
    assert parse_distances(text) == {1: 0.123, 2: 1.4, 10: 12.0}
    assert parse_distances("") == {} and parse_distances(float("nan")) == {}
