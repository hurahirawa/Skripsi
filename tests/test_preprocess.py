"""Uji filter sesi, pengecualian traffic type, FAI, dan sensitivitas."""
import numpy as np
import pandas as pd
import pytest

from qos.config import load_config
from qos.data.eda import feature_summary, sensitivity
from qos.features.preprocess import apply_session_filters, compute_fai, exclude_types, session_mask


@pytest.fixture
def flows():
    return pd.DataFrame(
        {
            "duration_us": [100.0, 200.0, 120e6, 119_999_500.0, 50.0],
            "iat_std_us": [0.0, 5.0, 0.0, 9.0, 0.0],
            "fwd_packets_count": [1, 10, 20, 50, 5],
            "bwd_packets_count": [0, 12, 20, 40, 5],
            "protocol": [17, 17, 6, 17, 17],
            "traffic_type": ["Tor", "VPN", "NonVPN", "tor", "VPN"],
        }
    )


def test_config_kunci_filter_sesi_final():
    cfg = load_config(use_user=False)
    assert cfg["data"]["min_session"] == {"fwd": 5, "bwd": 5}
    assert cfg["data"]["exclude_traffic_types"] == ["Tor"]
    assert cfg["data"]["voip_filter"]["min_flows_warning"] == 200
    assert cfg["hdbscan"]["grid_search"]["min_cluster_size"] == [3, 5, 8, 10, 15, 20, 30, 50]
    assert cfg["features"]["fai_clip_negative"] is True


def test_session_mask(flows):
    assert session_mask(flows, None, None).all()
    assert session_mask(flows, 10, 10).tolist() == [False, True, True, True, False]
    assert session_mask(flows, 5, None).sum() == 4


def test_exclude_types_case_insensitive(flows):
    assert exclude_types(flows, ["Tor"])["traffic_type"].tolist() == ["VPN", "NonVPN", "VPN"]
    assert len(exclude_types(flows, [])) == len(flows)
    with pytest.raises(ValueError):
        exclude_types(flows.drop(columns="traffic_type"), ["Tor"])


def test_apply_session_filters(flows):
    cfg = load_config(use_user=False)["data"]
    cfg["min_session"] = {"fwd": None, "bwd": None}
    cfg["exclude_traffic_types"] = []
    assert len(apply_session_filters(flows, cfg)) == len(flows)  # nonaktif
    cfg["min_session"] = {"fwd": 10, "bwd": 10}
    cfg["exclude_traffic_types"] = ["tor"]
    assert apply_session_filters(flows, cfg)["fwd_packets_count"].tolist() == [10, 20]


def test_compute_fai():
    fwd = pd.Series([150, 220, 10, 0])
    bwd = pd.Series([148, 175, 12, 3])
    raw = compute_fai(fwd, bwd, clip_negative=False)
    assert raw.iloc[0] == pytest.approx(1.33, abs=0.005)
    assert raw.iloc[1] == pytest.approx(20.45, abs=0.005)
    assert raw.iloc[2] == pytest.approx(-20.0)
    assert np.isnan(raw.iloc[3])
    clipped = compute_fai(fwd, bwd, clip_negative=True)
    assert clipped.iloc[2] == 0
    assert clipped.iloc[1] == pytest.approx(20.45, abs=0.005)
    assert np.isnan(clipped.iloc[3])


def test_feature_summary(flows):
    s = feature_summary(flows, cap_us=120e6, tol_us=1000.0, clip_negative=True)
    assert s["n"] == 5
    assert s["duration_us_median"] == 200.0
    assert s["iat_nol_pct"] == pytest.approx(60.0)
    assert s["durasi_batas_pct"] == pytest.approx(40.0)  # 120e6 dan 119_999_500
    assert s["fai_pct_median"] == pytest.approx(0.0)  # setelah clip: [100, 0, 0, 20, 0]


def test_feature_summary_kosong(flows):
    s = feature_summary(flows.iloc[0:0], 120e6, 1000.0, True)
    assert s["n"] == 0 and np.isnan(s["duration_us_median"])


def test_sensitivity(flows):
    res = sensitivity(flows, [1, 10], ["Tor"], 120e6, 1000.0, True)
    j = res["jumlah"]
    assert j.loc[1, "total"] == 4 and j.loc[1, "total_tanpa"] == 3
    assert j.loc[10, "total"] == 3 and j.loc[10, "total_tanpa"] == 2
    assert j.loc[10, "VPN"] == 1 and j.loc[10, "NonVPN"] == 1
    assert res["protokol"].loc[10, "proto_17"] == 2 and res["protokol"].loc[10, "proto_6"] == 1
    assert res["statistik_semua"].loc[10, "n"] == 3
    assert res["statistik_tanpa"].loc[10, "n"] == 2
